"""SPADE LLM coding multi-agent system — full implementation."""

from __future__ import annotations

import asyncio
import time
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

import spade
from spade.agent import Agent
from spade.message import Message
from spade.behaviour import OneShotBehaviour

from spade_llm import LLMAgent, LLMProvider

from coding_scenario.base import (
    CodingMASBase,
    ORCHESTRATION_GAP_MS_KEY,
    EVENT_CONCLUDE,
    EVENT_DECIDE_EXECUTION,
    EVENT_EXECUTE_CODE,
    EVENT_GENERATE_CODE,
    EVENT_PASS_TO_WRITER,
    EVENT_RECEIVE_TASK,
    EVENT_REDIRECT_WRITER,
    EVENT_SAFEGUARD_REVIEW,
    EVENT_SAFEGUARD_RULE_BLOCK,
    EVENT_SEND_CODE_TO_SAFEGUARD,
    EVENT_SKIP_EXECUTION,
    EVENT_WRITER_INTERPRET,
    PHASE_COORDINATION,
    PHASE_EXECUTION,
    PHASE_FINALIZATION,
    PHASE_GENERATION,
    WorkflowState,
)
from utils import (
    extract_code,
    execute_python,
    rule_based_safeguard,
    safe_json_parse,
)

load_dotenv()


class CodingRunContext:
    def __init__(self, mas, query, max_iterations):
        self.mas = mas
        self.user_query = query
        self.memory = mas.memory.copy()
        self.max_iterations = max_iterations
        self.attempt = 0
        self.commander_context = ""
        self.writer_code = ""
        self.writer_notes = ""
        self.safeguard_allowed = False
        self.safeguard_reason = ""
        self.requires_execution = True
        self.execution_output = ""
        self.execution_error = ""
        self.writer_interpretation = ""
        self.final_answer = ""
        self.finished = False
        self.completion_event = asyncio.Event()

    def to_workflow_state(self) -> WorkflowState:
        return {
            "user_query": self.user_query,
            "memory": self.memory,
            "max_iterations": self.max_iterations,
            "attempt": self.attempt,
            "commander_context": self.commander_context,
            "writer_code": self.writer_code,
            "writer_notes": self.writer_notes,
            "safeguard_allowed": self.safeguard_allowed,
            "safeguard_reason": self.safeguard_reason,
            "requires_execution": self.requires_execution,
            "execution_output": self.execution_output,
            "execution_error": self.execution_error,
            "writer_interpretation": self.writer_interpretation,
            "final_answer": self.final_answer,
            "finished": self.finished,
        }


class OrchestratorAgent(Agent):
    """The orchestrator agent drives the MAS execution."""
    def __init__(self, jid, password, ctx, run_workflow_coro, verify_security=False):
        super().__init__(jid, password, verify_security=verify_security)
        self.ctx = ctx
        self.run_workflow_coro = run_workflow_coro

    async def setup(self):
        class WorkflowBehav(OneShotBehaviour):
            async def run(self):
                await self.agent.run_workflow_coro(self)
        self.add_behaviour(WorkflowBehav())


class SpadeCodingMAS(CodingMASBase):
    """Spade Coding MAS full implementation using spade_llm"""
    def __init__(self, model_id: str, max_iterations: int) -> None:
        super().__init__(model_id, max_iterations)
        self._loop = asyncio.new_event_loop()
        self._between_steps_ms: float = 0.0
        self._last_step_end_mono: Optional[float] = None
        self._run_prompt_tokens: int = 0
        self._run_completion_tokens: int = 0
        self._run_total_tokens: int = 0

    def _reset_run_metrics(self) -> None:
        self._between_steps_ms = 0.0
        self._last_step_end_mono = None
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_total_tokens = 0

    def _accum_between_step_gap(self) -> None:
        if self._last_step_end_mono is None:
            return
        self._between_steps_ms += (time.perf_counter() - self._last_step_end_mono) * 1000.0

    def _mark_step_end(self) -> None:
        self._last_step_end_mono = time.perf_counter()

    def _record_usage(self, usage: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        self._run_prompt_tokens += int(usage.get("prompt_tokens", 0))
        self._run_completion_tokens += int(usage.get("completion_tokens", 0))
        self._run_total_tokens += int(usage.get("total_tokens", 0))
        return usage

    async def _workflow_logic(self, behav: OneShotBehaviour):
        ctx = behav.agent.ctx
        mas = ctx.mas

        # Step 1: receive task
        t0 = time.perf_counter()
        ctx.execution_output = ""
        ctx.execution_error = ""
        ctx.writer_interpretation = ""
        ctx.safeguard_allowed = False
        ctx.safeguard_reason = ""
        ctx.requires_execution = True
        dur_ms = (time.perf_counter() - t0) * 1000.0
        mas.log_conversation_event(
            event_type="action", actor="Commander", event_name=EVENT_RECEIVE_TASK,
            phase=PHASE_COORDINATION, duration_ms=dur_ms, payload={"step": EVENT_RECEIVE_TASK, "attempt": ctx.attempt}
        )
        mas._mark_step_end()

        mas._accum_between_step_gap()

        # Step 2: Pass to writer (Ask Commander for instructions)
        t_node = time.perf_counter()
        memory_text = "\n".join(f"- {item}" for item in ctx.memory[-5:]) or "(none)"
        human = (
            f"User query:\n{ctx.user_query}\n\n"
            f"Prior interaction memory:\n{memory_text}\n\n"
            "Produce clear instructions the Writer should follow (assumptions, deliverable: executable Python when coding is needed). "
            "If the user asks for a script to print a specific value, instruct the Writer to make sure the script prints EXACTLY the final answer without any additional text, labels, or formatting.\n"
            "Plain text only."
        )

        msg = Message(to="commander@localhost")
        msg.set_metadata("message_type", "llm")
        msg.body = human
        t_llm = time.perf_counter()
        await behav.send(msg)
        reply = await behav.receive(timeout=120)
        llm_ms = (time.perf_counter() - t_llm) * 1000.0

        ctx.commander_context = reply.body if reply else ""
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="pass", actor="Commander", target="Writer", event_name=EVENT_PASS_TO_WRITER,
            phase=PHASE_COORDINATION, duration_ms=node_ms, llm_api_duration_ms=llm_ms,
            payload={"step": EVENT_PASS_TO_WRITER, "commander_context": ctx.commander_context}
        )
        mas._mark_step_end()

        while not ctx.finished:
            mas._accum_between_step_gap()
            if ctx.attempt >= ctx.max_iterations:
                break
                
            # Step 3: Writer generates code
            t_node = time.perf_counter()
            writer_prompt = (
                f"User query:\n{ctx.user_query}\n\n"
                f"Commander instructions (passed from Commander to you):\n{ctx.commander_context}\n\n"
                "Return JSON only with keys: code, notes."
            )
            msg = Message(to="writer@localhost")
            msg.set_metadata("message_type", "llm")
            msg.body = writer_prompt
            t_llm = time.perf_counter()
            await behav.send(msg)
            reply = await behav.receive(timeout=120)
            llm_ms = (time.perf_counter() - t_llm) * 1000.0
            
            parsed = safe_json_parse(reply.body if reply else "{}")
            code = extract_code(str(parsed.get("code", "")))
            if not code:
                code = "print('Unable to generate valid code for this query.')"
            ctx.writer_code = code
            ctx.writer_notes = str(parsed.get("notes", "No additional notes."))
            node_ms = (time.perf_counter() - t_node) * 1000.0
            
            mas.log_conversation_event(
                event_type="agent_output", actor="Writer", target="Commander",
                event_name=EVENT_GENERATE_CODE, phase=PHASE_GENERATION,
                duration_ms=node_ms, llm_api_duration_ms=llm_ms,
                payload={"step": EVENT_GENERATE_CODE, "writer_notes": ctx.writer_notes, "writer_code": ctx.writer_code}
            )
            mas._mark_step_end()
            mas._accum_between_step_gap()

            # Step 4: Pass to safeguard
            t0 = time.perf_counter()
            ctx.execution_output = ""
            ctx.execution_error = ""
            dur_ms = (time.perf_counter() - t0) * 1000.0
            mas.log_conversation_event(
                event_type="pass", actor="Commander", target="Safeguard", event_name=EVENT_SEND_CODE_TO_SAFEGUARD,
                phase=PHASE_COORDINATION, duration_ms=dur_ms, payload={"step": EVENT_SEND_CODE_TO_SAFEGUARD, "writer_code": ctx.writer_code}
            )
            mas._mark_step_end()
            mas._accum_between_step_gap()

            # Step 5: Safeguard reviews code
            t_node = time.perf_counter()
            code = ctx.writer_code
            allowed, reason = rule_based_safeguard(code)
            if not allowed:
                t0 = time.perf_counter()
                ctx.safeguard_allowed = False
                ctx.safeguard_reason = reason
                dur_ms = (time.perf_counter() - t0) * 1000.0
                mas.log_conversation_event(
                    event_type="agent_output", actor="Safeguard", target="Commander", event_name=EVENT_SAFEGUARD_RULE_BLOCK,
                    phase=PHASE_COORDINATION, duration_ms=dur_ms, payload={"step": EVENT_SAFEGUARD_RULE_BLOCK, "allow": False, "reason": reason}
                )
                mas._mark_step_end()
            else:
                safeguard_prompt = (
                    f"Commander context (for awareness only):\n{ctx.commander_context[:1200]}\n\n"
                    f"Code to review:\n```python\n{code}\n```\n"
                    "Return strict JSON: {\"allow\": boolean, \"reason\": string}."
                )
                msg = Message(to="safeguard@localhost")
                msg.set_metadata("message_type", "llm")
                msg.body = safeguard_prompt
                t_llm = time.perf_counter()
                await behav.send(msg)
                reply = await behav.receive(timeout=120)
                llm_ms = (time.perf_counter() - t_llm) * 1000.0
                
                parsed = safe_json_parse(reply.body if reply else "{}")
                ctx.safeguard_allowed = bool(parsed.get("allow", False))
                ctx.safeguard_reason = str(parsed.get("reason", "No reason provided."))
                node_ms = (time.perf_counter() - t_node) * 1000.0
                
                mas.log_conversation_event(
                    event_type="agent_output", actor="Safeguard", target="Commander", event_name=EVENT_SAFEGUARD_REVIEW,
                    phase=PHASE_COORDINATION, duration_ms=node_ms, llm_api_duration_ms=llm_ms,
                    payload={"step": EVENT_SAFEGUARD_REVIEW, "allow": ctx.safeguard_allowed, "reason": ctx.safeguard_reason}
                )
                mas._mark_step_end()

            mas._accum_between_step_gap()

            if not ctx.safeguard_allowed:
                pass # Trigger redirect
            else:
                # Decide execution
                q = ctx.user_query
                if "Complete the following Python code" in q:
                    t0 = time.perf_counter()
                    ctx.requires_execution = False
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    mas.log_conversation_event(
                        event_type="action", actor="Commander", event_name=EVENT_DECIDE_EXECUTION,
                        phase=PHASE_COORDINATION, duration_ms=dur_ms, payload={"step": EVENT_DECIDE_EXECUTION, "requires_execution": False, "reason": "humaneval_completion_prompt"}
                    )
                    mas._mark_step_end()
                else:
                    t_node = time.perf_counter()
                    decide_prompt = (
                        f"User query:\n{q}\n\nWriter code:\n```python\n{ctx.writer_code}\n```\n"
                        "Decide whether running the Writer's code is required to answer the user.\n"
                        "CRITICAL: If the user explicitly asks for a script to 'calculate', 'read a file', 'print a result', "
                        "or execute any logic, execution IS REQUIRED. "
                        "Only skip execution if the user purely wants to read the source code and does not care about the output.\n"
                        "Return strict JSON: {\"requires_execution\": true|false, \"rationale\": string}."
                    )
                    msg = Message(to="commander@localhost")
                    msg.set_metadata("message_type", "llm")
                    msg.body = decide_prompt
                    t_llm = time.perf_counter()
                    await behav.send(msg)
                    reply = await behav.receive(timeout=120)
                    llm_ms = (time.perf_counter() - t_llm) * 1000.0
                    
                    parsed = safe_json_parse(reply.body if reply else "{}")
                    ctx.requires_execution = bool(parsed.get("requires_execution", True))
                    node_ms = (time.perf_counter() - t_node) * 1000.0
                    
                    mas.log_conversation_event(
                        event_type="action", actor="Commander", event_name=EVENT_DECIDE_EXECUTION,
                        phase=PHASE_COORDINATION, duration_ms=node_ms, llm_api_duration_ms=llm_ms,
                        payload={"step": EVENT_DECIDE_EXECUTION, "requires_execution": ctx.requires_execution}
                    )
                    mas._mark_step_end()

                mas._accum_between_step_gap()

                if ctx.requires_execution:
                    t0 = time.perf_counter()
                    output, error = execute_python(ctx.writer_code)
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    ctx.execution_output = output
                    ctx.execution_error = error
                    mas.log_conversation_event(
                        event_type="action", actor="Commander", event_name=EVENT_EXECUTE_CODE, phase=PHASE_EXECUTION, duration_ms=dur_ms,
                        payload={"step": EVENT_EXECUTE_CODE, "execution_output": output, "execution_error": error}
                    )
                    mas._mark_step_end()
                else:
                    t0 = time.perf_counter()
                    ctx.execution_output = "(Execution not run: Commander determined it was not required.)"
                    ctx.execution_error = ""
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    mas.log_conversation_event(
                        event_type="action", actor="Commander", event_name=EVENT_SKIP_EXECUTION, phase=PHASE_EXECUTION, duration_ms=dur_ms,
                        payload={"step": EVENT_SKIP_EXECUTION}
                    )
                    mas._mark_step_end()

                mas._accum_between_step_gap()

                # Interpret
                if ctx.execution_error and ctx.requires_execution:
                    ctx.writer_interpretation = ""
                else:
                    t_node = time.perf_counter()
                    interpret_prompt = (
                        f"User query:\n{ctx.user_query}\n\n"
                        f"Writer notes from coding step:\n{ctx.writer_notes}\n\n"
                        f"Stdout / summarized output:\n{ctx.execution_output}\n\n"
                        f"Execution error line (if any):\n{ctx.execution_error or '(none)'}\n"
                        "Interpret execution outcomes for the user query. Give the substantive technical answer the user needs (2–5 sentences). Mention uncertainty if assumptions were required."
                    )
                    msg = Message(to="writer@localhost")
                    msg.set_metadata("message_type", "llm")
                    msg.body = interpret_prompt
                    t_llm = time.perf_counter()
                    await behav.send(msg)
                    reply = await behav.receive(timeout=120)
                    llm_ms = (time.perf_counter() - t_llm) * 1000.0
                    
                    ctx.writer_interpretation = reply.body if reply else ""
                    node_ms = (time.perf_counter() - t_node) * 1000.0
                    
                    mas.log_conversation_event(
                        event_type="agent_output", actor="Writer", target="Commander", event_name=EVENT_WRITER_INTERPRET,
                        phase=PHASE_GENERATION, duration_ms=node_ms, llm_api_duration_ms=llm_ms,
                        payload={"step": EVENT_WRITER_INTERPRET, "writer_interpretation": ctx.writer_interpretation}
                    )
                    mas._mark_step_end()

                    if not (ctx.execution_error and ctx.requires_execution):
                        break # Success! Loop ends.

            ctx.attempt += 1
            mas._accum_between_step_gap()
            if ctx.attempt >= ctx.max_iterations:
                break

            t_node = time.perf_counter()
            logs = (
                f"Safeguard reason: {ctx.safeguard_reason}\n"
                f"Execution error: {ctx.execution_error}\n"
                f"Execution output: {ctx.execution_output}\n"
                f"Writer interpretation (if any): {ctx.writer_interpretation}\n"
            )
            redirect_prompt = (
                f"Original user query:\n{ctx.user_query}\n\n"
                f"Previous Writer code (for reference):\n```python\n{ctx.writer_code}\n```\n\n"
                f"Logs:\n{logs}\n"
                "Tell the Writer exactly what failed and what to change in the next code attempt. Plain text, actionable bullets."
            )
            msg = Message(to="commander@localhost")
            msg.set_metadata("message_type", "llm")
            msg.body = redirect_prompt
            t_llm = time.perf_counter()
            await behav.send(msg)
            reply = await behav.receive(timeout=120)
            llm_ms = (time.perf_counter() - t_llm) * 1000.0
            
            ctx.commander_context = reply.body if reply else ""
            node_ms = (time.perf_counter() - t_node) * 1000.0
            
            mas.log_conversation_event(
                event_type="pass", actor="Commander", target="Writer", event_name=EVENT_REDIRECT_WRITER,
                phase=PHASE_COORDINATION, duration_ms=node_ms, llm_api_duration_ms=llm_ms,
                payload={"step": EVENT_REDIRECT_WRITER, "attempt": ctx.attempt, "redirect_context": ctx.commander_context}
            )
            ctx.writer_interpretation = ""
            ctx.safeguard_allowed = False
            ctx.safeguard_reason = ""
            ctx.execution_output = ""
            ctx.execution_error = ""
            mas._mark_step_end()

        mas._accum_between_step_gap()

        # Step 8: Conclude
        t_step = time.perf_counter()
        llm_ms = None
        if (ctx.writer_interpretation or "").strip():
            conclude_prompt = (
                f"User query:\n{ctx.user_query}\n\n"
                f"Writer interpretation (substantive answer from Writer):\n{ctx.writer_interpretation}\n\n"
                f"Safeguard allowed: {ctx.safeguard_allowed}\n"
                f"Safeguard reason: {ctx.safeguard_reason}\n"
                f"Execution error: {ctx.execution_error or '(none)'}\n"
                f"Execution output: {ctx.execution_output or '(none)'}\n"
                "Furnish the user with the concluding answer. Lightly edit for clarity and user tone, preserving facts and numbers."
            )
            msg = Message(to="commander@localhost")
            msg.set_metadata("message_type", "llm")
            msg.body = conclude_prompt
            t_llm = time.perf_counter()
            await behav.send(msg)
            reply = await behav.receive(timeout=120)
            llm_ms = (time.perf_counter() - t_llm) * 1000.0
            final = reply.body if reply else ""
        else:
            if not ctx.safeguard_allowed and ctx.safeguard_reason:
                final = f"I could not complete this request: the Safeguard did not clear the proposed code after retries. Details: {ctx.safeguard_reason}"
            elif ctx.execution_error and ctx.requires_execution:
                final = f"I could not complete this request: code execution failed after retries. Last error: {ctx.execution_error}"
            else:
                final = "I could not produce a final answer for this query."
                
        conclude_dur_ms = (time.perf_counter() - t_step) * 1000.0
        ctx.final_answer = final
        ctx.finished = True
        
        mas.log_conversation_event(
            event_type="final", actor="Commander", event_name=EVENT_CONCLUDE, phase=PHASE_FINALIZATION,
            duration_ms=conclude_dur_ms, llm_api_duration_ms=llm_ms,
            payload={"step": EVENT_CONCLUDE, "final_answer": final, "safeguard_allowed": ctx.safeguard_allowed, "execution_error": ctx.execution_error}
        )
        mas._mark_step_end()
        mas.memory.append(f"Q: {ctx.user_query} | A: {final}")
        
        ctx.completion_event.set()


    async def _run_spade_workflow(self, ctx: CodingRunContext):
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        provider = LLMProvider(
            model=self.model_id,
            api_key=api_key,
        )

        try:
            # We explicitly pass verify_security=False to allow connection to the local untrusted XMPP server (spade run)
            commander = LLMAgent(
                jid="commander@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Commander. You coordinate the Writer and Safeguard agents. "
                    "Follow instructions strictly and provide only the requested output."
                ),
                verify_security=False
            )
            writer = LLMAgent(
                jid="writer@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Writer: you combine Coder and Interpreter. "
                    "In Coder mode: produce executable Python without markdown, explanations, or extra wrapper code. Return strict JSON with keys: code, notes. "
                    "In Interpreter mode: interpret execution outcomes and give the substantive technical answer the user needs (2-5 sentences)."
                ),
                verify_security=False
            )
            safeguard = LLMAgent(
                jid="safeguard@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Safeguard. You screen code and ascertain its safety. Allow only harmless data transforms and print-based output. "
                    "Return strict JSON: {\"allow\": boolean, \"reason\": string}."
                ),
                verify_security=False
            )

            orchestrator = OrchestratorAgent("orchestrator@localhost", "password", ctx, self._workflow_logic, verify_security=False)
        except TypeError:
            # Fallback if verify_security is not supported by LLMAgent in this version
            commander = LLMAgent(
                jid="commander@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Commander. You coordinate the Writer and Safeguard agents. "
                    "Follow instructions strictly and provide only the requested output."
                )
            )
            writer = LLMAgent(
                jid="writer@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Writer: you combine Coder and Interpreter. "
                    "In Coder mode: produce executable Python without markdown, explanations, or extra wrapper code. Return strict JSON with keys: code, notes. "
                    "In Interpreter mode: interpret execution outcomes and give the substantive technical answer the user needs (2-5 sentences)."
                )
            )
            safeguard = LLMAgent(
                jid="safeguard@localhost",
                password="password",
                provider=provider,
                system_prompt=(
                    "You are the Safeguard. You screen code and ascertain its safety. Allow only harmless data transforms and print-based output. "
                    "Return strict JSON: {\"allow\": boolean, \"reason\": string}."
                )
            )
            orchestrator = OrchestratorAgent("orchestrator@localhost", "password", ctx, self._workflow_logic)

        try:
            # Start agents using SPADE 4+ standard approach (no auto_register=True, no verify_security=False)
            await commander.start()
            await writer.start()
            await safeguard.start()
            await orchestrator.start()
        except spade.agent.DisconnectedException as e:
            print("\n" + "="*60, file=sys.stderr)
            print("CRITICAL ERROR: Unable to connect to the SPADE XMPP Server.", file=sys.stderr)
            print("Please ensure that you have started the local SPADE server in a separate terminal", file=sys.stderr)
            print("by running the following command:", file=sys.stderr)
            print("    spade run", file=sys.stderr)
            print("="*60 + "\n", file=sys.stderr)
            raise e

        # Wait for orchestrator to finish
        await ctx.completion_event.wait()

        # Stop agents
        await orchestrator.stop()
        await commander.stop()
        await writer.stop()
        await safeguard.stop()

        return ctx.to_workflow_state()


    def answer(self, query: str) -> Dict[str, Any]:
        self._reset_run_metrics()
        self.begin_conversation_log(query)
        wall_t0 = time.perf_counter()
        
        ctx = CodingRunContext(mas=self, query=query, max_iterations=self.max_iterations)
        
        # Start workflow using the persistent event loop to avoid LiteLLM context/logging_worker issue
        asyncio.set_event_loop(self._loop)
        state_output = self._loop.run_until_complete(self._run_spade_workflow(ctx))
        
        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        log_path = self.end_conversation_log(
            state_output.get("final_answer", ""),
            mas_total_duration_ms=mas_wall_ms,
        )
        state_output["conversation_log_path"] = log_path
        state_output[ORCHESTRATION_GAP_MS_KEY] = self._between_steps_ms
        state_output["token_usage"] = {
            "prompt_tokens": self._run_prompt_tokens,
            "completion_tokens": self._run_completion_tokens,
            "total_tokens": self._run_total_tokens,
        }
        
        return state_output