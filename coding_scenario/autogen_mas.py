"""AutoGen multi-agent system aligned with `workflow.md`.

Numbered flow (workflow.md):
1. Receiving the task from user by Commander
2. Passing the question by Commander to Writer
3. Writer generates code and passes it to the Commander
4. Commander communicates with the Safeguard to screen the code
5. Code obtains the Safeguard's clearance to the Commander
6. On safeguard red flag or execution failure, Commander redirects to Writer with logs
7. If execution is required: Commander executes code and passes results to Writer;
   the substantive answer is provided by the Writer (interpretation)
8. Commander furnishes the user with the concluding answer

Steps 3-6 may repeat until resolution or max iterations (stand-in for timeout).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from autogen import AssistantAgent

from coding_scenario.base import (
    CodingMASBase,
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
    to_text,
)

load_dotenv()


def _try_autogen_usage(agent: AssistantAgent) -> Optional[Dict[str, int]]:
    """Best-effort token usage from AutoGen agent (varies by version)."""
    for attr in ("client", "_client"):
        client = getattr(agent, attr, None)
        if client is None:
            continue
        usage = getattr(client, "total_usage_summary", None) or getattr(
            client, "actual_usage_summary", None
        )
        if usage is None and hasattr(client, "get_usage"):
            try:
                usage = client.get_usage()
            except Exception:
                usage = None
        if isinstance(usage, dict):
            pt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            ct = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0))
            )
            tt = int(usage.get("total_tokens", pt + ct))
            if pt or ct or tt:
                return {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                }
    return None


class AutoGenCodingMAS(CodingMASBase):
    def __init__(self, model_id: str, max_iterations: int) -> None:
        super().__init__(model_id, max_iterations)

        self._llm_config = {
            "config_list": [
                {
                    "model": model_id,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            ],
            "temperature": 0,
        }

        self.commander = AssistantAgent(
            name="Commander",
            llm_config=self._llm_config,
            system_message=(
                "You are the Commander. Coordinate Writer and Safeguard, maintain context, "
                "and produce the final user-facing response."
            ),
        )
        self.writer = AssistantAgent(
            name="Writer",
            llm_config=self._llm_config,
            system_message=(
                "You are the Writer. You can act as coder or interpreter depending on instruction."
            ),
        )
        self.safeguard = AssistantAgent(
            name="Safeguard",
            llm_config=self._llm_config,
            system_message=(
                "You are the Safeguard. Screen code for dangerous behavior and return strict JSON."
            ),
        )

    def _agent_text_timed(
        self, agent: AssistantAgent, system: str, human: str
    ) -> Tuple[str, float, Optional[Dict[str, int]]]:
        prompt = f"{system}\n\n{human}"
        t0 = time.perf_counter()
        reply = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        dur_ms = (time.perf_counter() - t0) * 1000.0
        text = to_text(reply)
        usage = _try_autogen_usage(agent)
        return text, dur_ms, usage

    # --- Step 1: task from user received by Commander ---
    def _commander_receive_task(self, state: WorkflowState) -> None:
        t0 = time.perf_counter()
        state["execution_output"] = ""
        state["execution_error"] = ""
        state["writer_interpretation"] = ""
        state["safeguard_allowed"] = False
        state["safeguard_reason"] = ""
        state["requires_execution"] = True
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_RECEIVE_TASK,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            payload={"step": EVENT_RECEIVE_TASK, "attempt": state.get("attempt", 0)},
        )

    # --- Step 2: Commander passes the question to Writer ---
    def _commander_pass_to_writer(self, state: WorkflowState) -> None:
        memory_text = "\n".join(f"- {item}" for item in state["memory"][-5:]) or "(none)"
        system = (
            "You are the Commander. Step 2 of the workflow: pass the user's question to the "
            "Writer agent. Produce clear instructions the Writer should follow (assumptions, "
            "deliverable: executable Python when coding is needed). Include relevant memory. "
            "Plain text only."
        )
        human = f"User query:\n{state['user_query']}\n\nPrior interaction memory:\n{memory_text}"
        text, dur_ms, usage = self._agent_text_timed(self.commander, system, human)
        state["commander_context"] = text
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_PASS_TO_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={"step": EVENT_PASS_TO_WRITER, "commander_context": state["commander_context"]},
        )

    # --- Step 3: Writer generates code and passes it to Commander (state holds code) ---
    def _writer_generate_code(self, state: WorkflowState) -> None:
        system = (
            "You are the Writer: you combine Coder and Interpreter. "
            "In this step you act only as Coder: produce executable Python for the Commander's "
            "instructions and the user query. Use simple built-ins; print a concise result summary. "
            "Return strict JSON with keys: code, notes."
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Commander instructions (passed from Commander to you):\n{state['commander_context']}\n\n"
            "Return JSON only."
        )
        raw, dur_ms, usage = self._agent_text_timed(self.writer, system, human)
        parsed = safe_json_parse(raw)
        code = extract_code(str(parsed.get("code", "")))
        if not code:
            code = "print('Unable to generate valid code for this query.')"
        state["writer_code"] = code
        state["writer_notes"] = str(parsed.get("notes", "No additional notes."))
        state["execution_output"] = ""
        state["execution_error"] = ""
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_GENERATE_CODE,
            phase=PHASE_GENERATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={
                "step": EVENT_GENERATE_CODE,
                "writer_notes": state["writer_notes"],
                "writer_code": state["writer_code"],
            },
        )
        t_pass = time.perf_counter()
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Safeguard",
            event_name=EVENT_SEND_CODE_TO_SAFEGUARD,
            phase=PHASE_COORDINATION,
            duration_ms=(time.perf_counter() - t_pass) * 1000.0,
            payload={
                "step": EVENT_SEND_CODE_TO_SAFEGUARD,
                "writer_code": state.get("writer_code", ""),
            },
        )

    # --- Steps 4-5: Commander communicates with Safeguard; clearance returns to Commander ---
    def _safeguard_step(self, state: WorkflowState) -> None:
        code = state["writer_code"]
        allowed, reason = rule_based_safeguard(code)
        if not allowed:
            t0 = time.perf_counter()
            state["safeguard_allowed"] = False
            state["safeguard_reason"] = reason
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self.log_conversation_event(
                event_type="agent_output",
                actor="Safeguard",
                target="Commander",
                event_name=EVENT_SAFEGUARD_RULE_BLOCK,
                phase=PHASE_COORDINATION,
                duration_ms=dur_ms,
                payload={
                    "step": EVENT_SAFEGUARD_RULE_BLOCK,
                    "allow": False,
                    "reason": reason,
                },
            )
            return

        system = (
            "You are the Safeguard. The Commander asks you to screen code from the Writer "
            "and ascertain its safety. Allow only harmless data transforms and print-based output. "
            'Return strict JSON: {"allow": boolean, "reason": string}.'
        )
        human = (
            f"Commander context (for awareness only):\n{state['commander_context'][:1200]}\n\n"
            f"Code to review:\n```python\n{code}\n```"
        )
        raw, dur_ms, usage = self._agent_text_timed(self.safeguard, system, human)
        parsed = safe_json_parse(raw)
        state["safeguard_allowed"] = bool(parsed.get("allow", False))
        state["safeguard_reason"] = str(parsed.get("reason", "No reason provided."))
        self.log_conversation_event(
            event_type="agent_output",
            actor="Safeguard",
            target="Commander",
            event_name=EVENT_SAFEGUARD_REVIEW,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={
                "step": EVENT_SAFEGUARD_REVIEW,
                "allow": state["safeguard_allowed"],
                "reason": state["safeguard_reason"],
            },
        )

    # --- Step 6: Commander redirects issue back to Writer with essential log information ---
    def _commander_redirect_writer(self, state: WorkflowState) -> None:
        state["attempt"] = state["attempt"] + 1
        logs = (
            f"Safeguard reason: {state.get('safeguard_reason', '')}\n"
            f"Execution error: {state.get('execution_error', '')}\n"
            f"Execution output: {state.get('execution_output', '')}\n"
            f"Writer interpretation (if any): {state.get('writer_interpretation', '')}\n"
        )
        system = (
            "You are the Commander. Step 6: redirect the issue back to the Writer with the "
            "essential information from the logs below. Tell the Writer exactly what failed "
            "and what to change in the next code attempt. Plain text, actionable bullets."
        )
        human = (
            f"Original user query:\n{state['user_query']}\n\n"
            f"Previous Writer code (for reference):\n```python\n{state.get('writer_code', '')}\n```\n\n"
            f"Logs:\n{logs}"
        )
        text, dur_ms, usage = self._agent_text_timed(self.commander, system, human)
        state["commander_context"] = text
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_REDIRECT_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={
                "step": EVENT_REDIRECT_WRITER,
                "attempt": state["attempt"],
                "redirect_context": state["commander_context"],
            },
        )
        state["writer_interpretation"] = ""
        state["safeguard_allowed"] = False
        state["safeguard_reason"] = ""
        state["execution_output"] = ""
        state["execution_error"] = ""

    # --- Step 7 (part): Commander decides whether execution is required ---
    def _commander_decide_execution(self, state: WorkflowState) -> None:
        query = state["user_query"]
        if "Complete the following Python code" in query:
            t0 = time.perf_counter()
            state["requires_execution"] = False
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self.log_conversation_event(
                event_type="action",
                actor="Commander",
                event_name=EVENT_DECIDE_EXECUTION,
                phase=PHASE_COORDINATION,
                duration_ms=dur_ms,
                payload={
                    "step": EVENT_DECIDE_EXECUTION,
                    "requires_execution": False,
                    "reason": "humaneval_completion_prompt",
                },
            )
            return

        system = (
            "You are the Commander. After Safeguard clearance, decide whether running the Writer's "
            "code is required to answer the user (e.g. optimization/numeric result). "
            'Return strict JSON: {"requires_execution": true|false, "rationale": string}.'
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Writer code:\n```python\n{state['writer_code']}\n```"
        )
        raw, dur_ms, usage = self._agent_text_timed(self.commander, system, human)
        parsed = safe_json_parse(raw)
        state["requires_execution"] = bool(parsed.get("requires_execution", True))
        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_DECIDE_EXECUTION,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={
                "step": EVENT_DECIDE_EXECUTION,
                "requires_execution": state["requires_execution"],
            },
        )

    def _writer_interpret(self, state: WorkflowState) -> None:
        if state.get("execution_error"):
            state["writer_interpretation"] = ""
            return

        system = (
            "You are the Writer in Interpreter mode only. Step 7: interpret execution outcomes "
            "(or the Commander's decision to skip execution) for the user query. Give the "
            "substantive technical answer the user needs (2-5 sentences). Mention uncertainty if "
            "assumptions were required."
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Writer notes from coding step:\n{state['writer_notes']}\n\n"
            f"Stdout / summarized output:\n{state['execution_output']}\n\n"
            f"Execution error line (if any):\n{state['execution_error'] or '(none)'}"
        )
        text, dur_ms, usage = self._agent_text_timed(self.writer, system, human)
        state["writer_interpretation"] = text
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_WRITER_INTERPRET,
            phase=PHASE_GENERATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=dur_ms,
            token_usage=usage,
            payload={
                "step": EVENT_WRITER_INTERPRET,
                "writer_interpretation": state["writer_interpretation"],
            },
        )

    # --- Step 8: Commander furnishes the concluding answer to the user ---
    def _commander_conclude(self, state: WorkflowState) -> None:
        usage: Optional[Dict[str, int]] = None
        dur_ms: float
        llm_api_ms: Optional[float] = None
        if (state.get("writer_interpretation") or "").strip():
            system = (
                "You are the Commander. Step 8: furnish the user with the concluding answer. "
                "You may lightly edit for clarity and user tone, but preserve the Writer's facts "
                "and numbers. If there is no Writer interpretation, explain briefly what went wrong "
                "(safeguard block or execution failure) using the log fields."
            )
            human = (
                f"User query:\n{state['user_query']}\n\n"
                f"Writer interpretation (substantive answer from Writer):\n"
                f"{state.get('writer_interpretation') or '(none - see errors below)'}\n\n"
                f"Safeguard allowed: {state.get('safeguard_allowed')}\n"
                f"Safeguard reason: {state.get('safeguard_reason', '')}\n"
                f"Execution error: {state.get('execution_error', '') or '(none)'}\n"
                f"Execution output: {state.get('execution_output', '') or '(none)'}\n"
            )
            final, dur_ms, usage = self._agent_text_timed(self.commander, system, human)
            llm_api_ms = dur_ms
        else:
            t0 = time.perf_counter()
            if not state.get("safeguard_allowed") and state.get("safeguard_reason"):
                final = (
                    "I could not complete this request: the Safeguard did not clear the proposed "
                    f"code after retries. Details: {state['safeguard_reason']}"
                )
            elif state.get("execution_error") and state.get("requires_execution"):
                final = (
                    "I could not complete this request: code execution failed after retries. "
                    f"Last error: {state['execution_error']}"
                )
            else:
                final = "I could not produce a final answer for this query."
            dur_ms = (time.perf_counter() - t0) * 1000.0

        state["final_answer"] = final
        state["finished"] = True
        self.log_conversation_event(
            event_type="final",
            actor="Commander",
            event_name=EVENT_CONCLUDE,
            phase=PHASE_FINALIZATION,
            duration_ms=dur_ms,
            llm_api_duration_ms=llm_api_ms,
            token_usage=usage,
            payload={
                "step": EVENT_CONCLUDE,
                "final_answer": final,
                "safeguard_allowed": state.get("safeguard_allowed"),
                "execution_error": state.get("execution_error", ""),
            },
        )
        self.memory.append(f"Q: {state['user_query']} | A: {final}")

    def answer(self, query: str) -> Dict[str, Any]:
        self.begin_conversation_log(query)
        wall_t0 = time.perf_counter()
        state: WorkflowState = {
            "user_query": query,
            "memory": self.memory.copy(),
            "max_iterations": self.max_iterations,
            "attempt": 0,
            "commander_context": "",
            "writer_code": "",
            "writer_notes": "",
            "safeguard_allowed": False,
            "safeguard_reason": "",
            "requires_execution": True,
            "execution_output": "",
            "execution_error": "",
            "writer_interpretation": "",
            "final_answer": "",
            "finished": False,
        }

        self._commander_receive_task(state)
        self._commander_pass_to_writer(state)

        while state["attempt"] < state["max_iterations"]:
            self._writer_generate_code(state)
            self._safeguard_step(state)

            if not state["safeguard_allowed"]:
                if state["attempt"] + 1 >= state["max_iterations"]:
                    break
                self._commander_redirect_writer(state)
                continue

            self._commander_decide_execution(state)
            if state["requires_execution"]:
                t_ex = time.perf_counter()
                output, error = execute_python(state["writer_code"])
                ex_ms = (time.perf_counter() - t_ex) * 1000.0
                state["execution_output"] = output
                state["execution_error"] = error
                self.log_conversation_event(
                    event_type="action",
                    actor="Commander",
                    event_name=EVENT_EXECUTE_CODE,
                    phase=PHASE_EXECUTION,
                    duration_ms=ex_ms,
                    payload={
                        "step": EVENT_EXECUTE_CODE,
                        "execution_output": output,
                        "execution_error": error,
                    },
                )
            else:
                t_sk = time.perf_counter()
                state["execution_output"] = (
                    "(Execution not run: Commander determined it was not required.)"
                )
                state["execution_error"] = ""
                sk_ms = (time.perf_counter() - t_sk) * 1000.0
                self.log_conversation_event(
                    event_type="action",
                    actor="Commander",
                    event_name=EVENT_SKIP_EXECUTION,
                    phase=PHASE_EXECUTION,
                    duration_ms=sk_ms,
                    payload={"step": EVENT_SKIP_EXECUTION},
                )

            self._writer_interpret(state)
            if state.get("execution_error") and state["requires_execution"]:
                if state["attempt"] + 1 >= state["max_iterations"]:
                    break
                self._commander_redirect_writer(state)
                continue
            break

        self._commander_conclude(state)
        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        log_path = self.end_conversation_log(
            state.get("final_answer", ""),
            mas_total_duration_ms=mas_wall_ms,
        )
        state["conversation_log_path"] = log_path
        state["token_usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return state
