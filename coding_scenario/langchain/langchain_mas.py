"""LangGraph multi-agent system aligned with `coding_workflow.md`.

Numbered flow (coding_workflow.md):
1. Receiving the task from user by Commander
2. Passing the question by Commander to Writer
3. Writer generates code and passes it to the Commander
4. Commander communicates with the Safeguard to screen the code
5. Code obtains the Safeguard's clearance to the Commander
6. On safeguard red flag or execution failure, Commander redirects to Writer with logs
7. If execution is required: Commander executes code and passes results to Writer;
   the substantive answer is provided by the Writer (interpretation)
8. Commander furnishes the user with the concluding answer

Steps 3–6 may repeat until resolution or max iterations (stand-in for timeout).

Each ``log_conversation_event`` sets ``actor`` to ``Commander``, ``Writer``, or ``Safeguard`` so
``metrics/conversation_log_metrics`` can sum **time outside the LLM call** vs **LLM API**
milliseconds per agent (``duration_ms`` minus ``llm_api_duration_ms`` when the latter is set).
``llm_api_duration_ms`` is wall time around each ``ChatOpenAI.invoke`` (provider I/O plus
LangChain runnable work for that call). Tighter framework-native bounds would use
``BaseCallbackHandler.on_llm_start`` / ``on_llm_end``; differences are usually small.
``phase`` remains a workflow hint in the log; aggregates use ``actor``, not ``phase``.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
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
)
from utils import (
    extract_code,
    execute_python,
    rule_based_safeguard,
    safe_json_parse,
    to_text,
)
from coding_scenario.base import WorkflowState
from coding_scenario.langchain.callbacks.time_between_nodes import TimeBetweenNodesCallback
from coding_scenario.langchain.callbacks.token_tracking import TokenTrackingCallback

load_dotenv()


class LangchainCodingMAS(CodingMASBase):
    def __init__(self, model_id: str, max_iterations: int) -> None:
        super().__init__(model_id, max_iterations)
        self.token_tracking_cb = TokenTrackingCallback()
        self.time_between_nodes_cb = TimeBetweenNodesCallback()
        self.commander_llm = ChatOpenAI(model=model_id, temperature=0, callbacks=[self.token_tracking_cb])
        self.writer_llm = ChatOpenAI(model=model_id, temperature=0, callbacks=[self.token_tracking_cb])
        self.safeguard_llm = ChatOpenAI(model=model_id, temperature=0, callbacks=[self.token_tracking_cb])
        
        self.graph = self._build_graph()

    def _snapshot_tokens(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.token_tracking_cb.prompt_tokens,
            "completion_tokens": self.token_tracking_cb.completion_tokens,
            "total_tokens": self.token_tracking_cb.total_tokens,
        }

    def _token_delta(self, before: Dict[str, int]) -> Dict[str, int]:
        after = self._snapshot_tokens()
        return {
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
        }

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)

        # Step 1
        graph.add_node("commander_receive_task", self._commander_receive_task)
        # Step 2
        graph.add_node("commander_pass_to_writer", self._commander_pass_to_writer)
        # Step 3 (Writer -> Commander)
        graph.add_node("writer_generate_code", self._writer_generate_code)
        graph.add_node("commander_receive_writer_code", self._commander_receive_writer_code)
        # Steps 4–5 (Commander channel to Safeguard)
        graph.add_node("safeguard", self._safeguard)
        # Step 6 (Commander redirects Writer with logs)
        graph.add_node("commander_redirect_writer", self._commander_redirect_writer)
        # Step 7 (Commander executes if needed; Writer interprets)
        graph.add_node("commander_decide_execution", self._commander_decide_execution)
        graph.add_node("commander_execute_code", self._commander_execute_code)
        graph.add_node("commander_skip_execution", self._commander_skip_execution)
        graph.add_node("writer_interpret", self._writer_interpret)
        # Step 8
        graph.add_node("commander_conclude", self._commander_conclude)

        graph.set_entry_point("commander_receive_task")
        graph.add_edge("commander_receive_task", "commander_pass_to_writer")
        graph.add_edge("commander_pass_to_writer", "writer_generate_code")
        graph.add_edge("writer_generate_code", "commander_receive_writer_code")
        graph.add_edge("commander_receive_writer_code", "safeguard")

        graph.add_conditional_edges(
            "safeguard",
            self._route_after_safeguard,
            {
                "commander_decide_execution": "commander_decide_execution",
                "commander_redirect_writer": "commander_redirect_writer",
                "commander_conclude": "commander_conclude",
            },
        )

        graph.add_edge("commander_redirect_writer", "writer_generate_code")

        graph.add_conditional_edges(
            "commander_decide_execution",
            self._route_after_decide_execution,
            {
                "commander_execute_code": "commander_execute_code",
                "commander_skip_execution": "commander_skip_execution",
            },
        )

        graph.add_edge("commander_execute_code", "writer_interpret")
        graph.add_edge("commander_skip_execution", "writer_interpret")

        graph.add_conditional_edges(
            "writer_interpret",
            self._route_after_writer_interpret,
            {
                "commander_conclude": "commander_conclude",
                "commander_redirect_writer": "commander_redirect_writer",
            },
        )

        graph.add_edge("commander_conclude", END)
        return graph.compile()

    def reset_token_counts(self):
        """Reset the token tracker counts for a new task."""
        self.token_tracking_cb.total_tokens = 0
        self.token_tracking_cb.prompt_tokens = 0
        self.token_tracking_cb.completion_tokens = 0

    # --- Step 1: task from user received by Commander ---
    def _commander_receive_task(self, state: WorkflowState) -> WorkflowState:
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
        return state

    # --- Step 2: Commander passes the question to Writer ---
    def _commander_pass_to_writer(self, state: WorkflowState) -> WorkflowState:
        t_node = time.perf_counter()
        memory_text = "\n".join(f"- {item}" for item in state["memory"][-5:]) or "(none)"
        system = (
            "You are the Commander. Step 2 of the workflow: pass the user's question to the "
            "Writer agent. Produce clear instructions the Writer should follow (assumptions, "
            "deliverable: executable Python when coding is needed). Include relevant memory. "
            "If the user asks for a script to print a specific value, instruct the Writer to "
            "make sure the script prints EXACTLY the final answer without any additional text, labels, or formatting.\n"
            "Plain text only."
        )
        human = f"User query:\n{state['user_query']}\n\nPrior interaction memory:\n{memory_text}"
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        state["commander_context"] = to_text(getattr(response, "content", response))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_PASS_TO_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
            payload={
                "step": EVENT_PASS_TO_WRITER,
                "commander_context": state["commander_context"],
            },
        )
        return state

    # --- Step 3: Writer generates code and passes it to Commander (state holds code) ---
    def _writer_generate_code(self, state: WorkflowState) -> WorkflowState:
        t_node = time.perf_counter()
        writer_prompt = (
            "You are the Writer: you combine Coder and Interpreter. "
            "In this step you act only as Coder: produce executable Python for the Commander's "
            "instructions and the user query. "
            "If the request is a code-completion benchmark task, return only the function body/completion "
            "without markdown, explanations, or extra wrapper code. "
            "Avoid adding print statements unless explicitly requested by the user. "
            "Return strict JSON with keys: code, notes."
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Commander instructions (passed from Commander to you):\n{state['commander_context']}\n\n"
            "Return JSON only."
        )
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.writer_llm.invoke(
            [SystemMessage(content=writer_prompt), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        code = extract_code(str(parsed.get("code", "")))
        if not code:
            code = "print('Unable to generate valid code for this query.')"
        state["writer_code"] = code
        state["writer_notes"] = str(parsed.get("notes", "No additional notes."))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_GENERATE_CODE,
            phase=PHASE_GENERATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
            payload={
                "step": EVENT_GENERATE_CODE,
                "writer_notes": state["writer_notes"],
                "writer_code": state["writer_code"],
            },
        )
        return state

    # --- Step 3 (cont.): Commander receives code from Writer before Safeguard ---
    def _commander_receive_writer_code(self, state: WorkflowState) -> WorkflowState:
        # Commander holds writer_code for the Safeguard step; clear stale run artifacts.
        t0 = time.perf_counter()
        state["execution_output"] = ""
        state["execution_error"] = ""
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Safeguard",
            event_name=EVENT_SEND_CODE_TO_SAFEGUARD,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            payload={
                "step": EVENT_SEND_CODE_TO_SAFEGUARD,
                "writer_code": state.get("writer_code", ""),
            },
        )
        return state

    # --- Steps 4–5: Commander communicates with Safeguard; clearance returns to Commander ---
    def _safeguard(self, state: WorkflowState) -> WorkflowState:
        t_node = time.perf_counter()
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
            return state

        system = (
            "You are the Safeguard. The Commander asks you to screen code from the Writer "
            "and ascertain its safety. Allow only harmless data transforms and print-based output. "
            "Return strict JSON: {\"allow\": boolean, \"reason\": string}."
        )
        human = (
            f"Commander context (for awareness only):\n{state['commander_context'][:1200]}\n\n"
            f"Code to review:\n```python\n{code}\n```"
        )
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.safeguard_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        state["safeguard_allowed"] = bool(parsed.get("allow", False))
        state["safeguard_reason"] = str(parsed.get("reason", "No reason provided."))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="agent_output",
            actor="Safeguard",
            target="Commander",
            event_name=EVENT_SAFEGUARD_REVIEW,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
            payload={
                "step": EVENT_SAFEGUARD_REVIEW,
                "allow": state["safeguard_allowed"],
                "reason": state["safeguard_reason"],
            },
        )
        return state

    def _route_after_safeguard(self, state: WorkflowState) -> str:
        if state["safeguard_allowed"]:
            return "commander_decide_execution"
        if state["attempt"] + 1 >= state["max_iterations"]:
            return "commander_conclude"
        return "commander_redirect_writer"

    # --- Step 6: Commander redirects issue back to Writer with essential log information ---
    def _commander_redirect_writer(self, state: WorkflowState) -> WorkflowState:
        t_node = time.perf_counter()
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
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        state["commander_context"] = to_text(getattr(response, "content", response))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_REDIRECT_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
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
        return state

    # --- Step 7 (part): Commander decides whether execution is required ---
    def _commander_decide_execution(self, state: WorkflowState) -> WorkflowState:
        query = state["user_query"]
        if "Complete the following Python code" in query:
            # HumanEval-style prompts are completion tasks where code can be non-standalone.
            # Let the benchmark harness execute the completion with tests.
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
            return state

        t_node = time.perf_counter()
        system = (
            "You are the Commander. After Safeguard clearance, decide whether running the Writer's "
            "code is required to answer the user.\n"
            "CRITICAL: If the user explicitly asks for a script to 'calculate', 'read a file', 'print a result', "
            "or execute any logic, execution IS REQUIRED. "
            "Only skip execution if the user purely wants to read the source code and does not care about the output.\n"
            "Return strict JSON: {\"requires_execution\": true|false, \"rationale\": string}."
        )
        human = (
            f"User query:\n{query}\n\n"
            f"Writer code:\n```python\n{state['writer_code']}\n```"
        )
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        # Default to True (safe fallback for multi-agent code evaluation) if parsing fails.
        state["requires_execution"] = bool(parsed.get("requires_execution", True))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_DECIDE_EXECUTION,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
            payload={
                "step": EVENT_DECIDE_EXECUTION,
                "requires_execution": state["requires_execution"],
            },
        )
        return state

    def _route_after_decide_execution(self, state: WorkflowState) -> str:
        if state["requires_execution"]:
            return "commander_execute_code"
        return "commander_skip_execution"

    def _commander_execute_code(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        output, error = execute_python(state["writer_code"])
        dur_ms = (time.perf_counter() - t0) * 1000.0
        state["execution_output"] = output
        state["execution_error"] = error
        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_EXECUTE_CODE,
            phase=PHASE_EXECUTION,
            duration_ms=dur_ms,
            payload={
                "step": EVENT_EXECUTE_CODE,
                "execution_output": output,
                "execution_error": error,
            },
        )
        return state

    def _commander_skip_execution(self, state: WorkflowState) -> WorkflowState:
        t0 = time.perf_counter()
        state["execution_output"] = "(Execution not run: Commander determined it was not required.)"
        state["execution_error"] = ""
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_SKIP_EXECUTION,
            phase=PHASE_EXECUTION,
            duration_ms=dur_ms,
            payload={"step": EVENT_SKIP_EXECUTION},
        )
        return state

    # --- Step 7 (part): Writer interprets logs / provides the Writer-side answer ---
    def _writer_interpret(self, state: WorkflowState) -> WorkflowState:
        t_node = time.perf_counter()
        if state.get("execution_error"):
            # Execution failed: Step 6 — Commander redirects to Writer (unless max attempts).
            if state["attempt"] + 1 >= state["max_iterations"]:
                state["writer_interpretation"] = ""
                return state
            # Defer redirect to routing after this node returns partial state.
            state["writer_interpretation"] = ""
            return state

        system = (
            "You are the Writer in Interpreter mode only. Step 7: interpret execution outcomes "
            "(or the Commander's decision to skip execution) for the user query. Give the "
            "substantive technical answer the user needs (2–5 sentences). Mention uncertainty if "
            "assumptions were required."
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Writer notes from coding step:\n{state['writer_notes']}\n\n"
            f"Stdout / summarized output:\n{state['execution_output']}\n\n"
            f"Execution error line (if any):\n{state['execution_error'] or '(none)'}"
        )
        tok_before = self._snapshot_tokens()
        t_llm = time.perf_counter()
        response = self.writer_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        state["writer_interpretation"] = to_text(getattr(response, "content", response))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_WRITER_INTERPRET,
            phase=PHASE_GENERATION,
            duration_ms=node_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=self._token_delta(tok_before),
            payload={
                "step": EVENT_WRITER_INTERPRET,
                "writer_interpretation": state["writer_interpretation"],
            },
        )
        return state

    def _route_after_writer_interpret(self, state: WorkflowState) -> str:
        # this should be added to overhead timing
        if state.get("execution_error") and state["requires_execution"]:
            if state["attempt"] + 1 >= state["max_iterations"]:
                return "commander_conclude"
            return "commander_redirect_writer"
        return "commander_conclude"

    # --- Step 8: Commander furnishes the concluding answer to the user ---
    def _commander_conclude(self, state: WorkflowState) -> WorkflowState:
        tok_before = self._snapshot_tokens()
        t_step = time.perf_counter()
        llm_api_ms: float | None = None
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
                f"{state.get('writer_interpretation') or '(none — see errors below)'}\n\n"
                f"Safeguard allowed: {state.get('safeguard_allowed')}\n"
                f"Safeguard reason: {state.get('safeguard_reason', '')}\n"
                f"Execution error: {state.get('execution_error', '') or '(none)'}\n"
                f"Execution output: {state.get('execution_output', '') or '(none)'}\n"
            )
            t_llm = time.perf_counter()
            response = self.commander_llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
            llm_api_ms = (time.perf_counter() - t_llm) * 1000.0
            final = to_text(getattr(response, "content", response))
        else:
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

        conclude_dur_ms = (time.perf_counter() - t_step) * 1000.0
        conclude_tokens = self._token_delta(tok_before)

        state["final_answer"] = final
        state["finished"] = True
        self.log_conversation_event(
            event_type="final",
            actor="Commander",
            event_name=EVENT_CONCLUDE,
            phase=PHASE_FINALIZATION,
            duration_ms=conclude_dur_ms,
            llm_api_duration_ms=llm_api_ms,
            token_usage=conclude_tokens if sum(conclude_tokens.values()) > 0 else None,
            payload={
                "step": EVENT_CONCLUDE,
                "final_answer": final,
                "safeguard_allowed": state.get("safeguard_allowed"),
                "execution_error": state.get("execution_error", ""),
            },
        )
        self.memory.append(f"Q: {state['user_query']} | A: {final}")
        return state

    def answer(self, query: str) -> Dict[str, Any]:
        self.reset_token_counts()
        self.begin_conversation_log(query)
        wall_t0 = time.perf_counter()
        start_state: WorkflowState = {
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

        state_output = self.graph.invoke(start_state, config={"callbacks": [self.time_between_nodes_cb, self.token_tracking_cb]})
        
        state_output[ORCHESTRATION_GAP_MS_KEY] = self.time_between_nodes_cb.between_nodes_ms
        state_output["token_usage"] = self._snapshot_tokens()  
        
        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        log_path = self.end_conversation_log(
            state_output.get("final_answer", ""),
            mas_total_duration_ms=mas_wall_ms,
        )
        state_output["conversation_log_path"] = log_path
        
        return state_output
