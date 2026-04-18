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
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from utils import (
    extract_code,
    execute_python,
    rule_based_safeguard,
    safe_json_parse,
    to_text,
)

load_dotenv()


class WorkflowState(TypedDict):
    user_query: str
    memory: List[str]
    max_iterations: int
    attempt: int
    commander_context: str
    writer_code: str
    writer_notes: str
    safeguard_allowed: bool
    safeguard_reason: str
    requires_execution: bool
    execution_output: str
    execution_error: str
    writer_interpretation: str
    final_answer: str
    finished: bool


class CommanderWriterSafeguardSystem:
    def __init__(self, model_id: str, max_iterations: int) -> None:
        self.model_id = model_id
        self.max_iterations = max_iterations
        self.memory: List[str] = []
        self.commander_llm = ChatOpenAI(model=model_id, temperature=0)
        self.writer_llm = ChatOpenAI(model=model_id, temperature=0)
        self.safeguard_llm = ChatOpenAI(model=model_id, temperature=0)
        self.graph = self._build_graph()

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

    # --- Step 1: task from user received by Commander ---
    def _commander_receive_task(self, state: WorkflowState) -> WorkflowState:
        state["execution_output"] = ""
        state["execution_error"] = ""
        state["writer_interpretation"] = ""
        state["safeguard_allowed"] = False
        state["safeguard_reason"] = ""
        state["requires_execution"] = True
        return state

    # --- Step 2: Commander passes the question to Writer ---
    def _commander_pass_to_writer(self, state: WorkflowState) -> WorkflowState:
        memory_text = "\n".join(f"- {item}" for item in state["memory"][-5:]) or "(none)"
        system = (
            "You are the Commander. Step 2 of the workflow: pass the user's question to the "
            "Writer agent. Produce clear instructions the Writer should follow (assumptions, "
            "deliverable: executable Python when coding is needed). Include relevant memory. "
            "Plain text only."
        )
        human = f"User query:\n{state['user_query']}\n\nPrior interaction memory:\n{memory_text}"
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        state["commander_context"] = to_text(getattr(response, "content", response))
        return state

    # --- Step 3: Writer generates code and passes it to Commander (state holds code) ---
    def _writer_generate_code(self, state: WorkflowState) -> WorkflowState:
        writer_prompt = (
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
        response = self.writer_llm.invoke(
            [SystemMessage(content=writer_prompt), HumanMessage(content=human)]
        )
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        code = extract_code(str(parsed.get("code", "")))
        if not code:
            code = "print('Unable to generate valid code for this query.')"
        state["writer_code"] = code
        state["writer_notes"] = str(parsed.get("notes", "No additional notes."))
        return state

    # --- Step 3 (cont.): Commander receives code from Writer before Safeguard ---
    def _commander_receive_writer_code(self, state: WorkflowState) -> WorkflowState:
        # Commander holds writer_code for the Safeguard step; clear stale run artifacts.
        state["execution_output"] = ""
        state["execution_error"] = ""
        return state

    # --- Steps 4–5: Commander communicates with Safeguard; clearance returns to Commander ---
    def _safeguard(self, state: WorkflowState) -> WorkflowState:
        code = state["writer_code"]
        allowed, reason = rule_based_safeguard(code)
        if not allowed:
            state["safeguard_allowed"] = False
            state["safeguard_reason"] = reason
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
        response = self.safeguard_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        state["safeguard_allowed"] = bool(parsed.get("allow", False))
        state["safeguard_reason"] = str(parsed.get("reason", "No reason provided."))
        return state

    def _route_after_safeguard(self, state: WorkflowState) -> str:
        if state["safeguard_allowed"]:
            return "commander_decide_execution"
        if state["attempt"] + 1 >= state["max_iterations"]:
            return "commander_conclude"
        return "commander_redirect_writer"

    # --- Step 6: Commander redirects issue back to Writer with essential log information ---
    def _commander_redirect_writer(self, state: WorkflowState) -> WorkflowState:
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
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        state["commander_context"] = to_text(getattr(response, "content", response))
        state["writer_interpretation"] = ""
        state["safeguard_allowed"] = False
        state["safeguard_reason"] = ""
        state["execution_output"] = ""
        state["execution_error"] = ""
        return state

    # --- Step 7 (part): Commander decides whether execution is required ---
    def _commander_decide_execution(self, state: WorkflowState) -> WorkflowState:
        system = (
            "You are the Commander. After Safeguard clearance, decide whether running the Writer's "
            "code is required to answer the user (e.g. optimization/numeric result). "
            'Return strict JSON: {"requires_execution": true|false, "rationale": string}.'
        )
        human = (
            f"User query:\n{state['user_query']}\n\n"
            f"Writer code:\n```python\n{state['writer_code']}\n```"
        )
        response = self.commander_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        parsed = safe_json_parse(to_text(getattr(response, "content", response)))
        state["requires_execution"] = bool(parsed.get("requires_execution", True))
        return state

    def _route_after_decide_execution(self, state: WorkflowState) -> str:
        if state["requires_execution"]:
            return "commander_execute_code"
        return "commander_skip_execution"

    def _commander_execute_code(self, state: WorkflowState) -> WorkflowState:
        output, error = execute_python(state["writer_code"])
        state["execution_output"] = output
        state["execution_error"] = error
        return state

    def _commander_skip_execution(self, state: WorkflowState) -> WorkflowState:
        state["execution_output"] = "(Execution not run: Commander determined it was not required.)"
        state["execution_error"] = ""
        return state

    # --- Step 7 (part): Writer interprets logs / provides the Writer-side answer ---
    def _writer_interpret(self, state: WorkflowState) -> WorkflowState:
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
        response = self.writer_llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        state["writer_interpretation"] = to_text(getattr(response, "content", response))
        return state

    def _route_after_writer_interpret(self, state: WorkflowState) -> str:
        if state.get("execution_error") and state["requires_execution"]:
            if state["attempt"] + 1 >= state["max_iterations"]:
                return "commander_conclude"
            return "commander_redirect_writer"
        return "commander_conclude"

    # --- Step 8: Commander furnishes the concluding answer to the user ---
    def _commander_conclude(self, state: WorkflowState) -> WorkflowState:
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
            response = self.commander_llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
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

        state["final_answer"] = final
        state["finished"] = True
        self.memory.append(f"Q: {state['user_query']} | A: {final}")
        return state

    def answer(self, query: str) -> Dict[str, Any]:
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
        return self.graph.invoke(start_state)
