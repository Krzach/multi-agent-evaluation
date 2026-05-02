"""AutoGen multi-agent system aligned with `coding_workflow.md` (same steps as LangchainCodingMAS).

Orchestration uses **GraphFlow** (AgentChat team): a :class:`~autogen_agentchat.teams.GraphFlow` over a
:class:`~autogen_agentchat.teams.DiGraphBuilder` graph with conditional edges for safeguard / execution
retries, matching the LangGraph-style workflow. Role behaviour stays in ``CommanderAgent`` /
``WriterAgent`` / ``SafeguardAgent``; graph nodes delegate to them and read/write shared
:class:`CodingRunContext`.

See https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/graph-flow.html
and https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/custom-agents.html
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_core import CancellationToken
from autogen_core.models import ModelFamily, ModelInfo, RequestUsage, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

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
    to_text,
)

load_dotenv()


def _openai_chat_client(model_id: str, api_key: str) -> OpenAIChatCompletionClient:
    """Instantiate :class:`OpenAIChatCompletionClient`, adding ``model_info`` when needed.

    ``autogen_ext`` only knows a fixed allow-list of model ids. Names like ``gpt-5.4`` or other
    preview aliases are valid at the API but missing from the registry; those require explicit
    ``model_info`` or construction raises ``ValueError``.
    """
    base: Dict[str, Any] = {"model": model_id, "api_key": api_key, "temperature": 0}
    try:
        return OpenAIChatCompletionClient(**base)
    except ValueError as exc:
        if "model_info is required" not in str(exc):
            raise
    # Conservative modern OpenAI-style capabilities (matches registered GPT-5 class models).
    fallback: ModelInfo = {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.GPT_5,
        "structured_output": True,
        "multiple_system_messages": True,
    }
    return OpenAIChatCompletionClient(**base, model_info=fallback)


@dataclass
class LlmTurnResult:
    text: str
    llm_duration_ms: float
    token_usage: Optional[Dict[str, int]]


@dataclass
class WriterCodeResult:
    writer_code: str
    writer_notes: str
    raw_response: str
    llm_duration_ms: float
    token_usage: Optional[Dict[str, int]]


@dataclass
class SafeguardReviewResult:
    blocked_by_rule: bool
    rule_reason: str
    raw_llm_response: str = ""
    allow: bool = False
    safeguard_reason: str = ""
    llm_duration_ms: float = 0.0
    token_usage: Optional[Dict[str, int]] = None


def _final_assistant_text(result: TaskResult) -> str:
    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage) and msg.source not in ("user", "Orchestrator"):
            return to_text(msg.content)
    for msg in reversed(result.messages):
        if isinstance(msg, TextMessage):
            return to_text(msg.content)
    return ""


def _usage_from_request_usage(usage: Optional[RequestUsage]) -> Optional[Dict[str, int]]:
    """Map autogen_core ``RequestUsage`` to conversation-log ``token_usage`` dict."""
    if usage is None:
        return None
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    if not pt and not ct:
        return None
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }


def _usage_from_task_result(result: TaskResult) -> Optional[Dict[str, int]]:
    for msg in reversed(result.messages):
        if not isinstance(msg, TextMessage):
            continue
        mu = msg.models_usage
        if mu is None:
            continue
        pt = int(getattr(mu, "prompt_tokens", 0) or 0)
        ct = int(getattr(mu, "completion_tokens", 0) or 0)
        if pt or ct:
            return {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            }
    return None


def _split_system_user_texts(messages: Sequence[BaseChatMessage]) -> Tuple[str, str]:
    texts: list[str] = []
    for m in messages:
        if isinstance(m, TextMessage):
            texts.append(str(m.content))
    if len(texts) >= 2:
        return texts[0], texts[1]
    if len(texts) == 1:
        return "", texts[0]
    return "", ""


@dataclass
class CodingRunContext:
    """Mutable workflow state shared by GraphFlow step agents and the MAS."""

    mas: Optional[AutoGenCodingMAS] = None
    user_query: str = ""
    memory: List[str] = field(default_factory=list)
    max_iterations: int = 3
    attempt: int = 0
    commander_context: str = ""
    writer_code: str = ""
    writer_notes: str = ""
    safeguard_allowed: bool = False
    safeguard_reason: str = ""
    requires_execution: bool = True
    execution_output: str = ""
    execution_error: str = ""
    writer_interpretation: str = ""
    final_answer: str = ""
    finished: bool = False

    def to_workflow_state(self) -> WorkflowState:
        return {
            "user_query": self.user_query,
            "memory": list(self.memory),
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


class _OpenAiCodingRoleAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(name, description)
        self._model_client = model_client

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        system_text, user_text = _split_system_user_texts(messages)
        if system_text:
            llm_messages = [
                SystemMessage(content=system_text),
                UserMessage(content=user_text, source="user"),
            ]
        else:
            llm_messages = [UserMessage(content=user_text, source="user")]

        result = await self._model_client.create(
            messages=llm_messages,
            cancellation_token=cancellation_token,
        )
        text = to_text(result.content)
        chat = TextMessage(content=text, source=self.name, models_usage=result.usage)
        return Response(chat_message=chat, inner_messages=[])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    async def run_system_user(
        self, system: str, human: str, cancellation_token: CancellationToken
    ) -> LlmTurnResult:
        """LLM timing is ``OpenAIChatCompletionClient.create`` only (same path as ``on_messages``).

        Using ``self.run()`` would include AgentChat routing overhead and inflate ``llm_api_duration_ms``.
        """
        llm_messages = [
            SystemMessage(content=system),
            UserMessage(content=human, source="user"),
        ]
        t_llm = time.perf_counter()
        result = await self._model_client.create(
            messages=llm_messages,
            cancellation_token=cancellation_token,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        return LlmTurnResult(
            text=to_text(result.content),
            llm_duration_ms=llm_ms,
            token_usage=_usage_from_request_usage(result.usage),
        )


class CommanderAgent(_OpenAiCodingRoleAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(
            "Commander",
            "Coordinates Writer and Safeguard; passes tasks, decides execution, concludes for the user.",
            model_client,
        )

    @staticmethod
    def is_humaneval_completion_query(user_query: str) -> bool:
        return "Complete the following Python code" in user_query

    async def pass_question_to_writer(
        self,
        *,
        user_query: str,
        memory_snippets: list[str],
        cancellation_token: CancellationToken,
    ) -> LlmTurnResult:
        memory_text = "\n".join(f"- {item}" for item in memory_snippets) or "(none)"
        system = (
            "You are the Commander. Step 2 of the workflow: pass the user's question to the "
            "Writer agent. Produce clear instructions the Writer should follow (assumptions, "
            "deliverable: executable Python when coding is needed). Include relevant memory. "
            "If the user asks for a script to print a specific value, instruct the Writer to "
            "make sure the script prints EXACTLY the final answer without any additional text, labels, or formatting.\n"
            "Plain text only."
        )
        human = f"User query:\n{user_query}\n\nPrior interaction memory:\n{memory_text}"
        return await self.run_system_user(system, human, cancellation_token)

    async def redirect_writer_after_failure(
        self,
        *,
        user_query: str,
        writer_code: str,
        logs: str,
        cancellation_token: CancellationToken,
    ) -> LlmTurnResult:
        system = (
            "You are the Commander. Step 6: redirect the issue back to the Writer with the "
            "essential information from the logs below. Tell the Writer exactly what failed "
            "and what to change in the next code attempt. Plain text, actionable bullets."
        )
        human = (
            f"Original user query:\n{user_query}\n\n"
            f"Previous Writer code (for reference):\n```python\n{writer_code}\n```\n\n"
            f"Logs:\n{logs}"
        )
        return await self.run_system_user(system, human, cancellation_token)

    async def decide_requires_execution(
        self,
        *,
        user_query: str,
        writer_code: str,
        cancellation_token: CancellationToken,
    ) -> LlmTurnResult:
        system = (
            "You are the Commander. After Safeguard clearance, decide whether running the Writer's "
            "code is required to answer the user.\n"
            "CRITICAL: If the user explicitly asks for a script to 'calculate', 'read a file', 'print a result', "
            "or execute any logic, execution IS REQUIRED. "
            "Only skip execution if the user purely wants to read the source code and does not care about the output.\n"
            "Return strict JSON: {\"requires_execution\": true|false, \"rationale\": string}."
        )
        human = f"User query:\n{user_query}\n\nWriter code:\n```python\n{writer_code}\n```"
        return await self.run_system_user(system, human, cancellation_token)

    async def furnish_final_answer(
        self,
        *,
        user_query: str,
        writer_interpretation: str,
        safeguard_allowed: bool,
        safeguard_reason: str,
        execution_error: str,
        execution_output: str,
        cancellation_token: CancellationToken,
    ) -> LlmTurnResult:
        system = (
            "You are the Commander. Step 8: furnish the user with the concluding answer. "
            "You may lightly edit for clarity and user tone, but preserve the Writer's facts "
            "and numbers. If there is no Writer interpretation, explain briefly what went wrong "
            "(safeguard block or execution failure) using the log fields."
        )
        human = (
            f"User query:\n{user_query}\n\n"
            f"Writer interpretation (substantive answer from Writer):\n"
            f"{writer_interpretation or '(none — see errors below)'}\n\n"
            f"Safeguard allowed: {safeguard_allowed}\n"
            f"Safeguard reason: {safeguard_reason}\n"
            f"Execution error: {execution_error or '(none)'}\n"
            f"Execution output: {execution_output or '(none)'}\n"
        )
        return await self.run_system_user(system, human, cancellation_token)

    @staticmethod
    def template_failure_message(
        *,
        safeguard_allowed: bool,
        safeguard_reason: str,
        execution_error: str,
        requires_execution: bool,
    ) -> str:
        if not safeguard_allowed and safeguard_reason:
            return (
                "I could not complete this request: the Safeguard did not clear the proposed "
                f"code after retries. Details: {safeguard_reason}"
            )
        if execution_error and requires_execution:
            return (
                "I could not complete this request: code execution failed after retries. "
                f"Last error: {execution_error}"
            )
        return "I could not produce a final answer for this query."


class WriterAgent(_OpenAiCodingRoleAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(
            "Writer",
            "Coder and interpreter: generates Python and explains execution results.",
            model_client,
        )

    async def generate_code_bundle(
        self,
        *,
        user_query: str,
        commander_context: str,
        cancellation_token: CancellationToken,
    ) -> WriterCodeResult:
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
            f"User query:\n{user_query}\n\n"
            f"Commander instructions (passed from Commander to you):\n{commander_context}\n\n"
            "Return JSON only."
        )
        turn = await self.run_system_user(writer_prompt, human, cancellation_token)
        parsed = safe_json_parse(turn.text)
        code = extract_code(str(parsed.get("code", "")))
        if not code:
            code = "print('Unable to generate valid code for this query.')"
        notes = str(parsed.get("notes", "No additional notes."))
        return WriterCodeResult(
            writer_code=code,
            writer_notes=notes,
            raw_response=turn.text,
            llm_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
        )

    async def interpret_execution(
        self,
        *,
        user_query: str,
        writer_notes: str,
        execution_output: str,
        execution_error: str,
        cancellation_token: CancellationToken,
    ) -> LlmTurnResult:
        system = (
            "You are the Writer in Interpreter mode only. Step 7: interpret execution outcomes "
            "(or the Commander's decision to skip execution) for the user query. Give the "
            "substantive technical answer the user needs (2–5 sentences). Mention uncertainty if "
            "assumptions were required."
        )
        human = (
            f"User query:\n{user_query}\n\n"
            f"Writer notes from coding step:\n{writer_notes}\n\n"
            f"Stdout / summarized output:\n{execution_output}\n\n"
            f"Execution error line (if any):\n{execution_error or '(none)'}"
        )
        return await self.run_system_user(system, human, cancellation_token)


class SafeguardAgent(_OpenAiCodingRoleAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__(
            "Safeguard",
            "Reviews Writer code for dangerous behavior; returns strict JSON allow/deny.",
            model_client,
        )

    async def review_code(
        self,
        *,
        commander_context: str,
        writer_code: str,
        cancellation_token: CancellationToken,
    ) -> SafeguardReviewResult:
        allowed, reason = rule_based_safeguard(writer_code)
        if not allowed:
            return SafeguardReviewResult(blocked_by_rule=True, rule_reason=reason)

        system = (
            "You are the Safeguard. The Commander asks you to screen code from the Writer "
            "and ascertain its safety. Allow only harmless data transforms and print-based output. "
            "Return strict JSON: {\"allow\": boolean, \"reason\": string}."
        )
        human = (
            f"Commander context (for awareness only):\n{commander_context[:1200]}\n\n"
            f"Code to review:\n```python\n{writer_code}\n```"
        )
        turn = await self.run_system_user(system, human, cancellation_token)
        parsed = safe_json_parse(turn.text)
        return SafeguardReviewResult(
            blocked_by_rule=False,
            rule_reason="",
            raw_llm_response=turn.text,
            allow=bool(parsed.get("allow", False)),
            safeguard_reason=str(parsed.get("reason", "No reason provided.")),
            llm_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
        )


# --- GraphFlow step agents (BaseChatAgent nodes; delegate to roles + ctx) ---


class _CtxStepAgent(BaseChatAgent):
    """Non-LLM graph node with access to :class:`CodingRunContext`."""

    def __init__(self, name: str, description: str, ctx: CodingRunContext) -> None:
        super().__init__(name, description)
        self._ctx = ctx

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class ReceiveTaskStepAgent(_CtxStepAgent):
    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        t0 = time.perf_counter()
        self._ctx.execution_output = ""
        self._ctx.execution_error = ""
        self._ctx.writer_interpretation = ""
        self._ctx.safeguard_allowed = False
        self._ctx.safeguard_reason = ""
        self._ctx.requires_execution = True
        dur_ms = (time.perf_counter() - t0) * 1000.0
        mas._accum_between_step_gap()
        mas.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_RECEIVE_TASK,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            payload={"step": EVENT_RECEIVE_TASK, "attempt": self._ctx.attempt},
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content="receive_task_ok", source=self.name),
            inner_messages=[],
        )


class PassWriterStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, commander: CommanderAgent) -> None:
        super().__init__(name, description, ctx)
        self._commander = commander

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t_node = time.perf_counter()
        mem = self._ctx.memory[-5:]
        turn = await self._commander.pass_question_to_writer(
            user_query=self._ctx.user_query,
            memory_snippets=list(mem),
            cancellation_token=cancellation_token,
        )
        mas._record_usage(turn.token_usage)
        self._ctx.commander_context = turn.text
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_PASS_TO_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
            payload={"step": EVENT_PASS_TO_WRITER, "commander_context": self._ctx.commander_context},
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content=self._ctx.commander_context, source=self.name),
            inner_messages=[],
        )


class WriterGenerateStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, writer: WriterAgent) -> None:
        super().__init__(name, description, ctx)
        self._writer = writer

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t_node = time.perf_counter()
        bundle = await self._writer.generate_code_bundle(
            user_query=self._ctx.user_query,
            commander_context=self._ctx.commander_context,
            cancellation_token=cancellation_token,
        )
        mas._record_usage(bundle.token_usage)
        self._ctx.writer_code = bundle.writer_code
        self._ctx.writer_notes = bundle.writer_notes
        self._ctx.execution_output = ""
        self._ctx.execution_error = ""
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_GENERATE_CODE,
            phase=PHASE_GENERATION,
            duration_ms=node_ms,
            llm_api_duration_ms=bundle.llm_duration_ms,
            token_usage=bundle.token_usage,
            payload={
                "step": EVENT_GENERATE_CODE,
                "writer_notes": self._ctx.writer_notes,
                "writer_code": self._ctx.writer_code,
            },
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content=bundle.raw_response[:2000], source=self.name),
            inner_messages=[],
        )


class CommanderReceiveCodeStepAgent(_CtxStepAgent):
    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t0 = time.perf_counter()
        self._ctx.execution_output = ""
        self._ctx.execution_error = ""
        dur_ms = (time.perf_counter() - t0) * 1000.0
        mas.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Safeguard",
            event_name=EVENT_SEND_CODE_TO_SAFEGUARD,
            phase=PHASE_COORDINATION,
            duration_ms=dur_ms,
            payload={
                "step": EVENT_SEND_CODE_TO_SAFEGUARD,
                "writer_code": self._ctx.writer_code,
            },
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content="code_received", source=self.name),
            inner_messages=[],
        )


class SafeguardStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, safeguard: SafeguardAgent) -> None:
        super().__init__(name, description, ctx)
        self._safeguard = safeguard

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t_node = time.perf_counter()
        review = await self._safeguard.review_code(
            commander_context=self._ctx.commander_context,
            writer_code=self._ctx.writer_code,
            cancellation_token=cancellation_token,
        )

        if review.blocked_by_rule:
            self._ctx.safeguard_allowed = False
            self._ctx.safeguard_reason = review.rule_reason
            t_rule = time.perf_counter()
            dur_ms = (time.perf_counter() - t_rule) * 1000.0
            mas.log_conversation_event(
                event_type="agent_output",
                actor="Safeguard",
                target="Commander",
                event_name=EVENT_SAFEGUARD_RULE_BLOCK,
                phase=PHASE_COORDINATION,
                duration_ms=dur_ms,
                payload={
                    "step": EVENT_SAFEGUARD_RULE_BLOCK,
                    "allow": False,
                    "reason": review.rule_reason,
                },
            )
            mas._mark_step_end()
            out = f"RULE_BLOCK:{review.rule_reason}"
        else:
            mas._record_usage(review.token_usage)
            self._ctx.safeguard_allowed = review.allow
            self._ctx.safeguard_reason = review.safeguard_reason
            node_ms = (time.perf_counter() - t_node) * 1000.0
            mas.log_conversation_event(
                event_type="agent_output",
                actor="Safeguard",
                target="Commander",
                event_name=EVENT_SAFEGUARD_REVIEW,
                phase=PHASE_COORDINATION,
                duration_ms=node_ms,
                llm_api_duration_ms=review.llm_duration_ms,
                token_usage=review.token_usage,
                payload={
                    "step": EVENT_SAFEGUARD_REVIEW,
                    "allow": self._ctx.safeguard_allowed,
                    "reason": self._ctx.safeguard_reason,
                },
            )
            mas._mark_step_end()
            out = f"LLM_REVIEW:allow={review.allow}"

        return Response(chat_message=TextMessage(content=out, source=self.name), inner_messages=[])


class CommanderDecideStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, commander: CommanderAgent) -> None:
        super().__init__(name, description, ctx)
        self._commander = commander

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        q = self._ctx.user_query
        if CommanderAgent.is_humaneval_completion_query(q):
            t0 = time.perf_counter()
            self._ctx.requires_execution = False
            dur_ms = (time.perf_counter() - t0) * 1000.0
            mas.log_conversation_event(
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
            mas._mark_step_end()
            return Response(
                chat_message=TextMessage(content="humaneval_skip_exec", source=self.name),
                inner_messages=[],
            )

        t_node = time.perf_counter()
        turn = await self._commander.decide_requires_execution(
            user_query=q,
            writer_code=self._ctx.writer_code,
            cancellation_token=cancellation_token,
        )
        mas._record_usage(turn.token_usage)
        parsed = safe_json_parse(turn.text)
        self._ctx.requires_execution = bool(parsed.get("requires_execution", True))
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_DECIDE_EXECUTION,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
            payload={
                "step": EVENT_DECIDE_EXECUTION,
                "requires_execution": self._ctx.requires_execution,
            },
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(
                content=f"decide:requires_execution={self._ctx.requires_execution}",
                source=self.name,
            ),
            inner_messages=[],
        )


class CommanderExecuteStepAgent(_CtxStepAgent):
    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t0 = time.perf_counter()
        output, error = execute_python(self._ctx.writer_code)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self._ctx.execution_output = output
        self._ctx.execution_error = error
        mas.log_conversation_event(
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
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content=f"executed:err={bool(error)}", source=self.name),
            inner_messages=[],
        )


class CommanderSkipExecStepAgent(_CtxStepAgent):
    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t0 = time.perf_counter()
        self._ctx.execution_output = (
            "(Execution not run: Commander determined it was not required.)"
        )
        self._ctx.execution_error = ""
        dur_ms = (time.perf_counter() - t0) * 1000.0
        mas.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_SKIP_EXECUTION,
            phase=PHASE_EXECUTION,
            duration_ms=dur_ms,
            payload={"step": EVENT_SKIP_EXECUTION},
        )
        mas._mark_step_end()
        return Response(chat_message=TextMessage(content="skip_exec", source=self.name), inner_messages=[])


class WriterInterpretStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, writer: WriterAgent) -> None:
        super().__init__(name, description, ctx)
        self._writer = writer

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        if self._ctx.execution_error:
            self._ctx.writer_interpretation = ""
            mas._mark_step_end()
            return Response(
                chat_message=TextMessage(content="interpret_skipped_error", source=self.name),
                inner_messages=[],
            )

        t_node = time.perf_counter()
        turn = await self._writer.interpret_execution(
            user_query=self._ctx.user_query,
            writer_notes=self._ctx.writer_notes,
            execution_output=self._ctx.execution_output,
            execution_error=self._ctx.execution_error,
            cancellation_token=cancellation_token,
        )
        mas._record_usage(turn.token_usage)
        self._ctx.writer_interpretation = turn.text
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_WRITER_INTERPRET,
            phase=PHASE_GENERATION,
            duration_ms=node_ms,
            llm_api_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
            payload={
                "step": EVENT_WRITER_INTERPRET,
                "writer_interpretation": self._ctx.writer_interpretation,
            },
        )
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content=self._ctx.writer_interpretation[:2000], source=self.name),
            inner_messages=[],
        )


class CommanderRedirectStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, commander: CommanderAgent) -> None:
        super().__init__(name, description, ctx)
        self._commander = commander

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t_node = time.perf_counter()
        self._ctx.attempt += 1
        logs = (
            f"Safeguard reason: {self._ctx.safeguard_reason}\n"
            f"Execution error: {self._ctx.execution_error}\n"
            f"Execution output: {self._ctx.execution_output}\n"
            f"Writer interpretation (if any): {self._ctx.writer_interpretation}\n"
        )
        turn = await self._commander.redirect_writer_after_failure(
            user_query=self._ctx.user_query,
            writer_code=self._ctx.writer_code,
            logs=logs,
            cancellation_token=cancellation_token,
        )
        mas._record_usage(turn.token_usage)
        self._ctx.commander_context = turn.text
        node_ms = (time.perf_counter() - t_node) * 1000.0
        mas.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_REDIRECT_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=node_ms,
            llm_api_duration_ms=turn.llm_duration_ms,
            token_usage=turn.token_usage,
            payload={
                "step": EVENT_REDIRECT_WRITER,
                "attempt": self._ctx.attempt,
                "redirect_context": self._ctx.commander_context,
            },
        )
        self._ctx.writer_interpretation = ""
        self._ctx.safeguard_allowed = False
        self._ctx.safeguard_reason = ""
        self._ctx.execution_output = ""
        self._ctx.execution_error = ""
        mas._mark_step_end()
        return Response(
            chat_message=TextMessage(content="redirected", source=self.name),
            inner_messages=[],
        )


class CommanderConcludeStepAgent(_CtxStepAgent):
    def __init__(self, name: str, description: str, ctx: CodingRunContext, commander: CommanderAgent) -> None:
        super().__init__(name, description, ctx)
        self._commander = commander

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        mas = self._ctx.mas
        assert mas is not None
        mas._accum_between_step_gap()
        t_step = time.perf_counter()
        llm_api_ms: Optional[float] = None
        usage: Optional[Dict[str, int]] = None
        if (self._ctx.writer_interpretation or "").strip():
            turn = await self._commander.furnish_final_answer(
                user_query=self._ctx.user_query,
                writer_interpretation=self._ctx.writer_interpretation,
                safeguard_allowed=self._ctx.safeguard_allowed,
                safeguard_reason=self._ctx.safeguard_reason,
                execution_error=self._ctx.execution_error,
                execution_output=self._ctx.execution_output,
                cancellation_token=cancellation_token,
            )
            mas._record_usage(turn.token_usage)
            self._ctx.final_answer = turn.text
            llm_api_ms = turn.llm_duration_ms
            usage = turn.token_usage
        else:
            self._ctx.final_answer = CommanderAgent.template_failure_message(
                safeguard_allowed=self._ctx.safeguard_allowed,
                safeguard_reason=self._ctx.safeguard_reason,
                execution_error=self._ctx.execution_error,
                requires_execution=self._ctx.requires_execution,
            )

        conclude_dur_ms = (time.perf_counter() - t_step) * 1000.0
        conclude_tokens: Optional[Dict[str, int]] = None
        if usage is not None and sum(usage.values()) > 0:
            conclude_tokens = usage

        self._ctx.finished = True
        mas.log_conversation_event(
            event_type="final",
            actor="Commander",
            event_name=EVENT_CONCLUDE,
            phase=PHASE_FINALIZATION,
            duration_ms=conclude_dur_ms,
            llm_api_duration_ms=llm_api_ms,
            token_usage=conclude_tokens,
            payload={
                "step": EVENT_CONCLUDE,
                "final_answer": self._ctx.final_answer,
                "safeguard_allowed": self._ctx.safeguard_allowed,
                "execution_error": self._ctx.execution_error,
            },
        )
        mas._mark_step_end()
        mas.memory.append(f"Q: {self._ctx.user_query} | A: {self._ctx.final_answer}")
        return Response(
            chat_message=TextMessage(content=self._ctx.final_answer[:2000], source=self.name),
            inner_messages=[],
        )


def _build_coding_graph_flow(
    ctx: CodingRunContext,
    commander: CommanderAgent,
    writer: WriterAgent,
    safeguard: SafeguardAgent,
) -> GraphFlow:
    receive = ReceiveTaskStepAgent("receive_task", "Reset run state and log receive_task.", ctx)
    pass_w = PassWriterStepAgent("pass_to_writer", "Commander passes instructions to Writer.", ctx, commander)
    writer_gen = WriterGenerateStepAgent("writer_generate", "Writer emits code JSON.", ctx, writer)
    recv_code = CommanderReceiveCodeStepAgent(
        "commander_receive_code", "Commander hands code toward Safeguard (log only).", ctx
    )
    sg = SafeguardStepAgent("safeguard_review", "Safeguard rule + LLM review.", ctx, safeguard)
    decide = CommanderDecideStepAgent("commander_decide", "Commander decides whether to execute code.", ctx, commander)
    execute = CommanderExecuteStepAgent("commander_execute", "Commander runs Python locally.", ctx)
    skip_ex = CommanderSkipExecStepAgent("commander_skip_exec", "Commander records skip-execution.", ctx)
    interpret = WriterInterpretStepAgent("writer_interpret", "Writer interprets stdout.", ctx, writer)
    redirect = CommanderRedirectStepAgent("commander_redirect", "Commander redirects Writer after failure.", ctx, commander)
    conclude = CommanderConcludeStepAgent("commander_conclude", "Commander final user answer.", ctx, commander)

    b = DiGraphBuilder()
    b.add_node(receive)
    b.add_node(pass_w)
    b.add_node(writer_gen)
    b.add_node(recv_code)
    b.add_node(sg)
    b.add_node(decide)
    b.add_node(execute)
    b.add_node(skip_ex)
    b.add_node(interpret)
    b.add_node(redirect)
    b.add_node(conclude)

    # Fan-in edges must use activation_condition="any": default "all" requires every parent
    # edge to fire before the target runs. Here pass_to_writer and commander_redirect both
    # feed writer_generate (only one runs per iteration); execute vs skip both feed interpret;
    # safeguard vs interpret both feed redirect/conclude on different branches.
    _g_writer_gen = "fanin_writer_generate"
    _g_interpret = "fanin_writer_interpret"
    _g_redirect = "fanin_commander_redirect"
    _g_conclude = "fanin_commander_conclude"

    b.add_edge(receive, pass_w)
    b.add_edge(pass_w, writer_gen, activation_group=_g_writer_gen, activation_condition="any")
    b.add_edge(writer_gen, recv_code)
    b.add_edge(recv_code, sg)

    def _safeguard_ok(_msg: BaseChatMessage) -> bool:
        return ctx.safeguard_allowed

    def _safeguard_retry(_msg: BaseChatMessage) -> bool:
        return (not ctx.safeguard_allowed) and (ctx.attempt + 1 < ctx.max_iterations)

    def _safeguard_terminal(_msg: BaseChatMessage) -> bool:
        return (not ctx.safeguard_allowed) and (ctx.attempt + 1 >= ctx.max_iterations)

    b.add_edge(sg, decide, condition=_safeguard_ok)
    b.add_edge(sg, redirect, condition=_safeguard_retry, activation_group=_g_redirect, activation_condition="any")
    b.add_edge(sg, conclude, condition=_safeguard_terminal, activation_group=_g_conclude, activation_condition="any")

    def _needs_exec(_msg: BaseChatMessage) -> bool:
        return ctx.requires_execution

    def _skip_exec(_msg: BaseChatMessage) -> bool:
        return not ctx.requires_execution

    b.add_edge(decide, execute, condition=_needs_exec)
    b.add_edge(decide, skip_ex, condition=_skip_exec)
    b.add_edge(execute, interpret, activation_group=_g_interpret, activation_condition="any")
    b.add_edge(skip_ex, interpret, activation_group=_g_interpret, activation_condition="any")

    def _retry_after_bad_run(_msg: BaseChatMessage) -> bool:
        return bool(ctx.execution_error) and ctx.requires_execution and (ctx.attempt + 1 < ctx.max_iterations)

    def _finish_iteration(_msg: BaseChatMessage) -> bool:
        return not (
            bool(ctx.execution_error) and ctx.requires_execution and (ctx.attempt + 1 < ctx.max_iterations)
        )

    b.add_edge(interpret, redirect, condition=_retry_after_bad_run, activation_group=_g_redirect, activation_condition="any")
    b.add_edge(interpret, conclude, condition=_finish_iteration, activation_group=_g_conclude, activation_condition="any")

    b.add_edge(redirect, writer_gen, activation_group=_g_writer_gen, activation_condition="any")

    b.set_entry_point(receive)
    graph = b.build()
    term = MaxMessageTermination(200)
    return GraphFlow(participants=b.get_participants(), graph=graph, termination_condition=term)


class AutoGenCodingMAS(CodingMASBase):
    def __init__(self, model_id: str, max_iterations: int) -> None:
        super().__init__(model_id, max_iterations)
        self._model_id = model_id
        self._model_client: Optional[OpenAIChatCompletionClient] = None

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

    async def _answer_async(self, query: str, cancellation_token: CancellationToken) -> Dict[str, Any]:
        self._reset_run_metrics()
        self.begin_conversation_log(query)
        wall_t0 = time.perf_counter()
        api_key = os.getenv("OPENAI_API_KEY") or ""
        self._model_client = _openai_chat_client(self._model_id, api_key)
        commander = CommanderAgent(self._model_client)
        writer = WriterAgent(self._model_client)
        safeguard = SafeguardAgent(self._model_client)

        ctx = CodingRunContext(
            mas=self,
            user_query=query,
            memory=self.memory.copy(),
            max_iterations=self.max_iterations,
            attempt=0,
        )

        flow = _build_coding_graph_flow(ctx, commander, writer, safeguard)
        try:
            await flow.run(task=query, cancellation_token=cancellation_token)
        finally:
            if self._model_client is not None:
                await self._model_client.close()
            self._model_client = None

        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        state = ctx.to_workflow_state()
        log_path = self.end_conversation_log(
            state.get("final_answer", ""),
            mas_total_duration_ms=mas_wall_ms,
        )
        state["conversation_log_path"] = log_path
        state[ORCHESTRATION_GAP_MS_KEY] = self._between_steps_ms
        state["token_usage"] = {
            "prompt_tokens": self._run_prompt_tokens,
            "completion_tokens": self._run_completion_tokens,
            "total_tokens": self._run_total_tokens,
        }
        return state

    def answer(self, query: str) -> Dict[str, Any]:
        cancel = CancellationToken()
        return asyncio.run(self._answer_async(query, cancel))
