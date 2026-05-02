"""SPADE LLM coding multi-agent system — mock implementation for overhead comparison."""

from __future__ import annotations

import time
from typing import Any, Dict

from coding_scenario.base import (
    CodingMASBase,
    EVENT_CONCLUDE,
    EVENT_DECIDE_EXECUTION,
    EVENT_EXECUTE_CODE,
    EVENT_GENERATE_CODE,
    EVENT_PASS_TO_WRITER,
    EVENT_RECEIVE_TASK,
    EVENT_SAFEGUARD_REVIEW,
    EVENT_SEND_CODE_TO_SAFEGUARD,
    EVENT_SKIP_EXECUTION,
    EVENT_WRITER_INTERPRET,
    PHASE_COORDINATION,
    PHASE_EXECUTION,
    PHASE_FINALIZATION,
    PHASE_GENERATION,
)


class SpadeCodingMAS(CodingMASBase):
    """Deterministic mock MAS emitting the same conversation log schema as other frameworks."""

    def answer(self, query: str) -> Dict[str, Any]:
        self.begin_conversation_log_with_mode(query, framework_mode="mock")
        wall_t0 = time.perf_counter()

        def _delay_ms() -> float:
            time.sleep(0.001)
            return 1.0

        self.log_conversation_event(
            event_type="action",
            actor="Commander",
            event_name=EVENT_RECEIVE_TASK,
            phase=PHASE_COORDINATION,
            duration_ms=_delay_ms(),
            payload={"step": EVENT_RECEIVE_TASK, "attempt": 0},
        )
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Writer",
            event_name=EVENT_PASS_TO_WRITER,
            phase=PHASE_COORDINATION,
            duration_ms=_delay_ms(),
            token_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            payload={"step": EVENT_PASS_TO_WRITER, "commander_context": "(mock instructions)"},
        )
        writer_code = "# mock generated code\npass\n"
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_GENERATE_CODE,
            phase=PHASE_GENERATION,
            duration_ms=_delay_ms(),
            token_usage={"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90},
            payload={
                "step": EVENT_GENERATE_CODE,
                "writer_notes": "mock",
                "writer_code": writer_code,
            },
        )
        self.log_conversation_event(
            event_type="pass",
            actor="Commander",
            target="Safeguard",
            event_name=EVENT_SEND_CODE_TO_SAFEGUARD,
            phase=PHASE_COORDINATION,
            duration_ms=_delay_ms(),
            payload={"step": EVENT_SEND_CODE_TO_SAFEGUARD, "writer_code": writer_code},
        )
        self.log_conversation_event(
            event_type="agent_output",
            actor="Safeguard",
            target="Commander",
            event_name=EVENT_SAFEGUARD_REVIEW,
            phase=PHASE_COORDINATION,
            duration_ms=_delay_ms(),
            token_usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            payload={"step": EVENT_SAFEGUARD_REVIEW, "allow": True, "reason": "mock"},
        )

        requires_execution = "Complete the following Python code" not in query
        execution_output = ""
        execution_error = ""
        if requires_execution:
            self.log_conversation_event(
                event_type="action",
                actor="Commander",
                event_name=EVENT_DECIDE_EXECUTION,
                phase=PHASE_COORDINATION,
                duration_ms=_delay_ms(),
                token_usage={"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
                payload={"step": EVENT_DECIDE_EXECUTION, "requires_execution": True},
            )
            t_ex = time.perf_counter()
            execution_output = "(mock stdout)"
            execution_error = ""
            ex_ms = (time.perf_counter() - t_ex) * 1000.0
            self.log_conversation_event(
                event_type="action",
                actor="Commander",
                event_name=EVENT_EXECUTE_CODE,
                phase=PHASE_EXECUTION,
                duration_ms=ex_ms,
                payload={
                    "step": EVENT_EXECUTE_CODE,
                    "execution_output": execution_output,
                    "execution_error": execution_error,
                },
            )
        else:
            self.log_conversation_event(
                event_type="action",
                actor="Commander",
                event_name=EVENT_DECIDE_EXECUTION,
                phase=PHASE_COORDINATION,
                duration_ms=_delay_ms(),
                payload={
                    "step": EVENT_DECIDE_EXECUTION,
                    "requires_execution": False,
                    "reason": "humaneval_completion_prompt",
                },
            )
            self.log_conversation_event(
                event_type="action",
                actor="Commander",
                event_name=EVENT_SKIP_EXECUTION,
                phase=PHASE_EXECUTION,
                duration_ms=_delay_ms(),
                payload={"step": EVENT_SKIP_EXECUTION},
            )
            execution_output = "(Execution not run: Commander determined it was not required.)"
            execution_error = ""

        interpretation = "Mock writer interpretation."
        self.log_conversation_event(
            event_type="agent_output",
            actor="Writer",
            target="Commander",
            event_name=EVENT_WRITER_INTERPRET,
            phase=PHASE_GENERATION,
            duration_ms=_delay_ms(),
            token_usage={"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
            payload={
                "step": EVENT_WRITER_INTERPRET,
                "writer_interpretation": interpretation,
            },
        )
        final_answer = "Mock final answer from SPADE."
        self.log_conversation_event(
            event_type="final",
            actor="Commander",
            event_name=EVENT_CONCLUDE,
            phase=PHASE_FINALIZATION,
            duration_ms=_delay_ms(),
            token_usage={"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30},
            payload={
                "step": EVENT_CONCLUDE,
                "final_answer": final_answer,
                "safeguard_allowed": True,
                "execution_error": execution_error,
            },
        )

        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        log_path = self.end_conversation_log(
            final_answer, mas_total_duration_ms=mas_wall_ms
        )
        self.memory.append(f"Q: {query} | A: {final_answer}")

        return {
            "user_query": query,
            "memory": self.memory.copy(),
            "max_iterations": self.max_iterations,
            "attempt": 0,
            "commander_context": "(mock instructions)",
            "writer_code": writer_code,
            "writer_notes": "mock",
            "safeguard_allowed": True,
            "safeguard_reason": "mock",
            "requires_execution": requires_execution,
            "execution_output": execution_output,
            "execution_error": execution_error,
            "writer_interpretation": interpretation,
            "final_answer": final_answer,
            "finished": True,
            "token_usage": {
                "prompt_tokens": 137,
                "completion_tokens": 118,
                "total_tokens": 255,
            },
            "conversation_log_path": log_path,
            "framework_mode": "mock",
        }
