from __future__ import annotations

import time
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from coding_scenario.base import (
    CodingMASBase,
    EVENT_GENERATE_CODE,
    PHASE_GENERATION,
)
from utils import extract_code, to_text

load_dotenv()

class SingleAgentMAS(CodingMASBase):
    """A simple single-agent implementation for coding tasks."""

    def __init__(self, model_id: str, max_iterations: int) -> None:
        super().__init__(model_id, max_iterations)
        self.llm = ChatOpenAI(model=model_id, temperature=0)

    def answer(self, query: str) -> Dict[str, Any]:
        self.begin_conversation_log(query)
        wall_t0 = time.perf_counter()

        system_prompt = (
            "You are a helpful coding assistant. "
            "Given a user query, you generate Python code to solve the problem. "
            "If the request is a code-completion benchmark task, return only the function body/completion "
            "without markdown, explanations, or extra wrapper code. "
            "Avoid adding print statements unless explicitly requested by the user."
        )
        
        human_prompt = f"User query:\n{query}"

        t_llm = time.perf_counter()
        response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        llm_ms = (time.perf_counter() - t_llm) * 1000.0
        
        generated_code = extract_code(to_text(getattr(response, "content", response)))
        
        token_usage = {}
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            tu = response.response_metadata['token_usage']
            token_usage = {
                "prompt_tokens": tu.get("prompt_tokens", 0),
                "completion_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
            }

        self.log_conversation_event(
            event_type="agent_output",
            actor="SingleAgent",
            event_name=EVENT_GENERATE_CODE,
            phase=PHASE_GENERATION,
            duration_ms=llm_ms,
            llm_api_duration_ms=llm_ms,
            token_usage=token_usage,
            payload={
                "step": EVENT_GENERATE_CODE,
                "writer_code": generated_code,
            },
        )

        final_answer = "The code has been generated."
        
        mas_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        log_path = self.end_conversation_log(
            final_answer,
            mas_total_duration_ms=mas_wall_ms,
        )
        
        output_state = {
            "writer_code": generated_code,
            "attempt": 1,
            "safeguard_allowed": True,
            "token_usage": token_usage,
            "conversation_log_path": log_path,
            "final_answer": final_answer,
            "user_query": query,
            "memory": [],
            "max_iterations": self.max_iterations,
            "commander_context": "",
            "writer_notes": "",
            "safeguard_reason": "",
            "requires_execution": True,
            "execution_output": "",
            "execution_error": "",
            "writer_interpretation": "",
            "finished": True,
        }
        
        return output_state
