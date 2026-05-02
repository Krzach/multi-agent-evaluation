"""LangChain callback for token usage (optional dependency)."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - optional dependency

    class BaseCallbackHandler:  # type: ignore[misc, no-redef]
        pass


class TokenTrackingCallback(BaseCallbackHandler):
    """Callback to track token usage across all LLM calls."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_tokens += usage.get("total_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
