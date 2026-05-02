"""LangGraph callback: perf_counter-based time between graph node handlers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Set

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover

    class BaseCallbackHandler:  # type: ignore[misc, no-redef]
        pass


class TimeBetweenNodesCallback(BaseCallbackHandler):
    """
    Accumulates wall time (``perf_counter``) from the end of one LangGraph node runnable until
    the start of the next, using ``on_chain_start`` / ``on_chain_end`` pairs whose ``metadata``
    contains ``langgraph_node`` (LangGraph-injected step metadata).

    Nested LLM / tool chains without ``langgraph_node`` are ignored. The outer ``LangGraph``
    wrapper is also ignored because it does not carry ``langgraph_node``.
    """

    def __init__(self) -> None:
        self.between_nodes_ms: float = 0.0
        self._last_node_end_mono: Optional[float] = None
        self._open_graph_run_ids: Set[Any] = set()

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        metadata = kwargs.get("metadata") or {}
        if not metadata.get("langgraph_node"):
            return
        run_id = kwargs.get("run_id")
        if run_id is None:
            return
        now = time.perf_counter()
        if self._last_node_end_mono is not None:
            self.between_nodes_ms += (now - self._last_node_end_mono) * 1000.0
        self._open_graph_run_ids.add(run_id)

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        if run_id is None or run_id not in self._open_graph_run_ids:
            return
        self._open_graph_run_ids.remove(run_id)
        self._last_node_end_mono = time.perf_counter()
