"""Minimal MASEval benchmark: one task, local stub agent, string-match evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from maseval import AgentAdapter, Benchmark, Environment, Evaluator, ModelAdapter, Task
from maseval.core.history import MessageHistory
from maseval.core.model import ChatResponse
from maseval.core.seeding import SeedGenerator


class TrivialEnvironment(Environment):
    """No tools; holds optional metadata from the task."""

    def setup_state(self, task_data: dict) -> Any:
        return dict(task_data)

    def create_tools(self) -> Dict[str, Any]:
        return {}


class StubModelAdapter(ModelAdapter):
    """Satisfies Benchmark.get_model_adapter when no real LLM is used."""

    def __init__(self, model_id: str = "stub", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return ChatResponse(content="(stub model — not used in this example)", role="assistant")


class ToyAgent:
    """Tiny non-LLM agent: answers a fixed arithmetic prompt."""

    def answer(self, query: str) -> str:
        if "2 + 2" in query or "2+2" in query.replace(" ", ""):
            return "4"
        return "I only know 2 + 2."


class ToyAgentAdapter(AgentAdapter):
    def _run_agent(self, query: str) -> str:
        history = MessageHistory()
        history.add_message("user", query)
        out = self.agent.answer(query)
        history.add_message("assistant", out)
        self.messages = history
        return out


class ExactMatchEvaluator(Evaluator):
    def __init__(self, task: Task, environment: Environment, user: Any = None) -> None:
        super().__init__(task, environment, user)
        self._expected = str(task.evaluation_data.get("expected_answer", ""))

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        return traces

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        got = (final_answer or "").strip()
        ok = got == self._expected
        return {"exact_match": ok, "expected": self._expected, "got": got}


class MinimalBenchmark(Benchmark):
    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Environment:
        return TrivialEnvironment(task.environment_data or {})

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Any,
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        adapter = ToyAgentAdapter(ToyAgent(), name="toy")
        return [adapter], {"toy": adapter}

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Any,
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        return [ExactMatchEvaluator(task, environment, user)]

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        adapter = StubModelAdapter(model_id=model_id or "stub")
        category = kwargs.get("register_category", "models")
        name = kwargs.get("register_name", model_id or "stub")
        self.register(category, name, adapter)
        return adapter

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for evaluator in evaluators:
            filtered = evaluator.filter_traces(traces)
            results.append(evaluator(filtered, final_answer=str(final_answer) if final_answer is not None else None))
        return results

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment, query: str) -> Any:
        return agents[0].run(query)


def main() -> None:
    task = Task(
        query="What is 2 + 2? Reply with just the number.",
        evaluation_data={"expected_answer": "4"},
        environment_data={"note": "demo"},
    )

    benchmark = MinimalBenchmark(progress_bar=False, n_task_repeats=1)
    reports = benchmark.run(tasks=[task], agent_data={})

    for r in reports:
        print(f"task_id={r['task_id']} status={r['status']} query={r['task']['query']!r}")
        if r.get("eval"):
            print(f"  eval={r['eval']}")
        if r.get("error"):
            print(f"  error={r['error']}")


if __name__ == "__main__":
    main()
