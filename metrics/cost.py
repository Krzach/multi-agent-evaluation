from __future__ import annotations

class CostTracker:
    """Tracks token usage and estimates total cost from configured pricing."""

    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0):
        self.input_tokens = 0
        self.output_tokens = 0
        self._cost_per_1k_input = float(cost_per_1k_input)
        self._cost_per_1k_output = float(cost_per_1k_output)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)

    def configure_pricing(self, *, cost_per_1k_input: float, cost_per_1k_output: float) -> None:
        self._cost_per_1k_input = float(cost_per_1k_input)
        self._cost_per_1k_output = float(cost_per_1k_output)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def get_total_cost(self) -> float:
        input_cost = (self.input_tokens / 1000.0) * self._cost_per_1k_input
        output_cost = (self.output_tokens / 1000.0) * self._cost_per_1k_output
        return input_cost + output_cost

    def get_summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.get_total_cost(),
            "pricing": {
                "input_per_1k": self._cost_per_1k_input,
                "output_per_1k": self._cost_per_1k_output,
            },
        }
