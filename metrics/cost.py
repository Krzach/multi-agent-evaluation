class CostTracker:
    def __init__(self, cost_per_1k_input: float = 0.0, cost_per_1k_output: float = 0.0):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        
    def add_tokens(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
    def get_total_cost(self) -> float:
        input_cost = (self.input_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = (self.output_tokens / 1000.0) * self.cost_per_1k_output
        return input_cost + output_cost
        
    def get_summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.get_total_cost()
        }
