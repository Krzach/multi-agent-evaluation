import time
from statistics import mean

class TimeTracker:
    def __init__(self):
        self.task_start_time = None
        self.task_end_time = None
        self.agent_response_times = []
        
    def start_task(self):
        self.task_start_time = time.time()
        
    def end_task(self):
        self.task_end_time = time.time()
        
    def add_agent_response_time(self, duration_seconds: float):
        self.agent_response_times.append(duration_seconds)
        
    def get_total_task_time(self) -> float:
        if self.task_start_time and self.task_end_time:
            return self.task_end_time - self.task_start_time
        return 0.0
        
    def get_average_agent_response_time(self) -> float:
        if not self.agent_response_times:
            return 0.0
        return mean(self.agent_response_times)
        
    def get_summary(self) -> dict:
        return {
            "total_task_time_seconds": self.get_total_task_time(),
            "average_agent_response_time_seconds": self.get_average_agent_response_time()
        }
