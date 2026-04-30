class CollaborationTracker:
    def __init__(self):
        self.messages_between_agents: int = 0
        self.conversation_turns: int = 0
        self.activated_agents: set[str] = set()
        self.information_flow_depth: int = 0
        
    def add_message(self, sender: str, receiver: str):
        self.messages_between_agents += 1
        self.activated_agents.add(sender)
        self.activated_agents.add(receiver)
        
    def increment_turn(self):
        self.conversation_turns += 1
        
    def update_flow_depth(self, depth: int):
        self.information_flow_depth = max(self.information_flow_depth, depth)
        
    def get_summary(self) -> dict:
        return {
            "messages_between_agents": self.messages_between_agents,
            "conversation_turns": self.conversation_turns,
            "num_activated_agents": len(self.activated_agents),
            "information_flow_depth": self.information_flow_depth
        }
