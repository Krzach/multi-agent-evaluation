from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CollaborationTracker:
    """Tracks collaboration behavior across agents in a conversation."""

    messages_between_agents: int = 0
    conversation_turns: int = 0
    activated_agents: set[str] = field(default_factory=set)
    information_flow_depth: int = 0

    def add_message(self, sender: str, receiver: str) -> None:
        self.messages_between_agents += 1
        self.activated_agents.update({sender, receiver})

    def increment_turn(self) -> None:
        self.conversation_turns += 1

    def update_flow_depth(self, depth: int) -> None:
        self.information_flow_depth = max(self.information_flow_depth, int(depth))

    @property
    def num_activated_agents(self) -> int:
        return len(self.activated_agents)

    def get_summary(self) -> dict:
        return {
            "messages_between_agents": self.messages_between_agents,
            "conversation_turns": self.conversation_turns,
            "num_activated_agents": self.num_activated_agents,
            "information_flow_depth": self.information_flow_depth,
        }
