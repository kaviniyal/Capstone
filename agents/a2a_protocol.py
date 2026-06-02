"""
Agent-to-Agent (A2A) communication protocol for escalation workflows.

Agents post typed messages to a shared in-process channel. Downstream agents
read those messages and adjust their decisions accordingly, enabling direct
agent-to-agent communication beyond simple shared state.

Message types:
  ESCALATE  — sender recommends immediate escalation (high fraud risk)
  FLAG      — sender flags a specific policy or pattern concern
  APPROVE   — sender indicates low risk, safe to approve
  INFO      — informational signal, no action required
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class A2AMessage:
    sender: str
    receiver: str
    message_type: str        # ESCALATE | FLAG | APPROVE | INFO
    subject: str             # short description
    payload: dict[str, Any]
    timestamp: str = ""      # set by send(); empty until sent


class A2AChannel:
    """
    In-process message bus for a single analysis run.
    Agents call send() to post messages; downstream agents call receive()
    to drain their inbox before making decisions.
    """

    def __init__(self) -> None:
        self._inbox: dict[str, list[A2AMessage]] = {}

    def send(self, message: A2AMessage) -> None:
        from datetime import datetime, timezone
        message.timestamp = datetime.now(timezone.utc).isoformat()
        self._inbox.setdefault(message.receiver, []).append(message)

    def receive(self, agent_name: str) -> list[A2AMessage]:
        """Drain and return all messages addressed to agent_name."""
        return self._inbox.pop(agent_name, [])

    def peek(self, agent_name: str) -> list[A2AMessage]:
        """Return messages without draining."""
        return list(self._inbox.get(agent_name, []))

    def to_serializable(self) -> dict[str, list[dict]]:
        """Convert entire channel to JSON-serialisable form for LangGraph state storage."""
        return {
            agent: [asdict(m) for m in msgs]
            for agent, msgs in self._inbox.items()
        }

    def clear(self) -> None:
        self._inbox.clear()


def make_channel() -> A2AChannel:
    """Factory — create a fresh channel for one analysis run."""
    return A2AChannel()


# ── Helpers to build common message types ────────────────────────────────────

def escalation_message(sender: str, receiver: str, fraud_probability: float, reason: str) -> A2AMessage:
    return A2AMessage(
        sender=sender,
        receiver=receiver,
        message_type="ESCALATE",
        subject=f"High fraud probability detected: {fraud_probability:.2f}",
        payload={"fraud_probability": fraud_probability, "reason": reason},
    )


def flag_message(sender: str, receiver: str, flag: str, detail: str) -> A2AMessage:
    return A2AMessage(
        sender=sender,
        receiver=receiver,
        message_type="FLAG",
        subject=flag,
        payload={"detail": detail},
    )


def approval_message(sender: str, receiver: str, confidence: float) -> A2AMessage:
    return A2AMessage(
        sender=sender,
        receiver=receiver,
        message_type="APPROVE",
        subject=f"Low risk — safe to approve (confidence={confidence:.2f})",
        payload={"confidence": confidence},
    )


def messages_to_state_list(messages: list[A2AMessage]) -> list[dict]:
    """Serialise for storage in LangGraph ClaimState (a2a_messages field)."""
    return [asdict(m) for m in messages]
