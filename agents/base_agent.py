# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Base Agent Interface
  All agents inherit from this abstract class to guarantee a uniform API.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from utils.logger import get_logger


@dataclass
class AgentResponse:
    """Standardised response returned by every agent.

    Attributes:
        success:       Whether the agent completed without errors.
        answer:        The textual answer (Arabic markdown).
        severity:      Risk level — "critique" | "élevée" | "modérée" | "faible" | None.
        metadata:      Arbitrary extra data (probabilities, flags, urls …).
        override_ui:   If not None, the Streamlit layer should render this
                       HTML block *instead of* the default chat bubble.
        agent_name:    Which agent produced this response.
    """
    success: bool = True
    answer: str = ""
    severity: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    override_ui: Optional[str] = None
    agent_name: str = ""


class BaseAgent(ABC):
    """Contract every SHIFA agent must honour."""

    name: str = "base"

    def __init__(self):
        self.logger = get_logger(f"shifa.agent.{self.name}")

    # ── Public API ──────────────────────────────────────────────
    @abstractmethod
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Execute the agent's primary task.

        Args:
            query:   The user's input (text, or a description of the task).
            context: Optional dict with session-level data
                     (history, images, coordinates, etc.).

        Returns:
            AgentResponse with the result.
        """
        ...

    def health_check(self) -> bool:
        """Return True if the agent's dependencies are available."""
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
