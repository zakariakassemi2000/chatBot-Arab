# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · LLM Agent
  Wraps the Groq LLaMA 3.3 70B generator.  Takes a user query
  plus KB context (from RAG Agent) and produces a concise Arabic
  medical answer honouring the system prompt rules.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from agents.base_agent import BaseAgent, AgentResponse
from engine.llm import GroqGenerator


class LLMAgent(BaseAgent):
    """Generative answer enrichment via Groq Cloud LLM."""

    name = "llm"

    def __init__(self):
        super().__init__()
        self._generator = GroqGenerator()

    # ── Core API ────────────────────────────────────────────────
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Generate a rich Arabic medical answer.

        Expected *context* keys:
            kb_context  (str)  — RAG-retrieved answer text.
            intent      (str)  — Classified intent label.
            history     (list) — Previous chat messages for memory.

        Returns:
            AgentResponse with LLM-generated answer.
        """
        ctx = context or {}
        kb_text = ctx.get("kb_context", "")
        intent  = ctx.get("intent", "")
        history = ctx.get("history", None)

        if not self.health_check():
            return AgentResponse(
                success=False,
                answer="",
                metadata={"reason": "GROQ_API_KEY not configured"},
                agent_name=self.name,
            )

        answer = self._generator.generate_answer(
            query=query,
            context=kb_text,
            intent=intent,
            history=history,
        )

        if not answer:
            return AgentResponse(
                success=False,
                answer="",
                metadata={"reason": "LLM returned empty response"},
                agent_name=self.name,
            )

        return AgentResponse(
            success=True,
            answer=answer,
            agent_name=self.name,
        )

    def health_check(self) -> bool:
        return bool(self._generator.client)
