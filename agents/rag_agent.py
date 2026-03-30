# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · RAG Agent (Production)
  ────────────────────────────────────────────────────────────────────
  Uses HybridRetriever (FAISS + BM25 + cross-encoder reranking)
  to return top-3 high-quality contexts for LLM enrichment.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from agents.base_agent import BaseAgent, AgentResponse
from engine.retriever import HybridRetriever
from engine.classifier import IntentClassifier, format_response


class RAGAgent(BaseAgent):
    """Production RAG agent: hybrid search + reranking."""

    name = "rag"

    def __init__(self):
        super().__init__()
        self._retriever = HybridRetriever(enable_reranker=False)
        self._classifier = IntentClassifier()
        self._db_ready = False

    # ── Lifecycle ───────────────────────────────────────────────
    def load(self) -> bool:
        """Load FAISS + BM25 index and intent classifier from disk."""
        self._db_ready = self._retriever.load()
        if self._db_ready:
            self._classifier.load()
        self.logger.info("RAG agent load → db_ready=%s", self._db_ready)
        return self._db_ready

    def setup(self, max_samples: int = 8000):
        """First-time build: download datasets → build indices → train classifier."""
        from data.knowledge_base import load_and_prepare_datasets

        df = load_and_prepare_datasets(max_samples=max_samples)
        embeddings = self._retriever.build_index(df, verbose=True)
        self._retriever.save()

        self._classifier.train(embeddings, df["intent"].tolist(), verbose=True)
        self._classifier.save()

        self._db_ready = True

    # ── Core API ────────────────────────────────────────────────
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Retrieve top-3 KB contexts for a query.

        context keys used:
            category  (str, optional) — filter results by medical category

        Returns:
            AgentResponse with metadata containing:
            - intent, confidence
            - kb_context    (str)   — best single answer for backward compat
            - kb_contexts   (list)  — top-3 answer dicts (question, answer, score, category)
            - category, score
        """
        if not self._db_ready:
            return AgentResponse(
                success=False,
                answer="قاعدة المعرفة غير متاحة حالياً.",
                agent_name=self.name,
            )

        ctx = context or {}
        category_filter = ctx.get("category")

        # Classify intent
        enc = self._retriever.encode_query(query)
        try:
            intent, intent_conf = self._classifier.predict(enc)
        except Exception:
            intent, intent_conf = "general", 0.0
            self.logger.warning("Intent classification failed", exc_info=True)

        # Retrieve top-3 contexts (hybrid pipeline)
        top_contexts = self._retriever.get_top_contexts(
            query, top_k=3, category=category_filter
        )

        if not top_contexts:
            return AgentResponse(
                success=False,
                answer="عذراً، لا أملك إجابة دقيقة حالياً. يرجى استشارة طبيب مختص.",
                metadata={"intent": intent, "confidence": intent_conf},
                agent_name=self.name,
            )

        best = top_contexts[0]
        kb_answer = best["answer"]
        formatted = format_response(kb_answer, intent)

        # Build combined context string for LLM (top-3 fused)
        combined_context = "\n\n---\n\n".join(
            c["answer"] for c in top_contexts
        )

        return AgentResponse(
            success=True,
            answer=formatted,
            metadata={
                "intent": intent,
                "confidence": intent_conf,
                "category": best.get("category", "عام"),
                "score": best.get("final_score", 0.0),
                "kb_context": combined_context,      # Top-3 fused for LLM
                "kb_contexts": top_contexts,          # Full structured list
                "retrieval_stages": {
                    "dense_score": best.get("dense_score"),
                    "bm25_score": best.get("bm25_score"),
                    "rrf_score": best.get("rrf_score"),
                    "ce_score": best.get("ce_score"),
                },
            },
            agent_name=self.name,
        )

    # ── Helpers exposed for other agents ────────────────────────
    def encode_query(self, query: str):
        """Return the embedding vector for a query."""
        return self._retriever.encode_query(query)

    def get_raw_answer(self, query: str):
        """Return (answer, score, category, intent) — backward compat."""
        return self._retriever.get_best_answer(query)

    def health_check(self) -> bool:
        return self._db_ready

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def classifier(self) -> IntentClassifier:
        return self._classifier
