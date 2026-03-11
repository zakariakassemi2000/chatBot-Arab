# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — RAG Pipeline (Retriever + Classifier)
  Validates FAISS retrieval, intent classification, and knowledge base.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier, format_response


@pytest.fixture(scope="module")
def retriever():
    """Load the pre-built FAISS retriever."""
    r = FAISSRetriever()
    loaded = r.load()
    if not loaded:
        pytest.skip("FAISS index not found — run setup.py first")
    return r


@pytest.fixture(scope="module")
def classifier():
    """Load the pre-trained intent classifier."""
    c = IntentClassifier()
    loaded = c.load()
    if not loaded:
        pytest.skip("Classifier model not found — run setup.py first")
    return c


class TestRetriever:
    """Test FAISS semantic retrieval."""

    def test_search_returns_results(self, retriever):
        results = retriever.search("ما هي أعراض السكري؟")
        assert len(results) > 0
        assert "answer" in results[0]
        assert "score" in results[0]

    def test_search_relevance(self, retriever):
        results = retriever.search("صداع شديد مع حرارة")
        assert len(results) > 0
        assert results[0]["score"] > 0.3

    def test_search_empty_query(self, retriever):
        results = retriever.search("")
        # Should either return empty or low-score results
        assert isinstance(results, list)

    def test_best_answer_returns_tuple(self, retriever):
        answer, score, category, intent = retriever.get_best_answer("ألم في المعدة")
        assert answer is not None or score == 0.0

    def test_encode_query_shape(self, retriever):
        emb = retriever.encode_query("سؤال تجريبي")
        assert emb.shape == (384,)  # MiniLM dimension


class TestClassifier:
    """Test intent classification."""

    def test_predict_returns_tuple(self, retriever, classifier):
        emb = retriever.encode_query("أشعر بألم شديد في رأسي")
        intent, confidence = classifier.predict(emb)
        assert isinstance(intent, str)
        assert 0.0 <= confidence <= 1.0

    def test_symptom_intent(self, retriever, classifier):
        emb = retriever.encode_query("أعاني من ألم في الصدر وضيق تنفس")
        intent, _ = classifier.predict(emb)
        assert intent in ["وصف_أعراض", "استشارة_طارئة"]

    def test_info_intent(self, retriever, classifier):
        emb = retriever.encode_query("ما هو مرض السكري وما أسبابه؟")
        intent, _ = classifier.predict(emb)
        assert intent in ["طلب_معلومات", "وصف_أعراض"]

    def test_top_k_returns_list(self, retriever, classifier):
        emb = retriever.encode_query("كيف أعالج الصداع")
        top_k = classifier.predict_top_k(emb, k=3)
        assert len(top_k) <= 3
        assert all(isinstance(t, tuple) for t in top_k)


class TestFormatResponse:
    """Test response formatting with intent templates."""

    def test_format_symptom(self):
        result = format_response("الصداع قد يكون بسبب التوتر", "وصف_أعراض")
        assert "الصداع" in result
        assert "نصيحة" in result  # Suffix should be added

    def test_format_emergency(self):
        result = format_response("اتصل بالإسعاف فوراً", "استشارة_طارئة")
        assert "طوارئ" in result or "عاجل" in result or "اتصل" in result
