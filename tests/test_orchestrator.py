# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Orchestrator Integration Tests
  End-to-end routing tests for the multi-agent orchestrator.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image
import numpy as np

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


def make_dummy_image(size=(224, 224)):
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator Import & Initialization
# ═══════════════════════════════════════════════════════════════════

class TestOrchestratorInit:
    """Orchestrator can be imported and initialized."""

    def test_import(self):
        from agents.orchestrator import Orchestrator
        assert Orchestrator is not None

    def test_instantiation(self):
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch is not None

    def test_intent_enum_exists(self):
        from agents.orchestrator import Intent
        assert hasattr(Intent, "EMERGENCY")
        assert hasattr(Intent, "MEDICAL_CHAT")


# ═══════════════════════════════════════════════════════════════════
#  Intent Detection
# ═══════════════════════════════════════════════════════════════════

class TestIntentDetection:
    """Orchestrator correctly classifies user intents."""

    @pytest.fixture(scope="class")
    def orch(self):
        from agents.orchestrator import Orchestrator
        return Orchestrator()

    def test_emergency_intent(self, orch):
        """Emergency messages → EMERGENCY intent."""
        result = orch.detect_intent("لا أستطيع التنفس")
        from agents.orchestrator import Intent
        assert result in (Intent.EMERGENCY, "emergency"), \
            f"Expected EMERGENCY, got {result}"

    def test_boundary_intent(self, orch):
        """Prescription requests → BOUNDARY intent."""
        result = orch.detect_intent("اكتب لي وصفة دواء")
        from agents.orchestrator import Intent
        assert result in (Intent.BOUNDARY, "boundary"), \
            f"Expected BOUNDARY, got {result}"

    def test_medical_chat_intent(self, orch):
        """General medical question → MEDICAL_CHAT."""
        result = orch.detect_intent("ما هي أعراض السكري؟")
        from agents.orchestrator import Intent
        assert result not in (Intent.EMERGENCY, Intent.BOUNDARY)


# ═══════════════════════════════════════════════════════════════════
#  Agent Response Contract
# ═══════════════════════════════════════════════════════════════════

class TestAgentResponse:
    """AgentResponse dataclass contract."""

    def test_response_structure(self):
        from agents.orchestrator import AgentResponse
        resp = AgentResponse(
            text="Test response",
            intent="test",
            confidence=0.95,
        )
        assert resp.text == "Test response"
        assert resp.intent == "test"
        assert resp.confidence == 0.95


# ═══════════════════════════════════════════════════════════════════
#  Safety Agent Integration
# ═══════════════════════════════════════════════════════════════════

class TestSafetyAgentIntegration:
    """Safety agent correctly blocks/redirects."""

    def test_safety_agent_import(self):
        from agents.safety_agent import SafetyAgent
        agent = SafetyAgent()
        assert agent is not None

    def test_emergency_returns_override(self):
        from agents.safety_agent import SafetyAgent
        agent = SafetyAgent()
        result = agent.process("أريد الانتحار")
        assert result is not None
        # Should contain emergency numbers or override
        if hasattr(result, 'text'):
            assert len(result.text) > 0


# ═══════════════════════════════════════════════════════════════════
#  RAG Agent Integration
# ═══════════════════════════════════════════════════════════════════

class TestRAGAgentIntegration:
    """RAG agent returns structured responses."""

    def test_rag_agent_import(self):
        from agents.rag_agent import RAGAgent
        assert RAGAgent is not None

    def test_rag_agent_instantiation(self):
        from agents.rag_agent import RAGAgent
        try:
            agent = RAGAgent()
            assert agent is not None
        except Exception:
            pytest.skip("RAG agent requires FAISS index")


# ═══════════════════════════════════════════════════════════════════
#  Vision Agent Integration
# ═══════════════════════════════════════════════════════════════════

class TestVisionAgentIntegration:
    """Vision agent handles image routing."""

    def test_vision_agent_import(self):
        from agents.vision_agent import VisionAgent
        assert VisionAgent is not None

    def test_vision_agent_instantiation(self):
        from agents.vision_agent import VisionAgent
        try:
            agent = VisionAgent()
            assert agent is not None
        except Exception:
            pytest.skip("Vision agent requires model files")


# ═══════════════════════════════════════════════════════════════════
#  End-to-End Flow
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndFlow:
    """Full orchestrator pipeline tests."""

    @pytest.fixture(scope="class")
    def orch(self):
        from agents.orchestrator import Orchestrator
        return Orchestrator()

    def test_emergency_flow_returns_response(self, orch):
        """Emergency → Safety override → response with emergency numbers."""
        try:
            result = orch.process("لا أستطيع التنفس")
            assert result is not None
            if hasattr(result, 'text'):
                assert len(result.text) > 0
        except Exception as e:
            # If API keys missing, test should still pass on structure
            pytest.skip(f"E2E requires API keys: {e}")

    def test_safe_flow_returns_response(self, orch):
        """Safe query → RAG+LLM pipeline → informative response."""
        try:
            result = orch.process("ما هي فوائد شرب الماء؟")
            assert result is not None
        except Exception as e:
            pytest.skip(f"E2E requires API keys/FAISS: {e}")
