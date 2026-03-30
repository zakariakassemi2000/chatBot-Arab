# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Vision Agent
  Delegates medical image analysis to the correct model via VisionRouter.
  Supports: dermato, xray, brain_mri, cancer, breast.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Optional
from PIL import Image
from agents.base_agent import BaseAgent, AgentResponse
from engine.vision_router import VisionRouter


# Arabic labels for vision types
VISION_TYPE_LABELS = {
    "dermato":   "🔴 فحص الجلد",
    "xray":      "🫁 أشعة الصدر",
    "brain_mri": "🧠 رنين الدماغ",
    "cancer":    "🩺 كشف السرطان",
    "breast":    "🔬 كثافة الثدي",
}


class VisionAgent(BaseAgent):
    """Medical image analysis with dynamic model loading."""

    name = "vision"

    def __init__(self):
        super().__init__()
        self._router = VisionRouter()

    # ── Core API ────────────────────────────────────────────────
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Analyse a medical image.

        Expected *context* keys:
            image        (PIL.Image)  — The uploaded image.
            vision_type  (str)        — "dermato" | "xray" | "brain_mri" | "cancer" | "breast"

        Returns:
            AgentResponse with classification, confidence, severity, etc.
        """
        ctx = context or {}
        image: Optional[Image.Image] = ctx.get("image")
        vision_type: str = ctx.get("vision_type", "")

        if image is None:
            return AgentResponse(
                success=False,
                answer="يرجى تحميل الصورة أولاً.",
                agent_name=self.name,
            )

        if vision_type not in VISION_TYPE_LABELS:
            return AgentResponse(
                success=False,
                answer=f"نوع التحليل غير مدعوم: {vision_type}",
                agent_name=self.name,
            )

        try:
            result = self._router.analyze(image, vision_type)
        except Exception as e:
            self.logger.error("Vision analysis error: %s", e, exc_info=True)
            return AgentResponse(
                success=False,
                answer="حدث خطأ أثناء تحليل الصورة. يرجى إعادة المحاولة.",
                agent_name=self.name,
            )

        is_valid = result.get("valid", False)
        severity = result.get("severity")
        confidence = result.get("confidence", 0.0)

        return AgentResponse(
            success=is_valid,
            answer=result.get("recommendation_ar", ""),
            severity=severity,
            metadata={
                "class":            result.get("class"),
                "confidence":       confidence,
                "all_probs":        result.get("all_probs", {}),
                "gradcam":          result.get("gradcam"),
                "urgency":          result.get("urgency"),
                "vision_type":      vision_type,
                "rejection_reason": result.get("rejection_reason"),
            },
            agent_name=self.name,
        )

    def health_check(self) -> bool:
        return True
