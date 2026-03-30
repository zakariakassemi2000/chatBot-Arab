# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Intelligent Orchestrator
  ────────────────────────────────────────────────────────────────────
  Central coordinator that:
    1. Detects the intent of every user input (text or image)
    2. Routes to the correct agent pipeline
    3. Combines + post-processes outputs into a final response

  Supported intents:
    EMERGENCY     → SafetyAgent stops everything, returns emergency response
    BOUNDARY      → SafetyAgent blocks (prescription, diagnosis requests)
    IMAGE_ANALYSIS→ VisionAgent runs the correct CV model
    LOCATION      → LocationAgent queries OpenStreetMap
    MEDICAL_CHAT  → RAGAgent + LLMAgent pipeline
    SYMPTOM_SCAN  → Structured symptom analysis via RAG + LLM
    GENERAL_CHAT  → Fallback RAG + LLM

  Design:
    • Every public method returns AgentResponse  (uniform contract)
    • engine/ modules are untouched              (zero regression)
    • Compatible with @st.cache_resource         (stateless facade)
    • Easy to extend: add a new agent, register its intent
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from PIL import Image

from agents.base_agent import AgentResponse
from agents.safety_agent import SafetyAgent
from agents.rag_agent import RAGAgent
from agents.llm_agent import LLMAgent
from agents.vision_agent import VisionAgent
from agents.location_agent import LocationAgent
from engine.triage import MedicalTriageClassifier, RiskLevel
from utils.logger import get_logger

_triage_classifier = MedicalTriageClassifier()

logger = get_logger("shifa.orchestrator")


# ═════════════════════════════════════════════════════════════════
#  Intent Detection
# ═════════════════════════════════════════════════════════════════

class Intent(Enum):
    """All intents the orchestrator can route."""
    EMERGENCY      = auto()
    BOUNDARY       = auto()
    BLOCKED        = auto()
    IMAGE_ANALYSIS = auto()
    LOCATION       = auto()
    SYMPTOM_SCAN   = auto()
    MEDICAL_CHAT   = auto()
    GENERAL_CHAT   = auto()


@dataclass
class DetectedIntent:
    """Result of intent detection."""
    intent: Intent
    confidence: float
    safety_resp: Optional[AgentResponse] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ── Keyword sets for rule-based detection ───────────────────────
_LOCATION_KW = [
    "مستشفى قريب", "أقرب مستشفى", "أين أقرب", "مستشفى بالقرب",
    "عيادة قريبة", "مركز صحي قريب", "طوارئ قريبة",
    "hôpital proche", "nearest hospital", "nearby clinic",
]

_SYMPTOM_SCAN_KW = [
    "فحص الأعراض", "تحليل الأعراض", "فاحص الأعراض",
    "أعاني من عدة أعراض", "scan symptoms",
]


class IntentDetector:
    """Rule-based + classifier-backed intent detector.

    Detection priority (highest → lowest):
      1. Safety override (emergency / boundary / blocked)
      2. Image analysis  (presence of PIL.Image in context)
      3. Location query   (keyword match)
      4. Symptom scan     (keyword match or structured form)
      5. Medical chat     (default — RAG classifier refines)
    """

    def __init__(self, safety: SafetyAgent):
        self._safety = safety

    def detect(
        self,
        query: str,
        *,
        image: Optional[Image.Image] = None,
        vision_type: Optional[str] = None,
        is_symptom_form: bool = False,
    ) -> DetectedIntent:
        """Detect the intent of a user request.

        Args:
            query:           User's text input.
            image:           Optional uploaded image.
            vision_type:     Optional vision model selector.
            is_symptom_form: True when called from the symptom scanner page.

        Returns:
            DetectedIntent with the routing decision.
        """
        # ── Priority 1: Safety pre-check ────────────────────────
        safety_resp = self._safety.run(query=query)
        level = safety_resp.metadata.get("level", "safe")

        if level == "emergency":
            return DetectedIntent(
                intent=Intent.EMERGENCY,
                confidence=1.0,
                safety_resp=safety_resp,
            )
        if level == "boundary":
            return DetectedIntent(
                intent=Intent.BOUNDARY,
                confidence=1.0,
                safety_resp=safety_resp,
            )
        if level == "blocked":
            return DetectedIntent(
                intent=Intent.BLOCKED,
                confidence=1.0,
                safety_resp=safety_resp,
            )

        # ── Priority 2: Image analysis ──────────────────────────
        if image is not None and vision_type:
            return DetectedIntent(
                intent=Intent.IMAGE_ANALYSIS,
                confidence=1.0,
                safety_resp=safety_resp,
                metadata={"image": image, "vision_type": vision_type},
            )

        # ── Priority 3: Location query ──────────────────────────
        q_lower = query.lower() if query else ""
        for kw in _LOCATION_KW:
            if kw in q_lower:
                return DetectedIntent(
                    intent=Intent.LOCATION,
                    confidence=0.9,
                    safety_resp=safety_resp,
                )

        # ── Priority 4: Symptom scan (form-based) ───────────────
        if is_symptom_form:
            return DetectedIntent(
                intent=Intent.SYMPTOM_SCAN,
                confidence=1.0,
                safety_resp=safety_resp,
            )
        for kw in _SYMPTOM_SCAN_KW:
            if kw in q_lower:
                return DetectedIntent(
                    intent=Intent.SYMPTOM_SCAN,
                    confidence=0.8,
                    safety_resp=safety_resp,
                )

        # ── Priority 5: Medical chat (default) ──────────────────
        return DetectedIntent(
            intent=Intent.MEDICAL_CHAT,
            confidence=0.7,
            safety_resp=safety_resp,
        )


# ═════════════════════════════════════════════════════════════════
#  Orchestrator
# ═════════════════════════════════════════════════════════════════

class Orchestrator:
    """Intelligent request router — detects intent, dispatches to agents,
    combines outputs into a final AgentResponse.

    Usage::

        # In app.py — cached singleton
        @st.cache_resource(show_spinner=False)
        def load_system():
            return Orchestrator.load()

        orch = load_system()
        response = orch.handle(user_msg)           # text
        response = orch.handle("", image=img,      # image
                               vision_type="xray")
    """

    def __init__(self):
        self.safety   = SafetyAgent()
        self.rag      = RAGAgent()
        self.llm      = LLMAgent()
        self.vision   = VisionAgent()
        self.location = LocationAgent()

        self._detector = IntentDetector(self.safety)

    # ── Factory ─────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "Orchestrator":
        """Build an Orchestrator and load all persistent state.

        Designed to be called inside ``@st.cache_resource``.
        """
        orch = cls()
        orch.rag.load()
        logger.info(
            "Orchestrator ready — rag=%s  llm=%s",
            orch.rag.health_check(),
            orch.llm.health_check(),
        )
        return orch

    # ═════════════════════════════════════════════════════════════
    #  UNIFIED ENTRY POINT
    # ═════════════════════════════════════════════════════════════
    def handle(
        self,
        query: str,
        *,
        image: Optional[Image.Image] = None,
        vision_type: Optional[str] = None,
        history: list | None = None,
        is_symptom_form: bool = False,
        symptom_meta: dict | None = None,
        location_meta: dict | None = None,
    ) -> AgentResponse:
        """Unified entry point — auto-detects intent and routes.

        Args:
            query:           User text input.
            image:           Optional PIL image for vision analysis.
            vision_type:     Vision model type (if image provided).
            history:         Chat history for LLM memory.
            is_symptom_form: True when calling from the scanner page.
            symptom_meta:    Dict with age, gender, duration, severity, etc.
            location_meta:   Dict with lat, lng, radius_km.

        Returns:
            AgentResponse with the combined result.
        """
        # ── Step 1: Detect intent ───────────────────────────────
        detected = self._detector.detect(
            query,
            image=image,
            vision_type=vision_type,
            is_symptom_form=is_symptom_form,
        )

        logger.info(
            "Intent detected: %s (confidence=%.2f)",
            detected.intent.name,
            detected.confidence,
        )

        # ── Step 2: Route to the correct pipeline ───────────────
        route_map = {
            Intent.EMERGENCY:      self._handle_emergency,
            Intent.BOUNDARY:       self._handle_safety_stop,
            Intent.BLOCKED:        self._handle_safety_stop,
            Intent.IMAGE_ANALYSIS: self._handle_image,
            Intent.LOCATION:       self._handle_location,
            Intent.SYMPTOM_SCAN:   self._handle_symptoms,
            Intent.MEDICAL_CHAT:   self._handle_chat,
            Intent.GENERAL_CHAT:   self._handle_chat,
        }

        handler = route_map.get(detected.intent, self._handle_chat)
        return handler(
            query=query,
            detected=detected,
            history=history,
            symptom_meta=symptom_meta,
            location_meta=location_meta,
        )

    # ═════════════════════════════════════════════════════════════
    #  ROUTE HANDLERS (private)
    # ═════════════════════════════════════════════════════════════

    def _handle_emergency(self, *, query, detected, **_kw) -> AgentResponse:
        """Emergency — return SafetyAgent response immediately."""
        resp = detected.safety_resp
        resp.metadata["intent"] = Intent.EMERGENCY.name
        return resp

    def _handle_safety_stop(self, *, query, detected, **_kw) -> AgentResponse:
        """Boundary or blocked — return SafetyAgent response."""
        resp = detected.safety_resp
        resp.metadata["intent"] = detected.intent.name
        return resp

    def _handle_image(self, *, query, detected, **_kw) -> AgentResponse:
        """Image analysis — delegate to VisionAgent."""
        meta = detected.metadata or {}
        resp = self.vision.run(
            query=query,
            context={
                "image": meta.get("image"),
                "vision_type": meta.get("vision_type"),
            },
        )
        resp.metadata["intent"] = Intent.IMAGE_ANALYSIS.name
        return resp

    def _handle_location(
        self, *, query, detected, location_meta=None, **_kw
    ) -> AgentResponse:
        """Location query — delegate to LocationAgent."""
        loc = location_meta or {}
        resp = self.location.run(
            query=query,
            context={
                "lat": loc.get("lat", 33.9716),
                "lng": loc.get("lng", -6.8498),
                "radius": loc.get("radius_km", 5) * 1000,
            },
        )
        resp.metadata["intent"] = Intent.LOCATION.name
        return resp

    def _handle_symptoms(
        self, *, query, detected, history=None, symptom_meta=None, **_kw
    ) -> AgentResponse:
        """Symptom scanner — triage pre-analysis + direct LLM clinical report."""
        meta = symptom_meta or {}

        # ── Step 1: Fast triage classification (no DB needed) ────
        triage = _triage_classifier.classify(query)
        risk_emoji  = triage.risk_level.emoji
        risk_label  = triage.risk_level.label_ar
        triage_actions = "\n".join(f"• {a}" for a in triage.actions)

        # ── Step 2: Build a rich clinical prompt for LLM ─────────
        chronic = meta.get('medical_history') or 'لا يوجد'
        prompt = (
            f"أنت طبيب مساعد ذكي تابع لمنصة SHIFA AI. قدِّم تقريراً سريرياً أولياً وافياً باللغة العربية."
            f"\n\n### بيانات المريض\n"
            f"- الجنس: {meta.get('gender', 'غير محدد')}\n"
            f"- العمر: {meta.get('age', '?')} سنة\n"
            f"- مدة الأعراض: {meta.get('duration', '')}\n"
            f"- شدة الألم: {meta.get('severity', '')}\n"
            f"- الأمراض المزمنة / الأدوية: {chronic}\n"
            f"\n### الأعراض الموصوفة\n{query}\n"
            f"\n### مستوى درجة الخطر (نظام SHIFA للتصنيف)\n"
            f"{risk_emoji} {risk_label}\n"
            f"\n### المهمة\n"
            f"1. حدد الحالات المحتملة (الأكثر ترجيحاً أولاً) مع مبرر طبي موجز لكل حالة.\n"
            f"2. اذكر الأعراض التحذيرية (علامات الخطر) التي تستوجب طلب الطوارئ فوراً.\n"
            f"3. قدم توصيات عملية (فحوصات مقترحة، تعديلات نمط حياة، أدوية OTC بحذر).\n"
            f"4. ضع في الحسبان التاريخ المرضي المزمن ({chronic}) وتأثيره على التشخيص.\n"
            f"5. لا تنسَ الإشارة إلى ضرورة مراجعة الطبيب المختص دائماً.\n"
            f"اكتب بأسلوب طبي مهني ومفهوم للمريض العادي."
        )

        # ── Step 3: Call LLM directly (bypass RAG for symptom forms) ─
        llm_resp = self.llm.run(
            query=prompt,
            context={"kb_context": "", "intent": "symptom_scan", "history": history},
        )

        if llm_resp.success and llm_resp.answer:
            llm_answer = llm_resp.answer
            llm_answer = self.safety.post_check(llm_answer)
        else:
            # Ultimate fallback: structured triage-only response
            llm_answer = (
                f"**التقرير الأولي بناءً على الأعراض المُدخلة:**\n\n"
                f"{risk_emoji} **مستوى الخطر:** {risk_label}\n\n"
                f"**التوصيات الفورية:**\n{triage_actions}\n\n"
                f"---\n"
                f"> ⚠️ هذا تقييم أولي آلي. يرجى مراجعة طبيب مختص للتشخيص الدقيق."
            )

        # ── Step 4: Prepend triage badge + actions ───────────────
        triage_header = (
            f"{risk_emoji} **مستوى الخطر:** {risk_label}  "
            f"| درجة الخطورة: `{triage.score:.0%}`\n\n"
            f"**الإجراءات الموصى بها:**\n{triage_actions}\n\n"
            f"---\n\n"
        )
        final_answer = triage_header + llm_answer
        final_answer = self.safety.add_disclaimer(final_answer)

        return AgentResponse(
            success=True,
            answer=final_answer,
            metadata={
                "intent": Intent.SYMPTOM_SCAN.name,
                "triage_score":  triage.score,
                "triage_risk":   triage.risk_level.value,
                "safety_level":  detected.safety_resp.metadata.get("level", "safe") if detected.safety_resp else "safe",
            },
            agent_name="orchestrator",
        )

    def _handle_chat(
        self, *, query, detected, history=None, **_kw
    ) -> AgentResponse:
        """Medical / general chat — the full RAG + LLM pipeline."""
        resp = self._run_chat_pipeline(query, history=history, safety_resp=detected.safety_resp)
        resp.metadata["intent"] = detected.intent.name
        return resp

    # ═════════════════════════════════════════════════════════════
    #  CORE CHAT PIPELINE (reusable)
    # ═════════════════════════════════════════════════════════════

    def _run_chat_pipeline(
        self,
        query: str,
        *,
        history: list | None = None,
        safety_resp: AgentResponse | None = None,
    ) -> AgentResponse:
        """Internal: safety → RAG → LLM → post-check.

        Args:
            query:       The user text (may be a synthetic prompt).
            history:     Chat history for LLM memory.
            safety_resp: Pre-computed safety response (avoids double-check).
        """
        # Safety level from pre-check (already done in detect phase)
        safety_level = "safe"
        if safety_resp:
            safety_level = safety_resp.metadata.get("level", "safe")

        # ── RAG retrieval ───────────────────────────────────────
        rag_resp = self.rag.run(query=query)

        if not rag_resp.success:
            # RAG failed (empty DB) — try LLM-only fallback before giving up
            logger.warning("RAG failed — attempting LLM-only fallback for query")
            if self.llm.health_check():
                llm_only = self.llm.run(
                    query=query,
                    context={"kb_context": "", "intent": "general", "history": history},
                )
                if llm_only.success and llm_only.answer:
                    answer = self.safety.post_check(llm_only.answer)
                    answer = self.safety.add_disclaimer(answer)
                    return AgentResponse(
                        success=True,
                        answer=answer,
                        metadata={"rag_intent": "general", "safety_level": safety_level, "llm_fallback": True},
                        agent_name="orchestrator",
                    )
            return AgentResponse(
                success=True,
                answer="عذراً، لا أملك إجابة دقيقة حالياً. يرجى استشارة طبيب مختص فاحص الاعراض.",
                agent_name="orchestrator",
            )

        intent     = rag_resp.metadata.get("intent", "")
        kb_context = rag_resp.metadata.get("kb_context", "")

        # ── LLM enrichment ──────────────────────────────────────
        if self.llm.health_check():
            llm_resp = self.llm.run(
                query=query,
                context={
                    "kb_context": kb_context,
                    "intent": intent,
                    "history": history,
                },
            )
            if llm_resp.success:
                answer = llm_resp.answer
                answer = self.safety.post_check(answer)
            else:
                answer = rag_resp.answer
        else:
            answer = rag_resp.answer

        # ── Safety decorations ──────────────────────────────────
        if safety_level == "caution":
            answer = self.safety.format_caution(answer)

        answer = self.safety.add_disclaimer(answer)

        return AgentResponse(
            success=True,
            answer=answer,
            metadata={
                "rag_intent": intent,
                "safety_level": safety_level,
                "safety_flags": (
                    safety_resp.metadata.get("flags", []) if safety_resp else []
                ),
            },
            agent_name="orchestrator",
        )

    # ═════════════════════════════════════════════════════════════
    #  CONVENIENCE METHODS (backward-compat + direct calls)
    # ═════════════════════════════════════════════════════════════

    def chat(self, query: str, *, history: list | None = None) -> AgentResponse:
        """Shortcut: text-only chat through handle()."""
        return self.handle(query, history=history)

    def analyze_image(self, image: Image.Image, vision_type: str) -> AgentResponse:
        """Shortcut: image analysis through handle()."""
        return self.handle("", image=image, vision_type=vision_type)

    def find_hospitals(
        self, lat: float = 33.9716, lng: float = -6.8498, radius_km: int = 5
    ) -> AgentResponse:
        """Shortcut: location through handle()."""
        return self.handle(
            "أقرب مستشفى",
            location_meta={"lat": lat, "lng": lng, "radius_km": radius_km},
        )

    def scan_symptoms(
        self,
        symptoms: str,
        *,
        age: int = 30,
        gender: str = "ذكر",
        duration: str = "",
        severity: str = "",
        medical_history: str = "",
    ) -> AgentResponse:
        """Shortcut: symptom scanner through handle()."""
        return self.handle(
            symptoms,
            is_symptom_form=True,
            symptom_meta={
                "age": age,
                "gender": gender,
                "duration": duration,
                "severity": severity,
                "medical_history": medical_history,
            },
        )

    # ── KB setup ────────────────────────────────────────────────
    def setup_knowledge_base(self, max_samples: int = 8000) -> None:
        """Download datasets, build FAISS, train classifier."""
        self.rag.setup(max_samples=max_samples)

    # ── Health ──────────────────────────────────────────────────
    @property
    def db_ready(self) -> bool:
        return self.rag.health_check()

    @property
    def llm_ready(self) -> bool:
        return self.llm.health_check()

    def status(self) -> dict:
        """Return health status of all agents."""
        return {
            "safety":   self.safety.health_check(),
            "rag":      self.rag.health_check(),
            "llm":      self.llm.health_check(),
            "vision":   self.vision.health_check(),
            "location": self.location.health_check(),
        }
