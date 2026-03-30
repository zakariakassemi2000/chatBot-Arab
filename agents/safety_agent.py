# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Safety Agent (with Triage Classifier)
  ────────────────────────────────────────────────────────────────────
  Dual-engine safety gate:
    Engine 1: SafetyGuard    — rule-based (boundaries, payload, cardiac)
    Engine 2: TriageClassifier — multi-signal scoring (5 signals)

  The two engines run in parallel.  The HIGHER severity wins.
  This prevents both false negatives (triage catches what rules miss)
  and false positives (rules enforce hard boundaries the scorer can't).
═══════════════════════════════════════════════════════════════════════
"""

from agents.base_agent import BaseAgent, AgentResponse
from engine.safety import SafetyGuard
from engine.triage import MedicalTriageClassifier, RiskLevel, triage_to_safety_level


# Severity ordering for "higher wins" merge
_SEVERITY_ORDER = {"safe": 0, "caution": 1, "boundary": 2, "blocked": 2, "emergency": 3}


class SafetyAgent(BaseAgent):
    """Pre-flight safety gate — runs BEFORE other agents.

    Combines:
      • SafetyGuard       (rule-based keyword + regex detection)
      • TriageClassifier   (multi-signal scored classification)

    Returns the more severe of the two assessments.
    """

    name = "safety"

    def __init__(self):
        super().__init__()
        self._guard = SafetyGuard()
        self._triage = MedicalTriageClassifier()

    # ── Core API ────────────────────────────────────────────────
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Screen the user message for emergencies, boundaries, and risk.

        Returns:
            AgentResponse where:
            - metadata["level"]        = "safe" | "emergency" | "caution" | "boundary" | "blocked"
            - metadata["flags"]        = list of triggered keywords
            - metadata["triage_score"] = float (0–1)
            - metadata["triage_risk"]  = "emergency" | "moderate" | "safe"
            - metadata["actions"]      = list of Arabic recommended actions
            - If emergency/boundary → answer is pre-filled, override_ui may be set
        """
        # ── Engine 1: SafetyGuard (rules) ───────────────────────
        guard_result = self._guard.check(query)
        guard_level  = guard_result["level"]
        guard_flags  = guard_result.get("flags", [])

        # ── Engine 2: TriageClassifier (scoring) ────────────────
        triage_result   = self._triage.classify(query)
        triage_as_guard = triage_to_safety_level(triage_result)
        triage_level    = triage_as_guard["level"]

        # ── Merge: higher severity wins ─────────────────────────
        guard_sev  = _SEVERITY_ORDER.get(guard_level, 0)
        triage_sev = _SEVERITY_ORDER.get(triage_level, 0)

        if guard_sev >= triage_sev:
            # Guard wins (or tie → prefer guard for its richer rules)
            final_level    = guard_level
            final_response = guard_result.get("override_response", "")
            final_flags    = guard_flags
        else:
            # Triage wins — scored higher severity
            final_level    = triage_level
            final_response = triage_as_guard.get("override_response", "")
            final_flags    = triage_as_guard.get("flags", [])

        # ── Build metadata ──────────────────────────────────────
        metadata = {
            "level":         final_level,
            "flags":         final_flags,
            "stop":          final_level in ("emergency", "boundary", "blocked"),
            "triage_score":  triage_result.score,
            "triage_risk":   triage_result.risk_level.value,
            "triage_flags":  triage_result.flags,
            "actions":       triage_result.actions,
            "guard_level":   guard_level,
        }

        # ── Emergency / boundary → build full response ──────────
        if metadata["stop"]:
            override_html = None
            is_emergency = final_level == "emergency" or guard_result.get("emergency")

            if is_emergency:
                override_html = self._emergency_html(
                    score=triage_result.score,
                    actions=triage_result.actions,
                )

            return AgentResponse(
                success=True,
                answer=final_response,
                severity="critique" if is_emergency else None,
                metadata=metadata,
                override_ui=override_html,
                agent_name=self.name,
            )

        # ── Safe / caution → pass through ───────────────────────
        return AgentResponse(
            success=True,
            answer="",
            severity=None,
            metadata=metadata,
            agent_name=self.name,
        )

    # ── Triage-only method (for external use) ───────────────────
    def triage(self, message: str):
        """Run the triage classifier only.  Returns a TriageResult."""
        return self._triage.classify(message)

    # ── Post-processing helpers (called by orchestrator) ────────
    def post_check(self, answer: str) -> str:
        """Scan an LLM-generated answer for forbidden patterns."""
        return self._guard.post_check(answer)

    def add_disclaimer(self, answer: str) -> str:
        """Append medical disclaimer if not already present."""
        return self._guard.add_disclaimer(answer)

    def format_for_severity(self, answer: str, severity: str) -> str:
        """Append severity-specific footer (SAMU, caution, etc.)."""
        return self._guard.format_response(answer, severity)

    def format_caution(self, answer: str) -> str:
        """Prepend caution notice to a normal answer."""
        return self._guard.format_caution_response(answer)

    # ── Private ─────────────────────────────────────────────────
    @staticmethod
    def _emergency_html(
        score: float = 1.0,
        actions: list[str] | None = None,
    ) -> str:
        """Build the animated emergency HTML block for Streamlit."""
        import html as _html
        action_items = ""
        if actions:
            items = "".join(f"<li>{_html.escape(str(a))}</li>" for a in actions[:4])
            action_items = f"""
            <ul style="text-align:right; color:#FCA5A5; font-size:14px;
                        list-style:none; padding:0; margin:12px 0 0;">
              {items}
            </ul>"""

        return f"""
        <div style="
          background: rgba(220,38,38,0.15);
          border: 2px solid #DC2626;
          border-radius: 16px;
          padding: 24px;
          text-align: center;
          direction: rtl;
          animation: pulse 1s infinite;
        ">
        <h2 style="color:#DC2626; margin:0;">🚨 حالة طوارئ</h2>
        <h3 style="color:#FCA5A5; margin:8px 0;">اتصل بالإسعاف فوراً — 📞 15</h3>
        <div style="background:rgba(220,38,38,0.1); border-radius:8px;
                    padding:4px 12px; display:inline-block; margin-top:8px;">
          <span style="color:#FCA5A5; font-size:12px;">
            درجة الخطورة: {score:.0%}
          </span>
        </div>
        {action_items}
        </div>
        <style>
        @keyframes pulse {{
          0%   {{ border-color: #DC2626; }}
          50%  {{ border-color: #FCA5A5; }}
          100% {{ border-color: #DC2626; }}
        }}
        </style>
        """

    def health_check(self) -> bool:
        return True
