# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Location Agent
  Finds nearby hospitals/clinics/doctors using OpenStreetMap Overpass API.
  100% free — no API key required.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from agents.base_agent import BaseAgent, AgentResponse
from engine.nearby_care import get_nearby_doctors


class LocationAgent(BaseAgent):
    """Geolocation-based hospital finder via OpenStreetMap."""

    name = "location"

    def __init__(self):
        super().__init__()

    # ── Core API ────────────────────────────────────────────────
    def run(self, *, query: str, context: dict | None = None) -> AgentResponse:
        """Search for nearby healthcare facilities.

        Expected *context* keys:
            lat     (float) — Latitude  (default: Rabat 33.9716).
            lng     (float) — Longitude (default: Rabat -6.8498).
            radius  (int)   — Search radius in metres (default: 5000).

        Returns:
            AgentResponse with a list of places in metadata["places"].
        """
        ctx = context or {}
        lat    = ctx.get("lat", 33.9716)
        lng    = ctx.get("lng", -6.8498)
        radius = ctx.get("radius", 5000)

        places = get_nearby_doctors(lat, lng, radius)

        if not places:
            return AgentResponse(
                success=False,
                answer="لم يتم العثور على مراكز صحية في هذا النطاق.\n📞 اتصل بـ SAMU مباشرة : **15**",
                metadata={"places": []},
                agent_name=self.name,
            )

        return AgentResponse(
            success=True,
            answer=f"تم العثور على {len(places)} مرافق صحية",
            metadata={"places": places},
            agent_name=self.name,
        )

    def health_check(self) -> bool:
        return True
