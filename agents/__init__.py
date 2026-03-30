# -*- coding: utf-8 -*-
"""
SHIFA AI — Agent-Based Architecture
Each agent encapsulates a single domain of responsibility.
The Orchestrator routes user input to the appropriate agent(s).
"""

from agents.orchestrator import Orchestrator

__all__ = ["Orchestrator"]
