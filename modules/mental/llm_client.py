# -*- coding: utf-8 -*-
"""
SHIFA-Mental · Client LLM (OpenRouter + Groq fallback)
Supporte le mode Darija/MSA avec system prompts dédiés.
Intègre le safety post-check.
"""

import os
import logging
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("shifa.mental.llm")


class MentalLLMClient:
    """
    LLM client for the mental health module.

    Priority chain:
      1. OpenRouter API (anthropic/claude-3-haiku)
      2. Groq API (llama-3.3-70b-versatile) as fallback
      3. Rule-based fallback (no API needed)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._groq_key = os.getenv("GROQ_API_KEY", "")

        # Import safety post-checker
        try:
            from engine.safety import SafetyGuard
            self._safety = SafetyGuard()
            logger.info("[MentalLLM] Safety layer loaded")
        except ImportError:
            self._safety = None
            logger.warning("[MentalLLM] Safety layer unavailable")

    def chat(
        self,
        messages: list[dict],
        system_prompt: str,
        dialect: str = "msa",
    ) -> str:
        """
        Send a conversation to the LLM with the given system prompt.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: The full system prompt (MSA or Darija)
            dialect: 'msa' or 'darija' — for logging

        Returns:
            The assistant response text.
        """
        # Try OpenRouter first
        if self._openrouter_key:
            response = self._call_openrouter(messages, system_prompt)
            if response:
                return self._post_process(response)

        # Fallback to Groq
        if self._groq_key:
            response = self._call_groq(messages, system_prompt)
            if response:
                return self._post_process(response)

        logger.warning("[MentalLLM] No API available, using rule-based fallback")
        return ""

    def _call_openrouter(self, messages: list[dict], system_prompt: str) -> Optional[str]:
        """Call OpenRouter API with the SHIFA-Mental system prompt."""
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://shifa-ai.ma",
            "X-Title": "SHIFA-Mental"
        }
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "anthropic/claude-3-haiku",
            "max_tokens": 512,
            "messages": full_messages,
            "temperature": 0.75
        }
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("[MentalLLM] OpenRouter response OK (model=claude-3-haiku)")
            return content
        except Exception as e:
            logger.error("[MentalLLM] OpenRouter error: %s", str(e)[:120])
            return None

    def _call_groq(self, messages: list[dict], system_prompt: str) -> Optional[str]:
        """Fallback: call Groq API directly."""
        try:
            from groq import Groq
            client = Groq(api_key=self._groq_key)
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                messages=full_messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            logger.info("[MentalLLM] Groq fallback response OK")
            return content
        except Exception as e:
            logger.error("[MentalLLM] Groq error: %s", str(e)[:120])
            return None

    def _post_process(self, response: str) -> str:
        """Apply safety post-checks on LLM output."""
        if self._safety:
            response = self._safety.post_check(response)
        return response
