# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Voice Pipeline
  ────────────────────────────────────────────────────────────────────
  Lightweight voice AI pipeline:
    1. Speech-to-Text (Arabic STT)
    2. Agent Orchestration (Intent -> Safety -> RAG -> LLM)
    3. Text-to-Speech (Arabic TTS)
═══════════════════════════════════════════════════════════════════════
"""

import os
import speech_recognition as sr
from gtts import gTTS
from dataclasses import dataclass

from agents.orchestrator import Orchestrator

@dataclass
class VoiceResponse:
    """Structured output for the Voice Pipeline."""
    success: bool
    user_text: str = ""
    bot_text: str = ""
    audio_path: str = ""
    error: str = ""

class VoicePipeline:
    """
    End-to-end Voice AI processing: Audio In -> Processing -> Audio Out.
    """
    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator
        self._recognizer = sr.Recognizer()

    def _speech_to_text(self, audio_file_path: str) -> str:
        """Converts an audio file (.wav format) to Arabic text."""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self._recognizer.record(source)
                # Use Google's free lightweight STT (Arabic)
                text = self._recognizer.recognize_google(audio_data, language="ar-SA")
                return text
        except sr.UnknownValueError:
            raise ValueError("لم أتمكن من فهم الصوت، يرجى التحدث بوضوح أكثر.")
        except sr.RequestError as e:
            raise ConnectionError(f"حدث خطأ في خدمة التعرف على الصوت: {e}")

    def _text_to_speech(self, text: str, output_path: str = "response.mp3") -> str:
        """Converts Arabic text to spoken audio using gTTS."""
        tts = gTTS(text=text, lang="ar", slow=False)
        tts.save(output_path)
        return output_path

    def process_audio(self, audio_file_path: str, output_audio_path: str = "response.mp3") -> VoiceResponse:
        """
        Runs the full Voice AI Pipeline: STT -> Agents -> TTS.
        """
        # 1. Transcribe the audio
        try:
            user_text = self._speech_to_text(audio_file_path)
        except Exception as e:
            return VoiceResponse(success=False, error=str(e))

        # 2. Route through Orchestrator
        try:
            agent_response = self._orchestrator.chat(user_text)
            bot_text = agent_response.answer
            
            clean_text = bot_text.replace("*", "").replace("#", "").replace("🚨", "")
            if not clean_text.strip():
                clean_text = "عذراً، لم أتمكن من إيجاد إجابة."
        except Exception as e:
            return VoiceResponse(success=False, user_text=user_text, error=f"فشل معالجة النص: {e}")

        # 3. Synthesize the voice response
        try:
            audio_out = self._text_to_speech(clean_text, output_audio_path)
        except Exception as e:
             return VoiceResponse(success=False, user_text=user_text, bot_text=bot_text, error=f"فشل توليد الصوت: {e}")

        return VoiceResponse(
            success=True,
            user_text=user_text,
            bot_text=bot_text,
            audio_path=audio_out
        )
