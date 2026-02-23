# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Audio Module — Arabic Speech-to-Text & Text-to-Speech
  STT : SpeechRecognition (Google Web Speech API — ar-MA / ar-SA)
  TTS : gTTS (Google Text-to-Speech — Arabic)
═══════════════════════════════════════════════════════════════════════
"""

import os
import io
import tempfile
import logging

logger = logging.getLogger(__name__)

# ── Text-to-Speech ──────────────────────────────────────────────────
def text_to_speech_arabic(text: str) -> bytes | None:
    """
    Convert Arabic text to speech audio bytes (MP3).
    Returns raw MP3 bytes or None on failure.
    """
    try:
        from gtts import gTTS
        # Clean text: remove markdown-style symbols for cleaner audio
        clean_text = (
            text.replace("**", "")
                .replace("*", "")
                .replace("##", "")
                .replace("#", "")
                .replace("🩺", "")
                .replace("⚠️", "")
                .replace("🚨", "")
                .replace("💊", "")
                .replace("🏥", "")
                .replace("📋", "")
                .replace("✅", "")
                .replace("❌", "")
                .strip()
        )

        tts = gTTS(text=clean_text, lang="ar", slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()

    except ImportError:
        logger.error("gTTS not installed. Run: pip install gTTS")
        return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


# ── Speech-to-Text ───────────────────────────────────────────────────
def speech_to_text_arabic(audio_bytes: bytes) -> tuple[str, str]:
    """
    Convert audio bytes (WAV/WebM) to Arabic text via Google Speech API.
    Returns (transcribed_text, error_message).
    """
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.pause_threshold = 0.8

        # Write raw bytes to a temp file so AudioFile can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)

            # Try Moroccan Arabic first, then Modern Standard Arabic
            for lang in ["ar-MA", "ar-SA", "ar"]:
                try:
                    text = recognizer.recognize_google(audio_data, language=lang)
                    if text:
                        return text, ""
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    return "", f"خطأ في الاتصال بخدمة التعرف على الصوت: {e}"

            return "", "لم أتمكن من فهم الكلام. حاول مرة أخرى."

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except ImportError:
        return "", "مكتبة SpeechRecognition غير مثبتة."
    except Exception as e:
        logger.error(f"STT error: {e}")
        return "", f"حدث خطأ: {str(e)}"


# ── Convert WebM/OGG to WAV (pydub helper) ──────────────────────────
def convert_audio_to_wav(audio_bytes: bytes, src_format: str = "webm") -> bytes | None:
    """
    Convert WebM/OGG audio (from browser recorder) to WAV for SpeechRecognition.
    Requires pydub and ffmpeg. Fallback to original bytes if it fails.
    """
    try:
        from pydub import AudioSegment
        audio_buffer = io.BytesIO(audio_bytes)
        # Attempt conversion
        segment = AudioSegment.from_file(audio_buffer, format=src_format)
        # Convert to mono 16kHz WAV
        segment = segment.set_channels(1).set_frame_rate(16000)
        wav_buffer = io.BytesIO()
        segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        # Check if it's a known ffmpeg issue
        err_str = str(e).lower()
        if "ffmpeg" in err_str or "avconv" in err_str:
            logger.warning("FFmpeg not found. Using raw audio bytes as fallback.")
        else:
            logger.error(f"Audio conversion error: {e}")
        return audio_bytes
