# -*- coding: utf-8 -*-
"""
LLM Module - Groq Integration
Handles generation of high-quality Arabic medical responses using retrieved context.
Supports multi-turn conversation history for contextual memory.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from utils.logger import get_logger

# Load environment variables (GROQ_API_KEY)
load_dotenv()

logger = get_logger("shifa.llm")

# System prompt — injected once as the "system" role
SYSTEM_PROMPT = """أنت مساعد طبي ذكي ومحترف اسمه "شِفاء". مهمتك هي الإجابة على استفسارات المستخدمين باللغة العربية الفصحى وبشكل ودود ومبسط.

التعليمات الأساسية:
1. كن دقيقاً طبياً ولكن استخدم لغة يفهمها الشخص العادي.
2. لا تقم بالتشخيص النهائي، بل قدم معلومات وتوجيهات عامة.
3. إذا كان هناك شك، انصح دائماً بمراجعة الطبيب المختص.
4. حافظ على نبرة احترافية ورحيمة ومتعاطفة.
5. لا تذكر أنك حصلت على معلومات من "سياق" أو "قاعدة بيانات"، تحدث كطبيب مباشرة.
6. تذكر سياق المحادثة السابقة وابنِ عليه في إجاباتك.
7. إذا سبق أن ذكر المستخدم أعراضاً أو معلومات صحية، استخدمها لتقديم إجابات أكثر دقة."""


class GroqGenerator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.client = None
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")

        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            logger.warning("GROQ_API_KEY not found — LLM engine disabled.")

    def generate_answer(self, query: str, context: str, intent: str = "", history: list = None) -> str | None:
        """
        Generates a professional Arabic medical answer using context and conversation history.

        Args:
            query:   The current user question.
            context: Retrieved knowledge base context.
            intent:  Detected intent label (optional).
            history: List of previous messages [{"role": "user"|"assistant", "content": "..."}]
                     Used to maintain conversational memory across turns.

        Returns:
            Generated response string, or None on failure.
        """
        if not self.client:
            return None

        # ── Build the messages list ──────────────────────────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject previous turns (limit to last 10 to control token usage)
        if history:
            recent_history = history[-10:]
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content", "")
                # Only include valid roles and non-empty content
                if role in ("user", "assistant") and content.strip():
                    messages.append({"role": role, "content": content})

        # Build the current user message with KB context
        intent_line = f"\nنية المستخدم (Intent): {intent}" if intent else ""
        current_user_msg = f"""السياق المستخرج من قاعدة المعرفة:
{context}
{intent_line}

سؤال المستخدم الحالي:
{query}

الإجابة:"""

        messages.append({"role": "user", "content": current_user_msg})

        # ── Call Groq API ────────────────────────────────────────────────
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.3,
                max_tokens=800,
            )
            response = chat_completion.choices[0].message.content
            logger.info("LLM response generated (intent=%s, history_turns=%d)",
                        intent or "none", len(history) if history else 0)
            return response

        except Exception as e:
            logger.error("Groq API error: %s", e, exc_info=True)
            return None
