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
SYSTEM_PROMPT = """
أنت مساعد طبي ذكي في نظام شفاء AI.

قواعد الإجابة الإلزامية :
1. الجواب القصير دائماً : 3 إلى 5 أسطر فقط — ممنوع الإطالة
2. البنية الثابتة :
   السطر 1 : التشخيص المحتمل مباشرة بدون مقدمة
   السطر 2 : توصية واحدة واضحة ومحددة
   السطر 3 : مستوى الخطورة (خفيف/متوسط/مرتفع/حرج)
   السطر 4 : رقم الطوارئ فقط إذا كانت الحالة خطيرة
3. ممنوع : مقدمات مثل "أنا هنا لمساعدتك"
4. ممنوع : تكرار الأعراض التي ذكرها المريض
5. ممنوع : نصائح lifestyle إلا إذا طُلب صراحةً
6. الاستخدام الإلزامي للسياق المُقدَّم من قاعدة المعرفة
7. تذكير واحد فقط في النهاية : هذا التحليل لا يغني عن الطبيب
"""


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


# ─── GroqVision — Analyse d'images médicales ────────────────────────

VISION_SYSTEM_PROMPT = """أنت مساعد طبي متخصص في تحليل الصور الطبية في نظام شفاء AI.

عند تحليل أي صورة طبية، اتبع هذا الهيكل الإلزامي:

**1. نوع الصورة:** (أشعة X / رنين مغناطيسي / صورة جلدية / فحص عيون / غير طبية)
**2. الملاحظات الرئيسية:** ما تراه في الصورة بدقة (3-5 نقاط)
**3. التشخيص المحتمل:** أكثر الاحتمالات الطبية منطقية
**4. مستوى الخطورة:** خفيف / متوسط / مرتفع / حرج
**5. التوصية الطبية:** خطوة واحدة واضحة ومحددة

قواعد صارمة:
- أجب دائماً باللغة العربية الفصحى
- لا تؤكد التشخيص القاطع — استخدم "يُحتمل" أو "يُشير إلى"
- أضف دائماً: "هذا التحليل لا يغني عن استشارة طبيب مختص"
- إذا كانت الصورة غير طبية، قل ذلك بوضوح
"""


class GroqVision:
    """
    Client Groq Vision pour l'analyse d'images médicales.
    Utilise le modèle llama-3.2-90b-vision-preview (Groq).
    """

    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self):
        self.client = None
        self.api_key = os.environ.get("GROQ_API_KEY")

        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("GroqVision initialized — model: %s", self.VISION_MODEL)
            except Exception as e:
                logger.error("GroqVision init error: %s", e)
        else:
            logger.warning("GROQ_API_KEY not found — GroqVision disabled.")

    def analyze_image(
        self,
        base64_image: str,
        prompt: str = "",
        image_type: str = "auto",
        history: list = None,
    ) -> str:
        """
        Analyse une image médicale encodée en base64.

        Args:
            base64_image: Image encodée en base64.
            prompt:       Question ou instruction de l'utilisateur.
            image_type:   Type d'image ('xray', 'mri', 'dermato', 'auto').
            history:      Historique de conversation précédent.

        Returns:
            Réponse textuelle en arabe, ou message d'erreur.
        """
        if not self.client:
            return "❌ خدمة تحليل الصور غير متاحة — يرجى التحقق من مفتاح GROQ_API_KEY."

        # Prompt automatique selon le type d'image
        type_hints = {
            "xray":    "هذه صورة أشعة X. حللها بدقة.",
            "mri":     "هذه صورة رنين مغناطيسي (MRI). حللها بدقة.",
            "dermato": "هذه صورة جلدية. حلل الحالة الجلدية الظاهرة.",
            "eye":     "هذه صورة فحص عيون. حلل ما تراه.",
            "auto":    "حلل هذه الصورة الطبية.",
        }
        type_hint = type_hints.get(image_type, type_hints["auto"])
        user_question = prompt.strip() if prompt.strip() else type_hint

        # Construction des messages
        messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT}]

        # Historique de conversation (limite 6 derniers tours)
        if history:
            for msg in history[-6:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content.strip():
                    messages.append({"role": role, "content": content})

        # Message avec image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": user_question,
                },
            ],
        })

        try:
            response = self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            logger.info("GroqVision analysis completed (type=%s)", image_type)
            return answer

        except Exception as e:
            logger.error("GroqVision API error: %s", e, exc_info=True)
            return f"❌ حدث خطأ أثناء تحليل الصورة: {str(e)}"

    def detect_image_type(self, base64_image: str) -> str:
        """
        Détecte automatiquement le type de l'image médicale.

        Returns:
            'xray' | 'mri' | 'dermato' | 'eye' | 'general' | 'non_medical'
        """
        if not self.client:
            return "general"

        detection_prompt = """انظر إلى هذه الصورة وحدد نوعها من بين الخيارات التالية فقط:
- xray (أشعة X)
- mri (رنين مغناطيسي)
- dermato (صورة جلدية)
- eye (فحص عيون)
- dental (أسنان)
- general (صورة طبية عامة)
- non_medical (غير طبية)

أجب بكلمة واحدة فقط من القائمة أعلاه."""

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": detection_prompt},
            ],
        }]

        try:
            response = self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().lower()
            valid = {"xray", "mri", "dermato", "eye", "dental", "general", "non_medical"}
            return result if result in valid else "general"
        except Exception:
            return "general"
