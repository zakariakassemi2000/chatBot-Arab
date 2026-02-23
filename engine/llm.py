# -*- coding: utf-8 -*-
"""
LLM Module - Groq Integration
Handles generation of high-quality Arabic medical responses using retrieved context.
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY)
load_dotenv()

class GroqGenerator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.client = None
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def generate_answer(self, query, context, intent=""):
        """
        Generates a professional Arabic medical answer using context.
        """
        if not self.client:
            return None

        prompt = f"""
أنت مساعد طبي ذكي ومحترف. مهمتك هي الإجابة على استفسارات المستخدمين باللغة العربية الفصحى وبشكل ودود ومبسط.
استخدم السياق (Context) المقدم أدناه للإجابة على السؤال. إذا لم يكن السياق كافياً، اعتمد على معرفتك الطبية العامة مع الحفاظ على الأمان.

السياق المستخرج من قاعدة المعرفة:
{context}

سؤال المستخدم:
{query}

نية المستخدم (Intent): 
{intent}

التعليمات:
1. كن دقيقاً طبياً ولكن استخدم لغة يفهمها الشخص العادي.
2. لا تقم بالتشخيص النهائي، بل قدم معلومات وتوجيهات.
3. إذا كان هناك شك، انصح دائماً بمراجعة الطبيب.
4. حافظ على نبرة احترافية ورحيمة.
5. لا تذكر أنك حصلت على معلومات من "سياق" أو "قاعدة بيانات"، تحدث كطبيب مباشرة.

الإجابة:
"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=800,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating with Groq: {e}")
            return None
