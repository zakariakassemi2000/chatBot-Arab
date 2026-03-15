import logging
from groq import AsyncGroq
from openai import AsyncOpenAI
from fastapi import HTTPException

from core.config import settings
from services.prompt_builder import build_diagnostic_prompt
from services.parser import parse_ai_response
from models.schemas import DiagnosticRequest, DiagnosticResponse

# Configure Groq client
groq_client = None
if settings.GROQ_API_KEY:
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)

# Configure OpenAI fallback
openai_client = None
if settings.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

logger = logging.getLogger("shifaa-ia.ai_service")

async def get_diagnostic_from_groq(prompt: str) -> str:
    if not groq_client:
        raise ValueError("GROQ_API_KEY is not set")
    try:
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are 'Shifaa IA', a medical assistant. Always reply with perfectly formatted JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise

async def get_diagnostic_from_openai(prompt: str) -> str:
    if not openai_client:
        raise ValueError("OPENAI_API_KEY is not set")
    try:
        response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are 'Shifaa IA', a medical assistant. Always reply with perfectly formatted JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

async def process_diagnostic_request(request: DiagnosticRequest) -> DiagnosticResponse:
    prompt = build_diagnostic_prompt(request)
    raw_response = None
    
    # Primary: Try Groq First
    try:
        raw_response = await get_diagnostic_from_groq(prompt)
    except Exception as groq_err:
        logger.warning(f"Groq failed, falling back to OpenAI: {groq_err}")
        # Secondary: Try OpenAI Fallback
        try:
            raw_response = await get_diagnostic_from_openai(prompt)
        except Exception as openai_err:
            logger.error(f"OpenAI fallback also failed: {openai_err}")
            raise HTTPException(status_code=503, detail="AI diagnostic services are currently unavailable.")
    
    return parse_ai_response(raw_response)
