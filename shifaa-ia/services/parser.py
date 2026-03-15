import json
import logging
from typing import Any, Dict
from fastapi import HTTPException
from models.schemas import DiagnosticResponse

logger = logging.getLogger("shifaa-ia.parser")

def clean_json_response(raw_response: str) -> str:
    """Removes markdown code blocks and whitespace from AI response."""
    raw_response = raw_response.strip()
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:]
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:]
        
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3]
    
    return raw_response.strip()

def parse_ai_response(raw_response: str) -> DiagnosticResponse:
    """Parses raw AI string into a validated Pydantic model."""
    cleaned_json = clean_json_response(raw_response)
    
    try:
        parsed_data = json.loads(cleaned_json)
        return DiagnosticResponse(**parsed_data)
    except json.JSONDecodeError as decode_err:
        logger.error(f"Failed to parse JSON response: {cleaned_json}")
        raise HTTPException(
            status_code=500, 
            detail="AI returned invalid JSON format."
        )
    except Exception as parse_err:
        logger.error(f"Failed to validate response: {parse_err}")
        raise HTTPException(
            status_code=500, 
            detail="AI response did not match expected diagnostic schema."
        )
