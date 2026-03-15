from fastapi import APIRouter, HTTPException, Depends
from models.schemas import DiagnosticRequest, DiagnosticResponse
from services.ai_service import process_diagnostic_request

router = APIRouter(prefix="/diagnostic", tags=["diagnostic"])

@router.post("/", response_model=DiagnosticResponse)
async def get_diagnostic(request: DiagnosticRequest):
    """
    Endpoint to receive patient symptoms and return a structured AI diagnosis.
    """
    try:
        response = await process_diagnostic_request(request)
        return response
    except HTTPException as http_exc:
        # Re-raise FastAPI HTTP exceptions (e.g. AI service unavailable)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during diagnosis: {str(e)}")
