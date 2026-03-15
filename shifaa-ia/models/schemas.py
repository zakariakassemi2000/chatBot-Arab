from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class GenderEnum(str, Enum):
    male = "male"
    female = "female"

class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class UrgencyEnum(str, Enum):
    home_care = "home_care"
    consult_doctor = "consult_doctor"
    emergency = "emergency"

class DiagnosticRequest(BaseModel):
    patient_age: int = Field(..., ge=0, le=120, description="Age of the patient")
    gender: GenderEnum = Field(..., description="Gender of the patient")
    symptoms: List[str] = Field(..., min_length=1, description="List of current symptoms")
    duration_days: int = Field(..., ge=1, description="Duration of symptoms in days")
    medical_history: Optional[str] = Field(None, description="Previous medical history if any")

class DiagnosticResponse(BaseModel):
    probable_disease: str = Field(..., description="The most probable disease or condition")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of AI inference")
    severity: SeverityEnum = Field(..., description="Assessed severity level")
    recommendation: str = Field(..., description="Brief recommended next steps")
    urgency: UrgencyEnum = Field(..., description="Evaluation of the medical urgency")
    disclaimer: str = Field(..., description="Standard medical AI disclaimer")
