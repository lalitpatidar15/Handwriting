from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import datetime


class BaseSchema(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

# --- Pydantic Schemas for Request/Response Validation ---

class DocumentBase(BaseSchema):
    filename: str
    document_type: str
    confidence_score: float
    ocr_provider: str = "local"
    layout_provider: str = "local"
    reasoning_provider: str = "local"
    processing_time_ms: float = 0.0

class DocumentCreate(DocumentBase):
    original_image_path: str
    processed_image_path: Optional[str] = None
    extracted_text: Optional[str] = None
    extracted_json: Optional[Dict[str, Any]] = None

class DocumentResponse(DocumentBase):
    id: int
    upload_time: datetime.datetime
    extracted_text: Optional[str] = None
    extracted_json: Optional[Dict[str, Any]] = None
    review_status: str = "pending_review"
    schema_version: str = "1.0"
    document_domain: str = "general"
    template_id: Optional[int] = None
    template_match_score: float = 0.0
    
class FieldPredictionResponse(BaseSchema):
    id: int
    document_id: int
    field_name: str
    field_type: str
    predicted_value: Optional[str] = None
    corrected_value: Optional[str] = None
    confidence_score: float
    source_engine: str
    review_status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

class ReviewTaskResponse(BaseSchema):
    id: int
    document_id: int
    field_prediction_id: int
    status: str
    priority: str
    predicted_value: Optional[str] = None
    corrected_value: Optional[str] = None
    reviewer_name: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

class ModelRunResponse(BaseSchema):
    id: int
    document_id: int
    stage: str
    model_name: str
    provider: str
    success: str
    duration_ms: float
    raw_output: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime


class TemplateResponse(BaseSchema):
    id: int
    template_name: str
    tenant_id: str
    document_type: str
    document_domain: str
    fingerprint: Dict[str, Any]
    sample_count: int
    approval_status: str
    last_matched_at: datetime.datetime
    created_at: datetime.datetime


class ExtractionBlockResponse(BaseSchema):
    id: int
    document_id: int
    page_number: int
    block_type: str
    text: Optional[str] = None
    bbox: Optional[Any] = None
    confidence_score: float
    source_engine: str
    created_at: datetime.datetime

class ReviewTaskCompleteRequest(BaseSchema):
    corrected_value: Optional[str] = None
    reviewer_name: Optional[str] = None
    review_notes: Optional[str] = None
    resolution: str = "corrected"


class DocumentDetailResponse(DocumentResponse):
    field_predictions: List[FieldPredictionResponse] = []
    review_tasks: List[ReviewTaskResponse] = []
    model_runs: List[ModelRunResponse] = []
    extraction_blocks: List[ExtractionBlockResponse] = []
    template: Optional[TemplateResponse] = None

class ExtractionResult(BaseModel):
    status: str
    message: str
    document_id: Optional[int] = None
    time_taken_ms: float
    data: Optional[Dict[str, Any]] = None


class SignupRequest(BaseSchema):
    username: str
    email: str
    password: str


class LoginRequest(BaseSchema):
    email: str
    password: str


class AuthResponse(BaseSchema):
    status: str
    user_id: int
    username: str
    email: str
    token: str
