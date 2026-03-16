from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import datetime

# --- Pydantic Schemas for Request/Response Validation ---

class DocumentBase(BaseModel):
    filename: str
    document_type: str
    confidence_score: float
    ocr_provider: str = "local"
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
    
    class Config:
        from_attributes = True

class ExtractionResult(BaseModel):
    status: str
    message: str
    document_id: Optional[int] = None
    time_taken_ms: float
    data: Optional[Dict[str, Any]] = None
