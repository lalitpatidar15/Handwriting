from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from database import Base
import datetime

class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Classification Information
    document_type = Column(String, default="UNKNOWN")  # e.g., HANDWRITTEN_NOTE, STRUCTURED_FORM
    confidence_score = Column(Float, default=0.0)
    
    # Storage URLs/Paths
    original_image_path = Column(String)
    processed_image_path = Column(String)
    
    # Extracted Intelligence Data
    extracted_text = Column(Text, nullable=True)
    extracted_json = Column(JSON, nullable=True) # Full structured extraction (tables, key-values)
    
    # Processing Metadata (Cloud vs Local)
    ocr_provider = Column(String, default="local") # 'local', 'gemini', 'openai'
    processing_time_ms = Column(Float, default=0.0)
    review_status = Column(String, default="pending_review")
    schema_version = Column(String, default="1.0")
    document_domain = Column(String, default="general")
    template_id = Column(Integer, nullable=True)
    template_match_score = Column(Float, default=0.0)
    layout_provider = Column(String, default="local")
    reasoning_provider = Column(String, default="local")


class TemplateRecord(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    template_name = Column(String, index=True, nullable=False)
    tenant_id = Column(String, default="default")
    document_type = Column(String, default="UNKNOWN")
    document_domain = Column(String, default="general")
    fingerprint = Column(JSON, nullable=False)
    sample_count = Column(Integer, default=1)
    approval_status = Column(String, default="learned")
    last_matched_at = Column(DateTime, default=datetime.datetime.utcnow)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ExtractionBlock(Base):
    __tablename__ = "extraction_blocks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    page_number = Column(Integer, default=1)
    block_type = Column(String, default="word")
    text = Column(Text, nullable=True)
    bbox = Column(JSON, nullable=True)
    confidence_score = Column(Float, default=0.0)
    source_engine = Column(String, default="local")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class FieldPrediction(Base):
    __tablename__ = "field_predictions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    field_name = Column(String, index=True, nullable=False)
    field_type = Column(String, default="text")
    predicted_value = Column(Text, nullable=True)
    corrected_value = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)
    source_engine = Column(String, default="local")
    review_status = Column(String, default="pending_review")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class ReviewTask(Base):
    __tablename__ = "review_tasks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    field_prediction_id = Column(Integer, ForeignKey("field_predictions.id"), index=True, nullable=False)
    status = Column(String, default="open")
    priority = Column(String, default="medium")
    predicted_value = Column(Text, nullable=True)
    corrected_value = Column(Text, nullable=True)
    reviewer_name = Column(String, nullable=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    stage = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    provider = Column(String, default="local")
    success = Column(String, default="true")
    duration_ms = Column(Float, default=0.0)
    raw_output = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class UserAccount(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
