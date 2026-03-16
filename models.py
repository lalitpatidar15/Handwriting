from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
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
