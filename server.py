from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
import time

# Import Enterprise Modules
from database import engine, Base, get_db
import models, schemas
from main import DocumentIntelligenceSystem
from universal_parser import UniversalDataIntelligence
from cloud_ocr import cloud_ocr

# Create Database Tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="DOC-INTEL Enterprise API",
    description="Intelligent Document Processing API layer",
    version="1.0.0"
)

# CORS config to allow Streamlit UI or Mobile Apps to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create UPLOADS directory for Cloud Storage simulation
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Lazy-load AI systems to optimize server boot
system = None
intelligence = None

@app.on_event("startup")
def startup_event():
    global system, intelligence
    print("🚀 API Booting up...")
    print("🧠 Warming up DOC-INTEL systems... (This may take 30s)")
    try:
        system = DocumentIntelligenceSystem()
        intelligence = UniversalDataIntelligence()
        print("✅ AI Engines Online!")
    except Exception as e:
        print(f"⚠️ Warning: Full AI Engine couldn't start: {e}. API will start anyway.")

@app.get("/")
def read_root():
    return {"message": "Welcome to DOC-INTEL Enterprise API. Status: Online"}

@app.post("/api/v1/extract", response_model=schemas.ExtractionResult)
async def process_document(
    file: UploadFile = File(...),
    provider: str = Form("local"), # local, gemini, openai
    force_mode: str = Form(None), # None, HANDWRITTEN_NOTE, STRUCTURED_FORM
    high_fidelity: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Main API Endpoint for processing a document.
    Saves the image locally (simulating S3), processes it, and stores the structured JSON in the DB.
    """
    start_time = time.time()
    
    # 1. Secure Storage (Simulation of S3)
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    unique_filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"[API] File saved safely to {file_path}")

    doc_analysis = {"type": "UNKNOWN", "confidence_score": 0.0}
    full_text = ""
    extracted_json = {}

    try:
        if provider in ["gemini", "openai"]:
            # Cloud Auto-Extraction Path
            full_text = cloud_ocr(file_path, provider=provider)
            doc_analysis["type"] = "CLOUD_PROCESSED"
            doc_analysis["confidence_score"] = 0.99
            extracted_json = {"full_text": full_text, "cloud_raw_output": True}
        else:
            # Local Vanguard Engine Path
            if system is None or intelligence is None:
                raise HTTPException(status_code=500, detail="Local AI engine not loaded.")
            
            # Step 1: Vision Enhancement
            clean_img = system.ocr.vision.enhance_image(file_path)
            enhanced_path = os.path.join(UPLOAD_DIR, f"clean_{unique_filename}")
            import cv2
            cv2.imwrite(enhanced_path, clean_img)

            # Step 2: OCR
            ocr_results = system.ocr.extract_text_from_image(
                file_path, pre_cleaned=clean_img, paragraph=False, high_fidelity=high_fidelity
            )

            # Step 3: Parse and Structure (Intelligence Layer)
            doc_analysis, extracted_data, all_rows = intelligence.parse_universal(
                ocr_results, force_mode=force_mode
            )
            
            full_text = extracted_data.get("full_text", "")
            if not full_text and "table" in extracted_data:
                # If table mode, create string representation as fallback text
                import json
                full_text = json.dumps(extracted_data, indent=2)
                
            extracted_json = extracted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Pipeline Failed: {str(e)}")

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000

    # 4. Save to Database
    db_document = models.DocumentRecord(
        filename=file.filename,
        document_type=str(doc_analysis.get("type", "UNKNOWN")),
        confidence_score=float(doc_analysis.get("confidence_score", 0.0)),
        original_image_path=file_path,
        processed_image_path=enhanced_path if provider == "local" else None,
        extracted_text=full_text,
        extracted_json=extracted_json,
        ocr_provider=provider,
        processing_time_ms=processing_time
    )
    
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # 5. Return Output
    return {
        "status": "success",
        "message": f"Successfully processed via {provider}",
        "document_id": db_document.id,
        "time_taken_ms": processing_time,
        "data": {
            "analysis": doc_analysis,
            "extraction": extracted_json
        }
    }

@app.get("/api/v1/documents", response_model=List[schemas.DocumentResponse])
def get_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Fetch history of processed documents."""
    documents = db.query(models.DocumentRecord).order_by(models.DocumentRecord.upload_time.desc()).offset(skip).limit(limit).all()
    return documents

@app.get("/api/v1/documents/{doc_id}", response_model=schemas.DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Fetch specific document."""
    document = db.query(models.DocumentRecord).filter(models.DocumentRecord.id == doc_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server on a different port than Streamlit (8501/8506)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
