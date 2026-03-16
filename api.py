import os
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine, Base
import models
from main import DocumentIntelligenceSystem
from universal_parser import UniversalDataIntelligence
from cloud_ocr import cloud_ocr
import uvicorn
import cv2

# Initialize DB
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="DOC-INTEL AI API", version="5.0")

# CORS for external apps/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances of our AI Engines
system = None
intelligence = None

@app.on_event("startup")
async def load_ai_models():
    global system, intelligence
    print("🚀 Initializing AI Engines for FastAPI...")
    import asyncio
    # Run the heavy loading in a separate thread to prevent blocking Uvicorn startup
    system, intelligence = await asyncio.to_thread(DocumentIntelligenceSystem), await asyncio.to_thread(UniversalDataIntelligence)
    print("✅ AI Engines Ready.")

# Create permanent upload storage directory (Local S3 equivalent)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"status": "DOC-INTEL API is running", "version": "5.0"}

@app.post("/api/v1/process_document")
async def process_document(
    file: UploadFile = File(...),
    cloud_mode: bool = Form(False),
    provider: str = Form("gemini"),
    force_mode: str = Form(None), # None, 'HANDWRITTEN_NOTE', or 'STRUCTURED_FORM'
    high_fidelity: bool = Form(True),
    db=Depends(get_db)
):
    """
    Enterprise Endpoint: Uploads file, locally saves it securely, processes via Vision + OCR + AI, and stores in DB.
    """
    try:
        # 1. Secure Storage Allocation (Local AWS S3 equivalent)
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid4().hex}.{file_extension}"
        
        # Absolute path string ensures OpenCV and background threads never get confused by relative paths
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))
        
        # Safely read entire file into RAM first to avoid cursor and async IO flush bugs
        file_contents = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_contents)
            
        print(f"📥 File saved securely to: {file_path}")

        # 2. Vision Enhancement (Always run local enhancement first)
        clean_img = system.ocr.vision.enhance_image(file_path)
        processed_path = file_path
        if clean_img is not None:
            processed_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"clean_{unique_filename}"))
            cv2.imwrite(processed_path, clean_img)
        else:
            print("⚠️ Vision enhancement failed or skipped. Using original image.")

        # 3. AI Extraction Logic (Cloud vs Local)
        doc_analysis = {"type": "UNKNOWN", "confidence_score": 0.0}
        data = {"full_text": "", "table": [], "kv": {}}
        
        if cloud_mode:
            print(f"☁️ Cloud Mode Activated: Routing to {provider.upper()} API...")
            try:
                ocr_data = cloud_ocr(file_path, provider=provider)
                if isinstance(ocr_data, dict):
                    data["full_text"] = ocr_data.get("full_raw_text", "")
                    data["kv"] = ocr_data.get("metadata", {})
                    data["table"] = ocr_data.get("extracted_data", [])
                    doc_analysis["type"] = ocr_data.get("document_type", "UNKNOWN")
                    doc_analysis["confidence_score"] = ocr_data.get("confidence_score", 0.99)
                else:
                    data["full_text"] = str(ocr_data)
            except Exception as e:
                print(f"❌ Cloud OCR Failed: {e}. Falling back to Local AI.")
                cloud_mode = False # Fallback

        if not cloud_mode:
            print("🧠 Local AI Mode Activated: Running DocTR + TrOCR + Fuzzy Logic...")
            if system is None or intelligence is None:
                raise HTTPException(status_code=503, detail="AI models are still loading into memory. Please try again in 30 seconds.")
                
            # Run heavy extraction in a background thread to avoid blocking the API loop
            import asyncio
            ocr_results = await asyncio.to_thread(
                system.ocr.extract_text_from_image,
                processed_path, 
                pre_cleaned=None, 
                paragraph=False, 
                high_fidelity=high_fidelity
            )
            doc_analysis, dict_data, _ = await asyncio.to_thread(
                intelligence.parse_universal,
                ocr_results, 
                force_mode=force_mode
            )
            data["full_text"] = dict_data.get("full_text", "")
            data["table"] = dict_data.get("table", [])
            data["kv"] = dict_data.get("kv", {})

        # 4. Save to Enterprise Database
        new_record = models.DocumentRecord(
            filename=file.filename,
            document_type=doc_analysis.get("type", "UNKNOWN"),
            confidence_score=doc_analysis.get('confidence_score', 0.0),
            original_image_path=file_path,
            processed_image_path=processed_path,
            extracted_text=data.get("full_text", ""),
            extracted_json=data,
            ocr_provider=provider if cloud_mode else "local",
            processing_time_ms=0.0
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        print(f"✅ Record #{new_record.id} saved to DB.")

        # 5. Return JSON to Client
        return {
            "status": "success",
            "record_id": new_record.id,
            "analysis": doc_analysis,
            "structured_data": data,
            "paths": {
                "original": file_path,
                "processed": processed_path
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
def get_history(limit: int = 50, db=Depends(get_db)):
    """Retrieve document history from the database."""
    records = db.query(models.DocumentRecord).order_by(models.DocumentRecord.id.desc()).limit(limit).all()
    return records

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
