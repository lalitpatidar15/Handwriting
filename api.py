import os
import time
import datetime
import hashlib
import hmac
import base64
from uuid import uuid4
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine, Base, ensure_schema_migrations
import models
import schemas
from main import DocumentIntelligenceSystem
from universal_parser import UniversalDataIntelligence
from cloud_ocr import cloud_ocr, run_enterprise_idp_pipeline
from mongo_store import MongoIDPStore
import uvicorn
import cv2

# Initialize DB
models.Base.metadata.create_all(bind=engine)
ensure_schema_migrations()

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
mongo_store = MongoIDPStore.from_env()

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


def _normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def _build_field_predictions(data, doc_analysis, source_engine):
    predictions = []
    base_confidence = float(doc_analysis.get("confidence_score", 0.0) or 0.0)

    for key, value in (data.get("kv", {}) or {}).items():
        value_text = _normalize_value(value)
        confidence = min(0.99, max(0.55, base_confidence - (0.12 if len(value_text.strip()) <= 2 else 0.03)))
        predictions.append({
            "field_name": f"kv.{key}",
            "field_type": "kv",
            "predicted_value": value_text,
            "confidence_score": confidence,
            "source_engine": source_engine,
        })

    for row_index, row in enumerate((data.get("table", []) or [])):
        if isinstance(row, dict):
            for column_name, value in row.items():
                predictions.append({
                    "field_name": f"table[{row_index}].{column_name}",
                    "field_type": "table",
                    "predicted_value": _normalize_value(value),
                    "confidence_score": min(0.95, max(0.50, base_confidence - 0.10)),
                    "source_engine": source_engine,
                })
        else:
            predictions.append({
                "field_name": f"table[{row_index}]",
                "field_type": "table",
                "predicted_value": _normalize_value(row),
                "confidence_score": min(0.92, max(0.50, base_confidence - 0.10)),
                "source_engine": source_engine,
            })

    if not predictions and data.get("full_text"):
        predictions.append({
            "field_name": "full_text",
            "field_type": "text",
            "predicted_value": _normalize_value(data.get("full_text")),
            "confidence_score": min(0.90, max(0.50, base_confidence - 0.08)),
            "source_engine": source_engine,
        })

    return predictions


def _infer_document_domain(data: dict[str, Any], doc_analysis: dict[str, Any], filename: str) -> str:
    text_blob = " ".join(
        [
            str(filename or ""),
            str(data.get("full_text", "") or ""),
            " ".join(str(key) for key in (data.get("kv", {}) or {}).keys()),
        ]
    ).lower()

    invoice_terms = ["invoice", "inv", "tax", "qty", "amount", "subtotal", "supplier", "gst", "price", "total"]
    prescription_terms = ["diagnosis", "prescription", "tablet", "tab ", "capsule", "mg", "bd", "tds", "doctor", "fever"]
    form_terms = ["name", "date", "address", "signature", "form", "application", "gender", "age"]

    if any(term in text_blob for term in prescription_terms):
        return "prescription"
    if any(term in text_blob for term in invoice_terms):
        return "invoice"
    if doc_analysis.get("type") == "STRUCTURED_FORM" or any(term in text_blob for term in form_terms):
        return "form"
    if doc_analysis.get("type") == "HANDWRITTEN_NOTE":
        return "handwritten_note"
    return "general"


def _extract_anchor_labels(data: dict[str, Any]) -> list[str]:
    anchors = [str(key).strip().lower() for key in (data.get("kv", {}) or {}).keys() if str(key).strip()]
    table = data.get("table", []) or []
    if table and isinstance(table[0], dict):
        anchors.extend(str(key).strip().lower() for key in table[0].keys() if str(key).strip())
    if not anchors and data.get("full_text"):
        anchors.extend(line.strip().lower() for line in str(data.get("full_text", "")).splitlines()[:5] if line.strip())
    return anchors[:12]


def _build_template_fingerprint(data: dict[str, Any], doc_analysis: dict[str, Any], ocr_results: list[Any], filename: str, domain: str) -> dict[str, Any]:
    anchor_labels = _extract_anchor_labels(data)
    avg_conf = 0.0
    if ocr_results:
        confidences = [float(item.get("confidence", 0.0) or 0.0) for item in ocr_results if isinstance(item, dict)]
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "document_type": doc_analysis.get("type", "UNKNOWN"),
        "document_domain": domain,
        "filename_hint": os.path.splitext(filename)[0].lower(),
        "anchor_labels": anchor_labels,
        "kv_count": len((data.get("kv", {}) or {})),
        "table_row_count": len((data.get("table", []) or [])),
        "table_column_count": len((data.get("table", [])[0] or {}).keys()) if data.get("table") and isinstance(data.get("table", [])[0], dict) else 0,
        "text_line_count": len([line for line in str(data.get("full_text", "")).splitlines() if line.strip()]),
        "ocr_token_count": len(ocr_results or []),
        "avg_ocr_confidence": avg_conf,
    }


def _fingerprint_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_anchors = set(left.get("anchor_labels", []) or [])
    right_anchors = set(right.get("anchor_labels", []) or [])
    anchor_score = len(left_anchors & right_anchors) / max(1, len(left_anchors | right_anchors))

    def closeness(a: float, b: float) -> float:
        return max(0.0, 1.0 - (abs(a - b) / max(1.0, max(a, b))))

    kv_score = closeness(float(left.get("kv_count", 0)), float(right.get("kv_count", 0)))
    row_score = closeness(float(left.get("table_row_count", 0)), float(right.get("table_row_count", 0)))
    column_score = closeness(float(left.get("table_column_count", 0)), float(right.get("table_column_count", 0)))
    type_score = 1.0 if left.get("document_type") == right.get("document_type") else 0.0

    return round((0.40 * anchor_score) + (0.20 * kv_score) + (0.15 * row_score) + (0.10 * column_score) + (0.15 * type_score), 4)


def _match_or_create_template(db, data: dict[str, Any], doc_analysis: dict[str, Any], ocr_results: list[Any], filename: str, domain: str):
    fingerprint = _build_template_fingerprint(data, doc_analysis, ocr_results, filename, domain)
    candidates = db.query(models.TemplateRecord).filter(
        models.TemplateRecord.document_type == fingerprint["document_type"],
        models.TemplateRecord.document_domain == domain,
    ).all()

    best_template = None
    best_score = 0.0
    for candidate in candidates:
        score = _fingerprint_similarity(fingerprint, candidate.fingerprint or {})
        if score > best_score:
            best_template = candidate
            best_score = score

    if best_template and best_score >= 0.72:
        best_template.sample_count += 1
        best_template.last_matched_at = datetime.datetime.utcnow()
        db.add(best_template)
        return best_template, best_score, fingerprint

    template_name = f"{domain}_{fingerprint['document_type'].lower()}_{max(len(candidates) + 1, 1)}"
    template = models.TemplateRecord(
        template_name=template_name,
        tenant_id="default",
        document_type=fingerprint["document_type"],
        document_domain=domain,
        fingerprint=fingerprint,
        sample_count=1,
        approval_status="learned",
    )
    db.add(template)
    db.flush()
    return template, 1.0, fingerprint


def _build_extraction_blocks(data: dict[str, Any], ocr_results: list[Any], source_engine: str):
    blocks = []
    if ocr_results:
        for item in ocr_results:
            box = None
            if isinstance(item, dict):
                if item.get("bbox"):
                    box = item.get("bbox")
                elif item.get("box"):
                    box = item.get("box")
            blocks.append({
                "page_number": 1,
                "block_type": "word",
                "text": _normalize_value(item.get("text", "")) if isinstance(item, dict) else "",
                "bbox": box,
                "confidence_score": float(item.get("confidence", 0.0) or 0.0) if isinstance(item, dict) else 0.0,
                "source_engine": source_engine,
            })
        return blocks

    lines = [line for line in str(data.get("full_text", "")).splitlines() if line.strip()]
    total_lines = max(1, len(lines))
    for index, line in enumerate(lines, start=1):
        top = round((index - 1) / total_lines, 4)
        bottom = round(index / total_lines, 4)
        blocks.append({
            "page_number": 1,
            "block_type": "line",
            "text": line,
            "bbox": [[0.05, top], [0.95, bottom]],
            "confidence_score": 0.75,
            "source_engine": source_engine,
        })
    return blocks


def _review_priority(confidence_score):
    if confidence_score < 0.75:
        return "high"
    if confidence_score < 0.92:
        return "medium"
    return "low"


def _review_status_for_confidence(confidence_score):
    if confidence_score >= 0.92:
        return "auto_approved"
    if confidence_score >= 0.75:
        return "review_required"
    return "reprocess_required"


def _apply_correction_to_payload(payload, field_name, corrected_value):
    if not isinstance(payload, dict):
        return payload

    if field_name == "full_text":
        payload["full_text"] = corrected_value
        return payload

    if field_name.startswith("kv."):
        key = field_name[3:]
        payload.setdefault("kv", {})
        payload["kv"][key] = corrected_value
        return payload

    if field_name.startswith("table["):
        try:
            row_index_text, remainder = field_name.split("]", 1)
            row_index = int(row_index_text[len("table["):])
        except (ValueError, IndexError):
            return payload

        payload.setdefault("table", [])
        while len(payload["table"]) <= row_index:
            payload["table"].append({})

        if remainder.startswith("."):
            column_name = remainder[1:]
            if not isinstance(payload["table"][row_index], dict):
                payload["table"][row_index] = {"value": payload["table"][row_index]}
            payload["table"][row_index][column_name] = corrected_value
        else:
            payload["table"][row_index] = corrected_value

    return payload


def _serialize_document(document, db):
    field_predictions = db.query(models.FieldPrediction).filter(models.FieldPrediction.document_id == document.id).order_by(models.FieldPrediction.id.asc()).all()
    review_tasks = db.query(models.ReviewTask).filter(models.ReviewTask.document_id == document.id).order_by(models.ReviewTask.status.asc(), models.ReviewTask.id.asc()).all()
    model_runs = db.query(models.ModelRun).filter(models.ModelRun.document_id == document.id).order_by(models.ModelRun.id.asc()).all()
    extraction_blocks = db.query(models.ExtractionBlock).filter(models.ExtractionBlock.document_id == document.id).order_by(models.ExtractionBlock.id.asc()).all()
    template = None
    if document.template_id:
        template = db.query(models.TemplateRecord).filter(models.TemplateRecord.id == document.template_id).first()

    return {
        "id": document.id,
        "filename": document.filename,
        "upload_time": document.upload_time,
        "document_type": document.document_type,
        "confidence_score": document.confidence_score,
        "ocr_provider": document.ocr_provider,
        "layout_provider": document.layout_provider,
        "reasoning_provider": document.reasoning_provider,
        "processing_time_ms": document.processing_time_ms,
        "original_image_path": document.original_image_path,
        "processed_image_path": document.processed_image_path,
        "extracted_text": document.extracted_text,
        "extracted_json": document.extracted_json,
        "review_status": document.review_status,
        "schema_version": document.schema_version,
        "document_domain": document.document_domain,
        "template_id": document.template_id,
        "template_match_score": document.template_match_score,
        "field_predictions": field_predictions,
        "review_tasks": review_tasks,
        "model_runs": model_runs,
        "extraction_blocks": extraction_blocks,
        "template": template,
    }


def _update_document_review_status(document, db):
    db.flush()
    open_tasks = db.query(models.ReviewTask).filter(
        models.ReviewTask.document_id == document.id,
        models.ReviewTask.status == "open",
    ).count()
    document.review_status = "review_required" if open_tasks else "approved"
    db.add(document)


def _password_salt() -> str:
    return os.getenv("AUTH_PASSWORD_SALT", "docintel_default_salt")


def _hash_password(password: str) -> str:
    return hashlib.sha256(f"{_password_salt()}::{password}".encode("utf-8")).hexdigest()


def _verify_password(password: str, password_hash: str) -> bool:
    expected = _hash_password(password)
    return hmac.compare_digest(expected, str(password_hash or ""))


def _build_auth_token(user_id: int, email: str) -> str:
    payload = f"{user_id}:{email}:{int(time.time())}"
    signature = hashlib.sha256(f"{payload}:{_password_salt()}".encode("utf-8")).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{signature}".encode("utf-8")).decode("utf-8")

@app.get("/")
def read_root():
    return {"status": "DOC-INTEL API is running", "version": "5.0"}


@app.post("/api/v1/auth/signup", response_model=schemas.AuthResponse)
def signup(payload: schemas.SignupRequest, db=Depends(get_db)):
    username = payload.username.strip()
    email = payload.email.strip().lower()
    password = payload.password.strip()

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = db.query(models.UserAccount).filter(
        (models.UserAccount.email == email) | (models.UserAccount.username == username)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="User with this email or username already exists")

    user = models.UserAccount(
        username=username,
        email=email,
        password_hash=_hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    user_id = int(getattr(user, "id"))
    user_name = str(getattr(user, "username"))
    user_email = str(getattr(user, "email"))

    return schemas.AuthResponse(
        status="success",
        user_id=user_id,
        username=user_name,
        email=user_email,
        token=_build_auth_token(user_id, user_email),
    )


@app.post("/api/v1/auth/login", response_model=schemas.AuthResponse)
def login(payload: schemas.LoginRequest, db=Depends(get_db)):
    email = payload.email.strip().lower()
    password = payload.password.strip()

    user = db.query(models.UserAccount).filter(models.UserAccount.email == email).first()
    if not user or not _verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = int(getattr(user, "id"))
    user_name = str(getattr(user, "username"))
    user_email = str(getattr(user, "email"))

    return schemas.AuthResponse(
        status="success",
        user_id=user_id,
        username=user_name,
        email=user_email,
        token=_build_auth_token(user_id, user_email),
    )


@app.get("/api/v1/providers/requirements")
def get_provider_requirements():
    return {
        "enterprise_ocr": {
            "documentai": [
                "GOOGLE_DOCUMENT_AI_PROJECT_ID",
                "GOOGLE_DOCUMENT_AI_LOCATION",
                "GOOGLE_DOCUMENT_AI_PROCESSOR_ID",
                "GOOGLE_DOCUMENT_AI_API_KEY",
            ],
            "textract": [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION",
            ],
            "azure": [
                "AZURE_DOCINT_ENDPOINT",
                "AZURE_DOCINT_API_KEY",
                "AZURE_DOCINT_API_VERSION",
            ],
        },
        "reasoning": {
            "gemini": [
                "GEMINI_API_KEY",
            ]
        },
    }


@app.get("/api/v1/providers/health")
def get_provider_health():
    def required(keys: list[str]) -> dict[str, Any]:
        values = {key: bool(os.getenv(key)) for key in keys}
        return {
            "configured": all(values.values()),
            "keys": values,
        }

    return {
        "documentai": required(
            [
                "GOOGLE_DOCUMENT_AI_PROJECT_ID",
                "GOOGLE_DOCUMENT_AI_LOCATION",
                "GOOGLE_DOCUMENT_AI_PROCESSOR_ID",
                "GOOGLE_DOCUMENT_AI_API_KEY",
            ]
        ),
        "textract": required(
            [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION",
            ]
        ),
        "azure": required(
            [
                "AZURE_DOCINT_ENDPOINT",
                "AZURE_DOCINT_API_KEY",
            ]
        ),
        "gemini_reasoning": required(["GEMINI_API_KEY"]),
        "celery": required(["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"]),
        "mongodb": mongo_store.health(),
    }


@app.get("/api/v1/mongo/health")
def get_mongo_health():
    return mongo_store.health()


@app.get("/api/v1/mongo/documents/{document_id}")
def get_mongo_document(document_id: int):
    record = mongo_store.get_document_bundle(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Mongo document bundle not found")
    return record


@app.post("/api/v1/mongo/vector/search")
def mongo_vector_search(payload: dict[str, Any]):
    query_vector = payload.get("query_vector")
    if not isinstance(query_vector, list) or not query_vector:
        raise HTTPException(status_code=400, detail="query_vector must be a non-empty float array")

    try:
        cast_vector = [float(v) for v in query_vector]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid query_vector values: {exc}")

    limit = int(payload.get("limit", 5) or 5)
    num_candidates = int(payload.get("num_candidates", 100) or 100)
    document_type = payload.get("document_type")
    if document_type is not None:
        document_type = str(document_type)

    results = mongo_store.vector_search(
        query_vector=cast_vector,
        limit=limit,
        num_candidates=num_candidates,
        document_type=document_type,
    )
    return {"status": "success", "count": len(results), "results": results}


@app.post("/api/v1/process_document/async")
async def process_document_async_enqueue(
    file: UploadFile = File(...),
    enterprise_ocr_provider: str = Form("documentai"),
    enterprise_reasoning_provider: str = Form("gemini"),
    db=Depends(get_db),
):
    try:
        from workers.document_tasks import process_document_async
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Celery worker is not available: {exc}")

    filename = file.filename or "upload.jpg"
    file_extension = filename.split(".")[-1] if "." in filename else "jpg"
    unique_filename = f"{uuid4().hex}.{file_extension}"
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))

    file_contents = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(file_contents)

    new_record = models.DocumentRecord(
        filename=filename,
        document_type="QUEUED",
        confidence_score=0.0,
        original_image_path=file_path,
        processed_image_path=file_path,
        extracted_text="",
        extracted_json={"status": "queued"},
        ocr_provider="queued",
        processing_time_ms=0.0,
        review_status="queued",
        schema_version="1.0",
        document_domain="general",
        template_match_score=0.0,
        layout_provider=enterprise_ocr_provider,
        reasoning_provider=enterprise_reasoning_provider,
    )
    db.add(new_record)
    db.flush()

    task = process_document_async.delay(new_record.id)

    db.add(
        models.ModelRun(
            document_id=new_record.id,
            stage="async_queue",
            model_name="celery_enqueue",
            provider="celery",
            success="true",
            duration_ms=0.0,
            raw_output={
                "task_id": task.id,
                "layout_provider": enterprise_ocr_provider,
                "reasoning_provider": enterprise_reasoning_provider,
            },
        )
    )

    db.commit()

    return {
        "status": "queued",
        "record_id": new_record.id,
        "task_id": task.id,
        "layout_provider": enterprise_ocr_provider,
        "reasoning_provider": enterprise_reasoning_provider,
    }

@app.post("/api/v1/process_document")
async def process_document(
    file: UploadFile = File(...),
    cloud_mode: bool = Form(False),
    provider: str = Form("gemini"),
    enterprise_mode: bool = Form(False),
    enterprise_ocr_provider: str = Form("documentai"),
    enterprise_reasoning_provider: str = Form("gemini"),
    force_mode: str = Form(None), # None, 'HANDWRITTEN_NOTE', or 'STRUCTURED_FORM'
    high_fidelity: bool = Form(True),
    db=Depends(get_db)
):
    """
    Enterprise Endpoint: Uploads file, locally saves it securely, processes via Vision + OCR + AI, and stores in DB.
    """
    try:
        started_at = time.perf_counter()
        model_runs = []
        ocr_results = []
        filename = file.filename or "upload.jpg"

        # 1. Secure Storage Allocation (Local AWS S3 equivalent)
        file_extension = filename.split(".")[-1] if "." in filename else "jpg"
        unique_filename = f"{uuid4().hex}.{file_extension}"
        
        # Absolute path string ensures OpenCV and background threads never get confused by relative paths
        file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))
        
        # Safely read entire file into RAM first to avoid cursor and async IO flush bugs
        file_contents = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(file_contents)
            
        print(f"📥 File saved securely to: {file_path}")

        if system is None:
            raise HTTPException(status_code=503, detail="AI models are still loading into memory. Please try again in 30 seconds.")
        ocr_engine = getattr(system, "ocr", None)
        if ocr_engine is None:
            raise HTTPException(status_code=503, detail="OCR engine is still loading. Please try again in 30 seconds.")

        # 2. Vision Enhancement (Always run local enhancement first)
        clean_img = ocr_engine.vision.enhance_image(file_path)
        processed_path = file_path
        if clean_img is not None:
            processed_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"clean_{unique_filename}"))
            cv2.imwrite(processed_path, clean_img)
        else:
            print("⚠️ Vision enhancement failed or skipped. Using original image.")

        # 3. AI Extraction Logic (Cloud vs Local)
        doc_analysis = {"type": "UNKNOWN", "confidence_score": 0.0}
        data = {"full_text": "", "table": [], "kv": {}}
        
        if enterprise_mode:
            print(
                f"🏢 Enterprise Mode Activated: OCR={enterprise_ocr_provider.upper()} + "
                f"REASONING={enterprise_reasoning_provider.upper()}"
            )
            enterprise_started_at = time.perf_counter()
            try:
                enterprise_output = run_enterprise_idp_pipeline(
                    image_path=file_path,
                    ocr_provider=enterprise_ocr_provider,
                    reasoning_provider=enterprise_reasoning_provider,
                    document_type_hint=force_mode or "UNKNOWN",
                )
                doc_analysis = enterprise_output.get("doc_analysis", doc_analysis)
                data = enterprise_output.get("data", data)
                ocr_results = enterprise_output.get("ocr_results", []) or []
                source_engine = enterprise_output.get("source_engine", f"{enterprise_ocr_provider}+{enterprise_reasoning_provider}")
                model_runs.extend(enterprise_output.get("model_runs", []))
                model_runs.append({
                    "stage": "enterprise_pipeline",
                    "model_name": "specialized_ocr_plus_reasoning",
                    "provider": "enterprise",
                    "success": "true",
                    "duration_ms": round((time.perf_counter() - enterprise_started_at) * 1000, 2),
                    "raw_output": {
                        "ocr_provider": enterprise_ocr_provider,
                        "reasoning_provider": enterprise_reasoning_provider,
                        "token_count": len(ocr_results),
                    },
                })
            except Exception as e:
                print(f"❌ Enterprise pipeline failed: {e}. Falling back to current pipeline.")
                model_runs.append({
                    "stage": "enterprise_pipeline",
                    "model_name": "specialized_ocr_plus_reasoning",
                    "provider": "enterprise",
                    "success": "false",
                    "duration_ms": round((time.perf_counter() - enterprise_started_at) * 1000, 2),
                    "raw_output": {"error": str(e)},
                })
                enterprise_mode = False

        if cloud_mode and not enterprise_mode:
            print(f"☁️ Cloud Mode Activated: Routing to {provider.upper()} API...")
            cloud_started_at = time.perf_counter()
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
                model_runs.append({
                    "stage": "cloud_extraction",
                    "model_name": provider,
                    "provider": provider,
                    "success": "true",
                    "duration_ms": round((time.perf_counter() - cloud_started_at) * 1000, 2),
                    "raw_output": ocr_data if isinstance(ocr_data, dict) else {"text": str(ocr_data)},
                })
            except Exception as e:
                print(f"❌ Cloud OCR Failed: {e}. Falling back to Local AI.")
                model_runs.append({
                    "stage": "cloud_extraction",
                    "model_name": provider,
                    "provider": provider,
                    "success": "false",
                    "duration_ms": round((time.perf_counter() - cloud_started_at) * 1000, 2),
                    "raw_output": {"error": str(e)},
                })
                cloud_mode = False # Fallback

        if not cloud_mode and not enterprise_mode:
            print("🧠 Local AI Mode Activated: Running DocTR + TrOCR + Fuzzy Logic...")
            if intelligence is None:
                raise HTTPException(status_code=503, detail="AI models are still loading into memory. Please try again in 30 seconds.")
                
            # Run heavy extraction in a background thread to avoid blocking the API loop
            import asyncio
            assert system is not None and intelligence is not None
            ocr_results = await asyncio.to_thread(
                ocr_engine.extract_text_from_image,
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
            model_runs.append({
                "stage": "local_pipeline",
                "model_name": "doctr_trocr_universal_parser",
                "provider": "local",
                "success": "true",
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "raw_output": {
                    "ocr_token_count": len(ocr_results or []),
                    "document_type": doc_analysis.get("type", "UNKNOWN"),
                },
            })

        source_engine = (
            f"{enterprise_ocr_provider}+{enterprise_reasoning_provider}"
            if enterprise_mode
            else (provider if cloud_mode else "local")
        )
        document_domain = _infer_document_domain(data, doc_analysis, filename)
        field_predictions = _build_field_predictions(data, doc_analysis, source_engine)
        extraction_blocks = _build_extraction_blocks(data, ocr_results, source_engine)
        open_review_count = sum(1 for item in field_predictions if item["confidence_score"] < 0.92)
        processing_time_ms = round((time.perf_counter() - started_at) * 1000, 2)

        template, template_match_score, template_fingerprint = _match_or_create_template(
            db,
            data,
            doc_analysis,
            ocr_results,
            filename,
            document_domain,
        )
        model_runs.append({
            "stage": "template_matching",
            "model_name": "heuristic_template_matcher",
            "provider": "local",
            "success": "true",
            "duration_ms": 0.0,
            "raw_output": {
                "template_id": template.id,
                "template_name": template.template_name,
                "template_match_score": template_match_score,
                "fingerprint": template_fingerprint,
            },
        })

        # 4. Save to Enterprise Database
        new_record = models.DocumentRecord(
            filename=filename,
            document_type=doc_analysis.get("type", "UNKNOWN"),
            confidence_score=doc_analysis.get('confidence_score', 0.0),
            original_image_path=file_path,
            processed_image_path=processed_path,
            extracted_text=data.get("full_text", ""),
            extracted_json=data,
            ocr_provider=source_engine,
            processing_time_ms=processing_time_ms,
            review_status="review_required" if open_review_count else "approved",
            schema_version="1.0",
            document_domain=document_domain,
            template_id=template.id,
            template_match_score=template_match_score,
            layout_provider=enterprise_ocr_provider if enterprise_mode else source_engine,
            reasoning_provider=enterprise_reasoning_provider if enterprise_mode else (provider if cloud_mode else "local"),
        )
        db.add(new_record)
        db.flush()

        for block in extraction_blocks:
            db.add(models.ExtractionBlock(
                document_id=new_record.id,
                page_number=block["page_number"],
                block_type=block["block_type"],
                text=block["text"],
                bbox=block["bbox"],
                confidence_score=block["confidence_score"],
                source_engine=block["source_engine"],
            ))

        saved_predictions = []
        for prediction in field_predictions:
            prediction_review_status = _review_status_for_confidence(prediction["confidence_score"])
            saved_prediction = models.FieldPrediction(
                document_id=new_record.id,
                field_name=prediction["field_name"],
                field_type=prediction["field_type"],
                predicted_value=prediction["predicted_value"],
                confidence_score=prediction["confidence_score"],
                source_engine=prediction["source_engine"],
                review_status=prediction_review_status,
            )
            db.add(saved_prediction)
            db.flush()
            saved_predictions.append(saved_prediction)

            if prediction_review_status != "auto_approved":
                db.add(models.ReviewTask(
                    document_id=new_record.id,
                    field_prediction_id=saved_prediction.id,
                    status="open",
                    priority=_review_priority(saved_prediction.confidence_score),
                    predicted_value=saved_prediction.predicted_value,
                ))

        for run in model_runs:
            db.add(models.ModelRun(
                document_id=new_record.id,
                stage=run["stage"],
                model_name=run["model_name"],
                provider=run["provider"],
                success=run["success"],
                duration_ms=run["duration_ms"],
                raw_output=run["raw_output"],
            ))

        db.commit()
        db.refresh(new_record)
        print(f"✅ Record #{new_record.id} saved to DB.")

        mongo_store.upsert_document_bundle(
            document_id=new_record.id,
            file_path=file_path,
            processed_path=processed_path,
            document_type=str(doc_analysis.get("type", "UNKNOWN") or "UNKNOWN"),
            status=str(new_record.review_status or "processed"),
            confidence_score=float(doc_analysis.get("confidence_score", 0.0) or 0.0),
            domain=document_domain,
            source_engine=source_engine,
            extracted_data=data,
            field_predictions=[
                {
                    "name": item.field_name,
                    "value": item.predicted_value,
                    "confidence": item.confidence_score,
                    "review_status": item.review_status,
                }
                for item in saved_predictions
            ],
            model_runs=model_runs,
            template_payload={
                "template_id": template.id,
                "template_name": template.template_name,
                "document_domain": template.document_domain,
                "match_score": template_match_score,
                "fingerprint": template_fingerprint,
            },
        )

        # 5. Return JSON to Client
        return {
            "status": "success",
            "record_id": new_record.id,
            "analysis": doc_analysis,
            "structured_data": data,
            "field_predictions": [
                {
                    "id": item.id,
                    "field_name": item.field_name,
                    "field_type": item.field_type,
                    "predicted_value": item.predicted_value,
                    "confidence_score": item.confidence_score,
                    "review_status": item.review_status,
                }
                for item in saved_predictions
            ],
            "review_summary": {
                "document_review_status": new_record.review_status,
                "open_tasks": open_review_count,
                "auto_approved_fields": len(field_predictions) - open_review_count,
            },
            "pipeline": {
                "enterprise_mode": enterprise_mode,
                "cloud_mode": cloud_mode,
                "layout_provider": new_record.layout_provider,
                "reasoning_provider": new_record.reasoning_provider,
            },
            "template": {
                "id": template.id,
                "template_name": template.template_name,
                "document_domain": template.document_domain,
                "match_score": template_match_score,
            },
            "layout_summary": {
                "block_count": len(extraction_blocks),
                "source_engine": source_engine,
            },
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


@app.get("/api/v1/documents/{document_id}", response_model=schemas.DocumentDetailResponse)
def get_document_detail(document_id: int, db=Depends(get_db)):
    document = db.query(models.DocumentRecord).filter(models.DocumentRecord.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return _serialize_document(document, db)


@app.get("/api/v1/review/tasks", response_model=list[schemas.ReviewTaskResponse])
def get_review_tasks(status: str = "open", limit: int = 50, db=Depends(get_db)):
    query = db.query(models.ReviewTask)
    if status != "all":
        query = query.filter(models.ReviewTask.status == status)
    tasks = query.order_by(models.ReviewTask.created_at.desc()).limit(limit).all()
    return tasks


@app.get("/api/v1/templates", response_model=list[schemas.TemplateResponse])
def get_templates(limit: int = 50, db=Depends(get_db)):
    templates = db.query(models.TemplateRecord).order_by(models.TemplateRecord.last_matched_at.desc()).limit(limit).all()
    return templates


@app.post("/api/v1/review/tasks/{task_id}/complete")
def complete_review_task(task_id: int, payload: schemas.ReviewTaskCompleteRequest, db=Depends(get_db)):
    task = db.query(models.ReviewTask).filter(models.ReviewTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Review task not found")

    prediction = db.query(models.FieldPrediction).filter(models.FieldPrediction.id == task.field_prediction_id).first()
    document = db.query(models.DocumentRecord).filter(models.DocumentRecord.id == task.document_id).first()
    if not prediction or not document:
        raise HTTPException(status_code=404, detail="Review task is linked to missing records")

    final_value = payload.corrected_value if payload.corrected_value is not None else prediction.predicted_value
    task.corrected_value = final_value
    task.reviewer_name = payload.reviewer_name
    task.review_notes = payload.review_notes
    task.status = "completed"
    task.completed_at = datetime.datetime.utcnow()

    prediction.corrected_value = final_value
    prediction.review_status = "approved" if payload.resolution == "approved" else "corrected"

    if document.extracted_json is None:
        document.extracted_json = {}
    document.extracted_json = _apply_correction_to_payload(document.extracted_json, prediction.field_name, final_value)
    if prediction.field_name == "full_text":
        document.extracted_text = final_value

    db.flush()
    _update_document_review_status(document, db)
    db.add(task)
    db.add(prediction)
    db.add(document)
    db.commit()

    mongo_store.save_review_correction(
        document_id=document.id,
        field_name=prediction.field_name,
        predicted_value=str(prediction.predicted_value or ""),
        corrected_value=str(final_value or ""),
        reviewer_name=payload.reviewer_name,
        review_notes=payload.review_notes,
    )

    return {
        "status": "success",
        "task_id": task.id,
        "document_id": document.id,
        "document_review_status": document.review_status,
        "field_name": prediction.field_name,
        "final_value": final_value,
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
