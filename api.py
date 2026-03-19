import os
import time
import datetime
import hashlib
import hmac
import base64
from uuid import uuid4
from typing import Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import schemas
from main import DocumentIntelligenceSystem
from universal_parser import UniversalDataIntelligence
from cloud_ocr import cloud_ocr, run_enterprise_idp_pipeline
from mongo_store import MongoIDPStore
from env_loader import load_env_file
import uvicorn
import cv2
from PIL import Image, ImageDraw, ImageFont

load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(title="DOC-INTEL AI API", version="5.0")
auth_scheme = HTTPBearer(auto_error=False)

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

    # Check MongoDB connectivity at startup so issues are visible in logs immediately
    mongo_health = mongo_store.health()
    if mongo_health.get("connected"):
        print(f"✅ MongoDB connected — database: '{mongo_health.get('database')}'")
    else:
        mongo_err = mongo_health.get("error", "unknown error")
        print(f"⚠️  MongoDB NOT connected: {mongo_err}")
        print("   → Documents will not be persisted until MongoDB is reachable.")

# Create permanent upload storage directory (Local S3 equivalent)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _ensure_mongo_connected() -> None:
    if not mongo_store.is_connected():
        error = getattr(mongo_store, "_last_error", None) or "Check MONGODB_URI in .env and ensure your IP is whitelisted in MongoDB Atlas."
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB is not connected: {error}",
        )


def _normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def _image_mime_type(file_path: str) -> str:
    lowered = str(file_path or "").lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _image_file_to_base64(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _structured_lines_from_data(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    # Prefer kv pairs that have actual non-null values
    kv_payload = data.get("kv", {}) or {}
    if isinstance(kv_payload, dict):
        for key, value in kv_payload.items():
            rendered = _normalize_value(value).strip()
            if rendered:
                lines.append(f"{key}: {rendered}")

    # Table rows: if every row only has a single "text" key, just emit the text
    table_payload = data.get("table", []) or []
    is_text_only_table = table_payload and all(
        isinstance(r, dict) and list(r.keys()) == ["text"] for r in table_payload
    )
    for row in table_payload:
        if isinstance(row, dict):
            if is_text_only_table:
                val = _normalize_value(row.get("text", "")).strip()
                if val:
                    lines.append(val)
            else:
                row_parts = [
                    f"{k}: {_normalize_value(v).strip() or '-'}"
                    for k, v in row.items()
                    if _normalize_value(v).strip()
                ]
                if row_parts:
                    lines.append("  |  ".join(row_parts))
        else:
            val = _normalize_value(row).strip()
            if val:
                lines.append(val)

    # Always fall back to full_text if nothing else was extracted
    if not lines:
        full_text = str(data.get("full_text", "") or "").strip()
        for text_line in full_text.splitlines():
            cleaned = text_line.strip()
            if cleaned:
                lines.append(cleaned)

    return lines[:80]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType system font at the given size; fall back to scaled default."""
    candidates = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _render_structured_digital_image(file_stub: str, data: dict[str, Any], doc_analysis: dict[str, Any]) -> Optional[str]:
    try:
        width = 1600
        margin = 72
        title_font = _load_font(36)
        meta_font = _load_font(24)
        body_font = _load_font(26)
        line_height = 44

        body_lines = _structured_lines_from_data(data)
        height = max(900, 260 + (len(body_lines) * line_height))

        image = Image.new("RGB", (width, height), color=(250, 251, 253))
        draw = ImageDraw.Draw(image)

        # Outer card border
        draw.rounded_rectangle(
            (24, 24, width - 24, height - 24),
            radius=20, outline=(200, 210, 225), width=2, fill=(255, 255, 255),
        )

        # Header bar
        draw.rectangle((24, 24, width - 24, 110), fill=(30, 41, 59))
        draw.text((margin, 40), "Digital Document Reconstruction", fill=(255, 255, 255), font=title_font)

        # Meta row
        doc_type = str(doc_analysis.get("type", "UNKNOWN") or "UNKNOWN")
        score = float(doc_analysis.get("confidence_score", 0.0) or 0.0)
        draw.text((margin, 124), f"Type: {doc_type}", fill=(71, 85, 105), font=meta_font)
        draw.text((margin + 420, 124), f"Confidence: {score * 100:.1f}%", fill=(71, 85, 105), font=meta_font)

        draw.line((margin, 168, width - margin, 168), fill=(226, 232, 240), width=2)

        y = 188
        content_width = width - (margin * 2)
        for line in body_lines:
            # Draw a faint rule behind alternate rows
            if body_lines.index(line) % 2 == 0:
                draw.rectangle((margin - 8, y - 4, width - margin + 8, y + line_height - 4), fill=(248, 250, 252))
            wrapped = _wrap_text(draw, line, body_font, content_width)
            for wrapped_line in wrapped:
                draw.text((margin, y), wrapped_line, fill=(17, 24, 39), font=body_font)
                y += line_height
            y += 8

        output_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"digital_{file_stub}.png"))
        image.save(output_path, format="PNG")
        return output_path
    except Exception as exc:
        print(f"⚠️ Digital image render failed: {exc}")
        return None


def _build_digital_output(file_stub: str, source_image_path: str, data: dict[str, Any], doc_analysis: dict[str, Any], ocr_results: list[Any], ocr_engine: Any) -> Optional[dict[str, Any]]:
    digital_path: Optional[str] = None

    # ── 1. Try Gemini image-generation (best quality) ──────────────────────
    try:
        from cloud_ocr import generate_digital_form_image_gemini
        lines = _structured_lines_from_data(data)
        extracted_text = "\n".join(lines) if lines else str(data.get("full_text", "") or "")
        if extracted_text.strip():
            gemini_out = os.path.abspath(os.path.join(UPLOAD_DIR, f"digital_{file_stub}_gemini.png"))
            digital_path = generate_digital_form_image_gemini(
                extracted_text=extracted_text,
                doc_type=str(doc_analysis.get("type", "UNKNOWN") or "UNKNOWN"),
                output_path=gemini_out,
            )
    except Exception as exc:
        print(f"⚠️ Gemini image-gen skipped: {exc}")

    # ── 2. Try OpenCV filled-form renderer (good for structured forms) ─────
    if not digital_path:
        try:
            if ocr_engine is not None and (doc_analysis.get("type") == "STRUCTURED_FORM" or data.get("kv") or data.get("table")):
                render_fn = getattr(ocr_engine, "_generate_filled_form_image", None)
                if callable(render_fn):
                    digital_path = render_fn(source_image_path, ocr_results or [])
        except Exception as exc:
            print(f"⚠️ Filled form render skipped: {exc}")

    # ── 3. PIL fallback: typed text on clean white canvas ──────────────────
    if not digital_path:
        print("[RENDER] Using PIL fallback renderer")
        digital_path = _render_structured_digital_image(file_stub, data, doc_analysis)

    if digital_path:
        print(f"[RENDER] Digital output ready: {digital_path}")
    else:
        print("[RENDER] ⚠️ All renderers failed, digital_output will be null")

    if not digital_path:
        return None

    return {
        "path": digital_path,
        "mime_type": _image_mime_type(digital_path),
        "base64": _image_file_to_base64(digital_path),
    }


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


def _is_placeholder_secret(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return True
    placeholder_markers = [
        "your_",
        "replace_",
        "changeme",
        "example",
        "dummy",
    ]
    return any(marker in lowered for marker in placeholder_markers)


def _validate_enterprise_credentials(ocr_provider: str, reasoning_provider: str) -> None:
    provider = str(ocr_provider or "").strip().lower()
    reasoning = str(reasoning_provider or "").strip().lower()

    required_by_ocr: dict[str, list[str]] = {
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
        ],
    }

    required_keys = list(required_by_ocr.get(provider, []))
    if reasoning == "gemini":
        required_keys.append("GEMINI_API_KEY")

    missing_or_placeholder = []
    for key in required_keys:
        value = os.getenv(key, "")
        if _is_placeholder_secret(value):
            missing_or_placeholder.append(key)

    if missing_or_placeholder:
        raise HTTPException(
            status_code=400,
            detail=(
                "Enterprise mode requires valid provider credentials. "
                f"Missing or placeholder values: {', '.join(missing_or_placeholder)}"
            ),
        )


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


def _match_or_create_template(data: dict[str, Any], doc_analysis: dict[str, Any], ocr_results: list[Any], filename: str, domain: str):
    fingerprint = _build_template_fingerprint(data, doc_analysis, ocr_results, filename, domain)
    candidates = mongo_store.list_templates_by_type_domain(fingerprint["document_type"], domain)

    best_template = None
    best_score = 0.0
    for candidate in candidates:
        score = _fingerprint_similarity(fingerprint, candidate.get("fingerprint", {}) or {})
        if score > best_score:
            best_template = candidate
            best_score = score

    if best_template and best_score >= 0.72:
        mongo_store.touch_template_match(int(best_template.get("template_id", 0)))
        best_template = mongo_store.get_template(int(best_template.get("template_id", 0))) or best_template
        return best_template, best_score, fingerprint

    template_name = f"{domain}_{fingerprint['document_type'].lower()}_{max(len(candidates) + 1, 1)}"
    template = mongo_store.create_template(
        {
            "template_name": template_name,
            "tenant_id": "default",
            "document_type": fingerprint["document_type"],
            "document_domain": domain,
            "fingerprint": fingerprint,
            "sample_count": 1,
            "approval_status": "learned",
        }
    )
    if not template:
        raise HTTPException(status_code=500, detail="Failed to create template in MongoDB")
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


def _serialize_document(document: dict[str, Any]):
    document_id = int(document.get("document_id", 0) or 0)
    field_predictions = mongo_store.list_field_predictions(document_id)
    review_tasks = mongo_store.list_review_tasks(status="all", limit=10000)
    review_tasks = [task for task in review_tasks if int(task.get("document_id", 0) or 0) == document_id]
    model_runs = mongo_store.list_model_runs(document_id)
    extraction_blocks = mongo_store.list_extraction_blocks(document_id)
    template = None
    if document.get("template_id") is not None:
        template = mongo_store.get_template(int(document.get("template_id") or 0))

    return {
        "id": document_id,
        "filename": document.get("filename", ""),
        "upload_time": document.get("upload_time", datetime.datetime.utcnow()),
        "document_type": document.get("document_type", "UNKNOWN"),
        "confidence_score": float(document.get("confidence_score", 0.0) or 0.0),
        "ocr_provider": document.get("ocr_provider", "local"),
        "layout_provider": document.get("layout_provider", "local"),
        "reasoning_provider": document.get("reasoning_provider", "local"),
        "processing_time_ms": float(document.get("processing_time_ms", 0.0) or 0.0),
        "original_image_path": document.get("original_image_path", ""),
        "processed_image_path": document.get("processed_image_path"),
        "extracted_text": document.get("extracted_text"),
        "extracted_json": document.get("extracted_json"),
        "review_status": document.get("review_status", "pending_review"),
        "schema_version": document.get("schema_version", "1.0"),
        "document_domain": document.get("document_domain", "general"),
        "template_id": document.get("template_id"),
        "template_match_score": float(document.get("template_match_score", 0.0) or 0.0),
        "field_predictions": [
            {
                "id": int(item.get("prediction_id", 0) or 0),
                "document_id": int(item.get("document_id", 0) or 0),
                "field_name": item.get("field_name", ""),
                "field_type": item.get("field_type", "text"),
                "predicted_value": item.get("predicted_value"),
                "corrected_value": item.get("corrected_value"),
                "confidence_score": float(item.get("confidence_score", 0.0) or 0.0),
                "source_engine": item.get("source_engine", "local"),
                "review_status": item.get("review_status", "pending_review"),
                "created_at": item.get("created_at", datetime.datetime.utcnow()),
                "updated_at": item.get("updated_at", datetime.datetime.utcnow()),
            }
            for item in field_predictions
        ],
        "review_tasks": [
            {
                "id": int(item.get("task_id", 0) or 0),
                "document_id": int(item.get("document_id", 0) or 0),
                "field_prediction_id": int(item.get("field_prediction_id", 0) or 0),
                "status": item.get("status", "open"),
                "priority": item.get("priority", "medium"),
                "predicted_value": item.get("predicted_value"),
                "corrected_value": item.get("corrected_value"),
                "reviewer_name": item.get("reviewer_name"),
                "review_notes": item.get("review_notes"),
                "created_at": item.get("created_at", datetime.datetime.utcnow()),
                "completed_at": item.get("completed_at"),
            }
            for item in review_tasks
        ],
        "model_runs": [
            {
                "id": int(item.get("run_id", 0) or 0),
                "document_id": int(item.get("document_id", 0) or 0),
                "stage": item.get("stage", ""),
                "model_name": item.get("model_name", ""),
                "provider": item.get("provider", "local"),
                "success": item.get("success", "true"),
                "duration_ms": float(item.get("duration_ms", 0.0) or 0.0),
                "raw_output": item.get("raw_output"),
                "created_at": item.get("created_at", datetime.datetime.utcnow()),
            }
            for item in model_runs
        ],
        "extraction_blocks": [
            {
                "id": int(item.get("block_id", 0) or 0),
                "document_id": int(item.get("document_id", 0) or 0),
                "page_number": int(item.get("page_number", 1) or 1),
                "block_type": item.get("block_type", "word"),
                "text": item.get("text"),
                "bbox": item.get("bbox"),
                "confidence_score": float(item.get("confidence_score", 0.0) or 0.0),
                "source_engine": item.get("source_engine", "local"),
                "created_at": item.get("created_at", datetime.datetime.utcnow()),
            }
            for item in extraction_blocks
        ],
        "template": (
            {
                "id": int(template.get("template_id", 0) or 0),
                "template_name": template.get("template_name", ""),
                "tenant_id": template.get("tenant_id", "default"),
                "document_type": template.get("document_type", "UNKNOWN"),
                "document_domain": template.get("document_domain", "general"),
                "fingerprint": template.get("fingerprint", {}),
                "sample_count": int(template.get("sample_count", 1) or 1),
                "approval_status": template.get("approval_status", "learned"),
                "last_matched_at": template.get("last_matched_at", datetime.datetime.utcnow()),
                "created_at": template.get("created_at", datetime.datetime.utcnow()),
            }
            if template
            else None
        ),
    }


def _update_document_review_status(document_id: int) -> str:
    open_tasks = mongo_store.count_open_review_tasks(document_id)
    review_status = "review_required" if open_tasks else "approved"
    mongo_store.update_document(document_id, {"review_status": review_status})
    return review_status


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


def _decode_auth_token(token: str) -> dict[str, Any]:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        user_id_text, email, issued_at_text, signature = decoded.split(":", 3)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid auth token format: {exc}")

    payload = f"{user_id_text}:{email}:{issued_at_text}"
    expected_signature = hashlib.sha256(f"{payload}:{_password_salt()}".encode("utf-8")).hexdigest()
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid auth token signature")

    issued_at = int(issued_at_text)
    max_age_seconds = int(os.getenv("AUTH_TOKEN_MAX_AGE_SECONDS", str(7 * 24 * 60 * 60)))
    if int(time.time()) - issued_at > max_age_seconds:
        raise HTTPException(status_code=401, detail="Auth token expired")

    return {
        "user_id": int(user_id_text),
        "email": email,
        "issued_at": issued_at,
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    _ensure_mongo_connected()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token_data = _decode_auth_token(credentials.credentials)
    user = mongo_store.find_user_by_id(int(token_data["user_id"]))
    if not user or str(user.get("email", "")).lower() != str(token_data["email"]).lower():
        raise HTTPException(status_code=401, detail="Authenticated user not found")
    return user

@app.get("/")
def read_root():
    return {"status": "DOC-INTEL API is running", "version": "5.0"}


@app.post("/api/v1/auth/signup", response_model=schemas.AuthResponse)
def signup(payload: schemas.SignupRequest):
    _ensure_mongo_connected()
    username = payload.username.strip()
    email = payload.email.strip().lower()
    password = payload.password.strip()

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if mongo_store.user_exists_by_email_or_username(email=email, username=username):
        raise HTTPException(status_code=409, detail="User with this email or username already exists")

    user = mongo_store.create_user(username=username, email=email, password_hash=_hash_password(password))
    if not user:
        raise HTTPException(status_code=500, detail="Unable to create user in MongoDB")

    user_id = int(user.get("user_id", 0) or 0)
    user_name = str(user.get("username", ""))
    user_email = str(user.get("email", ""))

    return schemas.AuthResponse(
        status="success",
        user_id=user_id,
        username=user_name,
        email=user_email,
        token=_build_auth_token(user_id, user_email),
    )


@app.post("/api/v1/auth/login", response_model=schemas.AuthResponse)
def login(payload: schemas.LoginRequest):
    _ensure_mongo_connected()
    email = payload.email.strip().lower()
    password = payload.password.strip()

    user = mongo_store.find_user_by_email(email)
    if not user or not _verify_password(password, str(user.get("password_hash", ""))):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = int(user.get("user_id", 0) or 0)
    user_name = str(user.get("username", ""))
    user_email = str(user.get("email", ""))

    return schemas.AuthResponse(
        status="success",
        user_id=user_id,
        username=user_name,
        email=user_email,
        token=_build_auth_token(user_id, user_email),
    )


@app.get("/api/v1/auth/me")
def auth_me(current_user=Depends(get_current_user)):
    return {
        "status": "success",
        "user_id": int(current_user.get("user_id", 0) or 0),
        "username": str(current_user.get("username", "")),
        "email": str(current_user.get("email", "")),
    }


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
        "celery": required(["CELERY_BROKER_URL"]),
        "mongodb": mongo_store.health(),
    }


@app.get("/api/v1/mongo/health")
def get_mongo_health():
    return mongo_store.health()


@app.get("/api/v1/mongo/documents/{document_id}")
def get_mongo_document(document_id: int, current_user=Depends(get_current_user)):
    record = mongo_store.get_document_bundle(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Mongo document bundle not found")
    return record


@app.post("/api/v1/mongo/vector/search")
def mongo_vector_search(payload: dict[str, Any], current_user=Depends(get_current_user)):
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
    current_user=Depends(get_current_user),
):
    _ensure_mongo_connected()
    try:
        from workers.document_tasks import process_document_async
        from celery_app import is_celery_broker_available
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Celery worker is not available: {exc}")

    broker_available, broker_error = is_celery_broker_available()
    if not broker_available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Async queue is unavailable because the Celery broker is not reachable. "
                f"Start Redis or update CELERY_BROKER_URL. Root cause: {broker_error}"
            ),
        )

    filename = file.filename or "upload.jpg"
    file_extension = filename.split(".")[-1] if "." in filename else "jpg"
    unique_filename = f"{uuid4().hex}.{file_extension}"
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))

    file_contents = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(file_contents)

    new_record = mongo_store.create_document(
        {
            "filename": filename,
            "document_type": "QUEUED",
            "confidence_score": 0.0,
            "original_image_path": file_path,
            "processed_image_path": file_path,
            "extracted_text": "",
            "extracted_json": {"status": "queued"},
            "ocr_provider": "queued",
            "processing_time_ms": 0.0,
            "review_status": "queued",
            "schema_version": "1.0",
            "document_domain": "general",
            "template_match_score": 0.0,
            "layout_provider": enterprise_ocr_provider,
            "reasoning_provider": enterprise_reasoning_provider,
        }
    )
    if not new_record:
        raise HTTPException(status_code=500, detail="Failed to create queued document in MongoDB")

    record_id = int(new_record.get("document_id", 0) or 0)
    try:
        task = process_document_async.apply_async(args=(record_id,), ignore_result=True)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Failed to enqueue async document job. "
                f"Check Redis / CELERY_BROKER_URL and restart the API if configuration changed. Root cause: {exc}"
            ),
        )

    mongo_store.create_model_run(
        {
            "document_id": record_id,
            "stage": "async_queue",
            "model_name": "celery_enqueue",
            "provider": "celery",
            "success": "true",
            "duration_ms": 0.0,
            "raw_output": {
                "task_id": task.id,
                "layout_provider": enterprise_ocr_provider,
                "reasoning_provider": enterprise_reasoning_provider,
            },
        }
    )

    return {
        "status": "queued",
        "record_id": record_id,
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
    current_user=Depends(get_current_user),
):
    """
    Enterprise Endpoint: Uploads file, locally saves it securely, processes via Vision + OCR + AI, and stores in DB.
    """
    _ensure_mongo_connected()
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
            _validate_enterprise_credentials(enterprise_ocr_provider, enterprise_reasoning_provider)
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
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Enterprise pipeline failed and strict enterprise mode is enabled. "
                        f"Fix provider credentials/config and retry. Root cause: {e}"
                    ),
                )

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
        digital_output = _build_digital_output(
            file_stub=unique_filename.rsplit(".", 1)[0],
            source_image_path=processed_path,
            data=data,
            doc_analysis=doc_analysis,
            ocr_results=ocr_results,
            ocr_engine=ocr_engine,
        )

        template, template_match_score, template_fingerprint = _match_or_create_template(
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
                "template_id": template.get("template_id"),
                "template_name": template.get("template_name"),
                "template_match_score": template_match_score,
                "fingerprint": template_fingerprint,
            },
        })

        # 4. Save to Enterprise Database
        new_record = mongo_store.create_document(
            {
                "filename": filename,
                "document_type": doc_analysis.get("type", "UNKNOWN"),
                "confidence_score": doc_analysis.get("confidence_score", 0.0),
                "original_image_path": file_path,
                "processed_image_path": processed_path,
                "extracted_text": data.get("full_text", ""),
                "extracted_json": data,
                "ocr_provider": source_engine,
                "processing_time_ms": processing_time_ms,
                "review_status": "review_required" if open_review_count else "approved",
                "schema_version": "1.0",
                "document_domain": document_domain,
                "template_id": int(template.get("template_id", 0) or 0),
                "template_match_score": template_match_score,
                "layout_provider": enterprise_ocr_provider if enterprise_mode else source_engine,
                "reasoning_provider": enterprise_reasoning_provider if enterprise_mode else (provider if cloud_mode else "local"),
            }
        )
        if not new_record:
            raise HTTPException(status_code=500, detail="Failed to persist document in MongoDB")

        record_id = int(new_record.get("document_id", 0) or 0)

        for block in extraction_blocks:
            mongo_store.create_extraction_block(
                {
                    "document_id": record_id,
                    "page_number": block["page_number"],
                    "block_type": block["block_type"],
                    "text": block["text"],
                    "bbox": block["bbox"],
                    "confidence_score": block["confidence_score"],
                    "source_engine": block["source_engine"],
                }
            )

        saved_predictions = []
        for prediction in field_predictions:
            prediction_review_status = _review_status_for_confidence(prediction["confidence_score"])
            saved_prediction = mongo_store.create_field_prediction(
                {
                    "document_id": record_id,
                    "field_name": prediction["field_name"],
                    "field_type": prediction["field_type"],
                    "predicted_value": prediction["predicted_value"],
                    "confidence_score": prediction["confidence_score"],
                    "source_engine": prediction["source_engine"],
                    "review_status": prediction_review_status,
                }
            )
            if not saved_prediction:
                continue
            saved_predictions.append(saved_prediction)

            if prediction_review_status != "auto_approved":
                mongo_store.create_review_task(
                    {
                        "document_id": record_id,
                        "field_prediction_id": int(saved_prediction.get("prediction_id", 0) or 0),
                        "status": "open",
                        "priority": _review_priority(float(saved_prediction.get("confidence_score", 0.0) or 0.0)),
                        "predicted_value": saved_prediction.get("predicted_value"),
                    }
                )

        for run in model_runs:
            mongo_store.create_model_run(
                {
                    "document_id": record_id,
                    "stage": run["stage"],
                    "model_name": run["model_name"],
                    "provider": run["provider"],
                    "success": run["success"],
                    "duration_ms": run["duration_ms"],
                    "raw_output": run["raw_output"],
                }
            )

        print(f"✅ Record #{record_id} saved to MongoDB.")

        mongo_store.upsert_document_bundle(
            document_id=record_id,
            file_path=file_path,
            processed_path=processed_path,
            document_type=str(doc_analysis.get("type", "UNKNOWN") or "UNKNOWN"),
            status=str(new_record.get("review_status", "processed") or "processed"),
            confidence_score=float(doc_analysis.get("confidence_score", 0.0) or 0.0),
            domain=document_domain,
            source_engine=source_engine,
            extracted_data=data,
            field_predictions=[
                {
                    "name": item.get("field_name"),
                    "value": item.get("predicted_value"),
                    "confidence": item.get("confidence_score"),
                    "review_status": item.get("review_status"),
                }
                for item in saved_predictions
            ],
            model_runs=model_runs,
            template_payload={
                "template_id": template.get("template_id"),
                "template_name": template.get("template_name"),
                "document_domain": template.get("document_domain"),
                "match_score": template_match_score,
                "fingerprint": template_fingerprint,
            },
        )

        # 5. Return JSON to Client
        return {
            "status": "success",
            "record_id": record_id,
            "analysis": doc_analysis,
            "structured_data": data,
            "field_predictions": [
                {
                    "id": int(item.get("prediction_id", 0) or 0),
                    "field_name": item.get("field_name", ""),
                    "field_type": item.get("field_type", "text"),
                    "predicted_value": item.get("predicted_value"),
                    "confidence_score": float(item.get("confidence_score", 0.0) or 0.0),
                    "review_status": item.get("review_status", "pending_review"),
                }
                for item in saved_predictions
            ],
            "review_summary": {
                "document_review_status": new_record.get("review_status", "pending_review"),
                "open_tasks": open_review_count,
                "auto_approved_fields": len(field_predictions) - open_review_count,
            },
            "pipeline": {
                "enterprise_mode": enterprise_mode,
                "cloud_mode": cloud_mode,
                "layout_provider": new_record.get("layout_provider", "local"),
                "reasoning_provider": new_record.get("reasoning_provider", "local"),
            },
            "template": {
                "id": int(template.get("template_id", 0) or 0),
                "template_name": template.get("template_name", ""),
                "document_domain": template.get("document_domain", "general"),
                "match_score": template_match_score,
            },
            "layout_summary": {
                "block_count": len(extraction_blocks),
                "source_engine": source_engine,
            },
            "paths": {
                "original": file_path,
                "processed": processed_path,
                "digital": digital_output.get("path") if digital_output else None,
            },
            "digital_output": digital_output,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
def get_history(limit: int = 50, current_user=Depends(get_current_user)):
    """Retrieve document history from MongoDB."""
    _ensure_mongo_connected()
    records = mongo_store.list_documents(limit=limit)
    response = []
    for record in records:
        response.append(
            {
                "id": int(record.get("document_id", 0) or 0),
                "filename": record.get("filename", ""),
                "upload_time": record.get("upload_time", datetime.datetime.utcnow()),
                "document_type": record.get("document_type", "UNKNOWN"),
                "confidence_score": float(record.get("confidence_score", 0.0) or 0.0),
                "ocr_provider": record.get("ocr_provider", "local"),
                "layout_provider": record.get("layout_provider", "local"),
                "reasoning_provider": record.get("reasoning_provider", "local"),
                "processing_time_ms": float(record.get("processing_time_ms", 0.0) or 0.0),
                "extracted_text": record.get("extracted_text"),
                "extracted_json": record.get("extracted_json"),
                "review_status": record.get("review_status", "pending_review"),
                "schema_version": record.get("schema_version", "1.0"),
                "document_domain": record.get("document_domain", "general"),
                "template_id": record.get("template_id"),
                "template_match_score": float(record.get("template_match_score", 0.0) or 0.0),
            }
        )
    return response


@app.get("/api/v1/documents/{document_id}", response_model=schemas.DocumentDetailResponse)
def get_document_detail(document_id: int, current_user=Depends(get_current_user)):
    _ensure_mongo_connected()
    document = mongo_store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return _serialize_document(document)


@app.get("/api/v1/review/tasks", response_model=list[schemas.ReviewTaskResponse])
def get_review_tasks(status: str = "open", limit: int = 50, current_user=Depends(get_current_user)):
    _ensure_mongo_connected()
    tasks = mongo_store.list_review_tasks(status=status, limit=limit)
    return [
        {
            "id": int(item.get("task_id", 0) or 0),
            "document_id": int(item.get("document_id", 0) or 0),
            "field_prediction_id": int(item.get("field_prediction_id", 0) or 0),
            "status": item.get("status", "open"),
            "priority": item.get("priority", "medium"),
            "predicted_value": item.get("predicted_value"),
            "corrected_value": item.get("corrected_value"),
            "reviewer_name": item.get("reviewer_name"),
            "review_notes": item.get("review_notes"),
            "created_at": item.get("created_at", datetime.datetime.utcnow()),
            "completed_at": item.get("completed_at"),
        }
        for item in tasks
    ]


@app.get("/api/v1/templates", response_model=list[schemas.TemplateResponse])
def get_templates(limit: int = 50, current_user=Depends(get_current_user)):
    _ensure_mongo_connected()
    templates = mongo_store.list_templates(limit=limit)
    return [
        {
            "id": int(item.get("template_id", 0) or 0),
            "template_name": item.get("template_name", ""),
            "tenant_id": item.get("tenant_id", "default"),
            "document_type": item.get("document_type", "UNKNOWN"),
            "document_domain": item.get("document_domain", "general"),
            "fingerprint": item.get("fingerprint", {}),
            "sample_count": int(item.get("sample_count", 1) or 1),
            "approval_status": item.get("approval_status", "learned"),
            "last_matched_at": item.get("last_matched_at", datetime.datetime.utcnow()),
            "created_at": item.get("created_at", datetime.datetime.utcnow()),
        }
        for item in templates
    ]


@app.post("/api/v1/review/tasks/{task_id}/complete")
def complete_review_task(task_id: int, payload: schemas.ReviewTaskCompleteRequest, current_user=Depends(get_current_user)):
    _ensure_mongo_connected()
    task = mongo_store.get_review_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Review task not found")

    prediction = mongo_store.get_field_prediction(int(task.get("field_prediction_id", 0) or 0))
    document = mongo_store.get_document(int(task.get("document_id", 0) or 0))
    if not prediction or not document:
        raise HTTPException(status_code=404, detail="Review task is linked to missing records")

    final_value = payload.corrected_value if payload.corrected_value is not None else prediction.get("predicted_value")
    mongo_store.update_review_task(
        task_id,
        {
            "corrected_value": final_value,
            "reviewer_name": payload.reviewer_name,
            "review_notes": payload.review_notes,
            "status": "completed",
            "completed_at": datetime.datetime.utcnow(),
        },
    )

    mongo_store.update_field_prediction(
        int(prediction.get("prediction_id", 0) or 0),
        {
            "corrected_value": final_value,
            "review_status": "approved" if payload.resolution == "approved" else "corrected",
        },
    )

    extracted_json = document.get("extracted_json", {}) or {}
    updated_payload = _apply_correction_to_payload(extracted_json, str(prediction.get("field_name", "")), final_value)
    document_update: dict[str, Any] = {"extracted_json": updated_payload}
    if str(prediction.get("field_name", "")) == "full_text":
        document_update["extracted_text"] = final_value
    mongo_store.update_document(int(document.get("document_id", 0) or 0), document_update)

    new_review_status = _update_document_review_status(int(document.get("document_id", 0) or 0))

    mongo_store.save_review_correction(
        document_id=int(document.get("document_id", 0) or 0),
        field_name=str(prediction.get("field_name", "")),
        predicted_value=str(prediction.get("predicted_value") or ""),
        corrected_value=str(final_value or ""),
        reviewer_name=payload.reviewer_name,
        review_notes=payload.review_notes,
    )

    return {
        "status": "success",
        "task_id": int(task.get("task_id", 0) or 0),
        "document_id": int(document.get("document_id", 0) or 0),
        "document_review_status": new_review_status,
        "field_name": str(prediction.get("field_name", "")),
        "final_value": final_value,
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
