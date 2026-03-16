import os
import time
from typing import Any

from sqlalchemy.orm import Session

from celery_app import celery_app
from cloud_ocr import run_enterprise_idp_pipeline
from database import SessionLocal
import models


def _save_async_result(db: Session, document_id: int, pipeline_output: dict[str, Any]) -> None:
    document: Any = db.query(models.DocumentRecord).filter(models.DocumentRecord.id == document_id).first()
    if not document:
        return

    doc_analysis = pipeline_output.get("doc_analysis", {})
    data = pipeline_output.get("data", {})
    source_engine = pipeline_output.get("source_engine", "enterprise")

    setattr(document, "document_type", str(doc_analysis.get("type", getattr(document, "document_type", "UNKNOWN")) or getattr(document, "document_type", "UNKNOWN")))
    setattr(document, "confidence_score", float(doc_analysis.get("confidence_score", getattr(document, "confidence_score", 0.0)) or getattr(document, "confidence_score", 0.0)))
    setattr(document, "extracted_text", str(data.get("full_text", getattr(document, "extracted_text", "") or "") or ""))
    setattr(document, "extracted_json", data)
    setattr(document, "ocr_provider", source_engine)
    setattr(document, "review_status", "review_required")

    db.add(document)


@celery_app.task(name="workers.process_document_async")
def process_document_async(document_id: int) -> dict[str, Any]:
    db = SessionLocal()
    started = time.perf_counter()
    try:
        document: Any = db.query(models.DocumentRecord).filter(models.DocumentRecord.id == document_id).first()
        if not document:
            return {"status": "error", "error": "Document not found", "document_id": document_id}

        image_path = str(getattr(document, "original_image_path", "") or "")
        if not image_path or not os.path.exists(image_path):
            return {"status": "error", "error": "Source document file missing", "document_id": document_id}

        # Route through enterprise OCR + reasoning stack.
        output = run_enterprise_idp_pipeline(
            image_path=image_path,
            ocr_provider=str(getattr(document, "layout_provider", "documentai") or "documentai"),
            reasoning_provider=str(getattr(document, "reasoning_provider", "gemini") or "gemini"),
            document_type_hint=str(getattr(document, "document_type", "UNKNOWN") or "UNKNOWN"),
        )
        _save_async_result(db, document_id, output)

        db.add(
            models.ModelRun(
                document_id=document_id,
                stage="async_enterprise_pipeline",
                model_name="specialized_ocr_plus_reasoning",
                provider="enterprise",
                success="true",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                raw_output={
                    "source_engine": output.get("source_engine", "enterprise"),
                    "token_count": len(output.get("ocr_results", []) or []),
                },
            )
        )

        db.commit()
        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        db.rollback()
        return {"status": "error", "document_id": document_id, "error": str(exc)}
    finally:
        db.close()
