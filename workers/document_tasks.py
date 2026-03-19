import os
import time
from typing import Any

from celery_app import celery_app
from cloud_ocr import run_enterprise_idp_pipeline
from mongo_store import MongoIDPStore
from env_loader import load_env_file


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_env_file(os.path.join(ROOT_DIR, ".env"))


mongo_store = MongoIDPStore.from_env()


def _save_async_result(document_id: int, pipeline_output: dict[str, Any]) -> None:
    document: Any = mongo_store.get_document(document_id)
    if not document:
        return

    doc_analysis = pipeline_output.get("doc_analysis", {})
    data = pipeline_output.get("data", {})
    source_engine = pipeline_output.get("source_engine", "enterprise")

    mongo_store.update_document(
        document_id,
        {
            "document_type": str(doc_analysis.get("type", document.get("document_type", "UNKNOWN")) or document.get("document_type", "UNKNOWN")),
            "confidence_score": float(doc_analysis.get("confidence_score", document.get("confidence_score", 0.0)) or document.get("confidence_score", 0.0)),
            "extracted_text": str(data.get("full_text", document.get("extracted_text", "") or "") or ""),
            "extracted_json": data,
            "ocr_provider": source_engine,
            "review_status": "review_required",
        },
    )


@celery_app.task(name="workers.process_document_async", ignore_result=True)
def process_document_async(document_id: int) -> dict[str, Any]:
    if not mongo_store.is_connected():
        return {"status": "error", "document_id": document_id, "error": "MongoDB is not connected"}

    started = time.perf_counter()
    try:
        document: Any = mongo_store.get_document(document_id)
        if not document:
            return {"status": "error", "error": "Document not found", "document_id": document_id}

        image_path = str(document.get("original_image_path", "") or "")
        if not image_path or not os.path.exists(image_path):
            return {"status": "error", "error": "Source document file missing", "document_id": document_id}

        # Route through enterprise OCR + reasoning stack.
        output = run_enterprise_idp_pipeline(
            image_path=image_path,
            ocr_provider=str(document.get("layout_provider", "documentai") or "documentai"),
            reasoning_provider=str(document.get("reasoning_provider", "gemini") or "gemini"),
            document_type_hint=str(document.get("document_type", "UNKNOWN") or "UNKNOWN"),
        )
        _save_async_result(document_id, output)

        mongo_store.create_model_run(
            {
                "document_id": document_id,
                "stage": "async_enterprise_pipeline",
                "model_name": "specialized_ocr_plus_reasoning",
                "provider": "enterprise",
                "success": "true",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "raw_output": {
                    "source_engine": output.get("source_engine", "enterprise"),
                    "token_count": len(output.get("ocr_results", []) or []),
                },
            }
        )
        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        return {"status": "error", "document_id": document_id, "error": str(exc)}
