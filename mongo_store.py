import datetime
import hashlib
import os
from typing import Any, Optional

try:
    from pymongo import MongoClient  # type: ignore[import-not-found]
    from pymongo.errors import PyMongoError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    MongoClient = None  # type: ignore[assignment]

    class PyMongoError(Exception):
        pass


class MongoIDPStore:
    def __init__(self, uri: Optional[str], db_name: str = "idp"):
        self.uri = uri
        self.db_name = db_name
        self.enabled = bool(uri and MongoClient is not None)
        self._client = None
        self._db = None

    @classmethod
    def from_env(cls) -> "MongoIDPStore":
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "idp")
        return cls(uri=uri, db_name=db_name)

    def _connect(self):
        if not self.enabled:
            return None
        if self._db is not None:
            return self._db

        assert self.uri is not None
        assert MongoClient is not None
        self._client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[self.db_name]

        self._db.documents.create_index("document_id", unique=True)
        self._db.extraction_results.create_index("document_id")
        self._db.ai_processing_logs.create_index("document_id")
        self._db.templates.create_index("template_id")
        self._db.review_corrections.create_index("document_id")
        self._db.embeddings.create_index("document_id")
        return self._db

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "configured": bool(self.uri),
                "connected": False,
                "error": "MONGODB_URI missing or pymongo not installed",
            }
        try:
            db = self._connect()
            assert db is not None
            db.command("ping")
            return {
                "enabled": True,
                "configured": True,
                "connected": True,
                "database": self.db_name,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "configured": True,
                "connected": False,
                "database": self.db_name,
                "error": str(exc),
            }

    def upsert_document_bundle(
        self,
        *,
        document_id: int,
        file_path: str,
        processed_path: str,
        document_type: str,
        status: str,
        confidence_score: float,
        domain: str,
        source_engine: str,
        extracted_data: dict[str, Any],
        field_predictions: list[dict[str, Any]],
        model_runs: list[dict[str, Any]],
        template_payload: dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False

        try:
            db = self._connect()
            assert db is not None

            now = datetime.datetime.utcnow()
            full_text = str(extracted_data.get("full_text", "") or "")

            db.documents.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "document_id": document_id,
                        "file_url": file_path,
                        "processed_file_url": processed_path,
                        "document_type": document_type,
                        "status": status,
                        "confidence_score": confidence_score,
                        "document_domain": domain,
                        "source_engine": source_engine,
                        "created_at": now,
                    }
                },
                upsert=True,
            )

            db.extraction_results.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "document_id": document_id,
                        "fields": field_predictions,
                        "raw": extracted_data,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )

            db.ai_processing_logs.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "document_id": document_id,
                        "runs": model_runs,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )

            db.templates.update_one(
                {
                    "document_id": document_id,
                    "template_id": template_payload.get("template_id"),
                },
                {
                    "$set": {
                        "document_id": document_id,
                        "template_id": template_payload.get("template_id"),
                        "template_name": template_payload.get("template_name"),
                        "document_domain": template_payload.get("document_domain"),
                        "match_score": template_payload.get("match_score", 0.0),
                        "fingerprint": template_payload.get("fingerprint", {}),
                        "updated_at": now,
                    }
                },
                upsert=True,
            )

            embedding = self._deterministic_embedding(full_text)
            db.embeddings.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "document_id": document_id,
                        "embedding": embedding,
                        "document_type": document_type,
                        "document_domain": domain,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )
            return True
        except PyMongoError:
            return False

    def save_review_correction(
        self,
        *,
        document_id: int,
        field_name: str,
        predicted_value: str,
        corrected_value: str,
        reviewer_name: Optional[str],
        review_notes: Optional[str],
    ) -> bool:
        if not self.enabled:
            return False

        try:
            db = self._connect()
            assert db is not None
            db.review_corrections.insert_one(
                {
                    "document_id": document_id,
                    "field_name": field_name,
                    "predicted_value": predicted_value,
                    "corrected_value": corrected_value,
                    "reviewer_name": reviewer_name,
                    "review_notes": review_notes,
                    "created_at": datetime.datetime.utcnow(),
                }
            )
            return True
        except PyMongoError:
            return False

    def get_document_bundle(self, document_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            assert db is not None
            doc = db.documents.find_one({"document_id": document_id}, {"_id": 0})
            if not doc:
                return None
            extraction = db.extraction_results.find_one({"document_id": document_id}, {"_id": 0}) or {}
            logs = db.ai_processing_logs.find_one({"document_id": document_id}, {"_id": 0}) or {}
            template = db.templates.find_one({"document_id": document_id}, {"_id": 0}) or {}
            return {
                "document": doc,
                "extraction_results": extraction,
                "ai_processing_logs": logs,
                "template": template,
            }
        except PyMongoError:
            return None

    def vector_search(
        self,
        query_vector: list[float],
        limit: int = 5,
        num_candidates: int = 100,
        document_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            assert db is not None

            search_stage: dict[str, Any] = {
                "$vectorSearch": {
                    "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            }

            pipeline: list[dict[str, Any]] = [search_stage]
            if document_type:
                pipeline.append({"$match": {"document_type": document_type}})
            pipeline.append(
                {
                    "$project": {
                        "_id": 0,
                        "document_id": 1,
                        "document_type": 1,
                        "document_domain": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                }
            )
            return list(db.embeddings.aggregate(pipeline))
        except PyMongoError:
            return []

    def _deterministic_embedding(self, text: str, dims: int = 64) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * dims

        values: list[float] = []
        seed = cleaned.encode("utf-8")
        for i in range(dims):
            digest = hashlib.sha256(seed + f"::{i}".encode("utf-8")).digest()
            int_value = int.from_bytes(digest[:4], "big", signed=False)
            values.append((int_value % 1_000_000) / 1_000_000.0)
        return values
