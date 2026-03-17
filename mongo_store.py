import datetime
import hashlib
import os
import time
from typing import Any, Optional

try:
    from pymongo import MongoClient  # type: ignore[import-not-found]
    from pymongo import ReturnDocument  # type: ignore[import-not-found]
    from pymongo.errors import PyMongoError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    MongoClient = None  # type: ignore[assignment]
    ReturnDocument = None  # type: ignore[assignment]

    class PyMongoError(Exception):
        pass

try:
    import mongomock  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency at runtime
    mongomock = None  # type: ignore[assignment]

# Special URI value that activates the in-memory mongomock backend.
_MONGOMOCK_URI = "mongomock://"


class MongoIDPStore:
    def __init__(self, uri: Optional[str], db_name: str = "idp"):
        self.uri = uri
        self.db_name = db_name
        # Use in-memory mongomock when no real URI is provided (local dev / CI)
        # or when the URI is explicitly set to the special "mongomock://" value.
        self._use_mock = bool(
            mongomock is not None
            and (not uri or (uri and uri.strip() == _MONGOMOCK_URI))
        )
        self.enabled = self._use_mock or bool(uri and MongoClient is not None)
        self._client = None
        self._db = None
        # Cache the last successful ping time so we don't ping MongoDB on
        # every single API request.  A re-check is done at most every 30 s.
        self._last_ping_ok: bool = False
        self._last_ping_time: float = 0.0

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

        if self._use_mock:
            assert mongomock is not None
            self._client = mongomock.MongoClient()
        else:
            assert self.uri is not None
            assert MongoClient is not None
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[self.db_name]

        self._db.documents.create_index("document_id", unique=True)
        self._db.users.create_index("user_id", unique=True)
        self._db.users.create_index("email", unique=True)
        self._db.users.create_index("username", unique=True)
        self._db.templates.create_index("template_id", unique=True)
        self._db.field_predictions.create_index([("document_id", 1), ("prediction_id", 1)])
        self._db.review_tasks.create_index([("document_id", 1), ("task_id", 1)])
        self._db.review_tasks.create_index("status")
        self._db.model_runs.create_index([("document_id", 1), ("run_id", 1)])
        self._db.extraction_blocks.create_index([("document_id", 1), ("block_id", 1)])
        self._db.extraction_results.create_index("document_id")
        self._db.ai_processing_logs.create_index("document_id")
        self._db.review_corrections.create_index("document_id")
        self._db.embeddings.create_index("document_id")
        self._db.counters.create_index("name", unique=True)
        return self._db

    def _get_db_or_none(self):
        try:
            return self._connect()
        except Exception:
            return None

    def _strip_mongo_id(self, payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not payload:
            return payload
        item = dict(payload)
        item.pop("_id", None)
        return item

    def _next_sequence(self, name: str) -> int:
        db = self._get_db_or_none()
        if db is None:
            raise RuntimeError("MongoDB is not available")
        counter = db.counters.find_one_and_update(
            {"name": name},
            {"$inc": {"value": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        value = (counter or {}).get("value")
        if isinstance(value, int):
            return value
        # First upsert can return old doc depending on server config, so read back.
        current = db.counters.find_one({"name": name}) or {}
        return int(current.get("value", 1))

    def is_connected(self) -> bool:
        """Return True if MongoDB is reachable.

        The result is cached for up to 30 seconds so that ordinary API
        requests do not incur an extra network round-trip for every call.
        """
        _CACHE_TTL_SECONDS = 30.0
        now = time.monotonic()
        if self._last_ping_ok and (now - self._last_ping_time) < _CACHE_TTL_SECONDS:
            return True
        health = self.health()
        connected = bool(health.get("enabled") and health.get("connected"))
        self._last_ping_ok = connected
        self._last_ping_time = now
        return connected

    def create_user(self, username: str, email: str, password_hash: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            now = datetime.datetime.utcnow()
            user_id = self._next_sequence("users")
            payload = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "created_at": now,
            }
            db.users.insert_one(payload)
            return payload
        except PyMongoError:
            return None

    def find_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.users.find_one({"email": email}))
        except PyMongoError:
            return None

    def find_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.users.find_one({"user_id": int(user_id)}))
        except PyMongoError:
            return None

    def user_exists_by_email_or_username(self, email: str, username: str) -> bool:
        if not self.enabled:
            return False
        try:
            db = self._connect()
            found = db.users.find_one(
                {
                    "$or": [
                        {"email": email},
                        {"username": username},
                    ]
                },
                {"_id": 1},
            )
            return bool(found)
        except PyMongoError:
            return False

    def create_document(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            document_id = self._next_sequence("documents")
            now = datetime.datetime.utcnow()
            record = {
                "document_id": document_id,
                "filename": payload.get("filename", "upload.jpg"),
                "upload_time": payload.get("upload_time", now),
                "document_type": payload.get("document_type", "UNKNOWN"),
                "confidence_score": float(payload.get("confidence_score", 0.0) or 0.0),
                "original_image_path": payload.get("original_image_path", ""),
                "processed_image_path": payload.get("processed_image_path", ""),
                "extracted_text": payload.get("extracted_text", ""),
                "extracted_json": payload.get("extracted_json", {}),
                "ocr_provider": payload.get("ocr_provider", "local"),
                "processing_time_ms": float(payload.get("processing_time_ms", 0.0) or 0.0),
                "review_status": payload.get("review_status", "pending_review"),
                "schema_version": payload.get("schema_version", "1.0"),
                "document_domain": payload.get("document_domain", "general"),
                "template_id": payload.get("template_id"),
                "template_match_score": float(payload.get("template_match_score", 0.0) or 0.0),
                "layout_provider": payload.get("layout_provider", "local"),
                "reasoning_provider": payload.get("reasoning_provider", "local"),
                "updated_at": now,
            }
            db.documents.insert_one(record)
            return record
        except PyMongoError:
            return None

    def update_document(self, document_id: int, updates: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            db = self._connect()
            updates = dict(updates)
            updates["updated_at"] = datetime.datetime.utcnow()
            result = db.documents.update_one({"document_id": int(document_id)}, {"$set": updates})
            return result.matched_count > 0
        except PyMongoError:
            return False

    def get_document(self, document_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.documents.find_one({"document_id": int(document_id)}))
        except PyMongoError:
            return None

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.documents.find({}, {"_id": 0}).sort("document_id", -1).limit(max(1, int(limit)))
            return list(cursor)
        except PyMongoError:
            return []

    def create_model_run(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            run_id = self._next_sequence("model_runs")
            now = datetime.datetime.utcnow()
            record = {
                "run_id": run_id,
                "document_id": int(payload.get("document_id", 0) or 0),
                "stage": payload.get("stage", "unknown"),
                "model_name": payload.get("model_name", "unknown"),
                "provider": payload.get("provider", "local"),
                "success": payload.get("success", "true"),
                "duration_ms": float(payload.get("duration_ms", 0.0) or 0.0),
                "raw_output": payload.get("raw_output", {}),
                "created_at": now,
            }
            db.model_runs.insert_one(record)
            return record
        except PyMongoError:
            return None

    def list_model_runs(self, document_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.model_runs.find({"document_id": int(document_id)}, {"_id": 0}).sort("run_id", 1)
            return list(cursor)
        except PyMongoError:
            return []

    def create_extraction_block(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            block_id = self._next_sequence("extraction_blocks")
            now = datetime.datetime.utcnow()
            record = {
                "block_id": block_id,
                "document_id": int(payload.get("document_id", 0) or 0),
                "page_number": int(payload.get("page_number", 1) or 1),
                "block_type": payload.get("block_type", "word"),
                "text": payload.get("text"),
                "bbox": payload.get("bbox"),
                "confidence_score": float(payload.get("confidence_score", 0.0) or 0.0),
                "source_engine": payload.get("source_engine", "local"),
                "created_at": now,
            }
            db.extraction_blocks.insert_one(record)
            return record
        except PyMongoError:
            return None

    def list_extraction_blocks(self, document_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.extraction_blocks.find({"document_id": int(document_id)}, {"_id": 0}).sort("block_id", 1)
            return list(cursor)
        except PyMongoError:
            return []

    def create_field_prediction(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            prediction_id = self._next_sequence("field_predictions")
            now = datetime.datetime.utcnow()
            record = {
                "prediction_id": prediction_id,
                "document_id": int(payload.get("document_id", 0) or 0),
                "field_name": payload.get("field_name", ""),
                "field_type": payload.get("field_type", "text"),
                "predicted_value": payload.get("predicted_value"),
                "corrected_value": payload.get("corrected_value"),
                "confidence_score": float(payload.get("confidence_score", 0.0) or 0.0),
                "source_engine": payload.get("source_engine", "local"),
                "review_status": payload.get("review_status", "pending_review"),
                "created_at": now,
                "updated_at": now,
            }
            db.field_predictions.insert_one(record)
            return record
        except PyMongoError:
            return None

    def update_field_prediction(self, prediction_id: int, updates: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            db = self._connect()
            updates = dict(updates)
            updates["updated_at"] = datetime.datetime.utcnow()
            result = db.field_predictions.update_one(
                {"prediction_id": int(prediction_id)},
                {"$set": updates},
            )
            return result.matched_count > 0
        except PyMongoError:
            return False

    def get_field_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.field_predictions.find_one({"prediction_id": int(prediction_id)}))
        except PyMongoError:
            return None

    def list_field_predictions(self, document_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.field_predictions.find({"document_id": int(document_id)}, {"_id": 0}).sort("prediction_id", 1)
            return list(cursor)
        except PyMongoError:
            return []

    def create_review_task(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            task_id = self._next_sequence("review_tasks")
            now = datetime.datetime.utcnow()
            record = {
                "task_id": task_id,
                "document_id": int(payload.get("document_id", 0) or 0),
                "field_prediction_id": int(payload.get("field_prediction_id", 0) or 0),
                "status": payload.get("status", "open"),
                "priority": payload.get("priority", "medium"),
                "predicted_value": payload.get("predicted_value"),
                "corrected_value": payload.get("corrected_value"),
                "reviewer_name": payload.get("reviewer_name"),
                "review_notes": payload.get("review_notes"),
                "created_at": payload.get("created_at", now),
                "completed_at": payload.get("completed_at"),
            }
            db.review_tasks.insert_one(record)
            return record
        except PyMongoError:
            return None

    def update_review_task(self, task_id: int, updates: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            db = self._connect()
            result = db.review_tasks.update_one(
                {"task_id": int(task_id)},
                {"$set": dict(updates)},
            )
            return result.matched_count > 0
        except PyMongoError:
            return False

    def get_review_task(self, task_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.review_tasks.find_one({"task_id": int(task_id)}))
        except PyMongoError:
            return None

    def list_review_tasks(self, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            query: dict[str, Any] = {}
            if status != "all":
                query["status"] = status
            cursor = db.review_tasks.find(query, {"_id": 0}).sort("created_at", -1).limit(max(1, int(limit)))
            return list(cursor)
        except PyMongoError:
            return []

    def count_open_review_tasks(self, document_id: int) -> int:
        if not self.enabled:
            return 0
        try:
            db = self._connect()
            return int(
                db.review_tasks.count_documents(
                    {"document_id": int(document_id), "status": "open"}
                )
            )
        except PyMongoError:
            return 0

    def list_templates_by_type_domain(self, document_type: str, document_domain: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.templates.find(
                {
                    "document_type": document_type,
                    "document_domain": document_domain,
                },
                {"_id": 0},
            ).sort("template_id", 1)
            return list(cursor)
        except PyMongoError:
            return []

    def get_template(self, template_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            return self._strip_mongo_id(db.templates.find_one({"template_id": int(template_id)}))
        except PyMongoError:
            return None

    def list_templates(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            db = self._connect()
            cursor = db.templates.find({}, {"_id": 0}).sort("last_matched_at", -1).limit(max(1, int(limit)))
            return list(cursor)
        except PyMongoError:
            return []

    def create_template(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            db = self._connect()
            template_id = self._next_sequence("templates")
            now = datetime.datetime.utcnow()
            record = {
                "template_id": template_id,
                "template_name": payload.get("template_name", f"template_{template_id}"),
                "tenant_id": payload.get("tenant_id", "default"),
                "document_type": payload.get("document_type", "UNKNOWN"),
                "document_domain": payload.get("document_domain", "general"),
                "fingerprint": payload.get("fingerprint", {}),
                "sample_count": int(payload.get("sample_count", 1) or 1),
                "approval_status": payload.get("approval_status", "learned"),
                "last_matched_at": payload.get("last_matched_at", now),
                "created_at": payload.get("created_at", now),
            }
            db.templates.insert_one(record)
            return record
        except PyMongoError:
            return None

    def touch_template_match(self, template_id: int) -> bool:
        if not self.enabled:
            return False
        try:
            db = self._connect()
            result = db.templates.update_one(
                {"template_id": int(template_id)},
                {
                    "$set": {"last_matched_at": datetime.datetime.utcnow()},
                    "$inc": {"sample_count": 1},
                },
            )
            return result.matched_count > 0
        except PyMongoError:
            return False

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
