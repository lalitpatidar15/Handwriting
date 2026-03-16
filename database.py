from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1 Crore Project Step 1: Secure SQLite DB (Upgradeable to PostgreSQL)
SQLALCHEMY_DATABASE_URL = "sqlite:///./docintel.db"
# For PostgreSQL later: "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_schema_migrations():
    """Apply lightweight additive migrations for the local SQLite database."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    if "documents" not in table_names:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    column_updates = {
        "review_status": "ALTER TABLE documents ADD COLUMN review_status VARCHAR DEFAULT 'pending_review'",
        "schema_version": "ALTER TABLE documents ADD COLUMN schema_version VARCHAR DEFAULT '1.0'",
        "document_domain": "ALTER TABLE documents ADD COLUMN document_domain VARCHAR DEFAULT 'general'",
        "template_id": "ALTER TABLE documents ADD COLUMN template_id INTEGER",
        "template_match_score": "ALTER TABLE documents ADD COLUMN template_match_score FLOAT DEFAULT 0.0",
        "layout_provider": "ALTER TABLE documents ADD COLUMN layout_provider VARCHAR DEFAULT 'local'",
        "reasoning_provider": "ALTER TABLE documents ADD COLUMN reasoning_provider VARCHAR DEFAULT 'local'",
    }

    with engine.begin() as connection:
        for column_name, statement in column_updates.items():
            if column_name not in existing_columns:
                connection.execute(text(statement))

# Dependency for FastAPI to get DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
