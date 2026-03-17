"""
MongoDB connection module for user authentication.
Uses Motor (async MongoDB driver) to connect to MongoDB Atlas or local MongoDB.
"""
import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection string - can be set via MONGODB_URL env variable
# Falls back to a local MongoDB instance for development
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "docintel")

# Global client and db references
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[object] = None


async def connect_to_mongo():
    """Initialize the MongoDB connection."""
    global _client, _db
    try:
        _client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Verify connection
        await _client.admin.command("ping")
        _db = _client[DB_NAME]
        print(f"✅ MongoDB connected: {DB_NAME}")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}. Auth features will be limited.")
        _client = None
        _db = None


async def close_mongo_connection():
    """Close the MongoDB connection."""
    global _client
    if _client:
        _client.close()
        print("MongoDB connection closed.")


def get_db():
    """Return the MongoDB database instance."""
    return _db


def get_users_collection():
    """Return the users collection."""
    if _db is None:
        return None
    return _db["users"]
