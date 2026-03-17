"""
Authentication routes for the DOC-INTEL API.
Provides signup, login, and user profile endpoints using MongoDB.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt

from auth_db import get_users_collection

# ── Security configuration ────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "docintel-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime


# ── Helper functions ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Dependency to extract and validate the JWT token."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    users_col = get_users_collection()
    if users_col is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")

    user = await users_col.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── Auth endpoints ────────────────────────────────────────────────────────────
@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest):
    """Register a new user account."""
    users_col = get_users_collection()
    if users_col is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please check your MongoDB connection.",
        )

    # Check for duplicate email
    existing = await users_col.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Create user document
    user_doc = {
        "name": body.name,
        "email": body.email.lower(),
        "hashed_password": hash_password(body.password),
        "created_at": datetime.now(timezone.utc),
        "is_active": True,
    }
    result = await users_col.insert_one(user_doc)
    user_id = str(result.inserted_id)

    token = create_access_token({"sub": body.email.lower(), "user_id": user_id})
    return TokenResponse(
        access_token=token,
        user={"id": user_id, "name": body.name, "email": body.email.lower()},
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate a user and return a JWT token."""
    users_col = get_users_collection()
    if users_col is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please check your MongoDB connection.",
        )

    user = await users_col.find_one({"email": body.email.lower()})
    if user is None or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    user_id = str(user["_id"])
    token = create_access_token({"sub": body.email.lower(), "user_id": user_id})
    return TokenResponse(
        access_token=token,
        user={"id": user_id, "name": user["name"], "email": user["email"]},
    )


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserProfile(
        id=str(current_user["_id"]),
        name=current_user["name"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )
