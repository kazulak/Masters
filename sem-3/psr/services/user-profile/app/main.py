from __future__ import annotations

import requests
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from shared.config import BOOK_CATALOG_URL, DEFAULT_USER_ID
from shared.events import publish
from shared.repositories import get_or_create_user, get_user_by_email, list_user_books, upsert_reading_list_entry, upsert_user

app = FastAPI(title="Book AI Library - User Profile Service", version="0.1.0")


class UserProfile(BaseModel):
    id: str = DEFAULT_USER_ID
    email: str = "demo@example.edu"
    display_name: str = "Demo User"
    preferences: dict = Field(default_factory=lambda: {"mood": "curious", "genres": []})


class SignInRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class SignUpRequest(SignInRequest):
    display_name: str = Field(min_length=1)
    mood: str = "curious"
    genres: list[str] = Field(default_factory=list)


class AddBookRequest(BaseModel):
    title: str = Field(min_length=1)
    author: str = "Unknown"
    isbn: str | None = None
    description: str = ""
    genres: list[str] = Field(default_factory=list)
    published_year: int | None = None
    source: str = "manual"
    openlibrary_key: str | None = None
    cover_url: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)


def _current_user_id(x_user_id: str | None) -> str:
    return x_user_id or DEFAULT_USER_ID


def _clean_email(email: str) -> str:
    return email.strip().lower()


def _public_user(row: dict) -> dict:
    preferences = dict(row.get("preferences") or {})
    preferences.pop("_password", None)
    return row | {"preferences": preferences}


def _password_for(row: dict) -> str | None:
    return (row.get("preferences") or {}).get("_password")


def _default_profile(user_id: str) -> dict:
    email = user_id if "@" in user_id else f"{user_id}@example.edu"
    return UserProfile(id=user_id, email=email).model_dump()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "user-profile"}


@app.post("/auth/signup", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
def signup(payload: SignUpRequest) -> UserProfile:
    email = _clean_email(payload.email)
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Account already exists for this email")

    profile = {
        "id": email,
        "email": email,
        "display_name": payload.display_name.strip(),
        "preferences": {
            "mood": payload.mood,
            "genres": payload.genres,
            "_password": payload.password,
        },
    }
    return UserProfile(**_public_user(upsert_user(profile)))


@app.post("/auth/signin", response_model=UserProfile)
def signin(payload: SignInRequest) -> UserProfile:
    user = get_user_by_email(_clean_email(payload.email))
    if not user or _password_for(user) != payload.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return UserProfile(**_public_user(user))


@app.get("/me", response_model=UserProfile)
def get_me(x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)
    return UserProfile(**_public_user(get_or_create_user(user_id, _default_profile(user_id))))


@app.post("/me", response_model=UserProfile)
def upsert_me(profile: UserProfile, x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)
    existing = get_or_create_user(user_id, _default_profile(user_id))
    profile.id = user_id
    preferences = dict(profile.preferences)
    if _password_for(existing):
        preferences["_password"] = _password_for(existing)
    profile.preferences = preferences
    return UserProfile(**_public_user(upsert_user(profile.model_dump())))


@app.get("/me/books")
def list_my_books(x_user_id: str | None = Header(default=None)) -> dict:
    user_id = _current_user_id(x_user_id)
    return {"user_id": user_id, "books": list_user_books(user_id)}


@app.post("/me/books")
def add_my_book(payload: AddBookRequest, x_user_id: str | None = Header(default=None)) -> dict:
    user_id = _current_user_id(x_user_id)
    try:
        response = requests.post(
            f"{BOOK_CATALOG_URL}/books",
            json=payload.model_dump(exclude={"rating"}),
            timeout=10,
        )
        response.raise_for_status()
        book = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Book Catalog unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Book Catalog returned invalid JSON: {response.text[:300]}") from exc

    if not isinstance(book, dict) or not book.get("id"):
        raise HTTPException(status_code=502, detail=f"Book Catalog returned an unexpected book payload: {book}")

    get_or_create_user(user_id, _default_profile(user_id))
    row = upsert_reading_list_entry(user_id, book["id"], payload.rating)
    event = publish("users", "UserBookAdded", {"user_id": user_id, "book_id": book["id"]})
    return {"user_id": user_id, "book": book, "reading_list_entry": row, "event": event}
