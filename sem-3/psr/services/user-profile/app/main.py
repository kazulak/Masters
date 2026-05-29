from __future__ import annotations

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from shared.config import BOOK_CATALOG_URL, DEFAULT_USER_ID
from shared.events import publish
from shared.repositories import get_or_create_user, list_user_books, upsert_reading_list_entry, upsert_user

app = FastAPI(title="Book AI Library - User Profile Service", version="0.1.0")


class UserProfile(BaseModel):
    id: str = DEFAULT_USER_ID
    email: str = "demo@example.edu"
    display_name: str = "Demo User"
    preferences: dict = Field(default_factory=lambda: {"mood": "curious", "genres": []})


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "user-profile"}


@app.get("/me", response_model=UserProfile)
def get_me(x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)
    return UserProfile(**get_or_create_user(user_id, UserProfile(id=user_id).model_dump()))


@app.post("/me", response_model=UserProfile)
def upsert_me(profile: UserProfile, x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)
    profile.id = user_id
    return UserProfile(**upsert_user(profile.model_dump()))


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
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Book Catalog unavailable: {exc}") from exc

    book = response.json()

    get_or_create_user(user_id, UserProfile(id=user_id).model_dump())
    row = upsert_reading_list_entry(user_id, book["id"], payload.rating)
    publish("users", "UserBookAdded", {"user_id": user_id, "book_id": book["id"]})
    return {"user_id": user_id, "book": book, "reading_list_entry": row}
