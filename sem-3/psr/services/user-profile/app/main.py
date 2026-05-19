from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from shared.config import BOOK_CATALOG_URL, DEFAULT_USER_ID
from shared.events import publish
from shared.storage import read_state, update_state

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
    rating: int | None = Field(default=None, ge=1, le=5)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_user_id(x_user_id: str | None) -> str:
    return x_user_id or DEFAULT_USER_ID


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "user-profile"}


@app.get("/me", response_model=UserProfile)
def get_me(x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)

    def mutate(state: dict) -> dict:
        user = state["users"].setdefault(
            user_id,
            UserProfile(id=user_id).model_dump() | {"created_at": _now()},
        )
        return user

    return UserProfile(**update_state(mutate))


@app.post("/me", response_model=UserProfile)
def upsert_me(profile: UserProfile, x_user_id: str | None = Header(default=None)) -> UserProfile:
    user_id = _current_user_id(x_user_id)
    profile.id = user_id

    def mutate(state: dict) -> dict:
        row = profile.model_dump() | {"created_at": state["users"].get(user_id, {}).get("created_at", _now())}
        state["users"][user_id] = row
        return row

    return UserProfile(**update_state(mutate))


@app.get("/me/books")
def list_my_books(x_user_id: str | None = Header(default=None)) -> dict:
    user_id = _current_user_id(x_user_id)
    state = read_state()
    rows = [row for row in state["reading_list"].values() if row["user_id"] == user_id]
    books = [state["books"][row["book_id"]] | {"rating": row.get("rating")} for row in rows if row["book_id"] in state["books"]]
    return {"user_id": user_id, "books": books}


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

    def mutate(state: dict) -> dict:
        state["users"].setdefault(user_id, UserProfile(id=user_id).model_dump() | {"created_at": _now()})
        for row in state["reading_list"].values():
            if row["user_id"] == user_id and row["book_id"] == book["id"]:
                row["rating"] = payload.rating
                return row
        row_id = str(uuid4())
        row = {
            "id": row_id,
            "user_id": user_id,
            "book_id": book["id"],
            "read_at": _now(),
            "rating": payload.rating,
        }
        state["reading_list"][row_id] = row
        return row

    row = update_state(mutate)
    publish("users", "UserBookAdded", {"user_id": user_id, "book_id": book["id"]})
    return {"user_id": user_id, "book": book, "reading_list_entry": row}
