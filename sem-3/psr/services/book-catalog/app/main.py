from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shared.events import publish
from shared.storage import read_state, update_state
from shared.text import normalize

app = FastAPI(title="Book AI Library - Book Catalog Service", version="0.1.0")


class BookCreate(BaseModel):
    title: str = Field(min_length=1)
    author: str = "Unknown"
    isbn: str | None = None
    description: str = ""
    genres: list[str] = Field(default_factory=list)
    published_year: int | None = None


class Book(BookCreate):
    id: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_key(book: BookCreate) -> str:
    if book.isbn:
        return f"isbn:{normalize(book.isbn)}"
    return f"title:{normalize(book.title)}|author:{normalize(book.author)}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "book-catalog"}


@app.get("/books", response_model=list[Book])
def search_books(query: str = Query(default="")) -> list[Book]:
    state = read_state()
    books = list(state["books"].values())
    if not query:
        return books
    needle = normalize(query)
    return [
        book for book in books
        if needle in normalize(book["title"]) or needle in normalize(book.get("author", ""))
    ]


@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: str) -> Book:
    book = read_state()["books"].get(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return Book(**book)


@app.post("/books", response_model=Book)
def create_book(payload: BookCreate) -> Book:
    def mutate(state: dict) -> dict:
        key = _dedupe_key(payload)
        for existing in state["books"].values():
            if existing.get("dedupe_key") == key:
                return {"book": existing, "created": False}

        book_id = str(uuid4())
        book = payload.model_dump()
        book.update({"id": book_id, "created_at": _now(), "dedupe_key": key})
        state["books"][book_id] = book
        return {"book": book, "created": True}

    result = update_state(mutate)
    book = result["book"]
    if result["created"]:
        publish("books", "BookCreated", {"book_id": book["id"]})
    return Book(**book)
