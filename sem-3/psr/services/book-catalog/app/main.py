from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shared.events import publish
from shared.open_library import OpenLibraryBook, find_book, search_open_library
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
    source: str = "manual"
    openlibrary_key: str | None = None
    cover_url: str | None = None


class Book(BookCreate):
    id: str
    created_at: str


class SeedRequest(BaseModel):
    queries: list[str] = Field(default_factory=lambda: ["science fiction", "fantasy", "mystery"])
    limit_per_query: int = Field(default=8, ge=1, le=20)


class SeedResponse(BaseModel):
    imported: int
    existing: int
    total_catalog_size: int


def _from_open_library(book: OpenLibraryBook) -> BookCreate:
    return BookCreate(
        title=book.title,
        author=book.author,
        isbn=book.isbn,
        description=book.description,
        genres=book.genres,
        published_year=book.published_year,
        source="openlibrary",
        openlibrary_key=book.openlibrary_key,
        cover_url=book.cover_url,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_key(book: BookCreate) -> str:
    if book.openlibrary_key:
        return f"openlibrary:{normalize(book.openlibrary_key)}"
    if book.isbn:
        return f"isbn:{normalize(book.isbn)}"
    return f"title:{normalize(book.title)}|author:{normalize(book.author)}"


def _merge_with_open_library(payload: BookCreate) -> BookCreate:
    has_enough_metadata = payload.description and payload.genres and payload.author != "Unknown"
    if payload.source == "openlibrary" or has_enough_metadata:
        return payload
    try:
        match = find_book(payload.title, payload.author if payload.author != "Unknown" else None)
    except Exception:
        return payload
    if not match:
        return payload
    enriched = _from_open_library(match)
    return BookCreate(
        title=payload.title or enriched.title,
        author=payload.author if payload.author != "Unknown" else enriched.author,
        isbn=payload.isbn or enriched.isbn,
        description=payload.description or enriched.description,
        genres=payload.genres or enriched.genres,
        published_year=payload.published_year or enriched.published_year,
        source="openlibrary+manual",
        openlibrary_key=enriched.openlibrary_key,
        cover_url=enriched.cover_url,
    )


def _upsert_book(payload: BookCreate) -> dict:
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
    if result["created"]:
        publish("books", "BookCreated", {"book_id": result["book"]["id"]})
    return result


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


@app.get("/external/openlibrary/search", response_model=list[BookCreate])
def external_search(query: str = Query(min_length=1), limit: int = Query(default=10, ge=1, le=20)) -> list[BookCreate]:
    try:
        return [_from_open_library(book) for book in search_open_library(query, limit=limit)]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Open Library unavailable: {exc}") from exc


@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: str) -> Book:
    book = read_state()["books"].get(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return Book(**book)


@app.post("/books", response_model=Book)
def create_book(payload: BookCreate) -> Book:
    enriched = _merge_with_open_library(payload)
    return Book(**_upsert_book(enriched)["book"])


@app.post("/catalog/seed/openlibrary", response_model=SeedResponse)
def seed_open_library(payload: SeedRequest) -> SeedResponse:
    imported = 0
    existing = 0
    for query in payload.queries:
        try:
            candidates = search_open_library(query, limit=payload.limit_per_query)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Open Library unavailable for {query}: {exc}") from exc
        for candidate in candidates:
            result = _upsert_book(_from_open_library(candidate))
            if result["created"]:
                imported += 1
            else:
                existing += 1

    total = len(read_state()["books"])
    return SeedResponse(imported=imported, existing=existing, total_catalog_size=total)
