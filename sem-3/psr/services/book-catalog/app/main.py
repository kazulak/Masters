from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from shared.demo_catalog import DEMO_BOOKS
from shared.events import publish
from shared.open_library import OpenLibraryBook, find_book, search_open_library
from shared.repositories import count_books, get_book as repo_get_book, list_books, upsert_book
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
    source: str = "openlibrary"


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
    result = upsert_book(payload.model_dump(), _dedupe_key(payload))
    if result["created"]:
        publish("books", "BookCreated", {"book_id": result["book"]["id"]})
    return result


def _etag(payload: dict) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f'"{hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]}"'


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "book-catalog"}


@app.get("/books", response_model=list[Book])
def search_books(query: str = Query(default="")) -> list[Book]:
    return [Book(**book) for book in list_books(query)]


@app.get("/external/openlibrary/search", response_model=list[BookCreate])
def external_search(query: str = Query(min_length=1), limit: int = Query(default=10, ge=1, le=20)) -> list[BookCreate]:
    try:
        return [_from_open_library(book) for book in search_open_library(query, limit=limit)]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Open Library unavailable: {exc}") from exc


@app.get("/books/{book_id}", response_model=Book)
def get_book(
    book_id: str,
    response: Response,
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
) -> Book | Response:
    book = repo_get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    etag = _etag(book)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
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

    total = count_books()
    return SeedResponse(imported=imported, existing=existing, total_catalog_size=total, source="openlibrary")


@app.post("/catalog/seed/demo", response_model=SeedResponse)
def seed_demo_catalog() -> SeedResponse:
    imported = 0
    existing = 0
    for book in DEMO_BOOKS:
        result = _upsert_book(BookCreate(**book))
        if result["created"]:
            imported += 1
        else:
            existing += 1

    return SeedResponse(imported=imported, existing=existing, total_catalog_size=count_books(), source="demo")
