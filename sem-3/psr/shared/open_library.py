from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from shared.config import OPEN_LIBRARY_BASE_URL, OPEN_LIBRARY_TIMEOUT_SECONDS

SEARCH_FIELDS = ",".join(
    [
        "key",
        "title",
        "author_name",
        "first_publish_year",
        "subject",
        "isbn",
        "cover_i",
        "first_sentence",
    ]
)


@dataclass(frozen=True)
class OpenLibraryBook:
    title: str
    author: str
    isbn: str | None
    description: str
    genres: list[str]
    published_year: int | None
    openlibrary_key: str | None
    cover_url: str | None


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _description(doc: dict[str, Any]) -> str:
    sentence = _first(doc.get("first_sentence"))
    if isinstance(sentence, str) and sentence.strip():
        return sentence.strip()
    subjects = [item for item in doc.get("subject", [])[:8] if isinstance(item, str)]
    if subjects:
        return "Subjects: " + ", ".join(subjects)
    return ""


def _genres(doc: dict[str, Any]) -> list[str]:
    genres = []
    for subject in doc.get("subject", []):
        if not isinstance(subject, str):
            continue
        normalized = subject.strip().lower()
        if not normalized or len(normalized) > 48:
            continue
        if any(blocked in normalized for blocked in ("accessible book", "protected daisy", "in library")):
            continue
        if normalized not in genres:
            genres.append(normalized)
        if len(genres) >= 6:
            break
    return genres


def _cover_url(doc: dict[str, Any]) -> str | None:
    cover_id = doc.get("cover_i")
    if not cover_id:
        return None
    return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"


def _map_doc(doc: dict[str, Any]) -> OpenLibraryBook | None:
    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    author = _first(doc.get("author_name")) or "Unknown"
    isbn = _first(doc.get("isbn"))
    return OpenLibraryBook(
        title=title.strip(),
        author=str(author).strip() or "Unknown",
        isbn=str(isbn).strip() if isbn else None,
        description=_description(doc),
        genres=_genres(doc),
        published_year=doc.get("first_publish_year"),
        openlibrary_key=doc.get("key"),
        cover_url=_cover_url(doc),
    )


def search_open_library(query: str, limit: int = 10) -> list[OpenLibraryBook]:
    safe_limit = max(1, min(limit, 40))
    response = requests.get(
        f"{OPEN_LIBRARY_BASE_URL}/search.json",
        params={
            "q": query,
            "limit": safe_limit,
            "fields": SEARCH_FIELDS,
            "lang": "en",
        },
        timeout=OPEN_LIBRARY_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    books = []
    for doc in payload.get("docs", []):
        mapped = _map_doc(doc)
        if mapped:
            books.append(mapped)
    return books


def find_book(title: str, author: str | None = None) -> OpenLibraryBook | None:
    query = f"{title} {author or ''}".strip()
    books = search_open_library(query, limit=5)
    return books[0] if books else None
