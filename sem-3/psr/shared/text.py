from __future__ import annotations

import hashlib
import math
import re

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def book_identity_keys(book: dict) -> set[str]:
    keys = set()
    isbn = normalize(str(book.get("isbn") or ""))
    if isbn:
        keys.add(f"isbn:{isbn}")
    openlibrary_key = normalize(str(book.get("openlibrary_key") or ""))
    if openlibrary_key:
        keys.add(f"openlibrary:{openlibrary_key}")
    title = normalize(str(book.get("title") or ""))
    author = normalize(str(book.get("author") or ""))
    if title and author:
        keys.add(f"title-author:{title}|{author}")
    elif title:
        keys.add(f"title:{title}")
    return keys


def is_same_book(left: dict, right: dict) -> bool:
    return bool(book_identity_keys(left).intersection(book_identity_keys(right)))


def book_text(book: dict) -> str:
    parts = [
        book.get("title", ""),
        book.get("author", ""),
        " ".join(book.get("genres", []) or []),
        book.get("description", ""),
    ]
    return " ".join(part for part in parts if part)


def deterministic_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def average(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    result = [0.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            result[index] += value
    return [value / len(vectors) for value in result]
