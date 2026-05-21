from __future__ import annotations

from shared.open_library import search_open_library


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "docs": [
                {
                    "key": "/works/OL123W",
                    "title": "A Wizard of Earthsea",
                    "author_name": ["Ursula K. Le Guin"],
                    "first_publish_year": 1968,
                    "subject": ["Fantasy", "Dragons", "Accessible book"],
                    "isbn": ["9780547773742"],
                    "cover_i": 12345,
                    "first_sentence": ["The island of Gont is a single mountain."],
                }
            ]
        }


def test_search_open_library_maps_docs(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("shared.open_library.requests.get", fake_get)

    result = search_open_library("earthsea", limit=10)

    assert calls[0]["url"].endswith("/search.json")
    assert calls[0]["params"]["q"] == "earthsea"
    assert result[0].title == "A Wizard of Earthsea"
    assert result[0].author == "Ursula K. Le Guin"
    assert result[0].genres == ["fantasy", "dragons"]
    assert result[0].cover_url == "https://covers.openlibrary.org/b/id/12345-M.jpg"
