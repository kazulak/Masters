from __future__ import annotations

import os

import requests
import streamlit as st

USER_PROFILE_URL = os.getenv("USER_PROFILE_URL", "http://127.0.0.1:8001")
BOOK_CATALOG_URL = os.getenv("BOOK_CATALOG_URL", "http://127.0.0.1:8002")
RECOMMENDATION_URL = os.getenv("RECOMMENDATION_URL", "http://127.0.0.1:8004")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "demo-user")

HEADERS = {"X-User-Id": DEFAULT_USER_ID}

st.set_page_config(page_title="Book AI Library", page_icon="book", layout="wide")
st.title("Book AI Library")

tab_discover, tab_add, tab_list, tab_recs = st.tabs(["Discover", "Add book", "Reading list", "Recommendations"])

with tab_discover:
    query = st.text_input("Open Library search", value="Ursula Le Guin")
    col_search, col_demo_seed, col_open_seed = st.columns([1, 1, 1])
    with col_search:
        search_clicked = st.button("Search Open Library")
    with col_demo_seed:
        demo_seed_clicked = st.button("Seed demo catalog")
    with col_open_seed:
        open_seed_clicked = st.button("Seed Open Library")

    if demo_seed_clicked:
        response = requests.post(f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=15)
        if response.ok:
            result = response.json()
            st.success(
                f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                f"{result['total_catalog_size']} books total."
            )
        else:
            st.error(response.text)

    if open_seed_clicked:
        response = requests.post(
            f"{BOOK_CATALOG_URL}/catalog/seed/openlibrary",
            json={"queries": ["science fiction", "fantasy", "mystery", "historical fiction"], "limit_per_query": 8},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            st.success(
                f"Catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                f"{result['total_catalog_size']} books total."
            )
        else:
            fallback = requests.post(f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=15)
            if fallback.ok:
                result = fallback.json()
                st.warning("Open Library is unavailable or slow. Seeded the local demo catalog instead.")
                st.success(
                    f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                    f"{result['total_catalog_size']} books total."
                )
            else:
                st.error(response.text)

    if search_clicked:
        response = requests.get(
            f"{BOOK_CATALOG_URL}/external/openlibrary/search",
            params={"query": query, "limit": 8},
            timeout=15,
        )
        if not response.ok:
            st.error(response.text)
        else:
            st.session_state["open_library_results"] = response.json()

    for index, book in enumerate(st.session_state.get("open_library_results", [])):
        with st.container(border=True):
            cols = st.columns([1, 4])
            if book.get("cover_url"):
                cols[0].image(book["cover_url"], width=90)
            cols[1].write(f"**{book['title']}** by {book.get('author', 'Unknown')}")
            cols[1].caption(", ".join(book.get("genres", [])[:5]))
            if cols[1].button("Add to reading list", key=f"add-open-library-{index}"):
                response = requests.post(
                    f"{USER_PROFILE_URL}/me/books",
                    json=book | {"rating": 4},
                    headers=HEADERS,
                    timeout=15,
                )
                if response.ok:
                    st.success(f"Added {response.json()['book']['title']}.")
                else:
                    st.error(response.text)

with tab_add:
    with st.form("add-book"):
        title = st.text_input("Title", value="Dune")
        author = st.text_input("Author", value="Frank Herbert")
        genres = st.text_input("Genres", value="science fiction, adventure")
        description = st.text_area(
            "Description",
            value="A desert planet, political intrigue, ecology, prophecy, and power.",
        )
        rating = st.slider("Rating", min_value=1, max_value=5, value=5)
        submitted = st.form_submit_button("Add to reading list")

    if submitted:
        payload = {
            "title": title,
            "author": author,
            "genres": [item.strip() for item in genres.split(",") if item.strip()],
            "description": description,
            "rating": rating,
        }
        response = requests.post(f"{USER_PROFILE_URL}/me/books", json=payload, headers=HEADERS, timeout=15)
        if response.ok:
            st.success(f"Added {response.json()['book']['title']}. Recommendations update asynchronously.")
        else:
            st.error(response.text)

with tab_list:
    response = requests.get(f"{USER_PROFILE_URL}/me/books", headers=HEADERS, timeout=10)
    if not response.ok:
        st.error(response.text)
    else:
        books = response.json()["books"]
        if not books:
            st.info("No books in the reading list yet.")
        for book in books:
            st.write(f"**{book['title']}** by {book.get('author', 'Unknown')} · rating: {book.get('rating') or '-'}")
            st.caption(", ".join(book.get("genres", [])))

with tab_recs:
    rec_type = st.radio("Type", ["similar", "widen", "mood"], horizontal=True)
    response = requests.get(
        f"{RECOMMENDATION_URL}/recommendations",
        params={"user_id": DEFAULT_USER_ID, "type": rec_type},
        timeout=10,
    )
    if not response.ok:
        st.error(response.text)
    else:
        payload = response.json()
        books = payload["books"]
        if not books:
            st.info("No recommendations yet. Add at least one book and let the async workers process it.")
        for book in books:
            with st.container(border=True):
                cols = st.columns([1, 4])
                if book.get("cover_url"):
                    cols[0].image(book["cover_url"], width=90)
                cols[1].subheader(book["title"])
                cols[1].write(f"by {book.get('author', 'Unknown')}")
                cols[1].caption(payload["explanations"].get(book["id"], "Recommended from your reading profile."))
