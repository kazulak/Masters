from __future__ import annotations

import os

import requests
import streamlit as st

USER_PROFILE_URL = os.getenv("USER_PROFILE_URL", "http://127.0.0.1:8001")
RECOMMENDATION_URL = os.getenv("RECOMMENDATION_URL", "http://127.0.0.1:8004")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "demo-user")

HEADERS = {"X-User-Id": DEFAULT_USER_ID}

st.set_page_config(page_title="Book AI Library", page_icon="book", layout="wide")
st.title("Book AI Library")

tab_add, tab_list, tab_recs = st.tabs(["Add book", "Reading list", "Recommendations"])

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
                st.subheader(book["title"])
                st.write(f"by {book.get('author', 'Unknown')}")
                st.caption(payload["explanations"].get(book["id"], "Recommended from your reading profile."))
