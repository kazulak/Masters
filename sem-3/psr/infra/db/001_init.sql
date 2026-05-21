CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id text PRIMARY KEY,
    email text UNIQUE NOT NULL,
    display_name text NOT NULL,
    preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS books (
    id text PRIMARY KEY,
    isbn text,
    title text NOT NULL,
    author text NOT NULL,
    description text NOT NULL DEFAULT '',
    genres text[] NOT NULL DEFAULT '{}',
    published_year integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL DEFAULT 'manual',
    openlibrary_key text,
    cover_url text,
    dedupe_key text UNIQUE
);

CREATE TABLE IF NOT EXISTS reading_list (
    id text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    book_id text NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    read_at timestamptz NOT NULL DEFAULT now(),
    rating smallint CHECK (rating BETWEEN 1 AND 5),
    UNIQUE (user_id, book_id)
);

CREATE TABLE IF NOT EXISTS book_embeddings (
    id text PRIMARY KEY,
    book_id text NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    embedding vector NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (book_id, model_version)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type text NOT NULL CHECK (type IN ('similar', 'widen', 'mood')),
    book_ids text[] NOT NULL DEFAULT '{}',
    explanations jsonb NOT NULL DEFAULT '{}'::jsonb,
    computed_at timestamptz,
    UNIQUE (user_id, type)
);

CREATE TABLE IF NOT EXISTS events (
    id text PRIMARY KEY,
    topic text NOT NULL,
    type text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS event_deliveries (
    event_id text NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    subscriber text NOT NULL,
    delivered_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, subscriber)
);

CREATE INDEX IF NOT EXISTS idx_events_topic_type ON events(topic, type, created_at);
CREATE INDEX IF NOT EXISTS idx_reading_list_user ON reading_list(user_id);
CREATE INDEX IF NOT EXISTS idx_book_embeddings_book ON book_embeddings(book_id);
