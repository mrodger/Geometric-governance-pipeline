-- Geometric Governance Pipeline schema
-- Requires: PostgreSQL 15+, PostGIS 3.3+, pgvector 0.5+

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- A corpus is one logical dataset (skills, chat history, support tickets, etc.)
CREATE TABLE IF NOT EXISTS corpora (
    slug          TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    embed_model   TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    dim           INT  NOT NULL DEFAULT 1536,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One point per text item. Geometry is the 3D UMAP projection in SRID 0
-- (cartesian semantic space, not geographic). The 1536D embedding is kept
-- in pgvector for true KNN; the geometry is for spatial UI and PostGIS
-- envelope tests.
CREATE TABLE IF NOT EXISTS points (
    id            BIGSERIAL PRIMARY KEY,
    corpus_slug   TEXT NOT NULL REFERENCES corpora(slug) ON DELETE CASCADE,
    external_id   TEXT,
    category      TEXT,
    label         TEXT,
    text          TEXT NOT NULL,
    embedding     vector(1536) NOT NULL,
    geom          GEOMETRY(POINTZ, 0) NOT NULL,
    local_x       DOUBLE PRECISION,
    local_y       DOUBLE PRECISION,
    local_z       DOUBLE PRECISION,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS points_corpus_idx       ON points (corpus_slug);
CREATE INDEX IF NOT EXISTS points_category_idx     ON points (corpus_slug, category);
CREATE INDEX IF NOT EXISTS points_geom_idx         ON points USING GIST (geom);

-- ivfflat for approximate KNN. Tune lists to ~sqrt(N) after bulk load,
-- then run ANALYZE.
CREATE INDEX IF NOT EXISTS points_embedding_idx
    ON points USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Per-category centroid cache. Populated by build.py after UMAP.
-- centroid_vec is the mean of all 1536D embeddings in the category;
-- used for cheap "which cluster does this query belong to" lookups.
CREATE TABLE IF NOT EXISTS centroids (
    corpus_slug    TEXT NOT NULL REFERENCES corpora(slug) ON DELETE CASCADE,
    category       TEXT NOT NULL,
    n              INT  NOT NULL,
    cx             DOUBLE PRECISION NOT NULL,
    cy             DOUBLE PRECISION NOT NULL,
    cz             DOUBLE PRECISION NOT NULL,
    r60            DOUBLE PRECISION NOT NULL,
    centroid_vec   vector(1536) NOT NULL,
    PRIMARY KEY (corpus_slug, category)
);

-- Optional: convex envelope per category for fast PostGIS containment tests.
CREATE TABLE IF NOT EXISTS envelopes (
    corpus_slug    TEXT NOT NULL REFERENCES corpora(slug) ON DELETE CASCADE,
    category       TEXT NOT NULL,
    hull           GEOMETRY(POLYGONZ, 0),
    PRIMARY KEY (corpus_slug, category)
);
