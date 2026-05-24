"""
ingest.py - Load text items into the pointcloud database.

Usage:
    python ingest.py --corpus skills --input skills.jsonl

Input format (JSONL, one object per line):
    {"external_id": "skill-001", "category": "coding", "label": "rename_function",
     "text": "Rename a Python function across all callers..."}

The script:
  1. Creates the corpus row if it does not exist.
  2. Embeds each text with OpenAI text-embedding-3-small (1536D).
  3. Inserts rows into the points table with a placeholder geometry (0,0,0).
  4. Leaves UMAP projection to build.py (run after ingest finishes).

Embeddings are cached in a local .npy keyed by the SHA1 of the text, so
re-runs only pay for new items. Cost is roughly 0.002 USD per 1000 items.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Iterator

import numpy as np
import psycopg2
from openai import OpenAI

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CACHE_DIR = pathlib.Path(os.environ.get("EMBED_CACHE", "./data/embed_cache"))
BATCH_SIZE = 64


def db_connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "pointcloud"),
        password=os.environ.get("PGPASSWORD", ""),
        dbname=os.environ.get("PGDATABASE", "pointcloud"),
    )


def cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cached_embed(text: str) -> np.ndarray | None:
    p = CACHE_DIR / f"{cache_key(text)}.npy"
    if p.exists():
        return np.load(p)
    return None


def cache_embed(text: str, vec: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / f"{cache_key(text)}.npy", vec)


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed_batch(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    out: list[np.ndarray | None] = [cached_embed(t) for t in texts]
    pending_idx = [i for i, v in enumerate(out) if v is None]
    if pending_idx:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[texts[i] for i in pending_idx],
        )
        for offset, item in enumerate(resp.data):
            i = pending_idx[offset]
            vec = np.asarray(item.embedding, dtype=np.float32)
            out[i] = vec
            cache_embed(texts[i], vec)
    return [v for v in out if v is not None]


def ensure_corpus(cur, slug: str, title: str) -> None:
    cur.execute(
        "INSERT INTO corpora (slug, title) VALUES (%s, %s) "
        "ON CONFLICT (slug) DO NOTHING",
        (slug, title),
    )


def insert_points(cur, slug: str, rows: list[dict], vecs: list[np.ndarray]) -> int:
    written = 0
    for row, vec in zip(rows, vecs):
        cur.execute(
            "INSERT INTO points "
            "(corpus_slug, external_id, category, label, text, embedding, geom) "
            "VALUES (%s, %s, %s, %s, %s, %s, ST_MakePoint(0,0,0)::geometry)",
            (
                slug,
                row.get("external_id"),
                row.get("category"),
                row.get("label"),
                row["text"],
                vec.tolist(),
            ),
        )
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Corpus slug")
    ap.add_argument("--title", default=None, help="Human readable title")
    ap.add_argument("--input", required=True, help="Path to JSONL input")
    args = ap.parse_args()

    title = args.title or args.corpus
    client = OpenAI()

    with db_connect() as conn, conn.cursor() as cur:
        ensure_corpus(cur, args.corpus, title)
        conn.commit()

        batch: list[dict] = []
        total = 0
        for row in iter_jsonl(args.input):
            if "text" not in row or not row["text"]:
                continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                vecs = embed_batch(client, [r["text"] for r in batch])
                total += insert_points(cur, args.corpus, batch, vecs)
                conn.commit()
                print(f"  ingested {total}", file=sys.stderr)
                batch = []

        if batch:
            vecs = embed_batch(client, [r["text"] for r in batch])
            total += insert_points(cur, args.corpus, batch, vecs)
            conn.commit()

    print(f"done; {total} points ingested into corpus '{args.corpus}'")
    print("next: python build.py --corpus", args.corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
