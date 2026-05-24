"""
server.py - FastAPI service for query-time pointcloud operations.

Endpoints:
  GET  /health
  GET  /api/corpora
  GET  /api/corpus/{slug}/clusters
  POST /api/corpus/{slug}/search          KNN nearest points
  POST /api/corpus/{slug}/check           IN / EDGE / OUT envelope verdict
  POST /api/corpus/{slug}/route           Single best cluster + skill
  POST /api/corpus/{slug}/route-multihop  Top-k unique skills

All POST endpoints take a JSON body: {"text": "..."} and return a JSON
object with the query's projected position plus the requested verdict.

Cluster routing uses 1536D cosine similarity to centroid_vec (not the
3D UMAP coordinates). The 3D coordinates are only used for spatial UI
and for the rho-coherence metric (3D distance / r60) once a cluster
has been locked.
"""

import os
from typing import Optional

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

app = FastAPI(title="Geometric Governance Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = OpenAI()


def db_connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "pointcloud"),
        password=os.environ.get("PGPASSWORD", ""),
        dbname=os.environ.get("PGDATABASE", "pointcloud"),
    )


def embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def parse_vec(v) -> np.ndarray:
    if isinstance(v, str):
        return np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def project_query(cur, slug: str, qvec: np.ndarray, k: int = 15):
    """
    Place the query in 3D by KNN-weighted-mean of its k nearest neighbours
    in 1536D embedding space. This is the cheap inference-time projection;
    we never refit UMAP at query time.
    """
    cur.execute(
        "SELECT ST_X(geom), ST_Y(geom), ST_Z(geom), "
        "       1 - (embedding <=> %s::vector) AS sim "
        "FROM points WHERE corpus_slug = %s "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        (qvec.tolist(), slug, qvec.tolist(), k),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    zs = np.array([r[2] for r in rows])
    w = np.array([max(r[3], 0.0) for r in rows])
    if w.sum() == 0:
        w = np.ones_like(w)
    return (
        float(np.average(xs, weights=w)),
        float(np.average(ys, weights=w)),
        float(np.average(zs, weights=w)),
    )


class TextIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/corpora")
def list_corpora():
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.slug, c.title, COUNT(p.id) "
            "FROM corpora c LEFT JOIN points p ON p.corpus_slug = c.slug "
            "GROUP BY c.slug, c.title ORDER BY c.slug"
        )
        return [
            {"slug": s, "title": t, "n": n}
            for (s, t, n) in cur.fetchall()
        ]


@app.get("/api/corpus/{slug}/clusters")
def get_clusters(slug: str):
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category, n, cx, cy, cz, r60 FROM centroids "
            "WHERE corpus_slug = %s ORDER BY category",
            (slug,),
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(404, f"corpus '{slug}' has no centroids")
        return [
            {"category": cat, "n": n, "cx": cx, "cy": cy, "cz": cz, "r60": r60}
            for (cat, n, cx, cy, cz, r60) in rows
        ]


@app.post("/api/corpus/{slug}/search")
def search(slug: str, body: TextIn, k: int = 5):
    q = embed(body.text)
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, category, label, text, "
            "       ST_X(geom), ST_Y(geom), ST_Z(geom), "
            "       1 - (embedding <=> %s::vector) AS sim "
            "FROM points WHERE corpus_slug = %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (q.tolist(), slug, q.tolist(), k),
        )
        hits = [
            {"id": i, "category": c, "label": lab, "text": txt[:200],
             "x": x, "y": y, "z": z, "sim": float(sim)}
            for (i, c, lab, txt, x, y, z, sim) in cur.fetchall()
        ]
        pos = project_query(cur, slug, q)
    return {"query_xyz": pos, "hits": hits}


@app.post("/api/corpus/{slug}/check")
def check(slug: str, body: TextIn):
    q = embed(body.text)
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category, cx, cy, cz, r60, centroid_vec "
            "FROM centroids WHERE corpus_slug = %s",
            (slug,),
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(404, "no centroids for corpus")

        sims = []
        for cat, cx, cy, cz, r60, cvec in rows:
            cvec = parse_vec(cvec)
            sim = float(np.dot(q, cvec) /
                        (np.linalg.norm(q) * np.linalg.norm(cvec) + 1e-9))
            sims.append((sim, cat, cx, cy, cz, r60))
        sims.sort(reverse=True)
        best_sim, best_cat, cx, cy, cz, r60 = sims[0]

        pos = project_query(cur, slug, q)
        if pos is None:
            raise HTTPException(404, "no points to project against")
        qx, qy, qz = pos
        d3 = float(np.sqrt((qx - cx) ** 2 + (qy - cy) ** 2 + (qz - cz) ** 2))
        rho = d3 / r60 if r60 > 0 else float("inf")

        if rho < 1.0:
            verdict = "in"
        elif rho < 1.5:
            verdict = "edge"
        else:
            verdict = "out"

    return {
        "best_cluster": best_cat,
        "sim_1536d": best_sim,
        "rho": rho,
        "verdict": verdict,
        "query_xyz": [qx, qy, qz],
        "centroid_xyz": [cx, cy, cz],
        "r60": r60,
    }


@app.post("/api/corpus/{slug}/route")
def route(slug: str, body: TextIn):
    """
    Pick the single best (category, label) match. Used by governance
    overlays that want a one-line decision.
    """
    q = embed(body.text)
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category, label, text, "
            "       1 - (embedding <=> %s::vector) AS sim "
            "FROM points WHERE corpus_slug = %s "
            "ORDER BY embedding <=> %s::vector LIMIT 1",
            (q.tolist(), slug, q.tolist()),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "empty corpus")
        cat, label, text, sim = row
    return {
        "category": cat,
        "label": label,
        "snippet": (text or "")[:200],
        "sim": float(sim),
    }


@app.post("/api/corpus/{slug}/route-multihop")
def route_multihop(slug: str, body: TextIn, k: int = 3):
    """
    Top-k unique skills (deduped by label). Used when the caller wants
    a shortlist of candidates rather than a single winner.
    """
    q = embed(body.text)
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category, label, text, "
            "       1 - (embedding <=> %s::vector) AS sim "
            "FROM points WHERE corpus_slug = %s "
            "ORDER BY embedding <=> %s::vector LIMIT 50",
            (q.tolist(), slug, q.tolist()),
        )
        seen: set[str] = set()
        out: list[dict] = []
        for cat, label, text, sim in cur.fetchall():
            key = label or text[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "category": cat,
                "label": label,
                "snippet": (text or "")[:200],
                "sim": float(sim),
            })
            if len(out) >= k:
                break
    return {"candidates": out}


@app.get("/api/corpus/{slug}/local")
def get_local(slug: str, category: str):
    with db_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, local_x, local_y, local_z, label, text "
            "FROM points WHERE corpus_slug = %s AND category = %s "
            "AND local_x IS NOT NULL",
            (slug, category),
        )
        rows = cur.fetchall()
    return [
        {"id": i, "x": x, "y": y, "z": z,
         "label": lab, "text": (txt or "")[:200]}
        for (i, x, y, z, lab, txt) in rows
    ]
