"""
build.py - Project a corpus into 3D, compute centroids, write envelopes.

Two-tier dimensional reduction:

  Global map  : PCA (linear, deterministic, stable across rebuilds)
                Used for the corpus-wide overview and for cluster centroids.
                Adding new points later does not shift the existing layout.
  Local maps  : UMAP per category (nonlinear, manifold-preserving)
                Used for within-cluster geometry where the fine structure
                matters for retrieval.

Run after ingest.py. Steps:

  1. Pull all (id, embedding, category) rows for the corpus from Postgres.
  2. Fit a global PCA on the 1536D embeddings and write the 3D projection
     to points.geom.
  3. For each category, fit a local UMAP on the cluster's embeddings and
     write local_x, local_y, local_z.
  4. Compute per-category centroids in 1536D (for cheap cluster routing)
     and in 3D (for spatial UI). Cache r60 = 60th percentile distance
     from 3D centroid.

Cost: zero (no API calls). Runtime is dominated by local UMAP fits;
expect ~30s for 10k points, ~5min for 100k.
"""

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import psycopg2
import umap
from sklearn.decomposition import PCA

UMAP_KWARGS = dict(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)


def db_connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "pointcloud"),
        password=os.environ.get("PGPASSWORD", ""),
        dbname=os.environ.get("PGDATABASE", "pointcloud"),
    )


def fetch_corpus(cur, slug: str):
    cur.execute(
        "SELECT id, category, embedding FROM points WHERE corpus_slug = %s",
        (slug,),
    )
    ids: list[int] = []
    cats: list[str] = []
    vecs: list[np.ndarray] = []
    for pid, cat, vec in cur.fetchall():
        ids.append(pid)
        cats.append(cat or "uncategorised")
        # pgvector returns a string like '[0.1,0.2,...]'; psycopg2 + the
        # pgvector extension's adapter handles parsing. If not adapted,
        # parse defensively here.
        if isinstance(vec, str):
            vec = np.fromstring(vec.strip("[]"), sep=",", dtype=np.float32)
        else:
            vec = np.asarray(vec, dtype=np.float32)
        vecs.append(vec)
    return ids, cats, np.vstack(vecs) if vecs else np.zeros((0, 1536))


def write_global_geom(cur, ids: list[int], coords: np.ndarray) -> None:
    for pid, (x, y, z) in zip(ids, coords):
        cur.execute(
            "UPDATE points SET geom = ST_MakePoint(%s, %s, %s)::geometry "
            "WHERE id = %s",
            (float(x), float(y), float(z), pid),
        )


def write_local_geom(cur, ids: list[int], coords: np.ndarray) -> None:
    for pid, (x, y, z) in zip(ids, coords):
        cur.execute(
            "UPDATE points SET local_x = %s, local_y = %s, local_z = %s "
            "WHERE id = %s",
            (float(x), float(y), float(z), pid),
        )


def upsert_centroids(cur, slug: str, cats: Iterable[str],
                     coords3d: np.ndarray, vecs1536: np.ndarray) -> None:
    cur.execute("DELETE FROM centroids WHERE corpus_slug = %s", (slug,))
    cats_arr = np.array(list(cats))
    for cat in sorted(set(cats_arr)):
        mask = cats_arr == cat
        pts = coords3d[mask]
        vec = vecs1536[mask].mean(axis=0)
        cx, cy, cz = pts.mean(axis=0).tolist()
        dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        r60 = float(np.percentile(dists, 60)) if len(dists) else 0.0
        cur.execute(
            "INSERT INTO centroids "
            "(corpus_slug, category, n, cx, cy, cz, r60, centroid_vec) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (slug, cat, int(mask.sum()), cx, cy, cz, r60, vec.tolist()),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--skip-local", action="store_true",
                    help="Skip per-category local UMAP (faster, less detail)")
    args = ap.parse_args()

    with db_connect() as conn, conn.cursor() as cur:
        ids, cats, vecs = fetch_corpus(cur, args.corpus)
        if not ids:
            print(f"no points found for corpus '{args.corpus}'", file=sys.stderr)
            return 1

        print(f"fitting global PCA on {len(ids)} points...", file=sys.stderr)
        global_coords = PCA(n_components=3, random_state=42).fit_transform(vecs)
        write_global_geom(cur, ids, global_coords)
        conn.commit()

        upsert_centroids(cur, args.corpus, cats, global_coords, vecs)
        conn.commit()

        if not args.skip_local:
            cats_arr = np.array(cats)
            for cat in sorted(set(cats_arr)):
                mask = cats_arr == cat
                cluster_vecs = vecs[mask]
                if len(cluster_vecs) < 10:
                    continue
                print(f"  local UMAP for '{cat}' ({len(cluster_vecs)} pts)",
                      file=sys.stderr)
                local_coords = umap.UMAP(**UMAP_KWARGS).fit_transform(cluster_vecs)
                cluster_ids = [i for i, m in zip(ids, mask) if m]
                write_local_geom(cur, cluster_ids, local_coords)
                conn.commit()

    print(f"done; corpus '{args.corpus}' projected and centroids cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
