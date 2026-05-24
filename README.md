# Geometric Governance Pipeline

End-to-end ingestion, projection, and query service for spatial agent governance. Text in, semantic point cloud out, plus a fire-and-forget governance log that classifies every agent action against the resulting topology.

This is the production sister of [Geometric-Semantic-Recursion-](https://github.com/mrodger/Geometric-Semantic-Recursion-), which focused on the visualisation primitives. This repo focuses on the pipeline that feeds them and the runtime that queries them.

![Pointcloud viewer with cluster wireframes](assets/search-04-wireframe-spheres.png)

## The Problem

Agents need to know whether the action they are about to take belongs in the conversation they are in. Routing decisions, tool access, and compliance checks all depend on that judgement, and string matching is not enough; "schedule a meeting" and "delete the production database" can both arrive as polite, well-formed English.

Classical computer vision solved the equivalent problem for physical space: embed sensor data, segment it into regions, and reason about proximity and membership. This pipeline applies the same approach to language. Every message, tool call, and shell command is projected into a 3D semantic space, compared to known cluster envelopes, and either accepted, flagged, or rejected by geometry rather than by keyword.

## What It Does

Each text item is embedded with `text-embedding-3-small` (1536D), projected to 3D via UMAP, and stored with PostGIS geometry and pgvector cosine indexing. The runtime exposes five operations:

- search: KNN over 1536D embeddings, with the 3D projection of each hit returned for spatial UI.
- check: classify a query as IN, EDGE, or OUT of the nearest cluster using rho-coherence (3D distance divided by the cluster's 60th-percentile radius).
- route: pick the single best (category, label) match.
- route-multihop: top-k unique labels, deduped, for shortlist-style routing.
- local: switch from the global UMAP frame to a per-cluster local UMAP, equivalent to a world-to-ego coordinate transform.

The `governance_logger.py` module wraps all five into a fire-and-forget interface for agents. It sanitises secrets, calls the routing endpoint, and writes a SQLite row per event. If the routing service is down, the event is still logged with `envelope_state='unknown'`.

## Architecture

```
  text items
      |
      v
  ingest.py              embed with text-embedding-3-small (1536D)
      |                  cache to .npy by SHA1 of text
      v
  Postgres + pgvector    points table: embedding vector(1536),
      |                  geom geometry(POINTZ, 0)
      v
  build.py               global UMAP (n=15, min_dist=0.1, seed=42)
      |                  per-category local UMAP for drill-down
      |                  centroid + r60 cache per cluster
      v
  server.py              FastAPI: /search /check /route
      |                  /route-multihop /local /clusters
      |                  KNN via pgvector cosine, envelope via PostGIS
      v
  governance_logger.py   fire-and-forget: sanitise -> route -> SQLite log
      |
      v
  agent runtime          one async call per user message, tool call,
                         or bash command
```

PostGIS holds the 3D coordinates in SRID 0 (cartesian semantic space, not geographic). pgvector holds the 1536D embedding for true KNN. The two views are kept in sync by `build.py`; cluster centroids are stored in both spaces so that runtime queries pay only one embedding cost.

## Pipeline Stages

### 1. Ingest

```
python ingest.py --corpus skills --input skills.jsonl
```

Input is one JSON object per line:

```
{"external_id": "skill-001", "category": "coding", "label": "rename_function",
 "text": "Rename a Python function across all callers..."}
```

Embeddings are cached on disk; re-runs only pay for new items. Cost is roughly 0.002 USD per 1000 items at current OpenAI pricing.

### 2. Project

```
python build.py --corpus skills
```

Fits a global UMAP, writes 3D coordinates to `points.geom`, fits a per-category local UMAP, and rebuilds the `centroids` table (mean position in 3D, mean embedding in 1536D, 60th-percentile cluster radius). Runtime is dominated by UMAP; expect ~30 seconds for 10k points.

### 3. Serve

```
uvicorn server:app --host 0.0.0.0 --port 8300
```

The endpoints below assume `POINTCLOUD_URL=http://localhost:8300`.

```
GET  /api/corpora
GET  /api/corpus/{slug}/clusters
POST /api/corpus/{slug}/search          {"text": "..."}
POST /api/corpus/{slug}/check           {"text": "..."}
POST /api/corpus/{slug}/route           {"text": "..."}
POST /api/corpus/{slug}/route-multihop  {"text": "..."}
GET  /api/corpus/{slug}/local?category=...
```

### 4. Log

```
from governance_logger import log_user_message, log_tool_call, fire_and_forget

fire_and_forget(log_user_message(text, session_id=sid))
fire_and_forget(log_tool_call("Bash", {"command": cmd}, session_id=sid))
```

Three SQLite tables (`user_messages`, `tool_calls`, `bash_calls`) each carry the original text, the routing verdict, and the projected x/y/z. Query the log with normal SQL or join it to the points table for spatial analytics.

## Envelope Verdicts

The check endpoint returns one of three verdicts based on rho, defined as the query's 3D distance to the nearest cluster centroid divided by that cluster's `r60`:

- IN: rho < 1.0; the query sits inside the cluster's working radius.
- EDGE: 1.0 <= rho < 1.5; the query is on the boundary.
- OUT: rho >= 1.5; the query is outside the cluster.

A topical query against a corpus of agent skills lands IN:

![IN verdict](assets/check-02-verdict-in.png)

An out-of-distribution query lands OUT:

![OUT verdict](assets/check-03-verdict-out.png)

An adjacent-but-not-quite query lands on the EDGE:

![EDGE verdict](assets/check-04-verdict-edge.png)

The same logic drives the envelope demo, where a sequence of messages is plotted live against the cluster hulls:

![Envelope verdicts across a session](assets/envelope-07-canvas-out-hull.png)

## Search and Routing

A query is embedded once and reused for KNN, cluster ranking, and projection:

![Search results with KNN hits](assets/search-02-results-mamba.png)

The route-multihop endpoint returns deduped top-k matches, useful for a "did you mean one of these tools" shortlist:

![Route-multihop result](assets/route-04-route-tools.png)

## Pipeline Build

The full ingest plus project plus serve flow is also exposed via an upload UI; a fresh corpus appears in the viewer once `build.py` finishes:

![Pipeline complete](assets/06-pipeline-complete.png)

## Database Schema

The pipeline assumes PostgreSQL 15+ with the PostGIS and pgvector extensions. The full DDL is in `schema.sql`:

```
corpora     (slug, title, embed_model, dim)
points      (id, corpus_slug, category, label, text,
             embedding vector(1536),
             geom geometry(POINTZ, 0),
             local_x, local_y, local_z)
centroids   (corpus_slug, category, n, cx, cy, cz, r60,
             centroid_vec vector(1536))
envelopes   (corpus_slug, category, hull geometry(POLYGONZ, 0))
```

Indexes: GIST on `points.geom`, ivfflat (cosine) on `points.embedding`. Re-run `VACUUM ANALYZE points` after bulk loads so the ivfflat index keeps its row estimates honest.

## Governance Logger Sanitisation

`governance_logger.py` strips the following from every event before it touches the network or disk:

- OpenAI keys (`sk-...`) and Anthropic keys (`sk-ant-...`)
- OAuth bearer tokens (`ya29....`)
- JWTs (three-segment base64)
- AWS access keys (`AKIA...`)
- GitHub PATs (`ghp_...`)
- URL-embedded credentials (`scheme://user:pass@host`)
- Absolute home-directory paths (collapsed to `~`)

If you add a new secret format, extend `_SECRET_PATTERNS` in one place.

## Quick Start

```
git clone https://github.com/mrodger/Geometric-governance-pipeline.git
cd Geometric-governance-pipeline

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Postgres setup (assumes psql in PATH)
createdb pointcloud
psql pointcloud -f schema.sql

cp .env.example .env
# edit .env with your OPENAI_API_KEY and Postgres credentials
set -a; source .env; set +a

python ingest.py --corpus skills --input examples/skills.jsonl
python build.py  --corpus skills
uvicorn server:app --host 0.0.0.0 --port 8300

curl -s -X POST http://localhost:8300/api/corpus/skills/check \
  -H 'Content-Type: application/json' \
  -d '{"text":"how do I rename a function across the codebase"}'
```

## What This Is Not

- Not a recommender. The geometry is a navigable semantic space, not a ranked list with implicit personalisation.
- Not a classifier. The routing endpoint returns the nearest known region; if nothing is close, the verdict is OUT, not a guess.
- Not a chat product. There is no UI, no auth, and no opinions about how you wire the verdict into your agent.

## Built With

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings): `text-embedding-3-small`
- [UMAP](https://umap-learn.readthedocs.io/): nonlinear projection from 1536D to 3D
- [PostgreSQL](https://www.postgresql.org/) with [PostGIS](https://postgis.net/) and [pgvector](https://github.com/pgvector/pgvector)
- [FastAPI](https://fastapi.tiangolo.com/): the runtime layer

## Licence

Apache 2.0. See [LICENSE](LICENSE).
