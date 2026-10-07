# GraphRAG (v1 – document GraphRAG)

Hierarchical GraphRAG: LLM extraction → Neo4j knowledge graph → hierarchical Leiden communities (GDS) → community-summary drill-down retrieval with a fact-checked, cited answer — and a UI that replays which nodes the model picked.

Runs fully in Docker. LLM and embeddings are switchable in `.env`: **local** (llama.cpp on an NVIDIA GPU) or **Gemini** (cloud).

## Run
1. Docker Desktop running.
2. Double-click `start.bat`. On the first run it copies `.env.example` to `.env` and opens it: set `NEO4J_PASSWORD` (≥ 8 chars), choose `LLM_PROVIDER` / `EMBED_PROVIDER`, and add `GEMINI_API_KEY` only if a provider is `gemini`. Save, then run `start.bat` again.
3. The browser opens at http://localhost:8000. Neo4j Browser is at http://localhost:7474.

Stop: `stop.bat` (keeps the data). Clear the graph: UI → **Reset graph**. Wipe everything including the local-model volume: `docker compose --profile local down -v`.

Note: Neo4j stores the password on its **first successful start**. To change `NEO4J_PASSWORD` later, run `docker compose down -v` (this wipes the graph), then `start.bat`.

## Layout
```
docker-compose.yml   neo4j (+GDS), backend (FastAPI), and — profile "local" — models-init, embed, llm (llama.cpp)
backend/app/         Python pipeline + API
frontend/            static UI served by the backend (Cytoscape.js + fcose vendored in frontend/vendor, MIT)
.env                 secrets + provider choice (never committed, never baked into images)
models/              optional: GGUF files here are copied once into the Docker volume `graphrag_models` (git-ignored)
```
Everything runs inside containers, so nothing is installed on the host.

## Build status
- [x] Step 1 – infrastructure, health check, start/stop CLI
- [x] Step 2 – ingestion pipeline
- [x] Step 3 – retrieval pipeline
- [x] Step 4 – UI (upload, graph, Q&A, node highlighting)

## Ingestion pipeline (Step 2)
| Step | Where | What |
|---|---|---|
| 1-3 | `ingest/text.py` | PDF/DOCX/TXT/MD → text → recursive chunks (`CHUNK_SIZE_CHARS`) |
| 4 | `ingest/extraction.py` | LLM structured output per chunk → entities + relationships (stored on the Chunk, so failed jobs resume) |
| 5 | `ingest/graph.py` | exact-name merge, then ANN (Neo4j HNSW vector index) candidates → LLM confirms → merge (old name kept as `:Alias`) |
| 6 | `ingest/graph.py` | `(:Entity)-[:RELATED {type, weight}]->(:Entity)`, `(:Chunk)-[:MENTIONS]->(:Entity)` |
| 7 | `ingest/communities.py` | GDS hierarchical Leiden (`includeIntermediateCommunities`); levels that don't split anything are collapsed |
| 8 | `ingest/communities.py` | bottom-up summaries + embeddings; `(:Entity)-[:IN_COMMUNITY]->(:Community)-[:CHILD_OF]->(:Community)`. Cached by content hash, so re-runs only pay for changed communities |

API (try it at http://localhost:8000/docs): `POST /api/query`, `POST /api/documents`, `GET /api/jobs/{id}`, `GET /api/documents`, `GET /api/stats`, `POST /api/communities/rebuild`, `DELETE /api/graph`.

## Retrieval pipeline (Step 3) — `retrieval/engine.py`
1. Embed the question twice (RETRIEVAL_QUERY for summaries, SEMANTIC_SIMILARITY for entities).
2–3. Score top-level communities; keep those ≥ `RETRIEVAL_THRESHOLD` (fallback: best one + near ties).
4. BFS drill-down: accepted community → summary into context → enqueue children (always descend into the best child).
   Leaf community → top entities by similarity, their relationships, and the source chunks that mention them.
5. Dedupe, one LLM **rerank** of summaries + source excerpts, trim to `CONTEXT_MAX_CHARS`.
6. Draft answer with `[C#][E#][R#][S#]` citations → LLM **fact-check** per claim → corrected final answer.

The response includes `trace` (every community scored/visited/accepted, leaf entities, rerank, cited nodes) — the UI in Step 4 replays it to colour the nodes.

## UI (Step 4) — http://localhost:8000
- **Left:** upload (drag & drop), live ingestion progress, document list, colour legend, reset.
- **Centre:** graph — entities (colour = type) + community hubs (bigger = higher level). Click a node for details; toggle communities; Fit.
- **Right:** ask a question (Ctrl+Enter). The traversal is replayed on the graph: top-level scores → visited (amber) → accepted (green) / rejected (grey) → leaf entities picked (cyan) → cited nodes (pink glow). Click a `[S1]`-style citation to see its source text and jump to its nodes. The *Traversal* list shows every score (use it to tune `RETRIEVAL_THRESHOLD`).

## Local models (llama.cpp) — `LLM_PROVIDER=local`, `EMBED_PROVIDER=local`
| Role | Model | File | Why |
|---|---|---|---|
| LLM (default) | Qwen3.5-9B | `Qwen3.5-9B-UD-Q4_K_XL.gguf` (6 GB) | Fits fully in 16 GB VRAM — fast and validated end-to-end |
| LLM (optional) | Qwen3.6-35B-A3B (MoE, ~3B active) | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GB) | Higher quality; `--fit` splits it between 16 GB VRAM and system RAM, so it is noticeably slower |
| Embeddings | Qwen3-Embedding-0.6B | `Qwen3-Embedding-0.6B-Q8_0.gguf` (0.6 GB) | Strong small embedder, instruction-aware (queries get an instruction), 1024 dims |

- Models live in the Docker volume `graphrag_models` (WSL's Linux disk), so llama.cpp can memory-map them fast. Loading them through the Windows mount (`D:\`) failed with `read error: Cannot allocate memory`.
- `start.bat` fills that volume once, copying from `.\models` if the file is there or otherwise downloading it (resumable), then starts `embed` → `llm` → backend. After that the `.\models` copy can be deleted.
- Structured output uses llama.cpp JSON-schema constrained decoding (the Pydantic schemas in `prompts.py`); thinking is off for these tasks.
- llama.cpp's own chat UI: http://localhost:8081 (useful to sanity-check the model).
- Switching provider for embeddings changes the vector space: the graph remembers which embedding model built it and refuses to mix — reset the graph after switching.
- Provider-specific defaults (measured): duplicate-candidate similarity 0.85 (gemini) / 0.70 (local); retrieval threshold 0.55 / 0.50.

**Windows prerequisites (once):** NVIDIA driver up to date; Docker Desktop with WSL2 backend; give WSL enough RAM for the CPU-side experts — `%UserProfile%\.wslconfig`:
```
[wsl2]
memory=24GB
```
then `wsl --shutdown` and restart Docker Desktop.

**Tip:** close other GPU apps (e.g. LM Studio) before starting — llama.cpp sizes itself to the free VRAM at load time.

## Security
- Secrets live only in `.env`, which is git-ignored and excluded from Docker build contexts (`.dockerignore`); the backend reads it at runtime via `env_file`.
- All ports (8000, 7474, 7687, 8081) are bound to `127.0.0.1` only.

## License
MIT — see `LICENSE`. Vendored front-end libraries (Cytoscape.js, fcose, cose-base, layout-base) are MIT; their licences are in `frontend/vendor/licenses/`.
