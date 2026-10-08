import hashlib
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from . import db
from .config import get_settings
from .datasets import catalog
from .datasets.profile import SUPPORTED_TABULAR
from .ingest.pipeline import ingest_document, rebuild_only, register_document
from .ingest.text import SUPPORTED
from .jobs import Job, JobQueue
from .llm import check_models
from .retrieval.engine import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("graphrag")

jobs = JobQueue({"ingest": ingest_document, "rebuild": rebuild_only, "profile": catalog.profile_job})


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("GraphRAG backend starting")
    Path(get_settings().upload_dir).mkdir(parents=True, exist_ok=True)
    catalog.datasets_dir()
    db.init_schema()
    yield
    db.close_driver()


app = FastAPI(title="GraphRAG", version="0.5.0", lifespan=lifespan)


# --------------------------------------------------------------------------- health / stats
@app.get("/api/health")
def health() -> dict:
    """Check every dependency independently so one failure doesn't hide the others."""
    result: dict = {"status": "ok"}
    try:
        result["neo4j"] = {"ok": True, **db.check_neo4j()}
    except Exception as e:  # noqa: BLE001
        result["neo4j"] = {"ok": False, "error": str(e)}
    result.update(check_models())                    # llm + embeddings (gemini or local)
    result["embedding_space"] = db.embedding_status()  # graph vs configured embedding model
    if not all(result[k]["ok"] for k in ("neo4j", "llm", "embeddings", "embedding_space")):
        result["status"] = "degraded"
    return result


@app.get("/api/stats")
def stats() -> dict:
    return db.graph_stats()


# --------------------------------------------------------------------------- uploads
async def _receive(file: UploadFile, supported: set[str], max_mb: int, target_dir: Path) -> tuple[str, str, Path]:
    """Stream an upload to disk (content-addressed). Returns (original name, sha256, final path)."""
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in supported:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {sorted(supported)}")
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp = target_dir / f".incoming-{hashlib.md5(name.encode()).hexdigest()}"
    h = hashlib.sha256()
    size = 0
    with tmp.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {max_mb} MB")
            h.update(chunk)
            out.write(chunk)
    digest = h.hexdigest()
    final = target_dir / f"{digest}{ext}"
    shutil.move(tmp, final)
    return name, digest, final


# --------------------------------------------------------------------------- documents
@app.post("/api/documents", status_code=202)
async def upload(file: UploadFile) -> dict:
    s = get_settings()
    name, digest, final = await _receive(file, SUPPORTED, s.max_upload_mb, Path(s.upload_dir))

    doc, is_new = register_document(name, digest, final)
    if not is_new and doc["status"] == "done":
        return {"message": "Document already ingested", "document": doc, "job": None}
    if not is_new and any(j.doc_id == doc["id"] and j.status in ("queued", "running") for j in jobs.list()):
        raise HTTPException(409, "This document is already being processed")

    job = jobs.submit(Job(kind="ingest", doc_id=doc["id"], filename=name))
    return {"message": "queued" if is_new else "resuming", "document": doc, "job": job.to_dict()}


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return db.run(
        """MATCH (d:Document)
           RETURN d.id AS id, d.name AS name, d.status AS status, d.chunks AS chunks,
                  d.created_at AS created_at, d.finished_at AS finished_at, d.error AS error
           ORDER BY d.created_at DESC"""
    )


# --------------------------------------------------------------------------- jobs
@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.to_dict() for j in jobs.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/communities/rebuild", status_code=202)
def rebuild_communities() -> dict:
    """Re-run Leiden + summaries (e.g. after changing LEIDEN_* settings). Cached summaries are reused."""
    return jobs.submit(Job(kind="rebuild")).to_dict()


# --------------------------------------------------------------------------- graph for the UI
@app.get("/api/graph")
def graph(max_entities: int = 2000) -> dict:
    """Entities (most connected first, capped) + all communities, with RELATED / IN_COMMUNITY / CHILD_OF edges."""
    entities = db.run(
        """MATCH (e:Entity)
           WITH e, COUNT { (e)-[:RELATED]-() } AS degree
           ORDER BY degree DESC LIMIT $n
           OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(c:Community)
           RETURN e.id AS id, e.name AS label, e.type AS type, e.description AS description,
                  degree, c.id AS community""",
        n=max_entities,
    )
    ids = [e["id"] for e in entities]
    communities = db.run(
        """MATCH (c:Community)
           OPTIONAL MATCH (c)-[:CHILD_OF]->(p:Community)
           RETURN c.id AS id, c.title AS label, c.summary AS summary, c.level AS level,
                  c.size AS size, c.is_top AS is_top, p.id AS parent"""
    )
    related = db.run(
        """MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids
           RETURN a.id AS source, b.id AS target, r.type AS type, r.weight AS weight""",
        ids=ids,
    )
    return {"entities": entities, "communities": communities, "related": related,
            "truncated": db.graph_stats()["entities"] > len(entities)}


# --------------------------------------------------------------------------- query
class QueryIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.post("/api/query")
def query(body: QueryIn) -> dict:
    """Hierarchical retrieval + fact-checked answer. `trace` lists every node the model visited/picked."""
    try:
        return answer_question(body.question)
    except Exception as e:  # noqa: BLE001  -> readable error instead of a bare 500
        log.exception("query failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e



# --------------------------------------------------------------------------- datasets (Excel / CSV catalog)
def _dataset_busy(ds_id: str) -> bool:
    return any(j.dataset_id == ds_id and j.status in ("queued", "running") for j in jobs.list(500))


@app.post("/api/datasets", status_code=202)
async def upload_dataset(file: UploadFile) -> dict:
    s = get_settings()
    name, digest, final = await _receive(file, SUPPORTED_TABULAR, s.max_dataset_upload_mb, catalog.datasets_dir())
    ds, is_new = catalog.register_dataset(name, digest, final)
    if not is_new and ds["status"] not in ("failed", "queued"):
        return {"message": "This file is already in the catalog", "dataset": ds, "job": None}
    if _dataset_busy(ds["id"]):
        raise HTTPException(409, "This file is already being processed")
    job = jobs.submit(Job(kind="profile", dataset_id=ds["id"], filename=name))
    return {"message": "queued", "dataset": ds, "job": job.to_dict()}


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    return catalog.list_datasets()


@app.get("/api/datasets/stats")
def datasets_stats() -> dict:
    return catalog.catalog_stats()


@app.get("/api/datasets/graph")
def datasets_graph() -> dict:
    return catalog.catalog_graph()


@app.get("/api/datasets/{ds_id}")
def get_dataset(ds_id: str) -> dict:
    ds = catalog.get_dataset(ds_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


class DraftIn(BaseModel):
    description: str | None = Field(default=None, max_length=4000)


@app.post("/api/datasets/{ds_id}/draft")
def draft_dataset(ds_id: str, body: DraftIn | None = None) -> dict:
    """AI proposal for the dataset title/description and every column (the user edits before saving)."""
    try:
        return catalog.draft_descriptions(ds_id, body.description if body else None)
    except KeyError:
        raise HTTPException(404, "Dataset not found") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("draft failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e


class ColumnDescIn(BaseModel):
    position: int
    description: str = Field(default="", max_length=2000)


class DatasetDescIn(BaseModel):
    name: str = Field(default="", max_length=200)
    description: str = Field(min_length=10, max_length=4000)
    columns: list[ColumnDescIn] = []


@app.put("/api/datasets/{ds_id}")
def save_dataset(ds_id: str, body: DatasetDescIn) -> dict:
    """Save the user's descriptions, embed dataset + columns, mark the dataset ready."""
    try:
        return catalog.save_descriptions(ds_id, body.name, body.description, [c.model_dump() for c in body.columns])
    except KeyError:
        raise HTTPException(404, "Dataset not found") from None
    except (ValueError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str) -> dict:
    if _dataset_busy(ds_id):
        raise HTTPException(409, "This file is still being processed")
    if not catalog.delete_dataset(ds_id):
        raise HTTPException(404, "Dataset not found")
    return {"message": "deleted"}


@app.delete("/api/datasets")
def reset_datasets() -> dict:
    if any(j.kind == "profile" and j.status in ("queued", "running") for j in jobs.list(500)):
        raise HTTPException(409, "A file is still being processed; wait for it to finish")
    catalog.reset_catalog()
    return {"message": "Dataset catalog cleared"}

# --------------------------------------------------------------------------- reset
@app.delete("/api/graph")
def reset() -> dict:
    if jobs.busy():
        raise HTTPException(409, "A job is running; wait for it to finish before resetting")
    db.reset_graph()
    for f in Path(get_settings().upload_dir).glob("*"):
        if f.is_file():
            f.unlink()
    return {"message": "Document graph and uploaded documents cleared (dataset catalog kept)"}


# Frontend (mounted last so /api/* routes take precedence)
app.mount("/", StaticFiles(directory=get_settings().static_dir, html=True), name="static")
