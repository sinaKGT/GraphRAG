"""Dataset catalog: metadata knowledge graph of tabular files.

    (:Dataset {id, file_name, name, description, row_count, status, embedding, ...})
        -[:HAS_COLUMN]->
    (:DatasetColumn {id, name, position, dtype, stats..., description, embedding})

Status flow: profiling -> needs_description -> ready   (or failed)
Only metadata + small profiles live in Neo4j; the file itself stays in /data/uploads/datasets
so later steps can filter its rows.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .. import db
from ..config import get_settings
from ..jobs import Job
from ..llm import embed, generate_json
from ..prompts import DATASET_DRAFT_SYSTEM, DATASET_DRAFT_USER, DatasetDraft
from .profile import profile_file

log = logging.getLogger("graphrag.datasets")

# properties the UI never needs (big lists / vectors)
_HIDDEN = ("embedding", "key_sample")   # "categories" is kept: the UI shows it


def datasets_dir() -> Path:
    p = Path(get_settings().upload_dir) / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _HIDDEN}


# =========================================================================== register + profile
def register_dataset(file_name: str, digest: str, path: Path) -> tuple[dict, bool]:
    """Create the Dataset node (or return the existing one for the same file content)."""
    existing = db.run("MATCH (d:Dataset {file_hash: $h}) RETURN d", h=digest)
    if existing:
        return _strip(existing[0]["d"]), False
    ds_id = f"ds_{digest[:12]}"
    rec = db.run(
        """CREATE (d:Dataset {id: $id, file_name: $name, file_hash: $h, path: $path,
                              name: $title, status: 'queued', created_at: $now, updated_at: $now})
           RETURN d""",
        id=ds_id, name=file_name, h=digest, path=str(path), title=Path(file_name).stem, now=time.time(),
    )
    return _strip(rec[0]["d"]), True


def profile_job(job: Job) -> dict:
    """Job handler: stream the file, store columns + profiles, then wait for the user's descriptions."""
    ds = db.run("MATCH (d:Dataset {id: $id}) RETURN d.path AS path, d.file_name AS name", id=job.dataset_id)
    if not ds:
        raise RuntimeError("Dataset was deleted")
    path = Path(ds[0]["path"])
    db.run("MATCH (d:Dataset {id: $id}) SET d.status = 'profiling', d.error = null", id=job.dataset_id)

    def progress(done: int, total: int | None) -> None:
        job.update("reading rows", done=done, total=total or 0)

    try:
        job.update("opening file")
        prof = profile_file(path, progress)
        if not prof["columns"]:
            raise ValueError("No columns found (is the first row empty?)")
        cols = [{**c, "id": f"{job.dataset_id}:{c['position']}"} for c in prof["columns"]]
        job.update("saving columns", done=0, total=len(cols))
        db.run(
            """MATCH (d:Dataset {id: $id})
               OPTIONAL MATCH (d)-[:HAS_COLUMN]->(old:DatasetColumn)
               DETACH DELETE old""",
            id=job.dataset_id,
        )
        db.run(
            """MATCH (d:Dataset {id: $id})
               SET d.sheet = $sheet, d.other_sheets = $others, d.row_count = $rows,
                   d.column_count = size($cols), d.status = 'needs_description', d.updated_at = $now
               WITH d
               UNWIND $cols AS c
               CREATE (d)-[:HAS_COLUMN]->(col:DatasetColumn)
               SET col = c""",
            id=job.dataset_id, sheet=prof["sheet"], others=prof["other_sheets"], rows=prof["row_count"],
            cols=cols, now=time.time(),
        )
        job.update("profiled", done=len(cols), total=len(cols))
    except Exception as e:
        db.run("MATCH (d:Dataset {id: $id}) SET d.status = 'failed', d.error = $err",
               id=job.dataset_id, err=f"{type(e).__name__}: {e}"[:500])
        raise
    return {"rows": prof["row_count"], "columns": len(cols), "sheet": prof["sheet"],
            "other_sheets": prof["other_sheets"]}


# =========================================================================== read
def list_datasets() -> list[dict]:
    return db.run(
        """MATCH (d:Dataset)
           RETURN d.id AS id, d.name AS name, d.file_name AS file_name, d.status AS status,
                  d.row_count AS row_count, d.column_count AS column_count, d.error AS error,
                  d.description AS description, d.created_at AS created_at
           ORDER BY d.created_at DESC"""
    )


def get_dataset(ds_id: str) -> dict | None:
    rec = db.run(
        """MATCH (d:Dataset {id: $id})
           OPTIONAL MATCH (d)-[:HAS_COLUMN]->(c:DatasetColumn)
           WITH d, c ORDER BY c.position
           RETURN d, collect(c) AS cols""",
        id=ds_id,
    )
    if not rec:
        return None
    out = _strip(rec[0]["d"])
    out.pop("path", None)
    out["columns"] = [_strip(c) for c in rec[0]["cols"]]
    return out


def catalog_stats() -> dict:
    return db.run(
        """CALL () { MATCH (d:Dataset) RETURN count(d) AS datasets,
                     count(CASE WHEN d.status = 'ready' THEN 1 END) AS ready }
           CALL () { MATCH (c:DatasetColumn) RETURN count(c) AS columns }
           RETURN datasets, ready, columns"""
    )[0]


def catalog_graph() -> dict:
    datasets = db.run(
        """MATCH (d:Dataset)
           RETURN d.id AS id, d.name AS label, d.status AS status, d.description AS description,
                  d.row_count AS row_count, d.column_count AS column_count, d.file_name AS file_name"""
    )
    columns = db.run(
        """MATCH (d:Dataset)-[:HAS_COLUMN]->(c:DatasetColumn)
           RETURN c.id AS id, c.name AS label, c.dtype AS dtype, c.description AS description,
                  c.is_unique AS is_unique, c.distinct AS distinct, c.distinct_capped AS distinct_capped,
                  c.examples AS examples, d.id AS dataset"""
    )
    return {"datasets": datasets, "columns": columns, "relations": []}


# =========================================================================== AI draft
def _column_line(c: dict) -> str:
    stats = [c["dtype"], f"{c['null_pct']}% empty",
             f"{c['distinct']}{'+' if c.get('distinct_capped') else ''} distinct" + (" (unique)" if c.get("is_unique") else "")]
    if c.get("min") is not None:
        stats.append(f"range {c['min']} .. {c['max']}")
    cats = c.get("categories") or []
    ex = ("all values: " if cats else "examples: ") + " | ".join(v[:60] for v in (cats[:30] if cats else (c.get("examples") or [])[:6]))
    note = f'\n    user note: {c["description"]}' if c.get("description") else ""
    return f"- {c['name']} [{', '.join(stats)}] {ex}{note}"


def draft_descriptions(ds_id: str, user_description: str | None = None) -> dict:
    """One LLM call that proposes a title, a dataset description and one line per column.
    The UI only fills fields the user left empty."""
    ds = get_dataset(ds_id)
    if not ds:
        raise KeyError(ds_id)
    if not ds["columns"]:
        raise ValueError("Dataset has not been profiled yet")
    notes = user_description or ds.get("description") or ""
    prompt = DATASET_DRAFT_USER.format(
        file_name=ds["file_name"],
        sheet=f" (sheet '{ds['sheet']}')" if ds.get("sheet") else "",
        rows=ds.get("row_count"),
        user_notes=f"User's description of the dataset: {notes}\n" if notes else "",
        columns="\n".join(_column_line(c) for c in ds["columns"]),
    )
    draft = generate_json(prompt, DatasetDraft, DATASET_DRAFT_SYSTEM)
    by_name = {c.name.strip().lower(): c.description for c in draft.columns}
    cols = []
    for i, c in enumerate(ds["columns"]):
        desc = by_name.get(c["name"].strip().lower())
        if desc is None and i < len(draft.columns):      # model renamed a column: fall back to order
            desc = draft.columns[i].description
        cols.append({"position": c["position"], "name": c["name"], "description": desc or ""})
    return {"name": draft.name, "description": draft.description, "columns": cols}


# =========================================================================== save descriptions
def _dataset_text(name: str, description: str, cols: list[dict]) -> str:
    col_part = "; ".join(f"{c['name']}: {c['description']}" if c.get("description") else c["name"] for c in cols)
    return f"Dataset: {name}\n{description}\nColumns: {col_part}"[:6000]


def _column_text(ds_name: str, c: dict) -> str:
    cats = c.get("categories") or []
    ex = ", ".join(cats[:30] if cats else (c.get("examples") or [])[:5])
    desc = c.get("description") or ""
    return f"Column '{c['name']}' in dataset '{ds_name}'. {desc} Type: {c['dtype']}. {'Values' if cats else 'Examples'}: {ex}"[:2000]


def save_descriptions(ds_id: str, name: str, description: str, columns: list[dict]) -> dict:
    """Store the user's descriptions, embed dataset + columns, mark the dataset ready."""
    db.embedding_guard()
    ds = get_dataset(ds_id)
    if not ds:
        raise KeyError(ds_id)
    if ds["status"] in ("queued", "profiling"):
        raise ValueError("Wait until the file has been profiled")
    given = {int(c["position"]): (c.get("description") or "").strip() for c in columns}
    cols = [{**c, "description": given.get(c["position"], c.get("description") or "")} for c in ds["columns"]]
    name = name.strip() or ds["name"]
    description = description.strip()

    texts = [_dataset_text(name, description, cols)] + [_column_text(name, c) for c in cols]
    vectors = embed(texts, "RETRIEVAL_DOCUMENT")
    db.run(
        """MATCH (d:Dataset {id: $id})
           SET d.name = $name, d.description = $desc, d.embedding = $vec,
               d.status = 'ready', d.updated_at = $now
           WITH d
           UNWIND $cols AS c
           MATCH (d)-[:HAS_COLUMN]->(col:DatasetColumn {id: c.id})
           SET col.description = c.description, col.embedding = c.embedding""",
        id=ds_id, name=name, desc=description, vec=vectors[0], now=time.time(),
        cols=[{"id": c["id"], "description": c["description"], "embedding": v} for c, v in zip(cols, vectors[1:])],
    )
    log.info("dataset %s saved (%d columns described)", ds_id, sum(1 for c in cols if c["description"]))
    return get_dataset(ds_id)


# =========================================================================== delete
def delete_dataset(ds_id: str) -> bool:
    rec = db.run(
        """MATCH (d:Dataset {id: $id})
           OPTIONAL MATCH (d)-[:HAS_COLUMN]->(c:DatasetColumn)
           WITH d, d.path AS path, collect(c) AS cols
           FOREACH (c IN cols | DETACH DELETE c)
           DETACH DELETE d
           RETURN path""",
        id=ds_id,
    )
    if not rec:
        return False
    if rec[0]["path"]:
        Path(rec[0]["path"]).unlink(missing_ok=True)
    return True


def reset_catalog() -> None:
    with db.get_driver().session() as session:
        session.run("MATCH (n) WHERE n:Dataset OR n:DatasetColumn "
                    "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
    for f in datasets_dir().glob("*"):
        if f.is_file():
            f.unlink()
    db.sync_embedding_space()   # nothing left that pins the old embedding model -> rebuild indexes if needed
