"""Ingestion orchestrator: runs steps 1-8 for one document and reports progress."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import db
from ..config import get_settings
from ..jobs import Job
from . import communities, graph
from .extraction import aggregate, extract_chunks
from .text import chunk_text, extract_text

log = logging.getLogger("graphrag.pipeline")


def _set_doc(doc_id: str, **props) -> None:
    db.run("MATCH (d:Document {id: $id}) SET d += $props", id=doc_id, props=props)


def ingest_document(job: Job) -> dict:
    db.embedding_guard()
    s = get_settings()
    doc = db.run("MATCH (d:Document {id: $id}) RETURN d", id=job.doc_id)[0]["d"]
    doc_id, name, path = doc["id"], doc["name"], Path(doc["path"])
    stats: dict = {}

    try:
        # ---- steps 1-3: load, extract text, chunk (skipped when resuming) ----
        n_chunks = db.run("MATCH (c:Chunk {doc_id: $id}) RETURN count(c) AS n", id=doc_id)[0]["n"]
        if n_chunks == 0:
            job.update("extracting text")
            _set_doc(doc_id, status="processing")
            text = extract_text(path)
            chunks = chunk_text(text, s.chunk_size_chars, s.chunk_overlap_chars)
            db.run(
                """MATCH (d:Document {id: $doc_id})
                   UNWIND $rows AS row
                   CREATE (c:Chunk {id: row.id, doc_id: $doc_id, index: row.index,
                                    text: row.text, start: row.start})
                   CREATE (c)-[:PART_OF]->(d)""",
                doc_id=doc_id,
                rows=[{"id": f"{doc_id}:{c.index}", "index": c.index, "text": c.text, "start": c.start} for c in chunks],
            )
            _set_doc(doc_id, chars=len(text), chunks=len(chunks))
            n_chunks = len(chunks)
        stats["chunks"] = n_chunks

        # ---- step 4: entity/relationship extraction per chunk ----
        if doc.get("graph_built") is not True:
            extract_chunks(doc_id, name, lambda d, t: job.update("extracting entities", d, t))
            entities, rels = aggregate(doc_id)
            stats["entities_extracted"], stats["relationships_extracted"] = len(entities), len(rels)

            # ---- step 5: entity resolution ----
            job.update("entity resolution")
            key_to_id, new_ids = graph.upsert_entities(entities, lambda m: job.update(m))
            stats["new_entities"] = len(new_ids)
            stats.update(graph.resolve_entities(new_ids, key_to_id, lambda m: job.update(m)))

            # ---- step 6: base graph ----
            job.update("building base graph")
            stats["edges_written"] = graph.write_relationships(rels, entities, key_to_id)
            graph.entity_descriptions_refresh()
            _set_doc(doc_id, graph_built=True)

        # ---- steps 7-8: communities + hierarchical summaries (global) ----
        stats["communities"] = communities.rebuild_communities(lambda m, d, t: job.update(m, d, t))

        _set_doc(doc_id, status="done", finished_at=datetime.now(timezone.utc).isoformat(), error=None)
        return stats
    except Exception as e:
        _set_doc(doc_id, status="failed", error=str(e)[:500])
        raise


def rebuild_only(job: Job) -> dict:
    db.embedding_guard()
    return {"communities": communities.rebuild_communities(lambda m, d, t: job.update(m, d, t))}


def register_document(filename: str, file_hash: str, path: Path) -> tuple[dict, bool]:
    """Create (or find) the Document node. Returns (doc, is_new)."""
    rows = db.run(
        """MERGE (d:Document {hash: $hash})
           ON CREATE SET d.id = $id, d.name = $name, d.path = $path, d.status = 'queued',
                         d.created_at = $now, d._new = true
           WITH d, coalesce(d._new, false) AS is_new
           REMOVE d._new
           RETURN d, is_new""",
        hash=file_hash, id=str(uuid.uuid4()), name=filename, path=str(path),
        now=datetime.now(timezone.utc).isoformat(),
    )
    return rows[0]["d"], rows[0]["is_new"]
