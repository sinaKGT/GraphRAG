"""Neo4j driver lifecycle, schema (constraints + vector indexes) and small helpers."""
from __future__ import annotations

import logging

from neo4j import Driver, GraphDatabase

from .config import get_settings

log = logging.getLogger("graphrag.db")
_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        s = get_settings()
        _driver = GraphDatabase.driver(
            s.neo4j_uri,
            auth=(s.neo4j_user, s.neo4j_password),
            # keep deprecation/hint warnings, drop "unknown property" noise on a fresh DB
            notifications_min_severity="WARNING",
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def run(query: str, **params) -> list[dict]:
    """Run a query in an auto-commit transaction and return records as dicts."""
    records, _, _ = get_driver().execute_query(query, params)
    return [r.data() for r in records]


# --------------------------------------------------------------------------- schema
_VECTOR_INDEXES = {"entity_embedding": "Entity", "community_embedding": "Community",
                   "dataset_embedding": "Dataset", "dataset_column_embedding": "DatasetColumn"}
_DOC_LABELS = ("Document", "Chunk", "Entity", "Alias", "Community")   # document graph (reset by "Reset graph")
_embedding_mismatch: str | None = None   # set when the graph was built with another embedding model


def init_schema() -> None:
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT document_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.hash IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        "CREATE CONSTRAINT alias_key IF NOT EXISTS FOR (a:Alias) REQUIRE a.key IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT meta_key IF NOT EXISTS FOR (m:Meta) REQUIRE m.key IS UNIQUE",
        "CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level)",
        "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT dataset_hash IF NOT EXISTS FOR (d:Dataset) REQUIRE d.file_hash IS UNIQUE",
        "CREATE CONSTRAINT dataset_column_id IF NOT EXISTS FOR (c:DatasetColumn) REQUIRE c.id IS UNIQUE",
    ]
    for q in statements:
        run(q)
    sync_embedding_space()


def _index_dims() -> dict[str, int]:
    rows = run("SHOW VECTOR INDEXES YIELD name, options RETURN name, options")
    return {r["name"]: int(r["options"]["indexConfig"]["vector.dimensions"]) for r in rows}


def _create_vector_indexes(dim: int) -> None:
    existing = _index_dims()
    for name, label in _VECTOR_INDEXES.items():
        if name in existing and existing[name] != dim:
            run(f"DROP INDEX {name} IF EXISTS")
        run(f"""CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding
                OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}""")
    run("CALL db.awaitIndexes(60)")


def sync_embedding_space() -> None:
    """The graph remembers which embedding model built it (:Meta {key:'embedding'}).
    Empty graph -> (re)create indexes for the current model. Non-empty graph built with another
    model -> refuse to mix vector spaces until the user resets or switches back."""
    global _embedding_mismatch
    s = get_settings()
    current, dim = s.embed_model_id, s.effective_embed_dim
    meta = run("MATCH (m:Meta {key: 'embedding'}) RETURN m.model_id AS id")
    has_data = run("MATCH (n) WHERE n:Entity OR n:Community OR (n:Dataset AND n.embedding IS NOT NULL) "
                   "RETURN count(n) > 0 AS x")[0]["x"]
    stored = meta[0]["id"] if meta else None

    if stored is None and has_data:
        # graph built before this guard existed -> it was built with Gemini at the index's dimension
        idx = _index_dims().get("entity_embedding", 768)
        stored = f"gemini:{s.gemini_embed_model}:{idx}"
        run("MERGE (m:Meta {key: 'embedding'}) SET m.model_id = $id", id=stored)

    if stored == current or not has_data:
        _create_vector_indexes(dim)
        run("MERGE (m:Meta {key: 'embedding'}) SET m.model_id = $id", id=current)
        _embedding_mismatch = None
        log.info("Neo4j schema ready (embeddings: %s)", current)
    else:
        _embedding_mismatch = (
            f"The graph was built with embeddings '{stored}', but the current setting is '{current}'. "
            "Vectors from different models can't be mixed: reset the graph (UI -> Reset graph) "
            "or switch EMBED_PROVIDER/EMBED_DIM back in .env."
        )
        log.warning(_embedding_mismatch)


def embedding_guard() -> None:
    """Call before any ingestion/query. Raises a readable error if vector spaces don't match."""
    if _embedding_mismatch:
        raise RuntimeError(_embedding_mismatch)


def embedding_status() -> dict:
    return {"ok": _embedding_mismatch is None, "model_id": get_settings().embed_model_id,
            **({"error": _embedding_mismatch} if _embedding_mismatch else {})}


def check_neo4j() -> dict:
    """Return Neo4j + GDS versions, raising on failure."""
    driver = get_driver()
    driver.verify_connectivity()
    neo4j_version = run("CALL dbms.components() YIELD versions RETURN versions[0] AS v")[0]["v"]
    gds_version = run("RETURN gds.version() AS v")[0]["v"]
    return {"neo4j_version": neo4j_version, "gds_version": gds_version}


def graph_stats() -> dict:
    q = """
    CALL () { MATCH (d:Document) RETURN count(d) AS documents }
    CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
    CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
    CALL () { MATCH (:Entity)-[r:RELATED]->(:Entity) RETURN count(r) AS relationships }
    CALL () { MATCH (c:Community) RETURN count(c) AS communities }
    CALL () { MATCH (c:Community) WHERE NOT (c)-[:CHILD_OF]->() RETURN count(c) AS top_communities }
    CALL () { MATCH (c:Community) RETURN coalesce(max(c.level), -1) AS max_level }
    RETURN documents, chunks, entities, relationships, communities, top_communities, max_level
    """
    return run(q)[0]


def reset_graph() -> None:
    """Delete the document graph (in batches, safe for large graphs). The dataset catalog is kept."""
    labels = " OR ".join(f"n:{label}" for label in _DOC_LABELS)
    # CALL {} IN TRANSACTIONS needs an implicit (auto-commit) transaction -> session.run
    with get_driver().session() as session:
        session.run(f"MATCH (n) WHERE {labels} CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 5000 ROWS").consume()
    sync_embedding_space()  # empty graph -> indexes rebuilt for the current embedding model
