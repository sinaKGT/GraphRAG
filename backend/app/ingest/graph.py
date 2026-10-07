"""Steps 5-6: entity resolution (ANN via Neo4j HNSW vector index + Gemini confirmation)
and base-graph construction (entities as nodes, weighted RELATED edges, MENTIONS provenance)."""
from __future__ import annotations

import logging
import uuid
from typing import Callable

from .. import db
from ..config import get_settings
from ..llm import embed, generate_json
from ..prompts import RESOLVE_SYSTEM, RESOLVE_USER, ResolutionResult
from .extraction import AggEntity, AggRelationship, entity_key

log = logging.getLogger("graphrag.graph")

DESC_CHARS = 800


def _entity_text(name: str, etype: str, description: str) -> str:
    return f"{name} ({etype}): {description}"


# --------------------------------------------------------------------------- lookup
def resolve_keys(keys: list[str]) -> dict[str, str]:
    """key -> entity id, via Entity.key or an Alias left behind by an earlier merge."""
    rows = db.run(
        """UNWIND $keys AS k
           OPTIONAL MATCH (e:Entity {key: k})
           OPTIONAL MATCH (:Alias {key: k})-[:ALIAS_OF]->(a:Entity)
           WITH k, coalesce(e.id, a.id) AS id WHERE id IS NOT NULL
           RETURN k, id""",
        keys=keys,
    )
    return {r["k"]: r["id"] for r in rows}


# --------------------------------------------------------------------------- step 5a: upsert
def upsert_entities(entities: dict[str, AggEntity], progress: Callable[[str], None]) -> tuple[dict[str, str], list[str]]:
    """Existing entities (exact key/alias match) get the new descriptions appended.
    New entities are created with a SEMANTIC_SIMILARITY embedding. Returns (key->id, new_ids)."""
    key_to_id = resolve_keys(list(entities))

    existing = [
        {"id": key_to_id[k], "descriptions": e.descriptions, "mentions": len(e.chunk_ids)}
        for k, e in entities.items() if k in key_to_id
    ]
    if existing:
        db.run(
            """UNWIND $rows AS row MATCH (e:Entity {id: row.id})
               WITH e, row, [d IN row.descriptions WHERE NOT d IN e.descriptions] AS extra
               SET e.descriptions = (e.descriptions + extra)[0..8],
                   e.mention_count = e.mention_count + row.mentions""",
            rows=existing,
        )

    new = [e for k, e in entities.items() if k not in key_to_id]
    if not new:
        return key_to_id, []

    progress(f"embedding {len(new)} new entities")
    texts = [_entity_text(e.name, e.type, " ".join(e.descriptions)[:DESC_CHARS]) for e in new]
    vectors = embed(texts, "SEMANTIC_SIMILARITY")

    rows = []
    for e, vec in zip(new, vectors):
        eid = str(uuid.uuid4())
        key_to_id[e.key] = eid
        rows.append({
            "id": eid, "key": e.key, "name": e.name, "type": e.type,
            "descriptions": e.descriptions, "mentions": len(e.chunk_ids), "embedding": vec,
        })
    db.run(
        """UNWIND $rows AS row
           CREATE (e:Entity {id: row.id, key: row.key, name: row.name, type: row.type,
                             descriptions: row.descriptions, mention_count: row.mentions})
           WITH e, row CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)""",
        rows=rows,
    )
    return key_to_id, [r["id"] for r in rows]


# --------------------------------------------------------------------------- step 5b: resolution
def find_candidate_groups(new_ids: list[str]) -> list[list[dict]]:
    """ANN (HNSW) neighbours of each new entity above the cosine threshold -> star groups."""
    s = get_settings()
    # Neo4j reports cosine similarity as (1 + cos) / 2 in [0, 1]
    min_score = (1 + s.er_cosine_threshold) / 2
    rows = db.run(
        """UNWIND $ids AS id
           MATCH (e:Entity {id: id})
           CALL (e) {
             MATCH (node:Entity)
               SEARCH node IN (VECTOR INDEX entity_embedding FOR e.embedding LIMIT $k) SCORE AS score
             RETURN node, score
           }
           WITH e, node, score WHERE node <> e AND score >= $min_score
           RETURN e.id AS src, collect(node.id)[0..$max] AS nbrs""",
        ids=new_ids, k=s.er_top_k + 1, min_score=min_score, max=s.er_max_group_size - 1,
    )
    groups: list[frozenset[str]] = []
    for r in rows:
        g = frozenset([r["src"], *r["nbrs"]])
        if len(g) > 1:
            groups.append(g)
    # drop duplicates and groups fully contained in another group
    groups = sorted(set(groups), key=len, reverse=True)
    kept: list[frozenset[str]] = []
    for g in groups:
        if not any(g <= k for k in kept):
            kept.append(g)
    if not kept:
        return []

    all_ids = sorted(set().union(*kept))
    info = {
        r["id"]: r for r in db.run(
            """UNWIND $ids AS id MATCH (e:Entity {id: id})
               RETURN e.id AS id, e.name AS name, e.type AS type,
                      e.descriptions AS descriptions, e.mention_count AS mentions""",
            ids=all_ids,
        )
    }
    return [[info[i] for i in sorted(g) if i in info] for g in kept]


def judge_groups(groups: list[list[dict]], progress: Callable[[str], None]) -> list[tuple[str, list[str]]]:
    """Ask Gemini which members of each candidate group are true duplicates."""
    s = get_settings()
    decisions: list[tuple[str, list[str]]] = []
    batches = [groups[i : i + s.er_groups_per_call] for i in range(0, len(groups), s.er_groups_per_call)]
    for bi, batch in enumerate(batches):
        progress(f"resolution: judging batch {bi + 1}/{len(batches)}")
        lines = []
        for gi, group in enumerate(batch):
            lines.append(f"Group {gi}:")
            for m in group:
                desc = " ".join(m["descriptions"] or [])[:300]
                lines.append(f'  - id={m["id"]} | name="{m["name"]}" | type={m["type"]} | {desc}')
        result = generate_json(RESOLVE_USER.format(groups="\n".join(lines)), ResolutionResult, system=RESOLVE_SYSTEM)
        for d in result.merges:
            if not (0 <= d.group < len(batch)):
                continue
            allowed = {m["id"] for m in batch[d.group]}
            members = [i for i in dict.fromkeys(d.member_ids) if i in allowed]  # guard hallucinated ids
            if len(members) > 1:
                decisions.append((d.canonical_name.strip(), members))
    return decisions


def apply_merges(decisions: list[tuple[str, list[str]]], key_to_id: dict[str, str]) -> int:
    """Union overlapping decisions, pick the most-mentioned node as survivor, merge the rest into it."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names: dict[str, str] = {}
    for canonical, members in decisions:
        for m in members[1:]:
            parent[find(m)] = find(members[0])
        names[members[0]] = canonical

    clusters: dict[str, list[str]] = {}
    for x in list(parent):
        clusters.setdefault(find(x), []).append(x)

    merged = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        stats = {r["id"]: r["mentions"] for r in db.run(
            "UNWIND $ids AS id MATCH (e:Entity {id:id}) RETURN e.id AS id, e.mention_count AS mentions",
            ids=members)}
        members = [m for m in members if m in stats]
        if len(members) < 2:
            continue
        survivor = max(members, key=lambda m: stats[m])
        canonical = next((names[m] for m in members if m in names), None)
        for dup in members:
            if dup != survivor:
                _merge_into(survivor, dup)
                merged += 1
        if canonical:
            db.run("MATCH (e:Entity {id: $id}) SET e.name = $name", id=survivor, name=canonical)
            _add_alias(entity_key(canonical), survivor)
        for k, v in key_to_id.items():  # keep the in-memory map pointing at survivors
            if v in members:
                key_to_id[k] = survivor
    return merged


def _add_alias(key: str, entity_id: str) -> None:
    db.run(
        """MATCH (e:Entity {id: $id})
           WHERE NOT EXISTS { MATCH (:Entity {key: $key}) } AND NOT EXISTS { MATCH (:Alias {key: $key}) }
           CREATE (:Alias {key: $key})-[:ALIAS_OF]->(e)""",
        id=entity_id, key=key,
    )


def _merge_into(survivor_id: str, dup_id: str) -> None:
    """Rewire every relationship of `dup` onto `survivor`, keep dup's key as an Alias, delete dup."""
    db.run(
        """
        MATCH (s:Entity {id: $sid}), (d:Entity {id: $did})
        // outgoing RELATED
        CALL (s, d) {
          MATCH (d)-[r:RELATED]->(t) WHERE t <> s
          MERGE (s)-[n:RELATED {type: r.type}]->(t)
            ON CREATE SET n.weight = r.weight, n.descriptions = r.descriptions, n.chunk_ids = r.chunk_ids
            ON MATCH SET n.weight = n.weight + r.weight,
                         n.descriptions = (n.descriptions + [x IN r.descriptions WHERE NOT x IN n.descriptions])[0..8],
                         n.chunk_ids = n.chunk_ids + [x IN r.chunk_ids WHERE NOT x IN n.chunk_ids]
        }
        // incoming RELATED
        CALL (s, d) {
          MATCH (t)-[r:RELATED]->(d) WHERE t <> s
          MERGE (t)-[n:RELATED {type: r.type}]->(s)
            ON CREATE SET n.weight = r.weight, n.descriptions = r.descriptions, n.chunk_ids = r.chunk_ids
            ON MATCH SET n.weight = n.weight + r.weight,
                         n.descriptions = (n.descriptions + [x IN r.descriptions WHERE NOT x IN n.descriptions])[0..8],
                         n.chunk_ids = n.chunk_ids + [x IN r.chunk_ids WHERE NOT x IN n.chunk_ids]
        }
        // provenance
        CALL (s, d) {
          MATCH (c:Chunk)-[:MENTIONS]->(d)
          MERGE (c)-[:MENTIONS]->(s)
        }
        // aliases
        CALL (s, d) {
          MATCH (a:Alias)-[x:ALIAS_OF]->(d)
          DELETE x
          CREATE (a)-[:ALIAS_OF]->(s)
        }
        SET s.descriptions = (s.descriptions + [x IN d.descriptions WHERE NOT x IN s.descriptions])[0..8],
            s.mention_count = s.mention_count + d.mention_count
        WITH s, d, d.key AS dkey
        DETACH DELETE d
        CREATE (:Alias {key: dkey})-[:ALIAS_OF]->(s)
        """,
        sid=survivor_id, did=dup_id,
    )


def resolve_entities(new_ids: list[str], key_to_id: dict[str, str], progress: Callable[[str], None]) -> dict:
    if not new_ids:
        return {"candidate_groups": 0, "merged": 0}
    progress("resolution: ANN candidate search")
    groups = find_candidate_groups(new_ids)
    if not groups:
        return {"candidate_groups": 0, "merged": 0}
    decisions = judge_groups(groups, progress)
    merged = apply_merges(decisions, key_to_id)
    log.info("resolution: %d candidate groups, %d entities merged", len(groups), merged)
    return {"candidate_groups": len(groups), "merged": merged}


# --------------------------------------------------------------------------- step 6: base graph
def write_relationships(rels: list[AggRelationship], entities: dict[str, AggEntity], key_to_id: dict[str, str]) -> int:
    edges: dict[tuple[str, str, str], dict] = {}
    for r in rels:
        a, b = key_to_id.get(r.source_key), key_to_id.get(r.target_key)
        if not a or not b or a == b:  # self-loop after merging -> skip
            continue
        e = edges.setdefault((a, b, r.type), {"a": a, "b": b, "type": r.type, "weight": 0.0,
                                              "descriptions": [], "chunk_ids": []})
        e["weight"] += r.weight
        e["descriptions"] += [d for d in r.descriptions if d not in e["descriptions"]]
        e["chunk_ids"] += [c for c in r.chunk_ids if c not in e["chunk_ids"]]

    db.run(
        """UNWIND $rows AS row
           MATCH (a:Entity {id: row.a}), (b:Entity {id: row.b})
           MERGE (a)-[r:RELATED {type: row.type}]->(b)
             ON CREATE SET r.weight = row.weight, r.descriptions = row.descriptions[0..8], r.chunk_ids = row.chunk_ids
             ON MATCH SET r.weight = r.weight + row.weight,
                          r.descriptions = (r.descriptions + [x IN row.descriptions WHERE NOT x IN r.descriptions])[0..8],
                          r.chunk_ids = r.chunk_ids + [x IN row.chunk_ids WHERE NOT x IN r.chunk_ids]""",
        rows=list(edges.values()),
    )

    mentions = [{"c": c, "e": key_to_id[k]} for k, ent in entities.items() if k in key_to_id for c in ent.chunk_ids]
    db.run(
        """UNWIND $rows AS row
           MATCH (c:Chunk {id: row.c}), (e:Entity {id: row.e})
           MERGE (c)-[:MENTIONS]->(e)""",
        rows=mentions,
    )
    return len(edges)


def entity_descriptions_refresh() -> None:
    """Keep a compact `description` string for display/prompting."""
    db.run(
        f"""MATCH (e:Entity)
            SET e.description = substring(reduce(s = '', d IN e.descriptions | s + d + ' '), 0, {DESC_CHARS})"""
    )

