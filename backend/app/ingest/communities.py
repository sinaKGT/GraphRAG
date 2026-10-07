"""Steps 7-8: hierarchical Leiden (Neo4j GDS) + bottom-up community summarisation.

The whole community layer is rebuilt after every ingestion (Leiden is a global algorithm), but
summaries/embeddings are cached by content hash, so only communities whose content actually
changed cost LLM calls."""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .. import db
from ..config import get_settings
from ..llm import embed, generate_json
from ..prompts import (
    SUMMARY_LEAF_USER,
    SUMMARY_PARENT_USER,
    SUMMARY_SYSTEM,
    CommunitySummary,
)

log = logging.getLogger("graphrag.communities")
GRAPH_NAME = "graphrag_entities"


@dataclass
class Comm:
    key: tuple[int, int]                     # (leiden level, leiden community id)
    level: int
    members: set[str] = field(default_factory=set)       # all entity ids (transitively)
    children: list["Comm"] = field(default_factory=list)  # sub-communities
    entities: list[str] = field(default_factory=list)     # entities attached directly (leaf level)
    parent: "Comm | None" = None
    id: str = ""
    content_hash: str = ""
    height: int = 0
    title: str = ""
    summary: str = ""
    embedding: list[float] | None = None


def _sha(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()


# --------------------------------------------------------------------------- step 7: Leiden
def run_leiden() -> dict[str, list[int]]:
    """Return entity id -> [community id per level] (finest level first)."""
    s = get_settings()
    db.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=GRAPH_NAME)
    n_entities = db.run("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]
    if n_entities == 0:
        return {}
    n_rels = db.run("MATCH (:Entity)-[r:RELATED]->(:Entity) RETURN count(r) AS n")[0]["n"]
    if n_rels == 0:  # Leiden needs edges; every entity is its own community
        ids = [r["id"] for r in db.run("MATCH (e:Entity) RETURN e.id AS id")]
        return {eid: [i] for i, eid in enumerate(ids)}

    db.run(
        """CALL gds.graph.project($g, 'Entity',
             {RELATED: {orientation: 'UNDIRECTED', aggregation: 'SUM',
                        properties: {weight: {property: 'weight', aggregation: 'SUM', defaultValue: 1.0}}}})
           YIELD graphName RETURN graphName""",
        g=GRAPH_NAME,
    )
    try:
        rows = db.run(
            """CALL gds.leiden.stream($g, {
                   relationshipWeightProperty: 'weight',
                   includeIntermediateCommunities: true,
                   maxLevels: $max_levels, gamma: $gamma,
                   randomSeed: $seed, concurrency: 1})
               YIELD nodeId, communityId, intermediateCommunityIds
               RETURN gds.util.asNode(nodeId).id AS id, communityId, intermediateCommunityIds AS levels""",
            g=GRAPH_NAME, max_levels=s.leiden_max_levels, gamma=s.leiden_gamma, seed=s.leiden_seed,
        )
    finally:
        db.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=GRAPH_NAME)
    return {r["id"]: (r["levels"] or [r["communityId"]]) for r in rows}


def build_hierarchy(assign: dict[str, list[int]]) -> list[Comm]:
    """Turn per-level assignments into a tree; collapse levels that don't split anything."""
    comms: dict[tuple[int, int], Comm] = {}
    for eid, path in assign.items():
        prev = None
        for lvl, cid in enumerate(path):
            c = comms.setdefault((lvl, cid), Comm(key=(lvl, cid), level=lvl))
            c.members.add(eid)
            if prev is not None:
                prev.parent = c
            prev = c

    # collapse: a community whose parent has exactly the same members is redundant
    def surviving(c: Comm | None) -> Comm | None:
        while c is not None and c.parent is not None and c.parent.members == c.members:
            c = c.parent
        return c

    for eid, path in assign.items():
        leaf = surviving(comms[(0, path[0])])
        if eid not in leaf.entities:
            leaf.entities.append(eid)

    alive: dict[tuple[int, int], Comm] = {}
    for c in comms.values():
        s = surviving(c)
        alive[s.key] = s
    for c in alive.values():
        c.parent = surviving(c.parent) if c.parent is not None else None
        c.children = []
    for c in alive.values():
        if c.parent is not None:
            c.parent.children.append(c)

    # stable ids + heights (0 = leaf)
    for c in alive.values():
        c.id = "comm-" + _sha(*sorted(c.members))[:16]

    def height(c: Comm) -> int:
        if not c.children:
            return 0
        return 1 + max(height(ch) for ch in c.children)

    for c in alive.values():
        c.height = height(c)
    return list(alive.values())


# --------------------------------------------------------------------------- step 8: summaries
def _leaf_context(c: Comm) -> tuple[str, str, str]:
    """Return (content_hash, entities text, relationships text) for a leaf community."""
    s = get_settings()
    ents = db.run(
        """UNWIND $ids AS id MATCH (e:Entity {id: id})
           OPTIONAL MATCH (e)-[r:RELATED]-()
           RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS d, count(r) AS deg
           ORDER BY deg DESC""",
        ids=c.entities,
    )
    rels = db.run(
        """MATCH (a:Entity)-[r:RELATED]->(b:Entity)
           WHERE a.id IN $ids AND b.id IN $ids
           RETURN a.name AS a, r.type AS t, b.name AS b, r.descriptions AS d, r.weight AS w
           ORDER BY w DESC""",
        ids=c.entities,
    )
    ent_lines = [f"- {e['name']} ({e['type']}): {e['d'] or ''}" for e in ents]
    rel_lines = [f"- {r['a']} -[{r['t']}]-> {r['b']}: {' '.join(r['d'] or [])}" for r in rels]
    content_hash = _sha("leaf", *sorted(ent_lines), *sorted(rel_lines))

    budget = s.summary_max_input_chars
    ent_txt = _truncate(ent_lines, budget * 6 // 10)
    rel_txt = _truncate(rel_lines, budget * 4 // 10)
    return content_hash, ent_txt, rel_txt


def _truncate(lines: list[str], budget: int) -> str:
    out, used = [], 0
    for line in lines:
        if used + len(line) > budget:
            out.append(f"... ({len(lines) - len(out)} more omitted)")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out) if out else "(none)"


def rebuild_communities(progress: Callable[[str, int, int], None]) -> dict:
    progress("community detection (Leiden)", 0, 0)
    assign = run_leiden()
    comms = build_hierarchy(assign)
    if not comms:
        db.run("MATCH (c:Community) DETACH DELETE c")
        return {"communities": 0, "levels": 0, "summarised": 0, "cached": 0}

    # cache of previous summaries by content hash
    cache = {
        r["h"]: r for r in db.run(
            "MATCH (c:Community) RETURN c.content_hash AS h, c.title AS title, c.summary AS summary, c.embedding AS emb"
        ) if r["h"]
    }

    by_height: dict[int, list[Comm]] = defaultdict(list)
    for c in comms:
        by_height[c.height].append(c)

    s = get_settings()
    total = sum(1 for c in comms if len(c.members) > 1)
    done = 0
    summarised = cached = 0

    for h in sorted(by_height):
        level_comms = by_height[h]
        todo: list[tuple[Comm, str]] = []

        for c in level_comms:
            if c.height == 0:
                ch, ent_txt, rel_txt = _leaf_context(c)
                c.content_hash = ch
                if len(c.members) == 1:  # singleton: no LLM needed
                    e = db.run("MATCH (e:Entity {id: $id}) RETURN e.name AS n, e.description AS d",
                               id=next(iter(c.members)))[0]
                    c.title, c.summary = e["n"], e["d"] or e["n"]
                    continue
                prompt = SUMMARY_LEAF_USER.format(entities=ent_txt, relationships=rel_txt)
            else:
                kids = sorted(c.children, key=lambda k: len(k.members), reverse=True)
                c.content_hash = _sha("parent", *sorted(k.content_hash for k in kids))
                child_lines = [f"- [{k.title}] {k.summary}" for k in kids]
                prompt = SUMMARY_PARENT_USER.format(children=_truncate(child_lines, s.summary_max_input_chars))

            hit = cache.get(c.content_hash)
            if hit:
                c.title, c.summary, c.embedding = hit["title"], hit["summary"], hit["emb"]
                cached += 1
                done += 1
                continue
            todo.append((c, prompt))

        def work(item: tuple[Comm, str]) -> None:
            c, prompt = item
            r = generate_json(prompt, CommunitySummary, system=SUMMARY_SYSTEM)
            c.title, c.summary = r.title.strip(), r.summary.strip()

        with ThreadPoolExecutor(max_workers=s.llm_concurrency) as pool:
            for _ in pool.map(work, todo):
                done += 1
                summarised += 1
                progress(f"summarising communities (height {h})", done, total)

    # embed summaries that are new
    need = [c for c in comms if c.embedding is None]
    if need:
        progress(f"embedding {len(need)} community summaries", done, total)
        vecs = embed([f"{c.title}: {c.summary}" for c in need], "RETRIEVAL_DOCUMENT")
        for c, v in zip(need, vecs):
            c.embedding = v

    _write(comms)
    stats = {
        "communities": len(comms),
        "top_level": sum(1 for c in comms if c.parent is None),
        "levels": 1 + max(c.height for c in comms),
        "summarised": summarised,
        "cached": cached,
    }
    log.info("communities rebuilt: %s", stats)
    return stats


def _write(comms: list[Comm]) -> None:
    rows = [{
        "id": c.id, "level": c.height, "leiden_level": c.level, "size": len(c.members),
        "title": c.title, "summary": c.summary, "content_hash": c.content_hash,
        "embedding": c.embedding, "is_top": c.parent is None,
    } for c in comms]
    db.run("MATCH (c:Community) DETACH DELETE c")
    db.run(
        """UNWIND $rows AS row
           CREATE (c:Community {id: row.id, level: row.level, leiden_level: row.leiden_level,
                                size: row.size, title: row.title, summary: row.summary,
                                content_hash: row.content_hash, is_top: row.is_top})
           WITH c, row CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)""",
        rows=rows,
    )
    db.run(
        """UNWIND $rows AS row
           MATCH (c:Community {id: row.id})
           UNWIND row.entities AS eid
           MATCH (e:Entity {id: eid})
           CREATE (e)-[:IN_COMMUNITY]->(c)""",
        rows=[{"id": c.id, "entities": c.entities} for c in comms if c.entities],
    )
    db.run(
        """UNWIND $rows AS row
           MATCH (child:Community {id: row.child}), (parent:Community {id: row.parent})
           CREATE (child)-[:CHILD_OF]->(parent)""",
        rows=[{"child": c.id, "parent": c.parent.id} for c in comms if c.parent is not None],
    )
