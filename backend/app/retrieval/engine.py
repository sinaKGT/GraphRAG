"""Retrieval algorithm (spec steps 1-7):

1. embed query
2. context = empty
3. top-level communities by vector similarity (threshold, with a min/max fallback)
4. BFS drill-down: pop community -> score -> accept summary -> enqueue children,
   or at a leaf: add the most relevant entities, relationships and source-text excerpts
5. prepare context: dedupe, one Gemini rerank of the shortlist, trim to the char budget
6. fact check  (answer draft -> claim-by-claim verification -> corrected final answer)
7. answer

Every decision is recorded in `trace`, which the UI replays to highlight the nodes the model picked.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field

from .. import db
from ..config import get_settings
from ..llm import embed, generate_json
from ..prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    FACTCHECK_SYSTEM,
    FACTCHECK_USER,
    RERANK_SYSTEM,
    RERANK_USER,
    AnswerDraft,
    FactCheckResult,
    RerankResult,
)

log = logging.getLogger("graphrag.retrieval")

# Neo4j's vector.similarity.cosine returns (1 + cos) / 2 -> convert back to raw cosine
_COS = "(2 * vector.similarity.cosine({a}, $q) - 1)"


@dataclass
class State:
    question: str
    trace: list[dict] = field(default_factory=list)
    communities: dict[str, dict] = field(default_factory=dict)   # accepted community summaries
    leaf_of_entity: dict[str, str] = field(default_factory=dict)  # entity id -> leaf community id
    entities: dict[str, dict] = field(default_factory=dict)
    relationships: dict[str, dict] = field(default_factory=dict)
    chunks: dict[str, dict] = field(default_factory=dict)
    chars: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def event(self, step: str, **data) -> None:
        self.trace.append({"step": step, **data})


# --------------------------------------------------------------------------- public entry point
def answer_question(question: str) -> dict:
    s = get_settings()
    db.embedding_guard()
    st = State(question=question.strip())
    t = time.perf_counter()

    if db.run("MATCH (c:Community) RETURN count(c) AS n")[0]["n"] == 0:
        return _result(st, "The knowledge graph is empty. Upload a document first.", None, None, [])

    # ---- step 1: query vectors (one per embedding space) ----
    q_doc = embed([st.question], "RETRIEVAL_QUERY")[0]        # vs community summaries (RETRIEVAL_DOCUMENT)
    q_sim = embed([st.question], "SEMANTIC_SIMILARITY")[0]    # vs entity vectors (SEMANTIC_SIMILARITY)
    st.timings["embed"] = _lap(t)

    # ---- step 3: top-level communities ----
    tops = db.run(
        f"""MATCH (c:Community) WHERE c.is_top
            RETURN c.id AS id, c.title AS title, c.summary AS summary, c.level AS level, c.size AS size,
                   {_COS.format(a="c.embedding")} AS score
            ORDER BY score DESC LIMIT $k""",
        q=q_doc, k=s.retrieval_top_k,
    )
    selected = [c for c in tops if c["score"] >= s.retrieval_threshold]
    if not selected and tops:
        # nothing clears the threshold -> take the best one, plus any within a small margin of it
        best = tops[0]["score"]
        selected = [c for c in tops if c["score"] >= best - s.retrieval_fallback_margin][: s.retrieval_min_top]
        for c in selected:
            c["forced"] = True
    sel_ids = {c["id"] for c in selected}
    st.event("top_level", threshold=s.retrieval_threshold, candidates=[
        {"id": c["id"], "title": c["title"], "score": round(c["score"], 4), "selected": c["id"] in sel_ids}
        for c in tops
    ])

    # ---- step 4: hierarchical BFS drill-down ----
    _traverse(st, selected, q_doc, q_sim)
    st.timings["traverse"] = _lap(t)

    # ---- step 5: prepare context ----
    items = _prepare_context(st)
    st.timings["rerank"] = _lap(t)
    if not items:
        return _result(st, "I couldn't find relevant information in the knowledge graph for this question.",
                       None, None, [])
    context_txt = "\n\n".join(i["text"] for i in items)

    # ---- steps 6-7: draft -> fact check -> final answer ----
    draft = generate_json(ANSWER_USER.format(question=st.question, context=context_txt),
                          AnswerDraft, system=ANSWER_SYSTEM)
    st.timings["answer"] = _lap(t)

    claims_txt = "\n".join(f"{i}. {c.claim}  (cites: {', '.join(c.citations) or 'none'})"
                           for i, c in enumerate(draft.claims))
    fc = generate_json(
        FACTCHECK_USER.format(question=st.question, context=context_txt, draft=draft.answer,
                              claims=claims_txt or "(no claims)"),
        FactCheckResult, system=FACTCHECK_SYSTEM,
    )
    st.timings["fact_check"] = _lap(t)
    return _result(st, fc.final_answer.strip() or draft.answer, draft, fc, items)


# --------------------------------------------------------------------------- step 4
def _traverse(st: State, start: list[dict], q_doc: list[float], q_sim: list[float]) -> None:
    s = get_settings()
    queue = deque(start)
    seen: set[str] = set()
    visits = 0

    while queue and visits < s.retrieval_max_visits and st.chars < s.context_max_chars:
        c = queue.popleft()
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        visits += 1
        accepted = c["score"] >= s.retrieval_threshold or c.get("forced", False)
        st.event("visit", id=c["id"], title=c["title"], score=round(c["score"], 4),
                 accepted=accepted, forced=c.get("forced", False), order=visits)
        if not accepted:
            continue

        st.communities[c["id"]] = c
        st.chars += len(c["summary"] or "")

        children = db.run(
            f"""MATCH (ch:Community)-[:CHILD_OF]->(:Community {{id: $id}})
                RETURN ch.id AS id, ch.title AS title, ch.summary AS summary, ch.level AS level,
                       ch.size AS size, {_COS.format(a="ch.embedding")} AS score
                ORDER BY score DESC""",
            id=c["id"], q=q_doc,
        )
        if children:
            # relevant parent but no child passes -> still descend into the best child
            if not any(ch["score"] >= s.retrieval_threshold for ch in children):
                children[0]["forced"] = True
            queue.extend(children)
        else:
            _add_leaf_details(st, c["id"], q_sim)


def _add_leaf_details(st: State, community_id: str, q_sim: list[float]) -> None:
    s = get_settings()
    ents = db.run(
        f"""MATCH (e:Entity)-[:IN_COMMUNITY]->(:Community {{id: $cid}})
            RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description,
                   {_COS.format(a="e.embedding")} AS score
            ORDER BY score DESC LIMIT $n""",
        cid=community_id, q=q_sim, n=s.leaf_max_entities,
    )
    ids = [e["id"] for e in ents]
    for e in ents:
        if e["id"] not in st.entities or st.entities[e["id"]]["score"] < e["score"]:
            st.entities[e["id"]] = e
            st.leaf_of_entity[e["id"]] = community_id
        st.chars += len(e["description"] or "") + len(e["name"])

    rels = db.run(
        """MATCH (a:Entity)-[r:RELATED]->(b:Entity)
           WHERE a.id IN $ids OR b.id IN $ids
           RETURN a.id AS a_id, a.name AS a, r.type AS type, b.id AS b_id, b.name AS b,
                  r.descriptions AS descriptions, r.weight AS weight,
                  (a.id IN $ids AND b.id IN $ids) AS internal
           ORDER BY internal DESC, weight DESC LIMIT $m""",
        ids=ids, m=s.leaf_max_relationships,
    )
    rel_keys = []
    for r in rels:
        key = f"{r['a_id']}|{r['type']}|{r['b_id']}"
        rel_keys.append(key)
        if key not in st.relationships:
            r["community_id"] = community_id
            st.relationships[key] = r
            st.chars += 80 + len(" ".join(r["descriptions"] or []))

    chunk_rows = db.run(
        """MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids
           MATCH (c)-[:PART_OF]->(d:Document)
           RETURN c.id AS id, c.text AS text, d.name AS doc, c.index AS index, collect(e.id) AS entity_ids""",
        ids=ids,
    )
    for row in chunk_rows:
        ch = st.chunks.setdefault(row["id"], {**row, "entity_ids": set()})
        ch["entity_ids"].update(row["entity_ids"])

    st.event("leaf", id=community_id, entity_ids=ids, relationship_keys=rel_keys,
             chunk_ids=[r["id"] for r in chunk_rows])


# --------------------------------------------------------------------------- step 5
def _prepare_context(st: State) -> list[dict]:
    s = get_settings()

    # rank source chunks by the relevance of the entities they mention
    for ch in st.chunks.values():
        ch["score"] = sum(max(0.0, st.entities[e]["score"]) for e in ch["entity_ids"] if e in st.entities)
    chunks = sorted(st.chunks.values(), key=lambda c: c["score"], reverse=True)[: s.max_source_chunks]
    for ch in chunks:
        ch["excerpt"] = _excerpt(ch, st.entities, s.source_excerpt_chars)

    comms = sorted(st.communities.values(), key=lambda c: c["score"], reverse=True)

    # ---- single Gemini rerank over the shortlist (community summaries + source excerpts) ----
    candidates = {f"C{i + 1}": ("community", c) for i, c in enumerate(comms)}
    candidates |= {f"S{i + 1}": ("chunk", c) for i, c in enumerate(chunks)}
    kept_labels = set(candidates)
    if len(candidates) > 1:
        lines = []
        for label, (kind, obj) in candidates.items():
            body = f"{obj['title']}: {obj['summary']}" if kind == "community" else obj["excerpt"]
            lines.append(f"[{label}] {body[:1200]}")
        rr = generate_json(RERANK_USER.format(question=st.question, items="\n\n".join(lines)),
                           RerankResult, system=RERANK_SYSTEM)
        # empty -> nothing relevant: skip the answer + fact-check calls entirely
        kept_labels = {l for l in rr.relevant_ids if l in candidates}

    kept_comm_ids = {obj["id"] for l, (k, obj) in candidates.items() if k == "community" and l in kept_labels}
    kept_chunk_ids = {obj["id"] for l, (k, obj) in candidates.items() if k == "chunk" and l in kept_labels}
    dropped = [obj["id"] for l, (k, obj) in candidates.items() if l not in kept_labels]
    st.event("rerank", kept=sorted(kept_comm_ids | kept_chunk_ids), dropped=dropped)

    # entities/relationships survive only if their leaf community survived
    entities = [e for e in st.entities.values() if st.leaf_of_entity.get(e["id"]) in kept_comm_ids]
    entities.sort(key=lambda e: e["score"], reverse=True)
    kept_entity_ids = {e["id"] for e in entities}
    rels = [r for r in st.relationships.values()
            if r["community_id"] in kept_comm_ids and (r["a_id"] in kept_entity_ids or r["b_id"] in kept_entity_ids)]
    rels.sort(key=lambda r: r["weight"], reverse=True)

    # ---- build labelled context under the char budget: sources, summaries, entities, relations ----
    items: list[dict] = []
    used = 0

    def add(label: str, kind: str, node_ids: list[str], title: str, text: str) -> bool:
        nonlocal used
        block = f"[{label}] {text}"
        if used + len(block) > s.context_max_chars:
            return False
        items.append({"label": label, "kind": kind, "node_ids": node_ids, "title": title, "text": block})
        used += len(block)
        return True

    n = 0
    for ch in chunks:
        if ch["id"] in kept_chunk_ids:
            n += 1
            add(f"S{n}", "chunk", [ch["id"], *sorted(ch["entity_ids"])],
                f"{ch['doc']} · chunk {ch['index']}", f"(source: {ch['doc']}, chunk {ch['index']}) {ch['excerpt']}")
    n = 0
    for c in comms:
        if c["id"] in kept_comm_ids:
            n += 1
            add(f"C{n}", "community", [c["id"]], c["title"], f"{c['title']}: {c['summary']}")
    n = 0
    for e in entities:
        n += 1
        add(f"E{n}", "entity", [e["id"]], e["name"], f"{e['name']} ({e['type']}): {e['description'] or ''}")
    n = 0
    for r in rels:
        n += 1
        add(f"R{n}", "relationship", [r["a_id"], r["b_id"]], f"{r['a']} {r['type']} {r['b']}",
            f"{r['a']} -[{r['type']}]-> {r['b']}: {' '.join(r['descriptions'] or [])}")

    st.event("context", items=[{"label": i["label"], "kind": i["kind"], "node_ids": i["node_ids"]} for i in items],
             chars=used)
    return items


def _excerpt(chunk: dict, entities: dict[str, dict], size: int) -> str:
    """Window of the chunk centred on the first mention of its most relevant entity."""
    text = chunk["text"]
    if len(text) <= size:
        return text
    ranked = sorted((entities[e] for e in chunk["entity_ids"] if e in entities), key=lambda e: e["score"], reverse=True)
    pos = -1
    for e in ranked:
        pos = text.lower().find(e["name"].lower())
        if pos >= 0:
            break
    start = max(0, min(len(text) - size, pos - size // 3)) if pos >= 0 else 0
    out = text[start : start + size]
    return ("…" if start > 0 else "") + out + ("…" if start + size < len(text) else "")


# --------------------------------------------------------------------------- output
_CITE = re.compile(r"\[([CERS]\d+)\]")


def _result(st: State, answer: str, draft: AnswerDraft | None, fc: FactCheckResult | None, items: list[dict]) -> dict:
    by_label = {i["label"]: i for i in items}
    cited_labels = list(dict.fromkeys(_CITE.findall(answer)))
    cited = [by_label[l] for l in cited_labels if l in by_label]
    st.event("cited", labels=cited_labels, node_ids=sorted({n for i in cited for n in i["node_ids"]}))

    verdicts = []
    if draft and fc:
        for v in fc.verdicts:
            if 0 <= v.claim_index < len(draft.claims):
                verdicts.append({"claim": draft.claims[v.claim_index].claim, "verdict": v.verdict, "note": v.note})

    return {
        "question": st.question,
        "answer": answer,
        "draft_answer": draft.answer if draft else None,
        "fact_check": verdicts,
        "citations": {i["label"]: {"kind": i["kind"], "title": i["title"], "node_ids": i["node_ids"],
                                   "text": i["text"]} for i in cited},
        "context": [{k: i[k] for k in ("label", "kind", "title", "node_ids")} for i in items],
        "trace": st.trace,
        "timings": st.timings,
    }


def _lap(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)
