"""Step 4: per-chunk entity/relationship extraction + aggregation per document."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .. import db
from ..config import get_settings
from ..llm import generate_json
from ..prompts import EXTRACT_SYSTEM, EXTRACT_USER, ExtractionResult

log = logging.getLogger("graphrag.extract")

MAX_DESCRIPTIONS = 8


# --------------------------------------------------------------------------- normalisation
def entity_key(name: str) -> str:
    """Canonical lookup key: exact-duplicate names collapse here; fuzzy ones go to resolution."""
    k = unicodedata.normalize("NFKC", name).lower().strip()
    k = k.strip(" \t\"'`“”‘’.,;:()[]{}")
    k = re.sub(r"^(the|a|an)\s+", "", k)
    k = re.sub(r"\s+", " ", k)
    return k


def norm_type(t: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "_", t or "").strip("_").upper()
    return t or "OTHER"


# --------------------------------------------------------------------------- extraction
def extract_chunks(doc_id: str, doc_name: str, progress: Callable[[int, int], None]) -> None:
    """Extract every not-yet-extracted chunk of the document. Results are stored on the Chunk
    node (`extraction` JSON), so a failed/interrupted ingestion resumes without re-paying."""
    pending = db.run(
        """MATCH (c:Chunk {doc_id: $doc_id}) WHERE c.extraction IS NULL
           RETURN c.id AS id, c.index AS index, c.text AS text ORDER BY c.index""",
        doc_id=doc_id,
    )
    total = db.run("MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n", doc_id=doc_id)[0]["n"]
    done = total - len(pending)
    progress(done, total)
    if not pending:
        return

    def work(ch: dict) -> tuple[str, str]:
        result = generate_json(
            EXTRACT_USER.format(doc_name=doc_name, index=ch["index"], text=ch["text"]),
            ExtractionResult,
            system=EXTRACT_SYSTEM,
        )
        return ch["id"], result.model_dump_json()

    s = get_settings()
    with ThreadPoolExecutor(max_workers=s.llm_concurrency) as pool:
        futures = [pool.submit(work, ch) for ch in pending]
        for fut in as_completed(futures):
            chunk_id, payload = fut.result()  # raises -> job fails, finished chunks are kept
            db.run("MATCH (c:Chunk {id: $id}) SET c.extraction = $p", id=chunk_id, p=payload)
            done += 1
            progress(done, total)


# --------------------------------------------------------------------------- aggregation
@dataclass
class AggEntity:
    key: str
    names: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        # most frequent surface form; ties -> longest
        return max(self.names.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    @property
    def type(self) -> str:
        return self.types.most_common(1)[0][0]


@dataclass
class AggRelationship:
    source_key: str
    target_key: str
    type: str
    weight: float = 0.0
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)


def aggregate(doc_id: str) -> tuple[dict[str, AggEntity], list[AggRelationship]]:
    rows = db.run(
        "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c.id AS id, c.extraction AS x ORDER BY c.index",
        doc_id=doc_id,
    )
    entities: dict[str, AggEntity] = {}
    rels: dict[tuple[str, str, str], AggRelationship] = {}
    dropped = 0

    parsed = [(row["id"], ExtractionResult.model_validate(json.loads(row["x"]))) for row in rows]

    # pass 1: entities from every chunk (so relationships can reference entities from any chunk)
    for chunk_id, x in parsed:
        for e in x.entities:
            k = entity_key(e.name)
            if not k:
                continue
            agg = entities.setdefault(k, AggEntity(k))
            agg.names[e.name.strip()] += 1
            agg.types[norm_type(e.type)] += 1
            d = e.description.strip()
            if d and d not in agg.descriptions and len(agg.descriptions) < MAX_DESCRIPTIONS:
                agg.descriptions.append(d)
            agg.chunk_ids.add(chunk_id)

    # pass 2: relationships
    for chunk_id, x in parsed:
        for r in x.relationships:
            sk, tk = entity_key(r.source), entity_key(r.target)
            if sk not in entities or tk not in entities or sk == tk:
                dropped += 1  # endpoint not extracted as an entity -> unreliable edge
                continue
            rt = norm_type(r.type)
            agg_r = rels.setdefault((sk, tk, rt), AggRelationship(sk, tk, rt))
            agg_r.weight += max(1, min(10, r.strength)) / 10.0
            d = r.description.strip()
            if d and d not in agg_r.descriptions and len(agg_r.descriptions) < MAX_DESCRIPTIONS:
                agg_r.descriptions.append(d)
            agg_r.chunk_ids.add(chunk_id)

    log.info("doc %s: %d entities, %d relationships (%d dropped)", doc_id, len(entities), len(rels), dropped)
    return entities, list(rels.values())
