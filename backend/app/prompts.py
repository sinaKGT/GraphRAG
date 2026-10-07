"""All LLM prompts + their structured-output schemas in one place."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# =========================================================================== Step 4: extraction
EXTRACT_SYSTEM = """You are an information-extraction engine that builds a knowledge graph from text.
Extract every meaningful ENTITY and every explicit or clearly implied RELATIONSHIP between them.

Entities
- name: the most complete, canonical form used in the text (e.g. "Nottingham Trent University", not "the university").
  Resolve pronouns and abbreviations to the full name when the text makes it clear.
- type: a short category in UPPER_SNAKE_CASE that you choose (e.g. PERSON, ORGANIZATION, LOCATION,
  TECHNOLOGY, METHOD, DATASET, CONCEPT, EVENT, METRIC). Reuse the same type for the same kind of thing.
- description: 1-3 sentences on what the text says about this entity (facts only, from this text).

Relationships
- source / target: must exactly match the `name` of an extracted entity.
- type: a short verb phrase in UPPER_SNAKE_CASE (e.g. DEVELOPED_BY, LOCATED_IN, USES, PART_OF).
- description: one sentence explaining the relationship as stated in the text.
- strength: integer 1-10, how strongly/explicitly the text supports it.

Rules
- Use only information in the text. Do not invent facts.
- Skip generic, non-informative entities (e.g. "the author", "this paper", "Figure 3").
- Prefer fewer, high-quality relationships over many vague ones."""

EXTRACT_USER = """Document: {doc_name}
Text chunk {index}:
<<<
{text}
>>>"""


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str
    description: str
    strength: int


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


# =========================================================================== Step 5: entity resolution
RESOLVE_SYSTEM = """You are an entity-resolution judge for a knowledge graph.
Each group below lists entities that are close in embedding space. Decide which of them refer to the
SAME real-world thing (same person, same organisation, same concept...). Be conservative:
- Different but related things are NOT duplicates (e.g. "Nottingham" vs "Nottingham Trent University",
  "Neo4j" vs "Neo4j GDS", "GPT-4" vs "GPT-4o").
- Spelling variants, abbreviations, and alternate names of the same thing ARE duplicates
  (e.g. "NTU" and "Nottingham Trent University" when the descriptions agree).
For every set of duplicates, return one merge with the best canonical name and the ids of all members.
Only use ids that appear in the same group. Omit groups that contain no duplicates."""

RESOLVE_USER = """Groups:
{groups}"""


class MergeDecision(BaseModel):
    group: int
    canonical_name: str
    member_ids: list[str]


class ResolutionResult(BaseModel):
    merges: list[MergeDecision]


# =========================================================================== Step 8: community summaries
SUMMARY_SYSTEM = """You write concise analytical summaries of communities in a knowledge graph.
A community is a cluster of closely connected entities (or of sub-communities).
Write a summary that a retrieval system can match against user questions:
- title: a short, specific name for the community (max 10 words).
- summary: 1-2 paragraphs covering the main entities, how they relate, and the key facts/claims.
  Mention important entity names explicitly. Use only the information provided."""

SUMMARY_LEAF_USER = """Entities:
{entities}

Relationships:
{relationships}"""

SUMMARY_PARENT_USER = """This community is made of the following sub-communities:
{children}"""


class CommunitySummary(BaseModel):
    title: str
    summary: str


# =========================================================================== Retrieval: rerank
RERANK_SYSTEM = """You are a relevance judge for a retrieval system.
Given a user question and a list of candidate context items (community summaries and source-text
excerpts), select the items that contain information useful for answering the question.
Be inclusive of items that give necessary background, exclusive of items that are off-topic.
Return the ids of the useful items, most useful first."""

RERANK_USER = """Question: {question}

Candidates:
{items}"""


class RerankResult(BaseModel):
    relevant_ids: list[str]


# =========================================================================== Retrieval: answer
ANSWER_SYSTEM = """You answer questions using ONLY the provided context from a knowledge graph.
Context items have ids in square brackets: [C#] community summaries, [E#] entities,
[R#] relationships, [S#] source-text excerpts (the original document text, most authoritative).

Rules
- Use only facts present in the context. If the context is insufficient, say what is missing.
- Cite every factual sentence with the ids it relies on, e.g. "... in Singapore [S2][E4]."
- Prefer [S#] source excerpts as evidence when available.
- Also list each atomic factual claim you made with its citations (used for fact-checking).
- Write clearly and concisely; use short paragraphs or bullets where helpful."""

ANSWER_USER = """Question: {question}

Context:
{context}"""


class Claim(BaseModel):
    claim: str
    citations: list[str]


class AnswerDraft(BaseModel):
    answer: str
    claims: list[Claim]


# =========================================================================== Retrieval: fact check
FACTCHECK_SYSTEM = """You are a strict fact-checker. For each numbered claim, check whether the
cited context items (and, if needed, any other context item) actually support it.
- SUPPORTED: fully backed by the context.
- PARTIAL: partly backed; something is overstated, imprecise or missing.
- UNSUPPORTED: not backed by the context (hallucinated or wrong).
Then write final_answer: the draft answer rewritten so that it keeps only supported content,
fixes partial claims to match the context exactly, removes unsupported claims, and keeps the
[id] citations. If nothing changes, return the draft unchanged."""

FACTCHECK_USER = """Question: {question}

Context:
{context}

Draft answer:
{draft}

Claims:
{claims}"""


class Verdict(BaseModel):
    claim_index: int
    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    note: str


class FactCheckResult(BaseModel):
    verdicts: list[Verdict]
    final_answer: str
