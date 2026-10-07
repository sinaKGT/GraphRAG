"""LLM + embedding client layer.

Two interchangeable backends behind the same two functions:
    generate_json(prompt, schema, system) -> pydantic model   (structured output)
    embed(texts, task_type)               -> list[list[float]] (L2-normalised)

    LLM_PROVIDER / EMBED_PROVIDER = "gemini" | "local"
    gemini: google-genai SDK, free-tier rate limiting + RetryInfo-aware backoff
    local : llama.cpp server (OpenAI-compatible) - JSON-schema constrained decoding,
            Qwen3-Embedding with task instructions
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from typing import TypeVar

import httpx
import numpy as np
from pydantic import BaseModel

from .config import get_settings

log = logging.getLogger("graphrag.llm")
T = TypeVar("T", bound=BaseModel)


# =========================================================================== shared helpers
class RateLimiter:
    """Thread-safe sliding-window limiter (requests per minute) + concurrency cap."""

    def __init__(self, rpm: int | None, concurrency: int):
        self.rpm = rpm
        self._times: list[float] = []
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(max(1, concurrency))

    def __enter__(self):
        self._sem.acquire()
        while self.rpm:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    break
                wait = 60 - (now - self._times[0]) + 0.05
            time.sleep(wait)
        return self

    def __exit__(self, *exc):
        self._sem.release()


def _normalise(vectors: list[list[float]], dim: int) -> list[list[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape[1] < dim:
        raise ValueError(f"Embedding model returned {arr.shape[1]} dims but EMBED_DIM={dim}")
    arr = arr[:, :dim]  # Matryoshka truncation (both Gemini and Qwen3-Embedding support it)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    return arr.tolist()


def _backoff(attempt: int) -> float:
    return min(60.0, 2 ** attempt + random.random())


# =========================================================================== Gemini backend
class GeminiBackend:
    def __init__(self):
        from google import genai  # imported lazily: not needed in fully-local mode

        s = get_settings()
        if not s.gemini_key_set:
            raise RuntimeError("GEMINI_API_KEY is not set in .env (needed because a provider is 'gemini')")
        self.client = genai.Client(api_key=s.gemini_api_key)
        self.llm_limiter = RateLimiter(s.llm_rpm, s.llm_concurrency)
        self.embed_limiter = RateLimiter(s.embed_rpm, 1)

    def _call(self, fn, limiter: RateLimiter, what: str):
        from google.genai import errors

        s = get_settings()
        for attempt in range(s.llm_max_retries + 1):
            try:
                with limiter:
                    return fn()
            except errors.APIError as e:
                if e.code not in {429, 500, 502, 503, 504} or attempt == s.llm_max_retries:
                    raise
                m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", str(e))
                delay = float(m.group(1)) + 1.0 if m else _backoff(attempt)
                log.warning("%s: HTTP %s, retry %d in %.1fs", what, e.code, attempt + 1, delay)
                time.sleep(delay)

    def generate_json(self, prompt: str, schema: type[T], system: str | None) -> T:
        from google.genai import types

        s = get_settings()
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=s.gemini_thinking_level),
        )

        def _do():
            resp = self.client.models.generate_content(model=s.gemini_llm_model, contents=prompt, config=config)
            return resp.parsed if resp.parsed is not None else schema.model_validate_json(resp.text or "{}")

        return self._call(_do, self.llm_limiter, f"gemini.generate[{schema.__name__}]")

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        s = get_settings()
        out: list[list[float]] = []
        for i in range(0, len(texts), s.embed_batch_size):
            batch = texts[i : i + s.embed_batch_size]
            config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=s.effective_embed_dim)
            resp = self._call(
                lambda b=batch, c=config: self.client.models.embed_content(model=s.gemini_embed_model, contents=b, config=c),
                self.embed_limiter, "gemini.embed",
            )
            out.extend(_normalise([e.values for e in resp.embeddings], s.effective_embed_dim))
        return out

    def check_llm(self) -> dict:
        s = get_settings()
        return {"provider": "gemini", "model": self.client.models.get(model=s.gemini_llm_model).name}

    def check_embed(self) -> dict:
        s = get_settings()
        return {"provider": "gemini", "model": self.client.models.get(model=s.gemini_embed_model).name,
                "dim": s.effective_embed_dim}


# =========================================================================== local (llama.cpp) backend
# Qwen3-Embedding is instruction-aware: queries get an instruction, documents don't.
_QWEN_INSTRUCT = {
    "RETRIEVAL_QUERY": "Given a question, retrieve summaries and passages from a knowledge graph that answer it",
    "SEMANTIC_SIMILARITY": "Retrieve entity descriptions that refer to the same or a closely related concept",
    "RETRIEVAL_DOCUMENT": None,
}


class LocalBackend:
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self):
        s = get_settings()
        self.http = httpx.Client(timeout=httpx.Timeout(s.local_request_timeout_s, connect=10))
        self.llm_limiter = RateLimiter(None, s.llm_concurrency)
        self.embed_limiter = RateLimiter(None, 1)

    # ---------------------------------------------------------------- plumbing
    def _post(self, url: str, body: dict, limiter: RateLimiter, what: str) -> dict:
        """POST with retries. Connection errors / 503 (model still loading) are retried until
        LOCAL_READY_TIMEOUT_S so a first-start download doesn't fail the job."""
        s = get_settings()
        deadline = time.monotonic() + s.local_ready_timeout_s
        attempt = 0
        while True:
            try:
                with limiter:
                    r = self.http.post(url, json=body)
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in self._RETRYABLE_STATUS:
                    raise RuntimeError(f"{what}: HTTP {r.status_code}: {r.text[:500]}")
                reason = f"HTTP {r.status_code}"
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                reason = type(e).__name__
            if time.monotonic() > deadline:
                raise RuntimeError(f"{what}: local model not reachable/ready ({reason}) after "
                                   f"{s.local_ready_timeout_s}s - check `docker compose logs llm embed`")
            delay = min(30.0, _backoff(attempt))
            attempt += 1
            log.warning("%s: %s, retrying in %.0fs (model loading?)", what, reason, delay)
            time.sleep(delay)

    # ---------------------------------------------------------------- generation
    def generate_json(self, prompt: str, schema: type[T], system: str | None) -> T:
        s = get_settings()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": "local-llm",
            "messages": messages,
            # grammar-constrained decoding: output is guaranteed to match the Pydantic schema
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
            "chat_template_kwargs": {"enable_thinking": s.local_enable_thinking},
            "temperature": s.local_temperature,
            "top_p": s.local_top_p,
            "top_k": s.local_top_k,
            "presence_penalty": 0.0,   # JSON repeats keys by design - never penalise repetition here
            "max_tokens": s.local_max_tokens,
        }
        what = f"local.generate[{schema.__name__}]"
        for attempt in range(3):
            data = self._post(f"{s.local_llm_url}/v1/chat/completions", body, self.llm_limiter, what)
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length":
                log.warning("%s: hit max_tokens=%d (attempt %d)", what, s.local_max_tokens, attempt + 1)
                continue
            try:
                return schema.model_validate_json(content)
            except ValueError as e:
                log.warning("%s: invalid JSON (attempt %d): %s", what, attempt + 1, str(e)[:200])
        raise RuntimeError(f"{what}: no valid structured output after 3 attempts "
                           f"(raise LOCAL_MAX_TOKENS or use a smaller CHUNK_SIZE_CHARS)")

    # ---------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        s = get_settings()
        instruct = _QWEN_INSTRUCT.get(task_type)
        inputs = [f"Instruct: {instruct}\nQuery: {t}" if instruct else t for t in texts]
        inputs = [t if t.strip() else "(empty)" for t in inputs]
        out: list[list[float]] = []
        batch_size = min(s.embed_batch_size, 32)
        for i in range(0, len(inputs), batch_size):
            data = self._post(f"{s.local_embed_url}/v1/embeddings",
                              {"model": "local-embed", "input": inputs[i : i + batch_size]},
                              self.embed_limiter, "local.embed")
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend(_normalise([d["embedding"] for d in rows], s.effective_embed_dim))
        return out

    # ---------------------------------------------------------------- health
    def _health(self, base: str) -> str:
        try:
            r = self.http.get(f"{base}/health", timeout=3)
            if r.status_code == 200:
                return "ok"
            return "loading" if r.status_code == 503 else f"HTTP {r.status_code}"
        except httpx.HTTPError:
            return "unreachable (downloading or starting - see `docker compose logs llm embed`)"

    def check_llm(self) -> dict:
        s = get_settings()
        state = self._health(s.local_llm_url)
        if state != "ok":
            raise RuntimeError(f"local LLM {state}")
        return {"provider": "local", "model": s.local_llm_file}

    def check_embed(self) -> dict:
        s = get_settings()
        state = self._health(s.local_embed_url)
        if state != "ok":
            raise RuntimeError(f"local embedding model {state}")
        return {"provider": "local", "model": s.local_embed_file, "dim": s.effective_embed_dim}


# =========================================================================== dispatch
_backends: dict[str, object] = {}
_lock = threading.Lock()


def _backend(provider: str):
    with _lock:
        if provider not in _backends:
            _backends[provider] = LocalBackend() if provider == "local" else GeminiBackend()
        return _backends[provider]


def generate_json(prompt: str, schema: type[T], system: str | None = None) -> T:
    return _backend(get_settings().llm_provider).generate_json(prompt, schema, system)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY | SEMANTIC_SIMILARITY"""
    return _backend(get_settings().embed_provider).embed(texts, task_type)


def check_models() -> dict:
    """Health of the configured LLM and embedding backends (each checked independently)."""
    s = get_settings()
    out = {}
    for key, provider, fn in (("llm", s.llm_provider, "check_llm"), ("embeddings", s.embed_provider, "check_embed")):
        try:
            out[key] = {"ok": True, **getattr(_backend(provider), fn)()}
        except Exception as e:  # noqa: BLE001
            out[key] = {"ok": False, "provider": provider, "error": str(e)[:300]}
    return out
