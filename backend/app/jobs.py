"""Single-worker background job queue.
Ingestions are serialised on purpose: community detection is global over the whole graph,
so two documents must never rebuild communities at the same time."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable

log = logging.getLogger("graphrag.jobs")


@dataclass
class Job:
    kind: str                           # "ingest" | "rebuild"
    doc_id: str | None = None
    filename: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"              # queued | running | done | failed
    stage: str = ""
    done: int = 0
    total: int = 0
    stats: dict = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def update(self, stage: str, done: int | None = None, total: int | None = None) -> None:
        self.stage = stage
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        log.info("[job %s] %s %s", self.id, stage, f"{self.done}/{self.total}" if self.total else "")

    def to_dict(self) -> dict:
        return asdict(self)


class JobQueue:
    def __init__(self, handlers: dict[str, Callable[[Job], dict]]):
        self._handlers = handlers
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="job-worker")
        self._thread.start()

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def busy(self) -> bool:
        return any(j.status in ("queued", "running") for j in self._jobs.values())

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            job.status, job.started_at = "running", time.time()
            try:
                job.stats = self._handlers[job.kind](job) or {}
                job.status, job.stage = "done", "finished"
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                log.error("[job %s] failed:\n%s", job.id, traceback.format_exc())
            finally:
                job.finished_at = time.time()
                self._q.task_done()
