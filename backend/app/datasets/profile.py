"""Column detection + profiling for tabular files (xlsx / xlsm / xls / ods / csv / tsv).

Rows are streamed once, so memory stays flat even for files with ~1M rows. Per column we keep
only small bounded structures: type counts, a capped distinct set, a capped frequency counter,
min/max/mean and a small "key sample" used later to detect join keys between datasets.
"""
from __future__ import annotations

import csv
import io
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Iterator

SUPPORTED_TABULAR = {".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv"}

DISTINCT_CAP = 50_000      # exact distinct counting up to this many values
COUNTER_CAP = 20_000       # values tracked for "most frequent"
EXAMPLES = 8               # distinct example values shown to the user / LLM
KEY_SAMPLE = 2_000         # distinct values kept for join-key detection (step 2)
CATEGORY_MAX = 50          # columns with <= this many distinct values keep their full value list


# =========================================================================== reading
def _clean(v):
    """Normalise a raw cell: '' -> None, integral floats -> int, trimmed strings."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer() and abs(v) < 2**53:
            return int(v)
    return v


def _header(raw: list) -> list[str]:
    names, seen = [], Counter()
    for i, h in enumerate(raw):
        h = _clean(h)
        name = str(h) if h is not None else f"column_{i + 1}"
        seen[name] += 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return names


def open_table(path: Path) -> tuple[str | None, list[str], Iterator[list], int | None, list[str]]:
    """Return (sheet_name, header, row_iterator, total_rows_or_None, other_sheet_names).
    The header is the first non-empty row; fully empty rows are skipped."""
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        return _open_csv(path, "\t" if ext == ".tsv" else None)

    from python_calamine import CalamineWorkbook  # fast Rust reader (xlsx/xls/ods)

    wb = CalamineWorkbook.from_path(str(path))
    names = wb.sheet_names
    for idx, name in enumerate(names):          # first sheet that has any content
        sheet = wb.get_sheet_by_index(idx)
        if sheet.height == 0:
            continue
        rows = sheet.iter_rows()
        for raw in rows:
            if any(_clean(c) is not None for c in raw):
                header = _header(raw)
                total = max(0, sheet.height - 1)
                others = [n for n in names if n != name]
                return name, header, rows, total, others
    raise ValueError("The file has no non-empty sheet")


def _open_csv(path: Path, delimiter: str | None):
    raw = path.read_bytes()[:200_000]
    encoding = "utf-8-sig"
    try:
        raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(raw.decode(encoding, errors="ignore"), delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    f = io.open(path, "r", encoding=encoding, newline="")
    reader = csv.reader(f, delimiter=delimiter)

    def rows():
        try:
            yield from reader
        finally:
            f.close()

    it = rows()
    for first in it:
        if any(c.strip() for c in first):
            return None, _header(first), it, None, []
    raise ValueError("The file is empty")


# =========================================================================== typing
def _parse_text(v: str):
    """CSV cells are strings: recover numbers / booleans / ISO dates."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        f = float(v.replace(",", "")) if v.count(",") and "." in v else float(v)
        return int(f) if f.is_integer() and abs(f) < 2**53 else f
    except ValueError:
        pass
    if 8 <= len(v) <= 32 and v[:4].isdigit() and v[4] in "-/":
        try:
            return datetime.fromisoformat(v.replace("/", "-"))
        except ValueError:
            pass
    return v


def _kind(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    if isinstance(v, (datetime, date)):
        return "datetime"
    if isinstance(v, time):
        return "time"
    if isinstance(v, timedelta):
        return "duration"
    return "text"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, datetime):
        return v.isoformat(sep=" ", timespec="seconds") if (v.hour or v.minute or v.second) else v.date().isoformat()
    if isinstance(v, (date, time)):
        return v.isoformat()
    return str(v)


# =========================================================================== profiling
@dataclass
class ColumnStats:
    name: str
    position: int
    non_null: int = 0
    kinds: Counter = field(default_factory=Counter)
    distinct: set = field(default_factory=set)
    distinct_capped: bool = False
    counter: Counter = field(default_factory=Counter)
    examples: list = field(default_factory=list)
    key_sample: list = field(default_factory=list)
    num_min: float | None = None
    num_max: float | None = None
    num_sum: float = 0.0
    num_n: int = 0
    other_min: object = None
    other_max: object = None
    len_sum: int = 0
    len_max: int = 0

    def add(self, v) -> None:
        self.non_null += 1
        k = _kind(v)
        self.kinds[k] += 1
        s = _fmt(v)
        if not self.distinct_capped:
            if s not in self.distinct:
                if len(self.distinct) >= DISTINCT_CAP:
                    self.distinct_capped = True
                else:
                    self.distinct.add(s)
                    if len(self.examples) < EXAMPLES:
                        self.examples.append(s)
                    if len(self.key_sample) < KEY_SAMPLE and k in ("text", "integer"):
                        self.key_sample.append(s)
        if s in self.counter or len(self.counter) < COUNTER_CAP:
            self.counter[s] += 1
        if k in ("integer", "float"):
            x = float(v)
            self.num_min = x if self.num_min is None else min(self.num_min, x)
            self.num_max = x if self.num_max is None else max(self.num_max, x)
            self.num_sum += x
            self.num_n += 1
        elif k in ("datetime", "time"):
            if isinstance(v, date) and not isinstance(v, datetime):
                v = datetime(v.year, v.month, v.day)
            try:
                self.other_min = v if self.other_min is None or v < self.other_min else self.other_min
                self.other_max = v if self.other_max is None or v > self.other_max else self.other_max
            except TypeError:
                pass
        n = len(s)
        self.len_sum += n
        self.len_max = max(self.len_max, n)

    def result(self, rows: int) -> dict:
        dtype = "empty"
        if self.non_null:
            top_kind, top_n = self.kinds.most_common(1)[0]
            numeric = self.kinds["integer"] + self.kinds["float"]
            if top_n / self.non_null >= 0.9:
                dtype = top_kind
                if top_kind == "integer" and self.kinds["float"]:
                    dtype = "float"
            elif numeric / self.non_null >= 0.9:
                dtype = "float"
            else:
                dtype = "mixed"
        distinct = len(self.distinct)
        top = self.counter.most_common(5)
        out = {
            "name": self.name,
            "position": self.position,
            "dtype": dtype,
            "non_null": self.non_null,
            "null_pct": round(100 * (1 - self.non_null / rows), 1) if rows else 0.0,
            "distinct": distinct,
            "distinct_capped": self.distinct_capped,
            # exact when not capped; past the cap: no repeated value seen among the tracked ones -> very likely unique
            "is_unique": self.non_null > 1 and (distinct == self.non_null if not self.distinct_capped
                                                else bool(top) and top[0][1] == 1),
            "examples": self.examples,
            "top_values": [f"{v} ({c})" for v, c in top] if top and top[0][1] > 1 else [],
            # full value list for low-cardinality columns: the filter vocabulary (e.g. role = 'Fire Safety Officer')
            "categories": [v for v, _ in self.counter.most_common(CATEGORY_MAX)]
                          if not self.distinct_capped and 0 < distinct <= CATEGORY_MAX else [],
            "key_sample": self.key_sample if dtype in ("text", "integer") and self.len_max <= 64 else [],
            "min": None, "max": None, "mean": None,
            "avg_len": round(self.len_sum / self.non_null, 1) if self.non_null else 0.0,
        }
        if self.num_n and dtype in ("integer", "float"):
            out.update(min=_fmt(self.num_min), max=_fmt(self.num_max), mean=round(self.num_sum / self.num_n, 4))
        elif self.other_min is not None:
            out.update(min=_fmt(self.other_min), max=_fmt(self.other_max))
        return out


def profile_file(path: Path, progress: Callable[[int, int | None], None] | None = None) -> dict:
    """Stream the file once and return {sheet, other_sheets, row_count, columns:[...]}."""
    sheet, header, rows, total, others = open_table(path)
    is_text_source = path.suffix.lower() in (".csv", ".tsv")
    cols = [ColumnStats(name=h, position=i) for i, h in enumerate(header)]
    width = len(cols)
    n = 0
    for raw in rows:
        cells = [_clean(c) for c in raw[:width]]
        if not any(c is not None for c in cells):
            continue
        n += 1
        for col, v in zip(cols, cells):
            if v is None:
                continue
            if is_text_source and isinstance(v, str):
                v = _parse_text(v)
            col.add(v)
        if progress and n % 20_000 == 0:
            progress(n, total)
    if progress:
        progress(n, n)
    columns = [c.result(n) for c in cols]
    # drop trailing auto-named columns that are completely empty (Excel formatting artefacts)
    while columns and columns[-1]["dtype"] == "empty" and columns[-1]["name"].startswith("column_"):
        columns.pop()
    return {"sheet": sheet, "other_sheets": others, "row_count": n, "columns": columns}
