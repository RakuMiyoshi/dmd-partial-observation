"""Atomic file output and provenance records for the analysis scripts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                  default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.csv.gz" if path.name.endswith(".csv.gz") else ".tmp.csv"
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{suffix}")
    try:
        compression = "gzip" if path.name.endswith(".gz") else None
        frame.to_csv(tmp, index=False, compression=compression)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)



def provenance(sources: Iterable[Path]) -> dict[str, Any]:
    """Git commit, uncommitted-change digest, and SHA-256 of the given sources."""
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=REPOSITORY_ROOT, check=True, text=True,
                                  capture_output=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tracked = ("src", "scripts", "figures", "configs")
    diff = command("git", "diff", "--binary", "HEAD", "--", *tracked)
    return {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_diff_sha256": None if diff is None else hashlib.sha256(
            diff.encode("utf-8")).hexdigest(),
        "git_status": command("git", "status", "--short", "--", *tracked),
        "source_sha256": {str(Path(path).resolve().relative_to(REPOSITORY_ROOT)):
                          sha256_file(Path(path))
                          for path in sources if Path(path).exists()},
    }
