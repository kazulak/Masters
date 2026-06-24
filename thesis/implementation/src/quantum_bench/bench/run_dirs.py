from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def create_run_dir(root_dir: Path, suite_id: str) -> Path:
    runs_dir = root_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{sanitize(suite_id)}"
    run_dir = runs_dir / base
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{base}_{suffix:02d}"
        suffix += 1
    for rel in ("config", "cases", "raw", "validation", "metrics", "plots"):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    update_latest_symlink(runs_dir, run_dir)
    return run_dir


def update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    latest = runs_dir / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            return
        latest.symlink_to(run_dir.name)
    except OSError:
        return


def resolve_run_dir(path: Path) -> Path:
    return path.resolve()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "suite"
