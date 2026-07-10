from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SNAPSHOT = ROOT / "thesis_results" / "current" / "snapshot_manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or prune generated thesis runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    prune = subparsers.add_parser("prune")
    prune.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        list_runs()
        return 0
    return prune_runs(apply=bool(args.apply))


def list_runs() -> None:
    rows = _run_rows()
    headings = ("kind", "suite", "route/type", "run", "records", "size", "selected")
    widths = [len(value) for value in headings]
    values: list[tuple[str, ...]] = []
    for row in rows:
        item = (
            row["kind"],
            row["suite"],
            row["route"],
            row["run_id"],
            str(row["record_count"]),
            _human_bytes(row["size_bytes"]),
            "yes" if row["selected"] else "",
        )
        values.append(item)
        widths = [max(width, len(value)) for width, value in zip(widths, item)]
    print("  ".join(value.ljust(width) for value, width in zip(headings, widths)))
    for item in values:
        print("  ".join(value.ljust(width) for value, width in zip(item, widths)))


def prune_runs(*, apply: bool) -> int:
    selected = _selected_paths(require_snapshot=True)
    candidates = [row for row in _run_rows() if row["path"].resolve() not in selected]
    for row in candidates:
        prefix = "REMOVE" if apply else "WOULD REMOVE"
        print(f"{prefix} {row['path'].relative_to(ROOT)} ({_human_bytes(row['size_bytes'])})")
    if apply:
        for row in candidates:
            shutil.rmtree(row["path"], ignore_errors=True)
        _remove_legacy_roots()
        _remove_broken_links(RUNS)
    else:
        print("Dry run only. Re-run with --apply to remove stale generated runs.")
    return 0


def _run_rows() -> list[dict[str, Any]]:
    selected = _selected_paths(require_snapshot=False)
    rows: list[dict[str, Any]] = []
    evidence_root = RUNS / "evidence"
    if evidence_root.exists():
        for manifest_path in evidence_root.glob("*/*/*/run_manifest.json"):
            run_dir = manifest_path.parent
            manifest = _read_json(manifest_path)
            rows.append(
                _row(
                    "evidence",
                    str(manifest.get("suite_id") or run_dir.parents[1].name),
                    str(manifest.get("route_label") or run_dir.parent.name),
                    run_dir,
                    selected,
                )
            )
    comparison_root = RUNS / "comparisons"
    if comparison_root.exists():
        for manifest_path in comparison_root.glob("*/*/benchmark_manifest.json"):
            run_dir = manifest_path.parent
            rows.append(_row("comparison", run_dir.parent.name, "research_pack", run_dir, selected))
        for manifest_path in comparison_root.glob("*/*/*/comparison_manifest.json"):
            run_dir = manifest_path.parent
            rows.append(_row("comparison", run_dir.parents[1].name, run_dir.parent.name, run_dir, selected))
    return sorted(rows, key=lambda row: (row["kind"], row["suite"], row["route"], row["run_id"]))


def _row(kind: str, suite: str, route: str, path: Path, selected: set[Path]) -> dict[str, Any]:
    records = path / "normalized_records.jsonl"
    record_count = sum(1 for line in records.read_text(encoding="utf-8").splitlines() if line.strip()) if records.exists() else 0
    return {
        "kind": kind,
        "suite": suite,
        "route": route,
        "run_id": path.name,
        "record_count": record_count,
        "size_bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
        "selected": path.resolve() in selected,
        "path": path,
    }


def _selected_paths(*, require_snapshot: bool) -> set[Path]:
    if not SNAPSHOT.exists():
        if require_snapshot:
            raise ValueError("create and verify thesis_results/current before pruning runs")
        return set()
    manifest = _read_json(SNAPSHOT)
    selected = {
        (ROOT / str(entry["source_run"])).resolve()
        for entry in manifest.get("selected_evidence") or ()
        if not Path(str(entry["source_run"])).is_absolute()
    }
    source_pack = manifest.get("source_pack")
    if source_pack and not Path(str(source_pack)).is_absolute():
        selected.add((ROOT / str(source_pack)).resolve())
    return selected


def _remove_legacy_roots() -> None:
    if not RUNS.exists():
        return
    for path in RUNS.iterdir():
        if path.name in {"evidence", "comparisons", "latest"} or path.is_symlink():
            continue
        if path.is_dir():
            print(f"REMOVE legacy {path.relative_to(ROOT)}")
            shutil.rmtree(path)


def _remove_broken_links(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("latest"):
        if path.is_symlink() and not path.exists():
            path.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GiB"


if __name__ == "__main__":
    raise SystemExit(main())
