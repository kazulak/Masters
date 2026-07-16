from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SNAPSHOT = ROOT / "thesis_results" / "current" / "snapshot_manifest.json"
ARCHIVE_MANIFEST_NAMES = (
    "run_manifest.json",
    "benchmark_manifest.json",
    "comparison_manifest.json",
)


def _repository_root() -> Path:
    """Return the checkout root from the active implementation root.

    Keep this derived rather than cached: tests and maintenance tooling can
    intentionally replace ``ROOT`` with an isolated checkout.
    """
    return ROOT.resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or safely manage generated thesis runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    prune = subparsers.add_parser("prune")
    prune.add_argument("--apply", action="store_true")
    archive = subparsers.add_parser("archive", help="copy one run outside the repository")
    archive.add_argument("path", type=Path)
    archive.add_argument("--archive-root", type=Path)
    archive.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        list_runs()
        return 0
    if args.command == "archive":
        archive_run(args.path, apply=bool(args.apply), archive_root=args.archive_root)
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
            if run_dir.is_symlink() or run_dir.name == "latest":
                continue
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
            if run_dir.is_symlink() or run_dir.name == "latest":
                continue
            rows.append(_row("comparison", run_dir.parent.name, "research_pack", run_dir, selected))
        for manifest_path in comparison_root.glob("*/*/*/comparison_manifest.json"):
            run_dir = manifest_path.parent
            if run_dir.is_symlink() or run_dir.name == "latest":
                continue
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
    manifests = _snapshot_manifests()
    if not manifests:
        if require_snapshot:
            raise ValueError("create and verify a thesis_results snapshot before pruning runs")
        return set()
    selected: set[Path] = set()
    for snapshot in manifests:
        manifest = _read_json(snapshot)
        for entry in manifest.get("selected_evidence") or ():
            source = Path(str(entry["source_run"]))
            selected.add((source if source.is_absolute() else ROOT / source).resolve())
        source_pack = manifest.get("source_pack")
        if source_pack:
            source = Path(str(source_pack))
            selected.add((source if source.is_absolute() else ROOT / source).resolve())
    # A current `latest` link is an intentional active working result, even
    # before it is promoted to a tracked snapshot. Keep its target so cleanup
    # preserves the one obvious current run per namespace.
    if RUNS.exists():
        for latest in RUNS.rglob("latest"):
            if latest.is_symlink() and latest.exists():
                selected.add(latest.resolve())
    return selected


def _snapshot_manifests() -> list[Path]:
    root = ROOT / "thesis_results"
    if not root.exists():
        return [SNAPSHOT] if SNAPSHOT.exists() else []
    return sorted(
        path for path in root.rglob("snapshot_manifest.json")
        if path.is_file() and not path.is_symlink()
    )


def archive_run(
    run_path: str | Path,
    *,
    apply: bool = False,
    archive_root: str | Path | None = None,
) -> Path:
    """Archive one explicitly named run, deleting it only after verified apply."""
    source_input = Path(run_path).expanduser()
    source_input = ROOT / source_input if not source_input.is_absolute() else source_input
    if source_input.is_symlink():
        raise ValueError(f"run path must be a real directory: {source_input}")
    source = source_input.resolve()
    root = _repository_root()
    if not source.is_dir():
        raise ValueError(f"run path must be a real directory: {source}")
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("run path must be inside the implementation repository") from exc
    if not any((source / name).is_file() for name in ARCHIVE_MANIFEST_NAMES):
        raise ValueError(
            "archive source must be an explicit evidence or comparison run directory "
            "containing a run manifest"
        )
    destination_root = (
        Path(archive_root).expanduser().resolve()
        if archive_root is not None
        else root.parent / "thesis-evidence-archive"
    )
    try:
        destination_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("archive root must be outside the implementation repository")
    destination = destination_root / relative
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"archive destination already exists: {destination}")
    checksums = _file_checksums(source)
    action = "ARCHIVE" if apply else "WOULD ARCHIVE"
    print(f"{action} {source.relative_to(root)} -> {destination}")
    if not apply:
        print("Dry run only. Re-run with --apply to copy, verify, and remove the original.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        if _file_checksums(destination) != checksums:
            raise ValueError("archive checksum verification failed; original was retained")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    shutil.rmtree(source)
    return destination


def _file_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[path.relative_to(root).as_posix()] = digest
    return checksums


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
