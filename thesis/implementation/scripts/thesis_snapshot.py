from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.research_benchmark_pack import report_pack  # noqa: E402


SNAPSHOT_SCHEMA_VERSION = "thesis_result_snapshot_v1"
ROLE_BY_SUITE = {
    "research_cpu_gpu_correctness": "full_state_correctness",
    "research_cpu_gpu": "full_state_performance",
    "research_cpu_tn": "cpu_tn",
    "research_planner_compare": "planner_paths",
    "thesis_upmem_quantization_boundary": "upmem_generic_boundary",
    "thesis_upmem_quantization_stress": "upmem_quantization_stress",
    "research_internal_parallelism": "internal_parallelism",
}
REQUIRED_EVIDENCE_FILES = ("run_manifest.json", "environment.json", "normalized_records.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote one research pack into tracked thesis results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--pack", type=Path, default=ROOT / "runs" / "comparisons" / "research_pack" / "latest")
    promote.add_argument("--out", type=Path, default=ROOT / "thesis_results" / "current")
    promote.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", type=Path, default=ROOT / "thesis_results" / "current")

    report = subparsers.add_parser("report")
    report.add_argument("--snapshot", type=Path, default=ROOT / "thesis_results" / "current")

    release = subparsers.add_parser("release")
    release.add_argument("--name", required=True)
    release.add_argument("--snapshot", type=Path, default=ROOT / "thesis_results" / "current")
    release.add_argument("--releases", type=Path, default=ROOT / "thesis_results" / "releases")

    args = parser.parse_args(argv)
    if args.command == "promote":
        return promote_snapshot(args.pack, args.out, allow_dirty=bool(args.allow_dirty))
    if args.command == "verify":
        verify_snapshot(args.snapshot)
        print(args.snapshot.resolve())
        return 0
    if args.command == "report":
        return regenerate_snapshot_report(args.snapshot)
    if args.command == "release":
        return release_snapshot(args.snapshot, args.releases, args.name)
    raise AssertionError(args.command)


def promote_snapshot(pack: Path, out: Path, *, allow_dirty: bool = False) -> int:
    pack = pack.resolve()
    manifest_path = pack / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"research pack manifest missing: {manifest_path}")
    pack_manifest = _read_json(manifest_path)
    evidence_inputs = [Path(value).resolve() for value in pack_manifest.get("evidence_inputs") or ()]
    if not evidence_inputs:
        raise ValueError("research pack contains no evidence inputs")

    head = _git("rev-parse", "HEAD")
    dirty_worktree = bool(_git("status", "--short", "--", "."))
    if not allow_dirty and dirty_worktree:
        raise ValueError("tracked thesis snapshots require a clean worktree")
    if not allow_dirty and str(pack_manifest.get("git_commit") or "") != head:
        raise ValueError("research pack git commit does not match current HEAD")
    if not allow_dirty and bool(pack_manifest.get("dirty_worktree")):
        raise ValueError("research pack was generated from a dirty worktree")

    staging = out.parent / f".{out.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "evidence").mkdir(parents=True)
    (staging / "suites").mkdir(parents=True)
    selected: list[dict[str, Any]] = []
    used_roles: set[str] = set()
    for source in evidence_inputs:
        entry = _copy_evidence_capsule(source, staging, used_roles)
        if not allow_dirty and entry["git_commit"] != head:
            raise ValueError(f"evidence commit mismatch for {entry['role']}")
        selected.append(entry)

    report_dir = staging / ".report"
    report_inputs = [staging / entry["snapshot_path"] for entry in selected]
    if report_pack(ROOT, report_dir, inputs=report_inputs, suite_filter=None) != 0:
        raise RuntimeError("snapshot report generation failed")
    _install_report(report_dir, staging, provenance=pack_manifest)

    snapshot_manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": "current",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "dirty_worktree": dirty_worktree,
        "evidence_level": "normalized_report_regenerable",
        "raw_tensor_artifacts_included": False,
        "selected_evidence": selected,
        "source_pack": _relative_or_string(pack),
        "report_entrypoint": "README.md",
        "tables_dir": "tables",
        "plots_dir": "plots",
    }
    _write_json(staging / "snapshot_manifest.json", snapshot_manifest)
    _write_json(staging / "checksums.json", _checksums(staging, exclude={"checksums.json"}))
    verify_snapshot(staging)

    backup = out.parent / f".{out.name}.previous"
    shutil.rmtree(backup, ignore_errors=True)
    if out.exists():
        out.rename(backup)
    staging.rename(out)
    shutil.rmtree(backup, ignore_errors=True)
    print(out.resolve())
    return 0


def verify_snapshot(snapshot: Path) -> None:
    snapshot = snapshot.resolve()
    manifest = _read_json(snapshot / "snapshot_manifest.json")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported thesis snapshot schema")
    selected = list(manifest.get("selected_evidence") or ())
    if not selected:
        raise ValueError("snapshot contains no selected evidence")
    for entry in selected:
        evidence = snapshot / str(entry["snapshot_path"])
        for name in REQUIRED_EVIDENCE_FILES:
            if not (evidence / name).is_file():
                raise ValueError(f"snapshot evidence file missing: {evidence / name}")
        if not any((evidence / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines()):
            raise ValueError(f"snapshot has empty normalized records: {evidence}")
    for forbidden in ("*.npy", "*.bin", "*.pyc"):
        if next(snapshot.rglob(forbidden), None) is not None:
            raise ValueError(f"snapshot contains forbidden artifact type: {forbidden}")
    if next(snapshot.rglob("runner_work"), None) is not None:
        raise ValueError("snapshot contains runner_work")
    if any(path.is_symlink() for path in snapshot.rglob("*")):
        raise ValueError("snapshot must be self-contained and cannot contain symlinks")
    if not (snapshot / "README.md").is_file() or not (snapshot / "plots").is_dir() or not (snapshot / "tables").is_dir():
        raise ValueError("snapshot report, tables, or plots are missing")
    expected = _read_json(snapshot / "checksums.json") if (snapshot / "checksums.json").exists() else None
    if expected is not None:
        actual = _checksums(snapshot, exclude={"checksums.json"})
        if expected != actual:
            raise ValueError("snapshot checksum verification failed")


def regenerate_snapshot_report(snapshot: Path, *, root: Path = ROOT) -> int:
    verify_snapshot(snapshot)
    snapshot = snapshot.resolve()
    manifest = _read_json(snapshot / "snapshot_manifest.json")
    inputs = [snapshot / str(entry["snapshot_path"]) for entry in manifest["selected_evidence"]]
    report_dir = snapshot.parent / f".{snapshot.name}.report"
    shutil.rmtree(report_dir, ignore_errors=True)
    if report_pack(root, report_dir, inputs=inputs, suite_filter=None) != 0:
        raise RuntimeError("snapshot report regeneration failed")
    shutil.rmtree(snapshot / "tables", ignore_errors=True)
    shutil.rmtree(snapshot / "plots", ignore_errors=True)
    for name in ("README.md", "plot_manifest.json", "report_manifest.json"):
        (snapshot / name).unlink(missing_ok=True)
    _install_report(report_dir, snapshot, provenance=manifest)
    _write_json(snapshot / "checksums.json", _checksums(snapshot, exclude={"checksums.json"}))
    verify_snapshot(snapshot)
    print(snapshot)
    return 0


def release_snapshot(snapshot: Path, releases: Path, name: str) -> int:
    verify_snapshot(snapshot)
    safe_name = _safe(name)
    target = releases / safe_name
    if target.exists():
        raise ValueError(f"release already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, target)
    manifest = _read_json(target / "snapshot_manifest.json")
    manifest["snapshot_id"] = safe_name
    _write_json(target / "snapshot_manifest.json", manifest)
    _write_json(target / "checksums.json", _checksums(target, exclude={"checksums.json"}))
    verify_snapshot(target)
    print(target.resolve())
    return 0


def _copy_evidence_capsule(source: Path, staging: Path, used_roles: set[str]) -> dict[str, Any]:
    for name in REQUIRED_EVIDENCE_FILES:
        if not (source / name).is_file():
            raise ValueError(f"required evidence file missing: {source / name}")
    run_manifest = _read_json(source / "run_manifest.json")
    suite_id = str(run_manifest.get("suite_id") or source.parent.parent.name)
    role = ROLE_BY_SUITE.get(suite_id, _safe(suite_id))
    if role in used_roles:
        raise ValueError(f"duplicate snapshot evidence role: {role}")
    used_roles.add(role)
    destination = staging / "evidence" / role
    destination.mkdir(parents=True)
    copied = list(REQUIRED_EVIDENCE_FILES)
    summary_name = run_manifest.get("summary")
    if summary_name and (source / str(summary_name)).is_file():
        copied.append(str(summary_name))
    suite_source = source / "config" / "resolved_suite.yml"
    for name in copied:
        shutil.copy2(source / name, destination / Path(name).name)
    if suite_source.is_file():
        suite_name = f"{role}.yml"
        shutil.copy2(suite_source, staging / "suites" / suite_name)
    _write_json(destination / "checksums.json", _checksums(destination, exclude={"checksums.json"}))
    records = sum(1 for line in (source / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "role": role,
        "suite_id": suite_id,
        "run_id": str(run_manifest.get("run_id") or source.name),
        "git_commit": str(run_manifest.get("git_commit") or ""),
        "record_count": records,
        "source_run": _relative_or_string(source),
        "snapshot_path": f"evidence/{role}",
    }


def _install_report(report_dir: Path, staging: Path, *, provenance: dict[str, Any]) -> None:
    tables = staging / "tables"
    tables.mkdir()
    plots = report_dir / "plots"
    if plots.is_dir():
        shutil.move(str(plots), staging / "plots")
    else:
        (staging / "plots").mkdir()
    for csv_path in sorted(report_dir.glob("*.csv")):
        shutil.move(str(csv_path), tables / csv_path.name)
    shutil.copy2(report_dir / "benchmark_summary.md", staging / "README.md")
    shutil.copy2(report_dir / "plot_manifest.json", staging / "plot_manifest.json")
    report_manifest = _read_json(report_dir / "benchmark_manifest.json")
    report_manifest["root"] = "."
    report_manifest["evidence_inputs"] = sorted(path.relative_to(staging).as_posix() for path in (staging / "evidence").iterdir())
    report_manifest["command_line"] = "make thesis-report"
    report_manifest["git_commit"] = provenance.get("git_commit")
    report_manifest["dirty_worktree"] = bool(provenance.get("dirty_worktree", False))
    report_manifest["provenance_scope"] = "selected_evidence"
    _write_json(staging / "report_manifest.json", report_manifest)
    shutil.rmtree(report_dir)


def _checksums(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _relative_or_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("_") or "snapshot"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
