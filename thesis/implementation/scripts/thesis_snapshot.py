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

from scripts.research_benchmark_pack import planner_semantic_context, report_pack  # noqa: E402


SNAPSHOT_SCHEMA_VERSION = "thesis_result_snapshot_v1"
BENCHMARK_SOURCE_SCOPE = "thesis/implementation"
PROVENANCE_ALIAS_DESCRIPTION = {
    "git_commit": "alias of benchmark_source_commit",
    "dirty_tree": "alias of benchmark_source_worktree_dirty",
    "dirty_worktree": "alias of benchmark_source_worktree_dirty",
}
ROLE_BY_SUITE = {
    "thesis_full_state_correctness": "full_state_correctness",
    "thesis_full_state_cpu_gpu": "full_state_performance",
    "thesis_cpu_tn_quimb": "cpu_tn",
    "thesis_tn_paths_quantization": "tn_path_quantization",
    "thesis_planner_compare": "planner_paths",
    "thesis_planner_sensitivity": "planner_sensitivity",
    "thesis_planner_semantic_v2": "planner_paths_v2",
    "thesis_planner_sensitivity_v2": "planner_sensitivity_v2",
    "research_cpu_gpu_correctness": "full_state_correctness",
    "research_cpu_gpu": "full_state_performance",
    "research_cpu_tn": "cpu_tn",
    "research_planner_compare": "planner_paths",
    "thesis_upmem_quantization_boundary": "upmem_generic_boundary",
    "thesis_upmem_quantization_stress": "upmem_quantization_stress",
    "research_internal_parallelism": "internal_parallelism",
    "upmem_hardware_mvp": "upmem_hardware_functionality",
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

    promotion_stage = _current_provenance()
    report_stage = dict(promotion_stage)
    head = promotion_stage["commit"]
    if promotion_stage["benchmark_source_worktree_dirty"]:
        raise ValueError("tracked thesis snapshots require a clean thesis/implementation source")
    source_manifest = _manifest_provenance(pack_manifest, fallback_commit=head)
    if not allow_dirty and source_manifest["commit"] != head:
        raise ValueError("research pack git commit does not match current HEAD")
    if source_manifest["worktree_dirty"]:
        raise ValueError("research pack was generated from dirty thesis/implementation source")

    staging = out.parent / f".{out.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "evidence").mkdir(parents=True)
    (staging / "suites").mkdir(parents=True)
    selected: list[dict[str, Any]] = []
    used_roles: set[str] = set()
    for source in evidence_inputs:
        entry = _copy_evidence_capsule(source, staging, used_roles)
        if not allow_dirty and entry["benchmark_source_commit"] != head:
            raise ValueError(f"evidence commit mismatch for {entry['role']}")
        if entry["benchmark_source_worktree_dirty"]:
            raise ValueError(f"evidence source is dirty for {entry['role']}")
        selected.append(entry)

    report_dir = staging / ".report"
    report_inputs = [staging / entry["snapshot_path"] for entry in selected]
    if report_pack(ROOT, report_dir, inputs=report_inputs, suite_filter=None) != 0:
        raise RuntimeError("snapshot report generation failed")
    _install_report(report_dir, staging, provenance=pack_manifest, report_stage=report_stage)
    report_manifest = _read_json(staging / "report_manifest.json")
    planner_semantics = report_manifest.get("planner_semantics") or planner_semantic_context([])

    source_commit = source_manifest["commit"]
    source_dirty = source_manifest["worktree_dirty"]
    source_repository_dirty = source_manifest["repository_worktree_dirty"] or any(
        entry["repository_worktree_dirty"] for entry in selected
    )
    provenance_stages = {
        "benchmark_source": _stage(source_commit, source_dirty, source_repository_dirty),
        "report_generation": _stage_from_provenance(report_stage),
        "snapshot_promotion": _stage_from_provenance(promotion_stage),
    }

    snapshot_manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": "current",
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Legacy aliases remain source-scoped; repository dirtiness is carried
        # separately so unrelated files cannot masquerade as dirty evidence.
        "git_commit": source_commit,
        "dirty_tree": source_dirty,
        "dirty_worktree": source_dirty,
        "benchmark_source_commit": source_commit,
        "benchmark_source_worktree_dirty": source_dirty,
        "repository_worktree_dirty": promotion_stage["repository_worktree_dirty"],
        "provenance_scope": BENCHMARK_SOURCE_SCOPE,
        "provenance_aliases": PROVENANCE_ALIAS_DESCRIPTION,
        "report_generation_commit": report_stage["commit"],
        "report_generation_worktree_dirty": report_stage["benchmark_source_worktree_dirty"],
        "snapshot_promotion_commit": promotion_stage["commit"],
        "snapshot_promotion_worktree_dirty": promotion_stage["benchmark_source_worktree_dirty"],
        "provenance_stages": provenance_stages,
        "evidence_level": "normalized_report_regenerable",
        "raw_tensor_artifacts_included": False,
        "selected_evidence": selected,
        "planner_semantics": planner_semantics,
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
    records: list[dict[str, Any]] = []
    for entry in selected:
        evidence = snapshot / str(entry["snapshot_path"])
        for name in REQUIRED_EVIDENCE_FILES:
            if not (evidence / name).is_file():
                raise ValueError(f"snapshot evidence file missing: {evidence / name}")
        if not any((evidence / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines()):
            raise ValueError(f"snapshot has empty normalized records: {evidence}")
        for line in (evidence / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        evidence_manifest = _read_json(evidence / "run_manifest.json")
        # Legacy snapshots predate the scoped field and may contain a stale
        # dirty_tree value from the run-generation worktree. Promotion still
        # rejects that legacy evidence; verification remains compatible with
        # already-tracked v1 snapshots until they are regenerated.
        if "benchmark_source_worktree_dirty" in evidence_manifest and _manifest_provenance(evidence_manifest)["worktree_dirty"]:
            raise ValueError(f"snapshot evidence has dirty thesis/implementation source: {evidence}")
    planner_semantics = planner_semantic_context(records)
    if planner_semantics["issues"]:
        raise ValueError("snapshot planner semantic versions are mixed: " + "; ".join(planner_semantics["issues"]))
    recorded_semantics = manifest.get("planner_semantics")
    if isinstance(recorded_semantics, dict) and recorded_semantics.get("semantic_versions") != planner_semantics["semantic_versions"]:
        raise ValueError("snapshot planner semantic context does not match evidence")
    if manifest.get("benchmark_source_worktree_dirty"):
        raise ValueError("snapshot benchmark source is dirty")
    if "provenance_stages" in manifest:
        stages = manifest["provenance_stages"]
        if not all(name in stages for name in ("benchmark_source", "report_generation", "snapshot_promotion")):
            raise ValueError("snapshot provenance stages are incomplete")
        if stages["benchmark_source"].get("worktree_dirty"):
            raise ValueError("snapshot benchmark source stage is dirty")
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
    report_stage = _current_provenance()
    if report_pack(root, report_dir, inputs=inputs, suite_filter=None) != 0:
        raise RuntimeError("snapshot report regeneration failed")
    shutil.rmtree(snapshot / "tables", ignore_errors=True)
    shutil.rmtree(snapshot / "plots", ignore_errors=True)
    for name in ("README.md", "plot_manifest.json", "report_manifest.json"):
        (snapshot / name).unlink(missing_ok=True)
    _install_report(report_dir, snapshot, provenance=manifest, report_stage=report_stage)
    report_manifest = _read_json(snapshot / "report_manifest.json")
    manifest["planner_semantics"] = report_manifest.get("planner_semantics") or planner_semantic_context([])
    _write_json(snapshot / "snapshot_manifest.json", manifest)
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
    hardware_profile = source / "config" / "hardware_profile.json"
    if hardware_profile.is_file():
        shutil.copy2(hardware_profile, destination / hardware_profile.name)
    if suite_source.is_file():
        suite_name = f"{role}.yml"
        shutil.copy2(suite_source, staging / "suites" / suite_name)
    _write_json(destination / "checksums.json", _checksums(destination, exclude={"checksums.json"}))
    records = sum(1 for line in (source / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "role": role,
        "suite_id": suite_id,
        "run_id": str(run_manifest.get("run_id") or source.name),
        "git_commit": _manifest_provenance(run_manifest)["commit"],
        "dirty_tree": _manifest_provenance(run_manifest)["worktree_dirty"],
        "dirty_worktree": _manifest_provenance(run_manifest)["worktree_dirty"],
        "benchmark_source_commit": _manifest_provenance(run_manifest)["commit"],
        "benchmark_source_worktree_dirty": _manifest_provenance(run_manifest)["worktree_dirty"],
        "repository_worktree_dirty": _manifest_provenance(run_manifest)["repository_worktree_dirty"],
        "provenance_scope": _manifest_provenance(run_manifest)["scope"],
        "record_count": records,
        "source_run": _relative_or_string(source),
        "snapshot_path": f"evidence/{role}",
    }


def _install_report(
    report_dir: Path,
    staging: Path,
    *,
    provenance: dict[str, Any],
    report_stage: dict[str, Any] | None = None,
) -> None:
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
    source_provenance = _manifest_provenance(provenance)
    report_stage = report_stage or _manifest_provenance(report_manifest)
    report_manifest["root"] = "."
    report_manifest["evidence_inputs"] = sorted(path.relative_to(staging).as_posix() for path in (staging / "evidence").iterdir())
    report_manifest["command_line"] = "make thesis-report"
    report_manifest.update(
        {
            "git_commit": source_provenance["commit"],
            "dirty_tree": source_provenance["worktree_dirty"],
            "dirty_worktree": source_provenance["worktree_dirty"],
            "benchmark_source_commit": source_provenance["commit"],
            "benchmark_source_worktree_dirty": source_provenance["worktree_dirty"],
            "repository_worktree_dirty": report_stage["repository_worktree_dirty"],
            "provenance_scope": BENCHMARK_SOURCE_SCOPE,
            "provenance_aliases": PROVENANCE_ALIAS_DESCRIPTION,
            "provenance_stage": "report_generation",
            # report_pack runs inside the snapshot staging tree. Its own
            # repository probe therefore sees the intentionally untracked
            # staging files. Preserve the provenance captured before staging
            # was created so the installed report describes the actual source
            # revision rather than its temporary packaging workspace.
            "report_generation_commit": report_stage["commit"],
            "report_generation_worktree_dirty": report_stage["benchmark_source_worktree_dirty"],
            "report_generation_repository_worktree_dirty": report_stage["repository_worktree_dirty"],
            "report_generation_provenance": report_stage,
            "report_generation": report_stage,
            "provenance_stages": {
                "benchmark_source": _stage(
                    source_provenance["commit"],
                    source_provenance["worktree_dirty"],
                    source_provenance["repository_worktree_dirty"],
                ),
                "report_generation": _stage_from_provenance(report_stage),
            },
        }
    )
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


def _manifest_provenance(manifest: dict[str, Any], *, fallback_commit: str | None = None) -> dict[str, Any]:
    return {
        "commit": str(manifest.get("benchmark_source_commit") or manifest.get("git_commit") or fallback_commit or ""),
        "worktree_dirty": bool(
            manifest.get("benchmark_source_worktree_dirty")
            if manifest.get("benchmark_source_worktree_dirty") is not None
            else manifest.get("dirty_tree", manifest.get("dirty_worktree", False))
        ),
        "repository_worktree_dirty": bool(manifest.get("repository_worktree_dirty", False)),
        "scope": str(manifest.get("provenance_scope") or BENCHMARK_SOURCE_SCOPE),
    }


def _stage(commit: str, worktree_dirty: bool, repository_worktree_dirty: bool) -> dict[str, Any]:
    return {
        "commit": commit,
        "worktree_dirty": bool(worktree_dirty),
        "repository_worktree_dirty": bool(repository_worktree_dirty),
        "scope": BENCHMARK_SOURCE_SCOPE,
    }


def _stage_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return _stage(provenance["commit"], provenance["benchmark_source_worktree_dirty"], provenance["repository_worktree_dirty"])


def _current_provenance() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "benchmark_source_worktree_dirty": bool(_git("status", "--short", "--", f":(top){BENCHMARK_SOURCE_SCOPE}")),
        "repository_worktree_dirty": bool(_git("status", "--short", "--", ":(top)**")),
    }


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
