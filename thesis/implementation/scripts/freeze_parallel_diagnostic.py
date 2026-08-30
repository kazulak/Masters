#!/usr/bin/env python3
"""Build and verify a portable bundle for the frozen parallel diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "7e3ca432a3b109da15710b32dcb1edec9e4771fb"
EXPECTED_TAG = "thesis-upmem-hierarchical-parallel-diagnostic-v1"
ALLOWED_DESCENDANT_PATHS = {
    "README.md",
    "STATUS.md",
    "docs/evidence_workflow.md",
    "docs/hierarchical_parallel_diagnostic.md",
    "scripts/freeze_parallel_diagnostic.py",
    "scripts/inspect_parallel_scaling.py",
    "src/quantum_bench/report.py",
    "tests/test_cli_report.py",
    "tests/test_qualification.py",
}

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_parallel_scaling import derive_summary  # noqa: E402
from quantum_bench.evidence import load_artifacts  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _copy(source: Path, staging: Path, relative: str) -> None:
    if not source.is_file():
        raise ValueError(f"required bundle input is missing: {source}")
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _check_descendant_paths(source_sha: str, reporting_sha: str) -> None:
    changed = {
        path
        for path in _git("diff", "--name-only", f"{source_sha}..{reporting_sha}").splitlines()
        if path
    }
    unexpected = sorted(changed - ALLOWED_DESCENDANT_PATHS)
    if unexpected:
        raise ValueError(
            "reporting descendant changed forbidden paths: " + ", ".join(unexpected)
        )


def _verify_inputs(
    *,
    raw_stage: Path,
    report_dir: Path,
    summary_path: Path,
    incident_path: Path,
    execution_tag: str,
    reporting_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    source_sha = _git("rev-list", "-n1", execution_tag)
    if source_sha != EXPECTED_SOURCE:
        raise ValueError(f"execution tag resolves to {source_sha}, expected {EXPECTED_SOURCE}")
    if _git("rev-parse", "HEAD") != reporting_source_commit:
        raise ValueError("reporting source commit does not match current HEAD")
    _check_descendant_paths(source_sha, reporting_source_commit)

    evidence = raw_stage / "evidence"
    if not evidence.is_dir():
        evidence = raw_stage
    verification = verify_artifacts(evidence)
    required_verification = {
        "status": "completed",
        "sample_count": 36,
        "session_count": 36,
        "success_count": 36,
        "failed_count": 0,
        "unsupported_count": 0,
        "accuracy_qualified": True,
    }
    for field, expected in required_verification.items():
        if verification.get(field) != expected:
            raise ValueError(f"canonical evidence {field} is {verification.get(field)!r}")
    manifest, samples, sessions = load_artifacts(evidence)
    summary = derive_summary(
        manifest=manifest,
        samples=samples,
        sessions=sessions,
        expected_source_commit=source_sha,
        reporting_tool_source_commit=reporting_source_commit,
    )
    supplied_summary = _json(summary_path)
    if supplied_summary != json.loads(json.dumps(summary, sort_keys=True)):
        raise ValueError("supplied parallel summary does not match canonical evidence")
    if summary.get("gate_passed") is not True:
        raise ValueError("parallel diagnostic gate did not pass")
    if summary.get("claim_policy") != "diagnostic_v1":
        raise ValueError("parallel diagnostic claim policy changed")
    if summary.get("claim_eligible") is not False:
        raise ValueError("diagnostic summary unexpectedly became claim eligible")
    if not incident_path.is_file():
        raise ValueError(f"incident record is missing: {incident_path}")
    required_reports = (
        "report.json",
        "aggregate.csv",
        "scaling.csv",
        "speedups.csv",
        "route_statistics.csv",
        "parallel_comparisons_descriptive.csv",
        "accuracy_summary.csv",
        "tasklet_runtime.png",
        "tasklet_speedup.png",
        "dpu_runtime.png",
        "dpu_speedup.png",
        "transfer_by_route.png",
    )
    for filename in required_reports:
        if not (report_dir / filename).is_file():
            raise ValueError(f"derived report input is missing: {report_dir / filename}")
    return manifest, summary, tuple(samples), tuple(sessions)


def _write_checksums(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _verify_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"bundle checksum mismatch: {relative}")


def _safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
        stream.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("bundle must contain exactly one root directory")
    return roots[0]


def _verify_archive(
    archive: Path, archive_digest: Path, source_sha: str, reporting_sha: str
) -> None:
    expected = archive_digest.read_text(encoding="utf-8").split()[0]
    if expected != _sha256(archive):
        raise ValueError("outer bundle checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="parallel-bundle-verify-") as directory:
        root = _safe_extract(archive, Path(directory))
        _verify_checksums(root)
        verification = verify_artifacts(root / "evidence")
        if verification.get("success_count") != 36:
            raise ValueError("extracted canonical evidence is incomplete")
        manifest, samples, sessions = load_artifacts(root / "evidence")
        summary = derive_summary(
            manifest=manifest,
            samples=samples,
            sessions=sessions,
            expected_source_commit=source_sha,
            reporting_tool_source_commit=reporting_sha,
        )
        if summary.get("gate_passed") is not True:
            raise ValueError("extracted parallel summary did not pass")


def build_bundle(args: argparse.Namespace) -> Path:
    reporting_sha = args.reporting_source_commit or _git("rev-parse", "HEAD")
    raw_stage = args.raw_stage.resolve()
    report_dir = args.report_dir.resolve()
    summary_path = args.summary.resolve()
    incident_path = args.incident.resolve()
    manifest, summary, _, _ = _verify_inputs(
        raw_stage=raw_stage,
        report_dir=report_dir,
        summary_path=summary_path,
        incident_path=incident_path,
        execution_tag=args.execution_tag,
        reporting_source_commit=reporting_sha,
    )
    evidence = raw_stage / "evidence"
    if not evidence.is_dir():
        evidence = raw_stage
    archive = args.output.resolve()
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        raise ValueError(f"bundle output already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="parallel-bundle-") as directory:
        staging = Path(directory) / "thesis-upmem-hierarchical-parallel-diagnostic-v1"
        staging.mkdir()
        for filename in ("manifest.json", "samples.jsonl", "sessions.jsonl"):
            _copy(evidence / filename, staging, f"evidence/{filename}")
        for filename in (
            "report.json",
            "aggregate.csv",
            "scaling.csv",
            "speedups.csv",
            "route_statistics.csv",
            "parallel_comparisons_descriptive.csv",
            "accuracy_summary.csv",
            "tasklet_runtime.png",
            "tasklet_speedup.png",
            "dpu_runtime.png",
            "dpu_speedup.png",
            "transfer_by_route.png",
        ):
            _copy(report_dir / filename, staging, f"report/{filename}")
        _copy(summary_path, staging, "report/parallel_scaling_summary.json")
        _copy(ROOT / "docs" / "hierarchical_parallel_diagnostic.md", staging, "docs/hierarchical_parallel_diagnostic.md")
        _copy(incident_path, staging, "incident.json")
        prepared = raw_stage / "prepared-config.yml"
        if prepared.is_file():
            _copy(prepared, staging, "provenance/prepared-config.yml")
        provenance = raw_stage / "provenance"
        if provenance.is_dir():
            for path in sorted(provenance.iterdir()):
                if path.is_file():
                    _copy(path, staging, f"provenance/{path.name}")
        metadata = {
            "physical_execution_source_commit": EXPECTED_SOURCE,
            "reporting_tool_source_commit": reporting_sha,
            "execution_tag": args.execution_tag,
            "experiment_id": manifest["experiment_id"],
            "run_id": manifest["run_id"],
            "raw_evidence_sha256s_sha256": _sha256(raw_stage / "SHA256SUMS")
            if (raw_stage / "SHA256SUMS").is_file()
            else None,
            "claim_policy": summary["claim_policy"],
            "claim_eligible": summary["claim_eligible"],
            "claim_ineligibility_reason": summary["claim_ineligibility_reason"],
            "allowed_claims": [
                "physical correctness",
                "descriptive tasklet and one-rank DPU scaling",
                "steady-execution timing composition",
            ],
            "prohibited_claims": [
                "final physical_performance_v1 estimates",
                "general UPMEM acceleration",
                "optimized-host performance",
                "multi-rank scaling",
                "energy efficiency",
            ],
        }
        (staging / "baseline_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_checksums(staging)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(staging, arcname=staging.name)

    digest_path = archive.with_name(archive.name + ".sha256")
    digest_path.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    _verify_archive(archive, digest_path, EXPECTED_SOURCE, reporting_sha)
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stage", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--execution-tag", default=EXPECTED_TAG)
    parser.add_argument("--reporting-source-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(build_bundle(args))
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
