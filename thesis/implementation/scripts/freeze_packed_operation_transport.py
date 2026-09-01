#!/usr/bin/env python3
"""Build and verify the packed-operation transport adoption bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "032b3ab5ba774fed0e61fc10eb02f60814c7a190"
EXPECTED_TAG = "thesis-upmem-packed-operation-transport-v1"
EXPECTED_TRANSPORT = "packed_operation_v1"
ALLOWED_DESCENDANT_PATHS = {
    "README.md",
    "STATUS.md",
    "docs/timing.md",
    "docs/packed_operation_transport_adoption.md",
    "scripts/freeze_packed_operation_transport.py",
    "scripts/inspect_packed_operation_transport.py",
    "scripts/inspect_parallel_scaling.py",
    "scripts/qualify_general_resources.py",
    "tests/test_inspect_packed_operation_transport.py",
}

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_packed_operation_transport import inspect as inspect_packed  # noqa: E402
from quantum_bench.evidence import load_artifacts  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"required directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"required bundle input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_binaries(diagnostic_stage: Path, destination: Path) -> list[str]:
    source = (
        diagnostic_stage.parent
        / "source"
        / "thesis"
        / "implementation"
        / "native"
        / "upmem"
        / "runtime"
        / "bin"
    )
    binaries = sorted(path for path in source.glob("*") if path.is_file())
    if not binaries:
        raise ValueError(f"built binary inventory is missing: {source}")
    names = []
    for path in binaries:
        target = destination / path.name
        _copy_required(path, target)
        names.append(path.name)
    return names


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _evidence(stage: Path) -> Path:
    candidate = stage / "evidence"
    if candidate.is_dir():
        return candidate
    if (stage / "manifest.json").is_file():
        return stage
    raise ValueError(f"stage has no canonical evidence directory: {stage}")


def _report(stage: Path) -> Path:
    candidate = stage / "report"
    if candidate.is_dir():
        return candidate
    candidate = stage / "local-report"
    if candidate.is_dir():
        return candidate
    raise ValueError(f"stage has no report directory: {stage}")


def _verify_source(reporting_source: str) -> None:
    if _git("rev-list", "-n1", EXPECTED_TAG) != EXPECTED_SOURCE:
        raise ValueError("execution tag does not resolve to the packed source")
    if _git("rev-parse", "HEAD") != reporting_source:
        raise ValueError("reporting source does not match current HEAD")
    if _git("status", "--porcelain"):
        raise ValueError("bundle requires a clean reporting worktree")
    changed = {
        path.removeprefix("thesis/implementation/")
        for path in _git(
            "diff", "--name-only", f"{EXPECTED_SOURCE}..{reporting_source}"
        ).splitlines()
        if path
    }
    unexpected = sorted(changed - ALLOWED_DESCENDANT_PATHS)
    if unexpected:
        raise ValueError(
            "reporting descendant changed forbidden paths: " + ", ".join(unexpected)
        )


def _verify_diagnostic(stage: Path, reporting_source: str) -> Mapping[str, Any]:
    evidence = _evidence(stage)
    verification = verify_artifacts(evidence)
    required = {
        "status": "completed",
        "sample_count": 36,
        "session_count": 36,
        "success_count": 36,
        "failed_count": 0,
        "unsupported_count": 0,
        "accuracy_qualified": True,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise ValueError(
                f"packed diagnostic {field} is {verification.get(field)!r}"
            )
    summary_path = stage / "report" / "packed_operation_transport_summary.json"
    if not summary_path.is_file():
        summary_path = stage / "packed_operation_transport_summary.json"
    if not summary_path.is_file():
        raise ValueError("packed diagnostic summary is missing")
    report_path = _report(stage) / "report.json"
    summary = inspect_packed(
        input_dir=evidence,
        summary_output=summary_path,
        output_dir=None,
        expected_source_commit=EXPECTED_SOURCE,
        report_path=report_path if report_path.is_file() else None,
    )
    if summary.get("gate_passed") is not True:
        raise ValueError("packed diagnostic gate did not pass")
    if summary.get("claim_eligible") is not False:
        raise ValueError("packed diagnostic is unexpectedly claim eligible")
    return dict(summary)


def _verify_general(stage: Path) -> Mapping[str, Any]:
    evidence = _evidence(stage)
    verification = verify_artifacts(evidence)
    required = {
        "status": "completed",
        "sample_count": 5,
        "session_count": 5,
        "success_count": 5,
        "failed_count": 0,
        "unsupported_count": 0,
        "accuracy_qualified": True,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise ValueError(
                f"general-resource {field} is {verification.get(field)!r}"
            )
    _, samples, sessions = load_artifacts(evidence)
    session_map = {str(session["session_instance_id"]): session for session in sessions}
    for sample in samples:
        facts = sample.get("backend_facts", {})
        if facts.get("request_transport") != EXPECTED_TRANSPORT:
            raise ValueError("general-resource evidence does not prove packed transport")
        terminal = session_map[str(sample["session_instance_id"])].get(
            "terminal_backend_facts", {}
        )
        if terminal.get("request_transport") != EXPECTED_TRANSPORT:
            raise ValueError("general-resource terminal facts do not prove packed transport")
    summary = stage / "provenance" / "resource_general_summary.json"
    if not summary.is_file():
        summary = stage / "resource_general_summary.json"
    if not summary.is_file():
        raise ValueError("general-resource summary is missing")
    value = _json(summary)
    if value.get("overall_pass") is not True:
        raise ValueError("general-resource qualification did not pass")
    return value


def _verify_ab(stage: Path) -> Mapping[str, Any]:
    verification = verify_artifacts(_evidence(stage))
    required = {
        "status": "completed",
        "sample_count": 72,
        "session_count": 72,
        "success_count": 72,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise ValueError(
                f"A/B evidence {field} is {verification.get(field)!r}"
            )
    return verification


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def _verify_checksums(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"bundle checksum mismatch: {relative}")


def _safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        stream.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("bundle must contain exactly one root directory")
    return roots[0]


def _verify_archive(archive: Path, digest_path: Path) -> None:
    expected = digest_path.read_text(encoding="ascii").split()[0]
    if expected != _sha256(archive):
        raise ValueError("outer archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="packed-transport-bundle-") as directory:
        root = _safe_extract(archive, Path(directory))
        _verify_checksums(root)
        diagnostic = root / "diagnostic"
        verification = verify_artifacts(diagnostic / "evidence")
        if verification.get("success_count") != 36:
            raise ValueError("extracted diagnostic evidence is incomplete")
        manifest, samples, sessions = load_artifacts(diagnostic / "evidence")
        summary = inspect_packed(
            input_dir=diagnostic / "evidence",
            summary_output=Path(directory) / "reinspection.json",
            output_dir=None,
            expected_source_commit=EXPECTED_SOURCE,
            report_path=diagnostic / "report" / "report.json",
        )
        if summary.get("gate_passed") is not True:
            raise ValueError("extracted packed transport inspection failed")
        if (
            manifest.get("source_commit") != EXPECTED_SOURCE
            or len(samples) != 36
            or len(sessions) != 36
        ):
            raise ValueError("extracted source/evidence identity mismatch")


def build_bundle(
    *,
    diagnostic_stage: Path,
    general_stage: Path,
    ab_stage: Path,
    output: Path,
    reporting_source: str,
) -> Path:
    _verify_source(reporting_source)
    diagnostic_summary = _verify_diagnostic(diagnostic_stage, reporting_source)
    general_summary = _verify_general(general_stage)
    ab_verification = _verify_ab(ab_stage)
    if (
        diagnostic_summary.get("packed_transport", {}).get("transport")
        != EXPECTED_TRANSPORT
    ):
        raise ValueError("diagnostic does not prove packed transport")
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise ValueError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="packed-transport-stage-") as directory:
        staging = Path(directory) / "thesis-upmem-packed-operation-transport-v1"
        staging.mkdir()
        _copy_tree(_evidence(diagnostic_stage), staging / "diagnostic" / "evidence")
        _copy_tree(_report(diagnostic_stage), staging / "diagnostic" / "report")
        _copy_tree(
            diagnostic_stage / "provenance", staging / "diagnostic" / "provenance"
        )
        _copy_tree(_evidence(general_stage), staging / "general-resource" / "evidence")
        _copy_tree(
            general_stage / "provenance", staging / "general-resource" / "provenance"
        )
        _copy_tree(_evidence(ab_stage), staging / "ab-baseline" / "evidence")
        if _report(ab_stage).is_dir():
            _copy_tree(_report(ab_stage), staging / "ab-baseline" / "report")
        for name in ("analysis", "provenance"):
            source = ab_stage / name
            if source.is_dir():
                _copy_tree(source, staging / "ab-baseline" / name)
        for relative in (
            "scripts/inspect_packed_operation_transport.py",
            "scripts/freeze_packed_operation_transport.py",
            "scripts/inspect_parallel_scaling.py",
            "scripts/qualify_general_resources.py",
            "tests/test_inspect_packed_operation_transport.py",
        ):
            _copy_required(ROOT / relative, staging / "source" / relative)
        binary_names = _copy_binaries(diagnostic_stage, staging / "binaries")
        doc = ROOT / "docs" / "packed_operation_transport_adoption.md"
        if doc.is_file():
            _copy_required(doc, staging / "docs" / doc.name)
        metadata = {
            "physical_execution_source_commit": EXPECTED_SOURCE,
            "reporting_tool_source_commit": reporting_source,
            "execution_tag": EXPECTED_TAG,
            "transport": EXPECTED_TRANSPORT,
            "binary_names": binary_names,
            "diagnostic_experiment_id": diagnostic_summary["experiment_id"],
            "diagnostic_run_id": diagnostic_summary["run_id"],
            "diagnostic_sample_count": 36,
            "diagnostic_session_count": 36,
            "general_resource_verification": general_summary,
            "ab_verification": ab_verification,
            "claim_policy": diagnostic_summary["claim_policy"],
            "claim_eligible": False,
            "allowed_claims": [
                "packed operation transport was physically qualified",
                "descriptive host-boundary reduction for the tested circuits and "
                "routes",
            ],
            "prohibited_claims": [
                "final physical_performance_v1 estimates",
                "general-purpose simulator acceleration",
                "arbitrary circuit or resource-count performance",
                "multi-rank or sliced performance",
            ],
        }
        (staging / "baseline_summary.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        _write_checksums(staging)
        with tarfile.open(output, "w:gz") as stream:
            stream.add(staging, arcname=staging.name)
    digest_path = output.with_name(output.name + ".sha256")
    digest_path.write_text(f"{_sha256(output)}  {output.name}\n", encoding="ascii")
    _verify_archive(output, digest_path)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-stage", type=Path, required=True)
    parser.add_argument("--general-stage", type=Path, required=True)
    parser.add_argument("--ab-stage", type=Path, required=True)
    parser.add_argument("--reporting-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(
            build_bundle(
                diagnostic_stage=args.diagnostic_stage.resolve(),
                general_stage=args.general_stage.resolve(),
                ab_stage=args.ab_stage.resolve(),
                output=args.output.resolve(),
                reporting_source=args.reporting_source_commit,
            )
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
