#!/usr/bin/env python3
"""Build and verify a portable circuit-sensitivity diagnostic bundle."""

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
EXPECTED_SOURCE = "89ecc5527f182f42dc471101f96edf86b0dadefa"
EXPECTED_TAG = "thesis-upmem-circuit-resource-sensitivity-diagnostic-v1"
EXPECTED_CASES = ("quantization_stress_18q_l2", "hs_18q_d1", "ghz_chain_18q")
EXPECTED_ROUTES = (
    "upmem_float32_1dpu_t1",
    "upmem_float32_1dpu_t4",
    "upmem_float32_1dpu_t8",
    "upmem_float32_1dpu_t12",
    "upmem_float32_2dpu_t8",
    "upmem_float32_3dpu_t8",
    "upmem_float32_4dpu_t8",
)
ALLOWED_DESCENDANT_PATHS = {
    "docs/circuit_resource_sensitivity_diagnostic.md",
    "scripts/freeze_circuit_resource_sensitivity.py",
    "scripts/inspect_circuit_resource_sensitivity.py",
    "tests/test_freeze_circuit_resource_sensitivity.py",
    "tests/test_inspect_circuit_resource_sensitivity.py",
}

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_circuit_resource_sensitivity import inspect  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
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


def _copy(source: Path, destination_root: Path, relative: str) -> None:
    if not source.is_file():
        raise ValueError(f"required bundle input is missing: {source}")
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination_root: Path, prefix: str) -> None:
    if not source.is_dir():
        raise ValueError(f"required bundle directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = Path(prefix) / path.relative_to(source)
            _copy(path, destination_root, relative.as_posix())


def _verify_checksum_file(root: Path, checksum_file: Path) -> None:
    if not checksum_file.is_file():
        raise ValueError(f"checksum manifest is missing: {checksum_file}")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {relative}")


def _check_descendant_paths(source_sha: str, reporting_sha: str) -> None:
    changed = set(_git("diff", "--name-only", f"{source_sha}..{reporting_sha}").splitlines())
    prefix = "thesis/implementation/"
    changed = {
        path.removeprefix(prefix) if path.startswith(prefix) else path
        for path in changed
    }
    unexpected = sorted(changed - ALLOWED_DESCENDANT_PATHS)
    if unexpected:
        raise ValueError("reporting descendant changed forbidden paths: " + ", ".join(unexpected))


def _verify_canonical(
    *,
    raw_stage: Path,
    summary_path: Path,
    selection_path: Path,
    incident_path: Path,
    reporting_sha: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_sha = _git("rev-list", "-n1", EXPECTED_TAG)
    if source_sha != EXPECTED_SOURCE:
        raise ValueError(f"execution tag resolves to {source_sha}, expected {EXPECTED_SOURCE}")
    if _git("rev-parse", "HEAD") != reporting_sha:
        raise ValueError("reporting source commit does not match current HEAD")
    _check_descendant_paths(source_sha, reporting_sha)

    _verify_checksum_file(raw_stage, raw_stage / "SHA256SUMS")
    evidence = raw_stage / "evidence"
    if not evidence.is_dir():
        evidence = raw_stage
    verification = verify_artifacts(evidence)
    required = {
        "status": "completed",
        "sample_count": 126,
        "session_count": 126,
        "success_count": 126,
        "failed_count": 0,
        "unsupported_count": 0,
        "accuracy_qualified": True,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise ValueError(f"canonical evidence {field} is {verification.get(field)!r}")

    with tempfile.TemporaryDirectory(prefix="circuit-sensitivity-inspect-") as directory:
        derived = inspect(
            input_dir=evidence,
            output_dir=Path(directory) / "analysis",
            selection_path=selection_path,
            expected_source_commit=EXPECTED_SOURCE,
        )
    supplied = _json(summary_path)
    if supplied != derived:
        raise ValueError("supplied circuit-sensitivity summary does not match raw evidence")
    if derived["selected_case_ids"] != list(EXPECTED_CASES):
        raise ValueError("selected case matrix changed")
    if derived["route_ids"] != list(EXPECTED_ROUTES):
        raise ValueError("route matrix changed")
    if derived["block_ids"] != list(range(6)) or derived["measurement_count_per_route"] != 5:
        raise ValueError("block or measurement matrix changed")
    if derived["claim_policy"] != "diagnostic_v1" or derived["claim_eligible"] is not False:
        raise ValueError("diagnostic claim boundary changed")
    environment = derived["environment"]
    if environment["host"] != "safari-baguette1" or environment["rank_paths"] != ["/dev/dpu_rank1"]:
        raise ValueError("physical host or rank changed")
    if environment["affinity"] != [0] or environment["observed_cpu_governors"].get("0") != "powersave":
        raise ValueError("diagnostic environment changed")
    if not incident_path.is_file():
        raise ValueError(f"incident record is missing: {incident_path}")
    return evidence, verification, derived


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsafe archive member: {member.name}")
        stream.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("bundle must contain exactly one root directory")
    return roots[0]


def _verify_archive(archive: Path, digest_path: Path, selection_name: str) -> None:
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    if expected != _sha256(archive):
        raise ValueError("outer archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="circuit-sensitivity-bundle-") as directory:
        root = _safe_extract(archive, Path(directory))
        _verify_checksum_file(root, root / "SHA256SUMS")
        evidence = root / "evidence"
        verification = verify_artifacts(evidence)
        if verification.get("success_count") != 126:
            raise ValueError("extracted evidence is incomplete")
        with tempfile.TemporaryDirectory(prefix="circuit-sensitivity-reinspect-") as output:
            summary = inspect(
                input_dir=evidence,
                output_dir=Path(output) / "analysis",
                selection_path=root / selection_name,
                expected_source_commit=EXPECTED_SOURCE,
            )
        if summary["gate_passed"] is not True:
            raise ValueError("extracted diagnostic did not pass")


def build_bundle(args: argparse.Namespace) -> Path:
    reporting_sha = args.reporting_source_commit or _git("rev-parse", "HEAD")
    raw_stage = args.raw_stage.resolve()
    evidence, verification, summary = _verify_canonical(
        raw_stage=raw_stage,
        summary_path=args.summary.resolve(),
        selection_path=args.selection.resolve(),
        incident_path=args.incident.resolve(),
        reporting_sha=reporting_sha,
    )
    report_dir = args.report_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    archive = args.output.resolve()
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        raise ValueError(f"bundle output already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="circuit-sensitivity-staging-") as directory:
        staging = Path(directory) / "thesis-upmem-circuit-resource-sensitivity-diagnostic-v1"
        staging.mkdir()
        for filename in ("manifest.json", "samples.jsonl", "sessions.jsonl"):
            _copy(evidence / filename, staging, f"evidence/{filename}")
        _copy_tree(report_dir, staging, "report")
        _copy_tree(analysis_dir, staging, "analysis")
        _copy(args.selection.resolve(), staging, "configuration/selection.json")
        _copy(args.characterization.resolve(), staging, "characterization/characterization.json")
        _copy(args.incident.resolve(), staging, "incident/invalid-allocation-incident.json")
        _copy(ROOT / "docs" / "circuit_resource_sensitivity_diagnostic.md", staging, "docs/circuit_resource_sensitivity_diagnostic.md")
        provenance = raw_stage / "provenance"
        _copy_tree(provenance, staging, "provenance")
        if (raw_stage / "SHA256SUMS").is_file():
            _copy(raw_stage / "SHA256SUMS", staging, "provenance/original-stage-SHA256SUMS")
        metadata = {
            "execution_tag": EXPECTED_TAG,
            "physical_execution_source_commit": EXPECTED_SOURCE,
            "reporting_tool_source_commit": reporting_sha,
            "experiment_id": summary["experiment_id"],
            "run_id": summary["run_id"],
            "sample_count": verification["sample_count"],
            "session_count": verification["session_count"],
            "claim_policy": summary["claim_policy"],
            "claim_eligible": summary["claim_eligible"],
            "claim_ineligibility_reason": summary["claim_ineligibility_reason"],
            "allowed_claims": [
                "physical correctness",
                "descriptive within-circuit tasklet scaling",
                "descriptive one-rank DPU scaling",
                "powersave steady-execution timing composition",
            ],
            "prohibited_claims": [
                "final physical_performance_v1 estimates",
                "general UPMEM acceleration",
                "machine-independent performance",
                "multi-rank scaling",
                "energy efficiency",
            ],
        }
        (staging / "bundle_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_checksums(staging)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(staging, arcname=staging.name)

    digest_path = archive.with_name(archive.name + ".sha256")
    digest_path.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")
    _verify_archive(archive, digest_path, "configuration/selection.json")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stage", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--characterization", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--execution-tag", default=EXPECTED_TAG)
    parser.add_argument("--reporting-source-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.execution_tag != EXPECTED_TAG:
        parser.error("only the frozen circuit-sensitivity execution tag is supported")
    try:
        print(build_bundle(args))
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
