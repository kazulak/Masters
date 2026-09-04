#!/usr/bin/env python3
"""Freeze and verify the completed quantized UPMEM diagnostic.

This tool handles evidence and derived reporting only.  It never opens an
UPMEM rank, invokes the SDK simulator, or executes a benchmark.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_quantized_upmem_execution import (  # noqa: E402
    OUTPUTS as ANALYSIS_OUTPUTS,
    derive as derive_analysis,
    write_outputs as write_analysis_outputs,
)
from qualify_quantized_upmem_execution import (  # noqa: E402
    inspect as inspect_quantized,
    verify_checksums as verify_raw_checksums,
)
from quantum_bench.evidence import load_artifacts  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


POLICY = "complex_int8_shared_scale_v1"
POLICY_SOURCE = "6c9ca849a5ccc246dc645b63598ee391da75c599"
PHYSICAL_SOURCE = "c0ec6c76439e418e537a953a6b768ce2e1ea0dc6"
EXPECTED_TAG = "thesis-upmem-quantized-execution-diagnostic-v1"
SCHEMA = "quantized_upmem_execution_release_v1"
PREPARED_CONFIG_SHA256 = (
    "e6bdc7129b2a5cf2c8513776f955bac0307c0bf0da1dbfbba319f33f1d622b21"
)
DIAGNOSTIC_EXPERIMENT_ID = (
    "7797fa21b27957e070633ed69219ce774ddd3861881477d610461f8b037f6ff2"
)
DIAGNOSTIC_RUN_ID = "49339c75-dac1-438d-9307-dc77ebe5805d"
PILOT_EXPERIMENT_ID = (
    "d639d2be0bb2062f305528c5b09a6b766697b98d089da6bc19b612f8da7fff4d"
)
PILOT_RUN_ID = "e8193029-fb03-45e3-af7c-6296a6d7e564"
INCIDENT_RUN_ID = "42b5d3e2-c604-49a0-9fc9-5592d8fc3d28"
RAW_LANES = {"rr", "ii", "ri", "ir"}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

BINARY_SHA256 = {
    "host_upmem_execution_plan_v4_t1": "6dc5b468a97902e85bb4c0f3e120ccde87ce4c763f4d0a1116712a7eca033dd3",
    "dpu_gemm_tile_v4_t1": "628f4ccbce7187f9b17950a271b2dac5922dc9cdf754786ca30c7a63e84a377f",
    "dpu_simplepim_management_init_t1": "5dd6f7bcd32c59b3dde59893c4f50800df4cabf741117a7ee6ae4935935f4f8a",
    "host_upmem_execution_plan_v4_t4": "ea266074677eefda84830a0ca981bbdfc543f4e944fab49c23ce1cb79bc98f31",
    "dpu_gemm_tile_v4_t4": "15753458d7347b1986b9aea3352ebf4ca3a1398d92542621087468a5f973580c",
    "dpu_simplepim_management_init_t4": "1c965551d7ad3263344bd832382425e711e764bc0b6f72d4774e5acd4051645d",
    "host_upmem_execution_plan_v4_t8": "5a3ac1ea225696f78bf3e29aac499625951d4a1fec09150e7c85d8a3eca06d57",
    "dpu_gemm_tile_v4_t8": "a198e1e0a4a2f809d2c57fdb55033c52896d847454563a3712a1dccbbd8f12cb",
    "dpu_simplepim_management_init_t8": "0d947160ed8935da3f1aaa40b8febc634ceda41abcf3e9a3d29b380cede824da",
}

ALLOWED_DESCENDANT_PATHS = {
    "thesis/implementation/scripts/analyze_quantized_upmem_execution.py",
    "thesis/implementation/tests/test_quantized_upmem_execution.py",
    "thesis/implementation/scripts/freeze_quantized_upmem_execution.py",
    "thesis/implementation/tests/test_freeze_quantized_upmem_execution.py",
    "thesis/implementation/docs/quantized_upmem_execution_diagnostic_v1.md",
    "thesis/implementation/docs/quantized_contraction_policy_v1.md",
}


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"required regular file is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"required directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source tree contains a symlink: {path}")
        if path.is_file():
            _copy_file(path, target / path.relative_to(source))


def _check_descendant_paths(reporting_sha: str) -> None:
    changed = set(_git("diff", "--name-only", f"{PHYSICAL_SOURCE}..{reporting_sha}").splitlines())
    unexpected = sorted(changed - ALLOWED_DESCENDANT_PATHS)
    if unexpected:
        raise ValueError("reporting descendant changed forbidden paths: " + ", ".join(unexpected))


def _verify_git(reporting_sha: str, execution_tag: str) -> None:
    if _git("rev-parse", "HEAD") != reporting_sha:
        raise ValueError("reporting source commit does not match HEAD")
    if _git("status", "--porcelain"):
        raise ValueError("reporting worktree is dirty")
    if _git("rev-list", "-n1", execution_tag) != PHYSICAL_SOURCE:
        raise ValueError("execution tag does not resolve to the physical source")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PHYSICAL_SOURCE, reporting_sha],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("reporting source is not descended from the physical source")
    _check_descendant_paths(reporting_sha)


def _verify_raw_lanes(samples: Sequence[Mapping[str, Any]]) -> int:
    checked_samples = 0
    for sample in samples:
        numeric = sample.get("numeric_facts")
        if not isinstance(numeric, Mapping):
            raise ValueError("sample lacks numeric facts")
        if numeric.get("numeric_policy") != POLICY:
            continue
        records = numeric.get("raw_lane_records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
            raise ValueError("int8 sample lacks exact raw lane records")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("raw lane record is not a mapping")
            if (
                record.get("exact") is not True
                or record.get("dtype") != "<i4"
                or record.get("lane") not in RAW_LANES
                or not isinstance(record.get("node_id"), str)
                or not isinstance(record.get("stable_tile_id"), str)
                or not HEX_SHA256.fullmatch(str(record.get("sha256", "")))
            ):
                raise ValueError("raw int32 lane record is not exact and typed")
        checked_samples += 1
    if checked_samples == 0:
        raise ValueError("diagnostic contains no int8 samples with raw lanes")
    return checked_samples


def _identity_map(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = f"{sample.get('case_id')}:{sample.get('route_id')}"
        identities = sample.get("identities")
        if not isinstance(identities, Mapping):
            raise ValueError("sample lacks identities")
        value = {
            "logical_plan_id": identities.get("logical_plan_id"),
            "physical_plan_id": identities.get("physical_plan_id"),
        }
        if key in result and result[key] != value:
            raise ValueError(f"plan identities vary within {key}")
        result[key] = value
    return result


def _verify_plan_path(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    document = _json(path)
    entries = document.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("regenerated plan has no entries")
    observed = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("plan entry is not a mapping")
        key = f"{entry.get('case_id')}:{entry.get('route_id')}"
        upmem = entry.get("upmem")
        if not isinstance(upmem, Mapping):
            raise ValueError("plan entry lacks UPMEM identity")
        observed[key] = {
            "logical_plan_id": entry.get("logical_plan_id"),
            "physical_plan_id": upmem.get("physical_plan_id"),
        }
    expected = _identity_map(samples)
    if observed != expected or len(observed) != 30:
        raise ValueError("regenerated plan identities do not match evidence")


def _verify_plan(stage: Path, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _verify_plan_path(
        stage / "provenance" / "regenerated-plan" / "plan.json", samples
    )
    return {"entry_count": 30, "identity_match": True}


def _verify_binaries(stage: Path) -> dict[str, str]:
    directory = stage / "provenance" / "binaries"
    observed = {}
    for name, expected in sorted(BINARY_SHA256.items()):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"recorded binary is missing: {name}")
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"recorded binary hash mismatch: {name}")
        observed[name] = digest
    return observed


def _verify_stage(stage: Path, analysis: Path) -> dict[str, Any]:
    pilot = stage / "pilot"
    diagnostic = stage / "diagnostic"
    incident = stage / "diagnostic-interrupted-incident"
    verify_raw_checksums(pilot)
    verify_raw_checksums(diagnostic)
    verify_raw_checksums(incident / "raw")
    pilot_result = inspect_quantized(pilot, kind="pilot", expected_source=PHYSICAL_SOURCE)
    diagnostic_result = inspect_quantized(
        diagnostic, kind="diagnostic", expected_source=PHYSICAL_SOURCE
    )
    if pilot_result.get("experiment_id") != PILOT_EXPERIMENT_ID:
        raise ValueError("pilot experiment identity changed")
    manifest, samples, sessions = load_artifacts(diagnostic)
    if manifest.get("experiment_id") != DIAGNOSTIC_EXPERIMENT_ID:
        raise ValueError("diagnostic experiment identity changed")
    if len(samples) != 180 or len(sessions) != 180:
        raise ValueError("diagnostic evidence is not 180/180")
    int8_sample_count = _verify_raw_lanes(samples)
    _verify_plan(stage, samples)
    binary_hashes = _verify_binaries(stage)

    prepared = stage / "provenance" / "prepared-config.yml"
    if _sha256(prepared) != PREPARED_CONFIG_SHA256:
        raise ValueError("prepared configuration hash mismatch")
    generic_report = _json(stage / "provenance" / "generic-report" / "report.json")
    generic_verification = generic_report.get("verification")
    if not isinstance(generic_verification, Mapping):
        raise ValueError("generic report lacks verification facts")
    if (
        generic_report.get("status") != "completed"
        or generic_report.get("experiment_id") != DIAGNOSTIC_EXPERIMENT_ID
        or generic_report.get("run_id") != DIAGNOSTIC_RUN_ID
        or generic_report.get("aggregate_count") != 30
        or generic_report.get("session_count") != 180
        or generic_verification.get("sample_count") != 180
        or generic_verification.get("session_count") != 180
    ):
        raise ValueError("generic report is not bound to the accepted diagnostic")

    verify = verify_artifacts(diagnostic)
    if verify.get("policy_reference_failure_count") != 0:
        raise ValueError("same-policy replay failures are present")
    if verify.get("failed_count") != 0 or verify.get("unsupported_count") != 0:
        raise ValueError("diagnostic contains failed or unsupported attempts")

    incident_record = _json(incident / "INCIDENT.json")
    if (
        incident_record.get("invalid_for_scientific_analysis") is not True
        or incident_record.get("run_id") != INCIDENT_RUN_ID
        or incident_record.get("sample_rows") != 18
        or incident_record.get("session_rows") != 19
    ):
        raise ValueError("interrupted incident is not explicitly invalid/excluded")

    for name in ANALYSIS_OUTPUTS:
        if not (analysis / name).is_file():
            raise ValueError(f"corrected analysis output is missing: {name}")
    with tempfile.TemporaryDirectory(prefix="quantized-analysis-verify-") as directory:
        regenerated = Path(directory)
        result = derive_analysis(manifest, samples, sessions)
        write_analysis_outputs(result, regenerated)
        for name in ANALYSIS_OUTPUTS:
            if (analysis / name).read_bytes() != (regenerated / name).read_bytes():
                raise ValueError(f"corrected analysis is not reproducible: {name}")

    return {
        "pilot": pilot_result,
        "diagnostic": diagnostic_result,
        "diagnostic_verification": verify,
        "int8_samples_with_exact_raw_lanes": int8_sample_count,
        "binary_sha256": binary_hashes,
        "plan": {"entry_count": 30, "identity_match": True},
        "prepared_config_sha256": PREPARED_CONFIG_SHA256,
        "generic_report": {
            "status": generic_report["status"],
            "schema_version": generic_report.get("schema_version"),
            "aggregate_count": generic_report.get("aggregate_count"),
            "sample_count": generic_verification.get("sample_count"),
            "session_count": generic_verification.get("session_count"),
        },
    }


def _write_checksums(root: Path) -> None:
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(root).as_posix()
            entries.append((relative, f"{_sha256(path)}  {relative}"))
    (root / "SHA256SUMS").write_text(
        "\n".join(entry for _, entry in sorted(entries)) + "\n", encoding="ascii"
    )


def _verify_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError("bundle SHA256SUMS is missing")
    listed: list[str] = []
    for line in checksum_file.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        path = Path(relative)
        if (
            separator != "  "
            or not HEX_SHA256.fullmatch(digest)
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or path.name == "SHA256SUMS"
        ):
            raise ValueError(f"malformed or unsafe checksum entry: {line}")
        normalized = path.as_posix()
        if normalized in listed:
            raise ValueError(f"duplicate checksum entry: {relative}")
        target = root / path
        if not target.is_file() or target.is_symlink() or _sha256(target) != digest:
            raise ValueError(f"bundle checksum mismatch: {relative}")
        listed.append(normalized)
    if listed != sorted(listed):
        raise ValueError("bundle checksums are not sorted")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    )
    if actual != listed:
        raise ValueError("bundle checksum inventory differs from files")


def _safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names: set[str] = set()
        for member in members:
            path = Path(member.name)
            if (
                not member.name
                or member.name in names
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name
                or member.name in {".", ".."}
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
            names.add(member.name)
        stream.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    top_level_files = [path for path in destination.iterdir() if not path.is_dir()]
    if len(roots) != 1 or top_level_files:
        raise ValueError("bundle must contain exactly one root directory")
    return roots[0]


def _verify_extracted_bundle(root: Path, reporting_sha: str) -> None:
    _verify_checksums(root)
    pilot = root / "evidence" / "pilot"
    diagnostic = root / "evidence" / "diagnostic"
    incident = root / "evidence" / "diagnostic-interrupted-incident"
    verify_raw_checksums(pilot)
    verify_raw_checksums(diagnostic)
    verify_raw_checksums(incident / "raw")
    inspect_quantized(pilot, kind="pilot", expected_source=PHYSICAL_SOURCE)
    verify = verify_artifacts(diagnostic)
    if (
        verify.get("status") != "completed"
        or verify.get("sample_count") != 180
        or verify.get("session_count") != 180
        or verify.get("policy_reference_failure_count") != 0
    ):
        raise ValueError("extracted canonical diagnostic verification failed")
    manifest, samples, sessions = load_artifacts(diagnostic)
    inspect_quantized(diagnostic, kind="diagnostic", expected_source=PHYSICAL_SOURCE)
    _verify_raw_lanes(samples)
    _verify_plan_path(root / "provenance" / "regenerated-plan" / "plan.json", samples)
    prepared = root / "provenance" / "prepared-config.yml"
    if _sha256(prepared) != PREPARED_CONFIG_SHA256:
        raise ValueError("extracted prepared configuration hash mismatch")
    for name, expected in BINARY_SHA256.items():
        path = root / "provenance" / "binaries" / name
        if _sha256(path) != expected:
            raise ValueError(f"extracted binary hash mismatch: {name}")
    generic_report = _json(root / "provenance" / "generic-report" / "report.json")
    generic_verification = generic_report.get("verification")
    if (
        generic_report.get("status") != "completed"
        or generic_report.get("experiment_id") != DIAGNOSTIC_EXPERIMENT_ID
        or generic_report.get("run_id") != DIAGNOSTIC_RUN_ID
        or generic_report.get("aggregate_count") != 30
        or generic_report.get("session_count") != 180
        or not isinstance(generic_verification, Mapping)
        or generic_verification.get("sample_count") != 180
    ):
        raise ValueError("extracted generic report is not bound to the diagnostic")
    incident_record = _json(incident / "INCIDENT.json")
    if incident_record.get("invalid_for_scientific_analysis") is not True:
        raise ValueError("extracted interrupted incident is not excluded")
    result = derive_analysis(manifest, samples, sessions)
    with tempfile.TemporaryDirectory(prefix="quantized-bundle-analysis-") as directory:
        regenerated = Path(directory)
        write_analysis_outputs(result, regenerated)
        for name in ANALYSIS_OUTPUTS:
            if (root / "analysis" / "corrected" / name).read_bytes() != (
                regenerated / name
            ).read_bytes():
                raise ValueError(f"extracted analysis mismatch: {name}")
    metadata = _json(root / "metadata.json")
    if (
        metadata.get("policy_source_commit") != POLICY_SOURCE
        or metadata.get("physical_execution_source_commit") != PHYSICAL_SOURCE
        or metadata.get("reporting_source_commit") != reporting_sha
        or metadata.get("numeric_policy") != POLICY
    ):
        raise ValueError("extracted provenance metadata is inconsistent")


def _stage_bundle(
    *, stage: Path, analysis: Path, reporting_sha: str, verification: Mapping[str, Any], root: Path
) -> None:
    for name in ("pilot", "diagnostic", "diagnostic-interrupted-incident"):
        _copy_tree(stage / name, root / "evidence" / name)
    for name in (
        "diagnostic-preflight.txt",
        "pilot-preflight.txt",
        "recovery-diagnostic-duration-s",
        "recovery-diagnostic-exit-code",
    ):
        _copy_file(stage / name, root / "provenance" / name)
    for name in ANALYSIS_OUTPUTS:
        _copy_file(analysis / name, root / "analysis" / "corrected" / name)
    superseded = analysis / "superseded-c0"
    if superseded.is_dir():
        _copy_tree(superseded, root / "analysis" / "superseded-c0")
    for relative in (
        Path("prepared-config.yml"),
        Path("regenerated-plan"),
        Path("generic-report"),
        Path("binaries"),
    ):
        source = stage / "provenance" / relative
        target = root / "provenance" / relative
        if source.is_dir():
            _copy_tree(source, target)
        else:
            _copy_file(source, target)
    for document in (
        ROOT / "docs" / "quantized_contraction_policy_v1.md",
        ROOT / "docs" / "quantized_upmem_execution_diagnostic_v1.md",
    ):
        _copy_file(document, root / "docs" / document.name)
    metadata = {
        "schema_version": SCHEMA,
        "numeric_policy": POLICY,
        "policy_source_commit": POLICY_SOURCE,
        "physical_execution_source_commit": PHYSICAL_SOURCE,
        "reporting_source_commit": reporting_sha,
        "execution_tag": EXPECTED_TAG,
        "diagnostic_experiment_id": DIAGNOSTIC_EXPERIMENT_ID,
        "diagnostic_run_id": DIAGNOSTIC_RUN_ID,
        "pilot_experiment_id": PILOT_EXPERIMENT_ID,
        "pilot_run_id": PILOT_RUN_ID,
        "interrupted_incident_run_id": INCIDENT_RUN_ID,
        "accepted_counts": {
            "pilot_samples": 4,
            "pilot_sessions": 4,
            "diagnostic_samples": 180,
            "diagnostic_sessions": 180,
            "diagnostic_measurements": 150,
            "diagnostic_warmups": 30,
        },
        "incident_excluded": True,
        "verification": verification,
        "claim_boundary": {
            "status": "diagnostic_descriptive_accuracy_unqualified",
            "physical_same_policy_replay_required": True,
            "best_routes": "best observed within the tested five-route grid",
            "prohibited": [
                "universal int8 speedup or accuracy",
                "global topology optimality",
                "CPU/GPU competitiveness",
                "logical compression presented as physical transfer reduction",
                "fully integer-resident end-to-end execution",
                "final thesis performance claim",
            ],
        },
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_deterministic_tar(source: Path, archive: Path) -> None:
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as stream:
                for path in sorted(source.rglob("*")):
                    arcname = path.relative_to(source.parent).as_posix()
                    info = stream.gettarinfo(str(path), arcname=arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if info.isfile():
                        with path.open("rb") as payload:
                            stream.addfile(info, payload)
                    else:
                        stream.addfile(info)


def build_bundle(
    *, stage: Path, analysis: Path, output: Path, reporting_sha: str, execution_tag: str
) -> Path:
    _verify_git(reporting_sha, execution_tag)
    verification = _verify_stage(stage, analysis)
    if output.exists() or output.with_name(output.name + ".sha256").exists():
        raise ValueError(f"bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quantized-upmem-bundle-") as directory:
        root = Path(directory) / EXPECTED_TAG
        root.mkdir()
        _stage_bundle(
            stage=stage,
            analysis=analysis,
            reporting_sha=reporting_sha,
            verification=verification,
            root=root,
        )
        _write_checksums(root)
        _write_deterministic_tar(root, output)
    digest_path = output.with_name(output.name + ".sha256")
    digest_path.write_text(f"{_sha256(output)}  {output.name}\n", encoding="ascii")
    with tempfile.TemporaryDirectory(prefix="quantized-upmem-archive-verify-") as directory:
        extracted = _safe_extract(output, Path(directory))
        _verify_extracted_bundle(extracted, reporting_sha)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-tag", default=EXPECTED_TAG)
    parser.add_argument("--reporting-source-commit", default=None)
    args = parser.parse_args(argv)
    try:
        reporting_sha = args.reporting_source_commit or _git("rev-parse", "HEAD")
        result = build_bundle(
            stage=args.stage.resolve(),
            analysis=args.analysis.resolve(),
            output=args.output.resolve(),
            reporting_sha=reporting_sha,
            execution_tag=args.execution_tag,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "archive": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
