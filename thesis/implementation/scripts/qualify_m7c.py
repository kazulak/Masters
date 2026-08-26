#!/usr/bin/env python3
"""Create an exact-head, software/SDK-only M7C preparation qualification bundle.

Physical UPMEM execution is intentionally absent.  Run this script only after
the M7C source is final and the designated SDK qualification machine is ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
M7B_TAG = "thesis-m7b-prephysical-software-ready-v1"
_SDK_CASE_IDS = {
    "scalar_1x1x1",
    "tail_m3_n35_k65",
    "t8_k65",
    "t8_k130",
    "direct_k257",
    "planned_k257",
    "int8_tail",
    "t24_functional",
}


def _run(
    command: list[str], *, env: Mapping[str, str], capture: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(env),
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + ("\n" + completed.stdout if completed.stdout else "")
            + ("\n" + completed.stderr if completed.stderr else "")
        )
    return completed


def _git_output(*arguments: str) -> str:
    return _run(["git", *arguments], env=os.environ, capture=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum_file(checksum_file: Path, artifact: Path) -> None:
    expected = None
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and fields[1].lstrip("*") == artifact.name:
            expected = fields[0]
            break
    if expected is None or _sha256(artifact) != expected:
        raise ValueError(f"outer SHA-256 verification failed for {artifact.name}")


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract only unique regular files/directories beneath destination."""

    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        names: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.name
                or pure.is_absolute()
                or ".." in pure.parts
                or member.name in names
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {member.name!r}")
            names.add(member.name)
            if not (destination / pure).resolve().is_relative_to(destination_root):
                raise ValueError(f"archive member escapes extraction root: {member.name!r}")
        for member in members:
            target = destination / PurePosixPath(member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name!r}")
            with source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)


def _release_assets(destination: Path) -> tuple[Path, Path]:
    _run(["gh", "release", "download", M7B_TAG, "--dir", str(destination)], env=os.environ)
    archive = destination / f"{M7B_TAG}.tar.gz"
    checksum = destination / f"{M7B_TAG}.tar.gz.sha256"
    if not archive.is_file() or not checksum.is_file():
        raise ValueError("M7B release lacks its archive or outer checksum asset")
    _verify_checksum_file(checksum, archive)
    return archive, checksum


def _verify_internal_hashes(root: Path) -> None:
    checksum_files = list(root.rglob("SHA256SUMS"))
    if len(checksum_files) != 1:
        raise ValueError("M7B release must contain exactly one internal SHA256SUMS")
    checksum_file = checksum_files[0]
    bundle_root = checksum_file.parent
    bundle_prefix = f"runs/{bundle_root.name}/"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        expected, name = fields[0], fields[1].lstrip("*")
        if not name.startswith(bundle_prefix):
            continue
        artifact = bundle_root / name.removeprefix(bundle_prefix)
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"internal M7B checksum mismatch: {name}")


def _json_command(command: list[str], env: Mapping[str, str]) -> dict[str, Any]:
    result = _run(command, env=env, capture=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"command did not emit JSON: {' '.join(command)}") from exc
    if not isinstance(payload, dict):
        raise ValueError("command JSON must be an object")
    return payload


def _assert_summary(summary: Mapping[str, Any], *, samples: int, sessions: int) -> None:
    expected = {
        "status": "completed",
        "sample_count": samples,
        "session_count": sessions,
        "success_count": samples,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"unexpected verification {field}: {summary.get(field)!r}")


def _assert_direct_cases(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    required = payload.get("required_case_ids")
    executed = payload.get("executed_case_ids")
    passed = payload.get("passed_case_ids")
    if not all(isinstance(value, list) for value in (required, executed, passed)):
        raise ValueError("direct SDK case summary has invalid case IDs")
    if (
        set(required) != _SDK_CASE_IDS
        or set(executed) != _SDK_CASE_IDS
        or set(passed) != _SDK_CASE_IDS
        or len(executed) != len(_SDK_CASE_IDS)
        or len(passed) != len(_SDK_CASE_IDS)
        or payload.get("failed_case_ids") != []
        or payload.get("skipped_case_ids") != []
    ):
        raise ValueError("direct SDK qualification matrix is incomplete")
    return payload


def _write_hashes(root: Path, paths: list[Path]) -> None:
    lines: list[str] = []
    for path in sorted({path for path in paths if path.is_file()}):
        try:
            label = path.relative_to(root)
        except ValueError:
            label = Path("repository") / path.relative_to(ROOT)
        lines.append(f"{_sha256(path)}  {label}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qualify(output: Path) -> Path:
    branch = _git_output("branch", "--show-current")
    if branch != "feature/m7c-physical-readiness":
        raise ValueError(f"M7C qualification requires feature/m7c-physical-readiness, got {branch}")
    if _git_output("status", "--porcelain"):
        raise ValueError("M7C qualification requires a clean Git worktree")
    source_commit = _git_output("rev-parse", "HEAD")
    if output.exists():
        raise ValueError(f"qualification output must be absent: {output}")
    for command in ("gh", "make", "jq", "sha256sum"):
        if shutil.which(command) is None:
            raise ValueError(f"required qualification tool is unavailable: {command}")

    output.mkdir(parents=True)
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        "MPLCONFIGDIR": str(output / "matplotlib"),
    }
    python = sys.executable

    release_dir = output / "m7b-release"
    archive, outer_checksum = _release_assets(release_dir)
    extracted = release_dir / "extracted"
    _safe_extract_tar(archive, extracted)
    _verify_internal_hashes(extracted)
    manifests = sorted(extracted.rglob("manifest.json"))
    if not manifests:
        raise ValueError("M7B release contains no evidence manifest")
    m7b_source = _git_output("rev-parse", f"{M7B_TAG}^{{}}")
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("source_commit") != m7b_source:
            raise ValueError("M7B release evidence does not bind to its tag source")
        _json_command(
            [python, "-m", "quantum_bench.cli", "verify", "--input", str(manifest.parent)],
            environment,
        )

    # Rebuild ignored native artifacts from a known clean generated state.
    _run(["make", "-C", "native/quest_cpu", "clean", "clean-quest"], env=environment)
    _run(["make", "-C", "native/upmem/runtime", "clean"], env=environment)
    _run(["make", "build-quest-cpu"], env=environment)
    for tasklets in (1, 8, 24):
        _run(["make", "build-upmem-runtime", f"UPMEM_TASKLETS={tasklets}"], env=environment)
    _run([python, "-m", "pytest", "-q"], env=environment)
    _run([python, "-m", "ruff", "check", "src", "tests", "scripts"], env=environment)
    _run([python, "-m", "build", "--no-isolation", "--outdir", str(output / "dist")], env=environment)

    direct_summary_path = output / "sdk-direct-cases.json"
    _run(
        [python, "-m", "pytest", "-q", "tests/test_upmem_kernel_simulator.py"],
        env={
            **environment,
            "UPMEM_REQUIRE_SDK_SIMULATOR": "1",
            "UPMEM_SDK_SIMULATOR_CASE_SUMMARY": str(direct_summary_path),
        },
    )
    direct_summary = _assert_direct_cases(direct_summary_path)

    cpu_plan, cpu_run, cpu_report = (output / name for name in ("cpu-plan", "cpu-run", "cpu-report"))
    sim_plan, sim_run, sim_report = (output / name for name in ("simulator-plan", "simulator-run", "simulator-report"))
    _run(["make", "plan", "CONFIG=configs/tn_benchmark_reset.yml", f"OUTPUT={cpu_plan}"], env=environment)
    _run(["make", "run", "CONFIG=configs/tn_benchmark_reset.yml", f"OUTPUT={cpu_run}"], env=environment)
    cpu_summary = _json_command(["make", "-s", "--no-print-directory", "verify", f"INPUT={cpu_run}"], environment)
    _assert_summary(cpu_summary, samples=12, sessions=0)
    _run(["make", "report", f"INPUT={cpu_run}", f"REPORT_OUTPUT={cpu_report}"], env=environment)

    _run(["make", "plan", "CONFIG=configs/tn_benchmark_simulator.yml", f"OUTPUT={sim_plan}"], env=environment)
    _run(["make", "run", "CONFIG=configs/tn_benchmark_simulator.yml", f"OUTPUT={sim_run}"], env=environment)
    simulator_summary = _json_command(["make", "-s", "--no-print-directory", "verify", f"INPUT={sim_run}"], environment)
    _assert_summary(simulator_summary, samples=12, sessions=12)
    _run(["make", "report", f"INPUT={sim_run}", f"REPORT_OUTPUT={sim_report}"], env=environment)

    planned_templates: list[dict[str, str]] = []
    for name in (
        "tn_benchmark_physical_smoke.yml",
        "tn_benchmark_physical_scaling_diagnostic.yml",
        "tn_benchmark_physical_scaling.yml",
        "tn_benchmark_physical_scaling_confirmation.yml",
    ):
        output_plan = output / f"plan-{Path(name).stem}"
        _run(["make", "plan", f"CONFIG=configs/{name}", f"OUTPUT={output_plan}"], env=environment)
        planned_templates.append({"config": name, "plan": str(output_plan / "plan.json")})
    _run(
        [
            python,
            "scripts/select_m7c_workload.py",
            "--check",
            "configs/m7c_workload_selection.json",
            "--config",
            "configs/tn_benchmark_physical_scaling_diagnostic.yml",
        ],
        env=environment,
    )

    artifact_paths = [
        archive,
        outer_checksum,
        direct_summary_path,
        ROOT / "configs/m7c_workload_selection.json",
        ROOT / "configs/tn_benchmark_physical_smoke.yml",
        ROOT / "configs/tn_benchmark_physical_scaling_diagnostic.yml",
        ROOT / "configs/tn_benchmark_physical_scaling.yml",
        ROOT / "configs/tn_benchmark_physical_scaling_confirmation.yml",
        cpu_plan / "plan.json",
        cpu_run / "manifest.json",
        cpu_run / "samples.jsonl",
        cpu_run / "sessions.jsonl",
        cpu_report / "report.json",
        sim_plan / "plan.json",
        sim_run / "manifest.json",
        sim_run / "samples.jsonl",
        sim_run / "sessions.jsonl",
        sim_report / "report.json",
        *(Path(item["plan"]) for item in planned_templates),
        ROOT / "native/quest_cpu/bin/quest_runner",
        *(ROOT / "native/upmem/runtime/bin" / name for name in (
            "dpu_gemm_tile_v4_t1", "dpu_gemm_tile_v4_t8", "dpu_gemm_tile_v4_t24",
        )),
        *(output / "dist").glob("*"),
    ]
    qualification = {
        "branch": branch,
        "source_commit": source_commit,
        "source_worktree_dirty": False,
        "physical_qualification_status": "pending",
        "m7b_release_verified": True,
        "cpu_verification": cpu_summary,
        "simulator_verification": simulator_summary,
        "direct_sdk_cases": direct_summary,
        "planned_templates": planned_templates,
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "qualification.txt").write_text(
        "\n".join(
            (
                f"branch: {branch}",
                f"source_commit: {source_commit}",
                "physical_qualification_status: pending",
                "cpu_samples: 12",
                "simulator_samples: 12",
                "simulator_sessions: 12",
                "direct_sdk_cases: 8/8 passed",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_hashes(
        output,
        [*artifact_paths, output / "qualification.json", output / "qualification.txt"],
    )
    if _git_output("status", "--porcelain"):
        raise ValueError("qualification changed the Git worktree")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = qualify(args.output.resolve())
    except (OSError, RuntimeError, ValueError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
