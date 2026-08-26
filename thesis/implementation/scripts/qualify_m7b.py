#!/usr/bin/env python3
"""Create an exact-head, software-only M7B qualification bundle.

This command never enables physical hardware.  It is intended for the
designated SDK qualification machine after the M7B source commit is clean.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
M7A_TAG = "thesis-m7a-wram-kernel-software-ready-v1"


def _run(
    command: list[str], *, env: dict[str, str] | None = None, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
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
    return _run(["git", *arguments], capture=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum_file(checksum_file: Path, artifact: Path) -> None:
    expected: str | None = None
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        name = fields[1].lstrip("*")
        if name == artifact.name:
            expected = fields[0]
            break
    if expected is None:
        raise ValueError(f"{checksum_file.name} lacks a digest for {artifact.name}")
    actual = _sha256(artifact)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {artifact.name}")


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract only regular files and directories below destination."""

    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        names: set[str] = set()
        for member in members:
            name = member.name
            pure = PurePosixPath(name)
            if (
                not name
                or pure.is_absolute()
                or ".." in pure.parts
                or name in names
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe archive member: {name!r}")
            names.add(name)
            target = (destination / pure).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"archive member escapes extraction root: {name!r}")
        for member in members:
            target = destination / PurePosixPath(member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _release_assets(destination: Path) -> tuple[Path, Path]:
    _run(["gh", "release", "download", M7A_TAG, "--dir", str(destination)])
    archive = destination / f"{M7A_TAG}.tar.gz"
    checksum = destination / f"{M7A_TAG}.tar.gz.sha256"
    if not archive.is_file() or not checksum.is_file():
        raise ValueError(
            "M7A release must provide the archive and a distinct outer .tar.gz.sha256 asset"
        )
    _verify_checksum_file(checksum, archive)
    return archive, checksum


def _verify_internal_hashes(root: Path) -> tuple[str, ...]:
    """Verify hashes for bundled evidence and list declared external provenance."""

    checksum_files = list(root.rglob("SHA256SUMS"))
    if len(checksum_files) != 1:
        raise ValueError("M7A archive must contain exactly one internal SHA256SUMS")
    checksum_file = checksum_files[0]
    bundle_root = checksum_file.parent
    bundle_prefix = f"runs/{bundle_root.name}/"
    external: list[str] = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        expected, name = fields[0], fields[1].lstrip("*")
        if not name.startswith(bundle_prefix):
            external.append(name)
            continue
        artifact = bundle_root / name.removeprefix(bundle_prefix)
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"internal M7A checksum mismatch: {name}")
    return tuple(sorted(external))


def _json_output(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    result = _run(command, env=env, capture=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"command did not emit JSON: {' '.join(command)}") from exc
    if not isinstance(payload, dict):
        raise ValueError("command JSON must be an object")
    return payload


def _assert_summary(
    summary: dict[str, Any], *, samples: int, sessions: int
) -> None:
    expected = {
        "status": "completed",
        "sample_count": samples,
        "session_count": sessions,
        "success_count": samples,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"unexpected verification {key}: {summary.get(key)!r}")


def _write_hashes(root: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths):
        if path.is_file():
            try:
                label = path.relative_to(root)
            except ValueError:
                label = Path("repository") / path.relative_to(ROOT)
            lines.append(f"{_sha256(path)}  {label}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qualify(output: Path) -> Path:
    branch = _git_output("branch", "--show-current")
    if branch != "feature/m7b-prephysical":
        raise ValueError(f"M7B qualification requires feature/m7b-prephysical, got {branch}")
    if _git_output("status", "--porcelain"):
        raise ValueError("M7B qualification requires a clean Git worktree")
    source_commit = _git_output("rev-parse", "HEAD")
    if output.exists():
        raise ValueError(f"qualification output must be absent: {output}")
    output.mkdir(parents=True)
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        "MPLCONFIGDIR": str(output / "matplotlib"),
    }
    python = sys.executable

    for command in (("jq",), ("script",), ("sha256sum",), ("gh",)):
        if shutil.which(command[0]) is None:
            raise ValueError(f"required qualification tool is unavailable: {command[0]}")

    release_dir = output / "m7a-release"
    archive, outer_checksums = _release_assets(release_dir)
    extracted = release_dir / "extracted"
    _safe_extract_tar(archive, extracted)
    m7a_external_provenance = _verify_internal_hashes(extracted)
    m7a_manifests = sorted(extracted.rglob("manifest.json"))
    if not m7a_manifests:
        raise ValueError("M7A archive contains no evidence manifest")
    m7a_source = _git_output("rev-parse", f"{M7A_TAG}^{{}}")
    for manifest in m7a_manifests:
        persisted = json.loads(manifest.read_text(encoding="utf-8"))
        if persisted.get("source_commit") != m7a_source:
            raise ValueError("M7A release evidence does not bind to the M7A tag source")
        _json_output(
            [python, "-m", "quantum_bench.cli", "verify", "--input", str(manifest.parent)],
            environment,
        )

    _run([python, "-m", "pytest", "-q"], env=environment)
    _run([python, "-m", "ruff", "check", "src", "tests", "scripts"], env=environment)
    _run(
        [python, "-m", "build", "--no-isolation", "--outdir", str(output / "dist")],
        env=environment,
    )
    _run(["make", "build-quest-cpu"], env=environment)
    for tasklets in (1, 8, 24):
        _run(["make", "build-upmem-runtime", f"UPMEM_TASKLETS={tasklets}"], env=environment)

    direct_summary = output / "sdk-direct-cases.json"
    _run(
        [python, "-m", "pytest", "-q", "tests/test_upmem_kernel_simulator.py"],
        env={
            **environment,
            "UPMEM_REQUIRE_SDK_SIMULATOR": "1",
            "UPMEM_SDK_SIMULATOR_CASE_SUMMARY": str(direct_summary),
        },
    )
    direct = json.loads(direct_summary.read_text(encoding="utf-8"))
    if (
        direct.get("required_case_ids") != direct.get("executed_case_ids")
        or direct.get("required_case_ids") != direct.get("passed_case_ids")
        or direct.get("failed_case_ids") != []
        or direct.get("skipped_case_ids") != []
    ):
        raise ValueError("direct SDK case matrix is incomplete")

    cpu_plan, cpu_run, cpu_report = (output / name for name in ("cpu-plan", "cpu-run", "cpu-report"))
    sim_plan, sim_run, sim_report = (
        output / name for name in ("simulator-plan", "simulator-run", "simulator-report")
    )
    _run(["make", "plan", "CONFIG=configs/tn_benchmark_reset.yml", f"OUTPUT={cpu_plan}"], env=environment)
    _run(["make", "run", "CONFIG=configs/tn_benchmark_reset.yml", f"OUTPUT={cpu_run}"], env=environment)
    cpu_summary = _json_output(
        ["make", "-s", "--no-print-directory", "verify", f"INPUT={cpu_run}"], environment
    )
    _assert_summary(cpu_summary, samples=12, sessions=0)
    _run(["make", "report", f"INPUT={cpu_run}", f"REPORT_OUTPUT={cpu_report}"], env=environment)

    _run(["make", "plan", "CONFIG=configs/tn_benchmark_simulator.yml", f"OUTPUT={sim_plan}"], env=environment)
    _run(["make", "run", "CONFIG=configs/tn_benchmark_simulator.yml", f"OUTPUT={sim_run}"], env=environment)
    simulator_summary = _json_output(
        ["make", "-s", "--no-print-directory", "verify", f"INPUT={sim_run}"], environment
    )
    _assert_summary(simulator_summary, samples=12, sessions=12)
    _run(["make", "report", f"INPUT={sim_run}", f"REPORT_OUTPUT={sim_report}"], env=environment)

    artifact_paths = [
        archive,
        outer_checksums,
        direct_summary,
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
        ROOT / "native/quest_cpu/bin/quest_runner",
        *(ROOT / "native/upmem/runtime/bin" / name for name in (
            "dpu_gemm_tile_v4_t1", "dpu_gemm_tile_v4_t8", "dpu_gemm_tile_v4_t24",
        )),
        *(output / "dist").glob("*"),
    ]
    _write_hashes(output, artifact_paths)
    qualification = {
        "branch": branch,
        "source_commit": source_commit,
        "source_worktree_dirty": False,
        "physical_qualification_status": "pending",
        "cpu_verification": cpu_summary,
        "simulator_verification": simulator_summary,
        "direct_sdk_cases": direct,
        "m7a_release_verified": True,
        "m7a_unbundled_provenance_hashes": list(m7a_external_provenance),
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if _git_output("status", "--porcelain"):
        raise ValueError("qualification changed the Git worktree")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        completed = qualify(args.output.resolve())
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "output": str(completed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
