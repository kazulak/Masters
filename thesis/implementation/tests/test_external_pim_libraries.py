from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.benchmark_matrix_report import run_benchmark_matrix_report
import quantum_bench.bench.benchmark_matrix_report as matrix_module
from quantum_bench.bench.upmem_external_libs_check import run_upmem_external_libs_check
import quantum_bench.bench.upmem_external_libs_check as external_check_module
from quantum_bench.targets.upmem.environment import CommandExecutionRecord
from quantum_bench.targets.upmem.external_libs import (
    EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
    build_external_pim_libraries_report,
    discover_pid_comm_source,
    inspect_pid_comm_capability,
    inspect_simplepim_capability,
)


ROOT = Path(__file__).resolve().parents[1]


def _fake_simplepim_tree(path: Path) -> Path:
    (path / "benchmarks" / "va").mkdir(parents=True)
    (path / "benchmarks" / "va" / "Makefile").write_text("va:\n\t@true\n", encoding="utf-8")
    (path / "lib" / "management").mkdir(parents=True)
    (path / "lib" / "communication").mkdir(parents=True)
    (path / "lib" / "communication" / "CommOps.h").write_text(
        "void simplepim_allreduce(void);\nvoid simplepim_broadcast(void);\n",
        encoding="utf-8",
    )
    (path / "lib" / "processing" / "map").mkdir(parents=True)
    (path / "lib" / "processing" / "zip").mkdir(parents=True)
    (path / "lib" / "processing" / "gen_red").mkdir(parents=True)
    (path / "lib" / "processing" / "map" / "map_dpu.c").write_text(
        "#include <stdint.h>\nvoid f(uint8_t *a, int32_t *out){ /* MRAM WRAM */ }\n",
        encoding="utf-8",
    )
    return path


def _passed(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
    return CommandExecutionRecord(
        command=tuple(command),
        cwd=kwargs.get("cwd_label") if isinstance(kwargs.get("cwd_label"), str) else None,
        return_code=0,
        timed_out=False,
        stdout_snippet="ok",
        stderr_snippet="",
        elapsed_time_s=0.001,
        status="passed",
    )


def _failed(command: tuple[str, ...] | list[str], **kwargs: object) -> CommandExecutionRecord:
    return CommandExecutionRecord(
        command=tuple(command),
        cwd=kwargs.get("cwd_label") if isinstance(kwargs.get("cwd_label"), str) else None,
        return_code=2,
        timed_out=False,
        stdout_snippet="",
        stderr_snippet="failed",
        elapsed_time_s=0.001,
        status="failed",
    )


def test_simplepim_missing_is_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    report = inspect_simplepim_capability(root, env={})

    assert report.status == "unavailable"
    assert report.simplepim_detected is False
    assert report.simplepim_blocker_reason == "simplepim_source_unavailable"
    assert report.candidate.top_level_benchmark_route is False


def test_simplepim_marker_evidence_does_not_prove_capability(tmp_path: Path) -> None:
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")
    (simplepim / "README.md").write_text("This README mentions gemm and matmul only.\n", encoding="utf-8")
    report = inspect_simplepim_capability(tmp_path, simplepim_home=str(simplepim), env={})

    assert report.management_api.evidence_detected is True
    assert report.communication_api.evidence_detected is True
    assert report.map_zip_reduce_api.evidence_detected is True
    assert report.int8.evidence_detected is True
    assert report.int8.capability_proven is False
    assert report.int32_accumulation.evidence_detected is True
    assert report.int32_accumulation.capability_proven is False
    assert report.ready_gemm_primitive_detected is False
    assert report.ready_gemm_capability_proven is False
    assert report.simplepim_gemm_ready is False


def test_simplepim_source_gemm_marker_is_evidence_not_proof(tmp_path: Path) -> None:
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")
    (simplepim / "lib" / "processing" / "map" / "gemm.c").write_text(
        "void gemm(int8_t *a, int8_t *b, int32_t *c) {}\n",
        encoding="utf-8",
    )
    report = inspect_simplepim_capability(tmp_path, simplepim_home=str(simplepim), env={})

    assert report.ready_gemm_primitive_detected is True
    assert report.ready_gemm_evidence_paths == ("lib/processing/map/gemm.c",)
    assert report.ready_gemm_capability_proven is False
    assert report.simplepim_gemm_ready is False


def test_pid_comm_missing_and_fake_collective_evidence(tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    missing = inspect_pid_comm_capability(root, env={})
    pid = tmp_path / "PID-Comm"
    pid.mkdir()
    (pid / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (pid / "collectives.c").write_text(
        "void broadcast(void){} void scatter(void){} void gather(void){} void allreduce(void){}\n",
        encoding="utf-8",
    )
    fake = inspect_pid_comm_capability(
        root,
        pid_comm_home=str(pid),
        check_build=True,
        env={},
        command_runner=_passed,
    )

    assert missing.status == "unavailable"
    assert missing.pid_comm_blocker_reason == "pid_comm_source_unavailable"
    assert fake.status == "available"
    assert fake.collective_api.evidence_detected is True
    assert fake.collective_capability_proven is False
    assert fake.build_check_status == "available"
    assert fake.candidate.top_level_benchmark_route is False


def test_pid_comm_build_dry_run_failure_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    pid = tmp_path / "PID-Comm"
    pid.mkdir()
    (pid / "Makefile").write_text("all:\n\tfalse\n", encoding="utf-8")
    report = inspect_pid_comm_capability(
        root,
        pid_comm_home=str(pid),
        check_build=True,
        env={},
        command_runner=_failed,
    )

    assert report.build_check_status == "build_failed"
    assert report.pid_comm_blocker_reason == "pid_comm_build_dry_run_failed"


def test_discover_pid_comm_source_order(tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    cli = tmp_path / "cli_pid"
    env = tmp_path / "env_pid"
    fallback = tmp_path / "legacy" / "extern" / "PID-Comm"
    cli.mkdir()
    env.mkdir()
    fallback.mkdir(parents=True)

    assert discover_pid_comm_source(root, pid_comm_home=str(cli), env={"PID_COMM_HOME": str(env)})["source"] == "cli"
    assert discover_pid_comm_source(root, env={"PID_COMM_HOME": str(env)})["source"] == "environment"
    assert discover_pid_comm_source(root, env={})["home"] == str(fallback)


def test_external_pim_libraries_report_serializes_without_execution_claims(tmp_path: Path) -> None:
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")
    report = build_external_pim_libraries_report(
        tmp_path,
        simplepim_home=str(simplepim),
        env={"UPMEM_HOME": "/sdk"},
        path_lookup=lambda command: f"/sdk/{command}" if command in {"dpu-upmem-dpurte-clang", "dpu-pkg-config"} else None,
        command_runner=_passed,
    )
    payload = report.to_json_dict()

    assert payload["schema_version"] == EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION
    assert payload["metadata"]["simplepim_kernel_executed"] is False
    assert payload["metadata"]["pid_comm_collective_executed"] is False
    assert all(candidate["top_level_benchmark_route"] is False for candidate in payload["l1_l2_compute_backend_candidates"])
    assert all(candidate["top_level_benchmark_route"] is False for candidate in payload["l3_communication_backend_candidates"])


def test_upmem_external_libs_check_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(external_check_module, "capture_environment", lambda root_dir: {})
    simplepim = _fake_simplepim_tree(tmp_path / "SimplePIM")

    run_dir, artifact_path, status = run_upmem_external_libs_check(
        tmp_path / "implementation",
        simplepim_home=str(simplepim),
        env={},
        command_runner=_passed,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert status in {"blocked", "available", "unavailable"}
    assert payload["schema_version"] == EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION
    assert (run_dir / "external_pim_libraries.csv").exists()
    assert (run_dir / "external_pim_libraries_summary.md").exists()


def test_benchmark_matrix_optional_external_report_keeps_upmem_unified(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(matrix_module, "capture_environment", lambda root_dir: {})
    report = build_external_pim_libraries_report(
        tmp_path,
        env={},
        path_lookup=lambda command: None,
        command_runner=_passed,
    )
    report_path = tmp_path / "external_pim_libraries.json"
    report_path.write_text(json.dumps(report.to_json_dict()), encoding="utf-8")

    default_run = run_benchmark_matrix_report(
        tmp_path,
        ROOT / "configs" / "benchmark_matrix.yml",
        output_plots=False,
    )
    linked_run = run_benchmark_matrix_report(
        tmp_path,
        ROOT / "configs" / "benchmark_matrix.yml",
        output_plots=False,
        external_libs_report_path=report_path,
    )
    default_payload = json.loads((default_run / "benchmark_matrix.json").read_text(encoding="utf-8"))
    linked_payload = json.loads((linked_run / "benchmark_matrix.json").read_text(encoding="utf-8"))
    default_upmem = next(row for row in default_payload["benchmark_matrix_rows"] if row["route_category"] == "upmem_tn_runtime")
    linked_upmem = next(row for row in linked_payload["benchmark_matrix_rows"] if row["route_category"] == "upmem_tn_runtime")
    linked_categories = {row["route_category"] for row in linked_payload["benchmark_matrix_rows"] if row["route_category"].startswith("upmem")}

    assert default_upmem["simplepim_candidate_status"] == "not_checked"
    assert linked_upmem["simplepim_candidate_status"] == "unavailable"
    assert linked_upmem["pid_comm_candidate_status"] == "unavailable"
    assert linked_categories == {"upmem_tn_runtime"}
