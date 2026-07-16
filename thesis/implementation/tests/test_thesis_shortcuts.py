from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_bench.core.records import CircuitOperation, CircuitSpec
from quantum_bench.tn import build_tensor_network


ROOT = Path(__file__).resolve().parents[1]


def test_makefile_shortcuts_are_defined() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "CPU_SUITE ?= configs/suites/cpu_evidence.yml" in text
    assert "GPU_VERIFY ?= quest-hip" in text
    assert (
        "UPMEM_HW_TASKGRAPH_RUN ?= "
        "runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph/latest"
    ) in text
    assert (
        "UPMEM_HW_TASKGRAPH_STUDY_SUITE ?= configs/suites/upmem_hardware_taskgraph_path_quantization.yml"
        in text
    )
    assert (
        "UPMEM_HW_TASKGRAPH_STUDY_RUN ?= "
        "runs/evidence/upmem_hardware_taskgraph_path_quantization/upmem_hw_taskgraph_study/latest"
    ) in text
    assert "UPMEM_HW_TASKGRAPH_RESIDENT_SUITE ?= configs/suites/upmem_hardware_taskgraph_resident_path_quantization.yml" in text
    assert "UPMEM_HW_TASKGRAPH_RESIDENT_RUN ?= runs/evidence/upmem_hardware_taskgraph_resident_path_quantization/upmem_hw_taskgraph_resident/latest" in text
    for target in (
        "help",
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "upmem-hw-mvp-plan",
        "upmem-hw-mvp",
        "upmem-hw-taskgraph-plan",
        "upmem-hw-taskgraph",
        "upmem-hw-taskgraph-report",
        "upmem-hw-taskgraph-study-plan",
        "upmem-hw-taskgraph-study",
        "upmem-hw-taskgraph-study-report",
        "upmem-hw-taskgraph-resident-plan",
        "upmem-hw-taskgraph-resident",
        "upmem-hw-taskgraph-resident-report",
        "evidence-inbox",
        "thesis-run",
        "thesis-promote",
        "thesis-promote-historical",
        "thesis-verify",
        "thesis-report",
        "thesis-clean",
        "thesis-release",
        "list-runs",
        "research-plan",
        "clean-generated",
    ):
        assert f"{target}:" in text
        assert f"  make {target}" in text
    assert ".PHONY: $(PUBLIC_TARGETS)" in text
    assert "bench-cpu: build-quest-cpu" in text
    assert "thesis-run: build-quest-cpu" in text
    assert "research_benchmark_pack.py run --full" in text
    assert "BENCH_CPU_THREADS ?=" in text
    assert "PYTHON_BIN_DIR := $(abspath $(dir $(PYTHON)))" in text
    assert "export PATH := $(PYTHON_BIN_DIR):$(PATH)" in text
    assert "OPENBLAS_NUM_THREADS=$(BENCH_CPU_THREADS)" in text
    assert "scripts/thesis_snapshot.py promote" in text
    assert "--historical" in text
    assert "scripts/thesis_snapshot.py report" in text
    assert "scripts/thesis_runs.py prune" in text
    assert "scripts/thesis_runs.py list" in text

    cpu_makefile = (ROOT / "native" / "quest_cpu" / "Makefile").read_text(
        encoding="utf-8"
    )
    assert "CC = gcc" in cpu_makefile
    assert "CXX = g++" in cpu_makefile
    assert "-DCMAKE_C_COMPILER=$(CC)" in cpu_makefile
    assert "-DCMAKE_CXX_COMPILER=$(CXX)" in cpu_makefile


def test_readme_documents_shortcut_targets() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Thesis Workflow" in text
    for command in (
        "make doctor",
        "make thesis-run",
        "make thesis-promote",
        "make thesis-verify",
        "make thesis-report",
        "make thesis-clean",
        "make thesis-release",
        "make list-runs",
    ):
        assert command in text
    assert "thesis_results/current" in text
    assert "contraction_plan_hash" in text


def test_duplicate_gate_wires_are_rejected() -> None:
    circuit = CircuitSpec(
        name="duplicate-wire",
        n_qubits=2,
        operations=(CircuitOperation("cx", (0, 0)),),
        source={},
    )

    with pytest.raises(ValueError, match="Duplicate wire 0"):
        build_tensor_network(circuit)


def test_makefile_targets_parse_with_dry_run() -> None:
    if shutil.which("make") is None:
        return
    for target in (
        "setup",
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "upmem-hw-mvp-plan",
        "upmem-hw-mvp",
        "upmem-hw-generic-plan",
        "upmem-hw-generic-mvp",
        "upmem-hw-taskgraph-plan",
        "upmem-hw-taskgraph",
        "upmem-hw-taskgraph-report",
        "upmem-hw-taskgraph-study-plan",
        "upmem-hw-taskgraph-study",
        "upmem-hw-taskgraph-study-report",
        "upmem-hw-taskgraph-resident-plan",
        "upmem-hw-taskgraph-resident",
        "upmem-hw-taskgraph-resident-report",
        "evidence-inbox",
        "thesis-run",
        "thesis-promote",
        "thesis-promote-historical",
        "thesis-verify",
        "thesis-report",
        "thesis-clean",
        "list-runs",
        "research-plan",
        "planner-evidence",
        "planner-report",
        "archive-evidence",
        "clean-generated",
    ):
        result = subprocess.run(
            ["make", "-n", target],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert target != "setup" or "clean-quest" in result.stdout
        assert (
            target != "bench-cpu" or "configs/suites/cpu_evidence.yml" in result.stdout
        )
        assert (
            target != "bench-gpu" or "configs/suites/gpu_evidence.yml" in result.stdout
        )
        assert (
            target != "bench-upmem-sim"
            or "configs/suites/upmem_sim_evidence.yml" in result.stdout
        )
        assert (
            target != "upmem-hw-mvp-plan"
            or "configs/suites/upmem_hardware_mvp.yml" in result.stdout
        )
        assert (
            target != "upmem-hw-mvp" or "UPMEM_ALLOW_PHYSICAL_HARDWARE" in result.stdout
        )
        assert (
            target != "upmem-hw-generic-plan"
            or "configs/suites/upmem_hardware_generic_mvp.yml" in result.stdout
        )
        assert (
            target != "upmem-hw-generic-mvp"
            or "UPMEM_ALLOW_PHYSICAL_HARDWARE" in result.stdout
        )
        assert (
            target != "upmem-hw-taskgraph-plan"
            or "configs/suites/upmem_hardware_taskgraph_correctness.yml"
            in result.stdout
        )
        assert (
            target != "upmem-hw-taskgraph"
            or "UPMEM_ALLOW_PHYSICAL_HARDWARE" in result.stdout
        )
        assert (
            target != "upmem-hw-taskgraph-report"
            or "research_benchmark_pack.py report" in result.stdout
        )
        assert (
            target != "upmem-hw-taskgraph-study-plan"
            or "configs/suites/upmem_hardware_taskgraph_path_quantization.yml"
            in result.stdout
        )
        assert (
            target != "upmem-hw-taskgraph-study"
            or "UPMEM_ALLOW_PHYSICAL_HARDWARE" in result.stdout
        )
        assert target != "upmem-hw-taskgraph-study-report" or (
            "normalized_records.jsonl" in result.stdout
            and "--label upmem_hw_taskgraph_study" in result.stdout
        )
        assert target != "upmem-hw-taskgraph-resident-plan" or "upmem_hardware_taskgraph_resident_path_quantization.yml" in result.stdout
        assert target != "upmem-hw-taskgraph-resident" or "Resident TaskGraph runtime is reserved" in result.stdout
        assert target != "upmem-hw-taskgraph-resident-report" or "upmem_hw_taskgraph_resident" in result.stdout
        assert target != "evidence-inbox" or "runs/inbox/eth" in result.stdout
        assert (
            target != "thesis-run"
            or "research_benchmark_pack.py run --full" in result.stdout
        )
        assert target != "thesis-run" or "BENCH_CPU_THREADS" in result.stdout
        assert (
            target != "thesis-promote" or "thesis_snapshot.py promote" in result.stdout
        )
        assert target != "thesis-promote-historical" or "thesis_snapshot.py promote --historical" in result.stdout
        assert target != "thesis-verify" or "thesis_snapshot.py verify" in result.stdout
        assert target != "thesis-report" or "thesis_snapshot.py report" in result.stdout
        assert target != "thesis-clean" or "thesis_runs.py prune" in result.stdout
        assert target != "list-runs" or "thesis_runs.py list" in result.stdout
        assert (
            target != "research-plan"
            or "research_benchmark_pack.py plan" in result.stdout
        )
        assert target != "planner-evidence" or "--label planner_v2" in result.stdout
        assert target != "planner-report" or "--label planner_v2" in result.stdout
        assert target != "archive-evidence" or "thesis_runs.py archive" in result.stdout
        assert "simulation_backend_compare_thesis_small.yml" not in result.stdout


def test_top_level_suite_family_is_canonical() -> None:
    top_level = {path.name for path in (ROOT / "configs" / "suites").glob("*.yml")}

    canonical_top_level = {
        "smoke.yml",
        "cpu_evidence.yml",
        "gpu_evidence.yml",
        "cpu_gpu_sweep.yml",
        "upmem_sim_evidence.yml",
        "upmem_generic_sweep.yml",
        "upmem_hardware_mvp.yml",
        "upmem_hardware_generic_mvp.yml",
        "upmem_hardware_taskgraph_correctness.yml",
        "upmem_hardware_taskgraph_path_quantization.yml",
        "manual_large.yml",
    }
    assert canonical_top_level <= top_level
    assert top_level - canonical_top_level <= {
        "upmem_hardware_taskgraph_resident_path_quantization.yml"
    }
    assert (
        ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml"
    ).exists()
    assert (
        ROOT
        / "configs"
        / "suites"
        / "diagnostics"
        / "simulation_backend_compare_quick.yml"
    ).exists()
    assert (ROOT / "configs" / "suites" / "manual" / "cpu_gpu_sweep_tier1.yml").exists()
    assert (ROOT / "configs" / "suites" / "manual" / "cpu_gpu_sweep_tier2.yml").exists()


def test_doctor_reports_prerequisites_without_benchmark_rows() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/doctor.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Thesis benchmark doctor" in result.stdout
    for marker in (
        "python",
        "dependency:quantum_bench",
        "cpu_benchmark_profile",
        "recommended_thesis_run",
        "quest_cpu",
        "gpu_rocm",
        "upmem_sdk",
    ):
        assert marker in result.stdout
    assert "normalized_records" not in result.stdout


def test_thesis_run_requires_explicit_positive_thread_count() -> None:
    if shutil.which("make") is None:
        return
    missing = subprocess.run(
        ["make", "-o", "build-quest-cpu", "thesis-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "Set BENCH_CPU_THREADS" in missing.stderr

    invalid = subprocess.run(
        ["make", "-o", "build-quest-cpu", "thesis-run", "BENCH_CPU_THREADS=zero"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "positive integer" in invalid.stderr


def test_evidence_shortcut_helper_validates_gpu_and_upmem_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"suite_id": "suite_a"}), encoding="utf-8"
    )
    records = [
        {
            "case_id": "gpu_case",
            "contraction_execution_target": "gpu",
            "gpu_backend_verified": True,
            "gpu_program_executed": True,
            "gpu_device_name": "AMD Radeon RX 6600",
            "validation_status": "passed",
        },
        {
            "case_id": "upmem_case",
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": "sdk_simulator",
            "upmem_program_executed": True,
            "cpu_fallback_used": False,
        },
    ]
    (run_dir / "normalized_records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    suite = subprocess.run(
        [sys.executable, "scripts/evidence_shortcuts.py", "suite-id", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    gpu = subprocess.run(
        [sys.executable, "scripts/evidence_shortcuts.py", "check-gpu", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    upmem = subprocess.run(
        [sys.executable, "scripts/evidence_shortcuts.py", "check-upmem", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert suite.returncode == 0
    assert suite.stdout.strip() == "suite_a"
    assert gpu.returncode == 0
    assert "Verified GPU benchmark rows: 1" in gpu.stdout
    assert upmem.returncode == 0
    assert "Verified UPMEM SDK simulator benchmark rows: 1" in upmem.stdout

    bad_run_dir = tmp_path / "bad_gpu"
    bad_run_dir.mkdir()
    (bad_run_dir / "normalized_records.jsonl").write_text(
        json.dumps(
            {
                "case_id": "gpu_case",
                "contraction_execution_target": "gpu",
                "gpu_backend_verified": True,
                "gpu_program_executed": True,
                "gpu_device_name": "AMD Radeon RX 6600",
                "validation_status": "failed",
            }
        ),
        encoding="utf-8",
    )
    bad_gpu = subprocess.run(
        [
            sys.executable,
            "scripts/evidence_shortcuts.py",
            "check-gpu",
            str(bad_run_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_gpu.returncode == 2


def test_evidence_shortcut_helper_reports_missing_verified_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "normalized_records.jsonl").write_text(
        json.dumps({"contraction_execution_target": "cpu"}), encoding="utf-8"
    )

    gpu = subprocess.run(
        [sys.executable, "scripts/evidence_shortcuts.py", "check-gpu", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    upmem = subprocess.run(
        [sys.executable, "scripts/evidence_shortcuts.py", "check-upmem", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert gpu.returncode == 2
    assert "GPU blocker" in gpu.stderr
    assert upmem.returncode == 2
    assert "UPMEM SDK simulator blocker" in upmem.stderr


def test_evidence_shortcut_helper_validates_hardware_mvp_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "hardware"
    run_dir.mkdir()
    record = {
        "case_id": "dense_l1_2x2",
        "contraction_execution_target": "upmem",
        "target_requested": "hardware",
        "target_observed": "hardware",
        "backend_id": "upmem_sdk_hardware_dense",
        "hardware_profile_version": "hardware_mvp_l1_v2",
        "sdk_allocation_profile": "backend=hw",
        "sdk_allocation_profile_verified": True,
        "requested_dpu_count": 1,
        "allocated_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "hardware_allocation_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "exact_integer_match": True,
        "validation_status": "passed",
        "hardware_speedup_applicable": False,
    }
    (run_dir / "normalized_records.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    hardware = subprocess.run(
        [
            sys.executable,
            "scripts/evidence_shortcuts.py",
            "check-upmem-hardware",
            str(run_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert hardware.returncode == 0
    assert "Verified UPMEM hardware MVP rows: 1" in hardware.stdout
