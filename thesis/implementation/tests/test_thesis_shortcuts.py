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
    for target in (
        "help",
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "thesis-run",
        "thesis-promote",
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
    assert "scripts/thesis_snapshot.py promote" in text
    assert "scripts/thesis_snapshot.py report" in text
    assert "scripts/thesis_runs.py prune" in text
    assert "scripts/thesis_runs.py list" in text


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
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "thesis-run",
        "thesis-promote",
        "thesis-verify",
        "thesis-report",
        "thesis-clean",
        "list-runs",
        "research-plan",
        "clean-generated",
    ):
        result = subprocess.run(["make", "-n", target], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert target != "bench-cpu" or "configs/suites/cpu_evidence.yml" in result.stdout
        assert target != "bench-gpu" or "configs/suites/gpu_evidence.yml" in result.stdout
        assert target != "bench-upmem-sim" or "configs/suites/upmem_sim_evidence.yml" in result.stdout
        assert target != "thesis-run" or "research_benchmark_pack.py run --full" in result.stdout
        assert target != "thesis-promote" or "thesis_snapshot.py promote" in result.stdout
        assert target != "thesis-verify" or "thesis_snapshot.py verify" in result.stdout
        assert target != "thesis-report" or "thesis_snapshot.py report" in result.stdout
        assert target != "thesis-clean" or "thesis_runs.py prune" in result.stdout
        assert target != "list-runs" or "thesis_runs.py list" in result.stdout
        assert target != "research-plan" or "research_benchmark_pack.py plan" in result.stdout
        assert "simulation_backend_compare_thesis_small.yml" not in result.stdout


def test_top_level_suite_family_is_canonical() -> None:
    top_level = {path.name for path in (ROOT / "configs" / "suites").glob("*.yml")}

    assert top_level == {
        "smoke.yml",
        "cpu_evidence.yml",
        "gpu_evidence.yml",
        "cpu_gpu_sweep.yml",
        "upmem_sim_evidence.yml",
        "upmem_generic_sweep.yml",
        "manual_large.yml",
    }
    assert (ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml").exists()
    assert (ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_quick.yml").exists()
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
    for marker in ("python", "dependency:quantum_bench", "quest_cpu", "gpu_rocm", "upmem_sdk"):
        assert marker in result.stdout
    assert "normalized_records" not in result.stdout


def test_evidence_shortcut_helper_validates_gpu_and_upmem_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"suite_id": "suite_a"}), encoding="utf-8")
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
    (run_dir / "normalized_records.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

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
        [sys.executable, "scripts/evidence_shortcuts.py", "check-gpu", str(bad_run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_gpu.returncode == 2


def test_evidence_shortcut_helper_reports_missing_verified_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "normalized_records.jsonl").write_text(json.dumps({"contraction_execution_target": "cpu"}), encoding="utf-8")

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
