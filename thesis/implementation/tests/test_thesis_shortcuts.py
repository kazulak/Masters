from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_makefile_shortcuts_are_defined() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "CPU_SUITE ?= configs/suites/cpu_evidence.yml" in text
    assert "GPU_VERIFY ?= quest-hip" in text
    for target in (
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "thesis-benchmark",
        "thesis-results",
        "thesis-results-resume",
        "thesis-report",
        "parallelism-report",
        "research-plan",
        "research-benchmarks",
        "research-report",
        "report-latest",
        "compare-latest",
        "clean-generated",
    ):
        assert f"{target}:" in text
    assert "bench-cpu: build-quest-cpu" in text
    assert "compare-results --inputs \"$$cpu_run\" \"$$upmem_run\"" in text
    assert "THESIS_FULL_STATE_SUITE ?= configs/suites/manual/thesis_full_state_cpu_gpu.yml" in text
    assert "THESIS_TN_QUIMB_SUITE ?= configs/suites/manual/thesis_cpu_tn_quimb.yml" in text
    assert "THESIS_TN_QUANT_SUITE ?= configs/suites/manual/thesis_tn_paths_quantization.yml" in text
    assert "THESIS_UPMEM_SUITE ?= configs/suites/manual/thesis_upmem_quantization_boundary.yml" in text
    assert "PARALLELISM_SLICING_FRONTIER_SUITE ?= configs/suites/diagnostics/cpu_slicing_vs_frontier_quick.yml" in text
    assert "PARALLELISM_HYBRID_SUITE ?= configs/suites/diagnostics/cpu_hybrid_quick.yml" in text
    assert "PARALLELISM_UPMEM_ASSIGNMENT_SUITE ?= configs/suites/upmem_sim_evidence.yml" in text
    assert "PARALLELISM_INCLUDE_UPMEM_FRONTIER ?= 0" in text
    assert "scripts/thesis_report.py --inputs \"$$full_state_run\" \"$$tn_quimb_run\" \"$$tn_quant_run\" \"$$upmem_run\"" in text
    assert "upmem-multi-dpu-assignment --suite $(PARALLELISM_UPMEM_ASSIGNMENT_SUITE)" in text
    assert "upmem-taskgraph-frontier-runtime --case $(PARALLELISM_UPMEM_FRONTIER_CASE)" in text
    assert "compare-results --inputs $$inputs" in text
    assert "FULL_STATE_RUN=runs/evidence/thesis_full_state_cpu_gpu/simulation_backend_compare/<run_id>" in text
    assert "runs/comparisons/$$suite_id/report_run/$$timestamp" in text
    assert "report-run --input runs/latest --out" in text
    assert "runs/comparisons/$$suite_id/latest_single_run/$$timestamp" in text
    assert "CLEAN_RUNS=1" in text


def test_readme_documents_shortcut_targets() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Thesis Evidence Shortcuts" in text
    for command in (
        "make doctor",
        "make build-quest-cpu",
        "make bench-cpu",
        "make bench-gpu",
        "make bench-upmem-sim",
        "make thesis-benchmark",
        "make thesis-results",
        "make thesis-results-resume",
        "make thesis-report",
        "make parallelism-report",
        "make research-plan",
        "make research-benchmarks",
        "make research-report",
        "make report-latest",
        "make compare-latest",
        "make clean-generated",
    ):
        assert command in text
    assert "## Canonical Suites" in text
    for suite_name in (
        "smoke.yml",
        "cpu_evidence.yml",
        "gpu_evidence.yml",
        "cpu_gpu_sweep.yml",
        "upmem_sim_evidence.yml",
        "upmem_generic_sweep.yml",
        "manual_large.yml",
    ):
        assert f"configs/suites/{suite_name}" in text
    assert "configs/suites/diagnostics/" in text
    assert "configs/suites/manual/" in text


def test_makefile_targets_parse_with_dry_run() -> None:
    if shutil.which("make") is None:
        return
    for target in (
        "build-quest-cpu",
        "doctor",
        "bench-cpu",
        "bench-gpu",
        "bench-upmem-sim",
        "thesis-benchmark",
        "thesis-results",
        "thesis-results-resume",
        "thesis-report",
        "parallelism-report",
        "research-plan",
        "research-benchmarks",
        "research-report",
        "report-latest",
        "compare-latest",
        "clean-generated",
    ):
        result = subprocess.run(["make", "-n", target], cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert target != "bench-cpu" or "configs/suites/cpu_evidence.yml" in result.stdout
        assert target != "bench-gpu" or "configs/suites/gpu_evidence.yml" in result.stdout
        assert target != "bench-upmem-sim" or "configs/suites/upmem_sim_evidence.yml" in result.stdout
        assert target != "thesis-benchmark" or "thesis_benchmark_cpu_upmem" in result.stdout
        assert target != "thesis-results" or "configs/suites/manual/thesis_full_state_cpu_gpu.yml" in result.stdout
        assert target != "thesis-results" or "scripts/thesis_report.py --inputs" in result.stdout
        assert target != "thesis-results-resume" or "Set FULL_STATE_RUN=" in result.stdout
        assert target != "thesis-report" or "thesis_report.py --inputs" in result.stdout
        assert target != "parallelism-report" or "cpu_slicing_vs_frontier_quick.yml" in result.stdout
        assert target != "parallelism-report" or "upmem-multi-dpu-assignment" in result.stdout
        assert target != "parallelism-report" or "PARALLELISM_INCLUDE_UPMEM_FRONTIER" in (ROOT / "Makefile").read_text(encoding="utf-8")
        assert target != "parallelism-report" or "parallelism_evidence" in result.stdout
        assert target != "research-plan" or "research_benchmark_pack.py plan" in result.stdout
        assert target != "research-benchmarks" or "research_benchmark_pack.py run" in result.stdout
        assert target != "research-report" or "research_benchmark_pack.py report" in result.stdout
        assert "simulation_backend_compare_thesis_small.yml" not in result.stdout

    opt_in = subprocess.run(
        ["make", "-n", "parallelism-report", "PARALLELISM_INCLUDE_UPMEM_FRONTIER=1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert opt_in.returncode == 0, opt_in.stderr
    assert "upmem-taskgraph-frontier-runtime" in opt_in.stdout
    assert "--frontier-worker-count 1" in opt_in.stdout


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
