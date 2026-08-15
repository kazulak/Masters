from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_provider_qualification_cli_is_public() -> None:
    result = _run("-m", "quantum_bench.bench", "--help")
    assert result.returncode == 0
    assert "provider-qualification" in result.stdout


def test_provider_make_shortcuts_are_exposed() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-provider-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    qualify = subprocess.run(
        ["make", "-n", "upmem-provider-qualify", "PROVIDER=simplepim"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plan.returncode == 0
    assert "provider-qualification" in plan.stdout
    assert "--prepare-only" in plan.stdout
    assert qualify.returncode == 0
    assert "--execute" in qualify.stdout
    assert "--provider simplepim" in qualify.stdout


def test_make_help_is_a_concise_current_workflow_summary() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if not line.startswith("make[")]
    assert len(lines) <= 18
    for text in (
        "make setup",
        "make doctor",
        "make test",
        "make thesis-run",
        "make thesis-report",
        "make thesis-verify",
        "make m5-circuit-plan",
        "make m5-circuit-smoke",
        "make m5-circuit-study",
        "make m5-circuit-report",
        "M5_CIRCUIT_ROUTES=",
        "make evidence-inbox",
        "make list-runs",
        "make help-all",
    ):
        assert text in result.stdout


def test_make_help_all_keeps_historical_catalog() -> None:
    result = subprocess.run(
        ["make", "help-all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for target in (
        "make upmem-hw-m2-1-plan",
        "make upmem-hw-m4-1-plan",
        "make upmem-hw-m5-4-plan",
        "make upmem-provider-plan",
    ):
        assert target in result.stdout


def test_m2_1_make_shortcuts_use_canonical_suite() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m2-1-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-m2-1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_sliced_resident_m2_1.yml"
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout


def test_m2_2_make_shortcuts_use_canonical_suite() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m2-2-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-m2-2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_sliced_resident_m2_2.yml"
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout


def test_m2_2_report_shortcut_uses_configurable_evidence_run() -> None:
    help_result = subprocess.run(
        ["make", "help-all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = subprocess.run(
        ["make", "-n", "upmem-hw-m2-2-report", "UPMEM_HW_M2_2_RUN=/tmp/m2-2-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "UPMEM_HW_M2_2_RUN=runs/inbox/eth/m2_2/<timestamp>" in help_result.stdout
    assert report.returncode == 0
    assert "scripts/upmem_m2_2_report.py" in report.stdout
    assert "--input /tmp/m2-2-run" in report.stdout


def test_m2_3_make_shortcuts_use_canonical_suite() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m2-3-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-m2-3", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_sliced_resident_m2_3.yml"
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout


def test_m2_3_report_shortcut_documents_eth_inbox_override() -> None:
    help_result = subprocess.run(
        ["make", "help-all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = subprocess.run(
        ["make", "-n", "upmem-hw-m2-3-report", "UPMEM_HW_M2_3_RUN=/tmp/m2-3-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "UPMEM_HW_M2_3_RUN=runs/inbox/eth/m2_3/<timestamp>" in help_result.stdout
    assert report.returncode == 0
    assert "scripts/upmem_m2_3_report.py" in report.stdout
    assert "--input /tmp/m2-3-run" in report.stdout


def test_m5_shortcuts_use_plan_execute_and_focused_report_contract() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m5-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-m5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = subprocess.run(
        [
            "make",
            "-n",
            "upmem-hw-m5-report",
            "UPMEM_HW_M5_RUN=/tmp/m5-physical-latest",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_distributed_m5.yml"
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout
    assert report.returncode == 0
    assert "scripts/upmem_m5_report.py" in report.stdout
    assert "--input /tmp/m5-physical-latest" in report.stdout


def test_m5_4_shortcuts_use_corrected_suite_and_bounded_smoke_gate() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m5-4-plan"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    smoke = subprocess.run(
        [
            "make",
            "-n",
            "upmem-hw-m5-4-smoke",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=/dev/dpu_rank0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    full = subprocess.run(
        [
            "make",
            "-n",
            "upmem-hw-m5-4",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=/dev/dpu_rank0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = subprocess.run(
        [
            "make",
            "-n",
            "upmem-hw-m5-4-report",
            "UPMEM_HW_M5_4_RUN=/tmp/m5-4-physical-latest",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    suite = "configs/suites/upmem_hardware_distributed_m5_4.yml"
    assert suite in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert suite in smoke.stdout
    assert "--execute --dpu-counts 1,2,4,8" in smoke.stdout
    assert suite in full.stdout
    assert "--execute" in full.stdout
    assert "scripts/upmem_m5_report.py" in report.stdout
    assert "--input /tmp/m5-4-physical-latest" in report.stdout
    assert "--output-root ." in report.stdout
    assert "report-run" not in report.stdout
    assert " --out " not in report.stdout


def test_m3_1_frontier_shortcuts_use_canonical_suite_and_opt_in() -> None:
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-frontier-m3-1-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-frontier-m3-1", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_frontier_m3_1.yml"
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "upmem-hardware-frontier-m3-1" in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE=1" in execute.stdout


def test_m6a_frontier_shortcuts_use_canonical_suite_and_opt_in() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("UPMEM_ALLOW_PHYSICAL_HARDWARE", None)
    clean_env.pop("UPMEM_HW_RANK_PATH", None)

    missing_opt_in = subprocess.run(
        [
            "make",
            "upmem-hw-m6a",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=",
            "UPMEM_HW_RANK_PATH=",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    missing_rank = subprocess.run(
        [
            "make",
            "upmem-hw-m6a",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-m6a-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        [
            "make",
            "-n",
            "upmem-hw-m6a",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=/dev/dpu_rank1",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    suite = "configs/suites/upmem_hardware_frontier_m6a.yml"
    assert missing_opt_in.returncode == 2
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE=1" in missing_opt_in.stderr
    assert "quantum_bench.bench" not in missing_opt_in.stdout
    assert missing_rank.returncode == 2
    assert "UPMEM_HW_RANK_PATH" in missing_rank.stderr
    assert "quantum_bench.bench" not in missing_rank.stdout
    assert plan.returncode == 0
    assert suite in plan.stdout
    assert "upmem-hardware-frontier-m6a" in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert execute.returncode == 0
    assert suite in execute.stdout
    assert "--execute" in execute.stdout
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE=1" in execute.stdout
    assert "UPMEM_HW_RANK_PATH" in execute.stdout


def test_m45_simplepim_execution_requires_rank_and_plan_remains_rank_free() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("UPMEM_ALLOW_PHYSICAL_HARDWARE", None)
    clean_env.pop("UPMEM_HW_RANK_PATH", None)
    missing_rank = subprocess.run(
        [
            "make",
            "upmem-simplepim-run",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    plan = subprocess.run(
        ["make", "-n", "upmem-simplepim-plan"],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        [
            "make",
            "-n",
            "upmem-simplepim-run",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=/dev/dpu_rank1",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_rank.returncode == 2
    assert "UPMEM_HW_RANK_PATH" in missing_rank.stderr
    assert "quantum_bench.bench" not in missing_rank.stdout
    assert plan.returncode == 0
    assert "upmem-simplepim-taskgraph" in plan.stdout
    assert "--prepare-only --build" in plan.stdout
    assert "UPMEM_HW_RANK_PATH" not in plan.stdout
    assert execute.returncode == 0
    assert "UPMEM_HW_RANK_PATH" in execute.stdout


def test_m5_circuit_shortcuts_delegate_route_aware_guards_to_the_cli() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("UPMEM_ALLOW_PHYSICAL_HARDWARE", None)
    clean_env.pop("UPMEM_HW_RANK_PATH", None)
    clean_env.pop("UPMEM_HW_RANK_PATHS", None)
    plan = subprocess.run(
        ["make", "-n", "m5-circuit-plan"],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    missing_opt_in = subprocess.run(
        [
            "make",
            "m5-circuit-smoke",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=",
            "UPMEM_HW_RANK_PATH=",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        [
            "make",
            "-n",
            "m5-circuit-study",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
            "UPMEM_HW_RANK_PATH=/dev/dpu_rank0",
        ],
        cwd=ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plan.returncode == 0
    assert "--prepare-only --build" in plan.stdout
    assert "configs/suites/m5_circuit_smoke.yml" in plan.stdout
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" not in plan.stdout
    assert missing_opt_in.returncode == 2
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE=1" in missing_opt_in.stderr
    assert "quantum_bench.bench m5-circuit-study" in missing_opt_in.stdout
    assert execute.returncode == 0
    assert "--rank-paths /dev/dpu_rank0" in execute.stdout
    assert "env -u DPU_BACKEND -u UPMEM_EXECUTION_MODE" not in execute.stdout


def test_m5_circuit_plan_suite_is_overridable() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "m5-circuit-plan",
            "M5_CIRCUIT_PLAN_SUITE=configs/suites/m5_circuit_scaling.yml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "configs/suites/m5_circuit_scaling.yml" in result.stdout
    assert "configs/suites/m5_circuit_smoke.yml" not in result.stdout


def test_m5_circuit_make_routes_expand_to_repeatable_cli_arguments() -> None:
    route_ids = (
        "opt_einsum_greedy__float32_real__numpy_cpu "
        "opt_einsum_greedy__host_packed_int8__numpy_cpu"
    )
    for target in ("m5-circuit-plan", "m5-circuit-smoke", "m5-circuit-study"):
        result = subprocess.run(
            [
                "make",
                "-n",
                target,
                "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
                "UPMEM_HW_RANK_PATH=/dev/dpu_rank0",
                f"M5_CIRCUIT_ROUTES={route_ids}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--route opt_einsum_greedy__float32_real__numpy_cpu" in result.stdout
        assert "--route opt_einsum_greedy__host_packed_int8__numpy_cpu" in result.stdout


def test_m5_circuit_physical_shortcuts_reject_backend_environment() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("UPMEM_ALLOW_PHYSICAL_HARDWARE", None)
    clean_env.pop("UPMEM_HW_RANK_PATH", None)
    clean_env.pop("UPMEM_HW_RANK_PATHS", None)
    for variable in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE"):
        for value in ("simulator", ""):
            assignment = f"{variable}={value}"
            result = subprocess.run(
                [
                    "make",
                    "m5-circuit-smoke",
                    "UPMEM_ALLOW_PHYSICAL_HARDWARE=1",
                    "UPMEM_HW_RANK_PATH=/dev/dpu_rank0",
                    assignment,
                ],
                cwd=ROOT,
                env=clean_env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 2
            assert variable in result.stderr
            assert "quantum_bench.bench m5-circuit-study" in result.stdout


def test_m5_circuit_report_default_is_timestamped_and_overrideable() -> None:
    default = subprocess.run(
        ["make", "-n", "m5-circuit-report", "M5_CIRCUIT_RUN=/tmp/m5-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    override = subprocess.run(
        [
            "make",
            "-n",
            "m5-circuit-report",
            "M5_CIRCUIT_RUN=/tmp/m5-run",
            "M5_CIRCUIT_REPORT=/tmp/m5-report",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert default.returncode == 0
    match = re.search(
        r"--output runs/comparisons/m5_circuit_study/"
        r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{9})",
        default.stdout,
    )
    assert match is not None
    assert "latest" not in match.group(0)
    assert override.returncode == 0
    assert "--output /tmp/m5-report" in override.stdout


def test_m5_circuit_report_requires_an_explicit_run_and_accepts_baselines() -> None:
    missing = subprocess.run(
        ["make", "m5-circuit-report"],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    report = subprocess.run(
        [
            "make",
            "-n",
            "m5-circuit-report",
            "M5_CIRCUIT_RUN=/tmp/m5-run",
            "M5_CIRCUIT_BASELINES=/tmp/quest.jsonl,/tmp/quimb.jsonl",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "M5_CIRCUIT_RUN" in missing.stderr
    assert report.returncode == 0
    assert 'M5_CIRCUIT_BASELINES="/tmp/quest.jsonl,/tmp/quimb.jsonl"' in report.stdout
