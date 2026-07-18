"""Public Make, CLI, and active-suite behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import quantum_bench.routing as routing
import scripts.research_benchmark_pack as benchmark_pack
from quantum_bench.bench.config import load_suite


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PUBLIC_TARGETS = (
    "help",
    "test",
    "setup",
    "build-quest-cpu",
    "doctor",
    "bench-cpu",
    "bench-gpu",
    "bench-upmem-sim",
    "upmem-hw-taskgraph-resident-plan",
    "upmem-hw-taskgraph-resident",
    "upmem-hw-taskgraph-resident-report",
    "evidence-inbox",
    "research-plan",
    "planner-evidence",
    "planner-report",
    "thesis-run",
    "thesis-promote",
    "thesis-promote-historical",
    "thesis-verify",
    "thesis-report",
    "list-runs",
    "thesis-clean",
    "archive-evidence",
    "thesis-release",
    "clean-generated",
)


def _command(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"})
    return subprocess.run(
        list(args),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("target", PUBLIC_TARGETS)
def test_public_make_target_has_a_dry_run(target: str) -> None:
    result = _command("make", "-n", f"PYTHON={PYTHON}", target)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_upmem_simulator_suite_is_generic_only() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "upmem_sim_evidence.yml")
    route = next(
        config
        for config in suite["_route_configs"]
        if config["id"] == "upmem_tn_sdk_simulator_quantized"
    )

    assert route["options"]["policy"] == "generic-only"


def test_research_registry_cannot_select_deleted_suite() -> None:
    assert "internal_parallelism" not in benchmark_pack.RESEARCH_SUITES
    assert "internal_parallelism" not in benchmark_pack.SUITE_COMMAND_ORDER
    with pytest.raises(ValueError, match="unknown research suite group"):
        benchmark_pack._selected_suites(["internal_parallelism"])


def test_routing_public_exports_have_no_retired_shadow_policy() -> None:
    assert "evaluate_shadow_route_policy" not in routing.__all__


def test_public_cli_help_and_research_plan() -> None:
    help_result = _command(PYTHON, "-m", "quantum_bench.bench", "--help")
    plan_result = _command(PYTHON, "scripts/research_benchmark_pack.py", "plan")

    assert help_result.returncode == 0
    assert "upmem-hardware-taskgraph-resident" in help_result.stdout
    assert plan_result.returncode == 0, plan_result.stderr
    assert "internal_parallelism" not in plan_result.stdout


def test_public_cpu_smoke_executes_end_to_end() -> None:
    result = _command(PYTHON, "-m", "quantum_bench.bench", "run", "--suite", "configs/suites/smoke.yml")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["record_count"] == 4
    assert all(row["status"] == "passed" for row in summary["rows"])
