"""Small tests for the active Makefile and CLI workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import quantum_bench.cli as cli


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _command(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
    return subprocess.run(
        list(args),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _numpy_config() -> str:
    return """\
schema_version: tn_benchmark_v1
experiment_id: public-workflow
defaults:
  warmups: 0
  repetitions: 1
  timeout_s: 5.0
cases:
  bell:
    circuit:
      kind: builtin
      name: bell_2q
      path: null
      parameters: {}
plans:
  greedy:
    planner:
      engine: opt_einsum
      mode: greedy
      max_repeats: 1
      seed: 0
    slicing: null
routes:
  numpy:
    executor: numpy_dag
    numeric_policy: split_complex_float32_v1
    options: {}
matrix:
  - case_id: bell
    plan_id: greedy
    route_ids: [numpy]
"""


def _physical_config() -> str:
    return _numpy_config().replace(
        "  numpy:\n    executor: numpy_dag\n    numeric_policy: split_complex_float32_v1\n    options: {}",
        "  physical:\n"
        "    executor: upmem_physical\n"
        "    numeric_policy: split_complex_float32_v1\n"
        "    options:\n"
        "      dpu_count: 1\n"
        "      rank_count: 1\n"
        "      tasklets_per_dpu: 1\n"
        "      session_root: native/upmem/runtime\n"
        "      host_binary: native/upmem/runtime/bin/host_upmem_execution_plan_v4_t1\n"
        "      dpu_binary: native/upmem/runtime/bin/dpu_gemm_tile_v4_t1\n"
        "      initialization_binary: native/upmem/runtime/bin/dpu_simplepim_management_init_t1\n"
        "      rank_paths: [/dev/dpu_rank0]",
    ).replace("route_ids: [numpy]", "route_ids: [physical]")


def test_make_help_lists_only_active_workflow() -> None:
    result = _command("make", "help")

    assert result.returncode == 0, result.stderr
    for target in (
        "make plan",
        "make run",
        "make report",
        "make verify",
        "make build-upmem-runtime",
        "make qualify",
    ):
        assert target in result.stdout
    for obsolete in ("bench-cpu", "thesis-run", "upmem-hw-m5", "research-plan"):
        assert obsolete not in result.stdout


@pytest.mark.parametrize(
    ("target", "arguments", "fragment"),
    (
        ("plan", ("CONFIG=config.yml", "OUTPUT=/tmp/plan"), "quantum_bench.cli plan"),
        ("run", ("CONFIG=config.yml", "OUTPUT=/tmp/run"), "quantum_bench.cli run"),
        (
            "report",
            ("INPUT=/tmp/run", "REPORT_OUTPUT=/tmp/report"),
            "quantum_bench.cli report",
        ),
        ("verify", ("INPUT=/tmp/run",), "quantum_bench.cli verify"),
        (
            "qualify",
            ("CONFIG=config.yml", "OUTPUT=/tmp/qualify"),
            "quantum_bench.cli qualify",
        ),
    ),
)
def test_make_dry_run_constructs_active_cli_command(
    target: str, arguments: tuple[str, ...], fragment: str
) -> None:
    result = _command("make", "-n", target, *arguments)

    assert result.returncode == 0, result.stderr
    assert fragment in result.stdout


def test_make_dry_run_build_passes_tasklet_count() -> None:
    result = _command("make", "-n", "build-upmem-runtime", "UPMEM_TASKLETS=8")

    assert result.returncode == 0, result.stderr
    assert "NR_TASKLETS=8" in result.stdout


@pytest.mark.parametrize("target", ("bench-cpu", "thesis-run", "research-plan"))
def test_obsolete_make_targets_are_absent(target: str) -> None:
    result = _command("make", "-n", target)

    assert result.returncode != 0


def test_numpy_plan_run_verify_report_lifecycle(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yml"
    config.write_text(_numpy_config(), encoding="utf-8")
    plan_dir = tmp_path / "plan"
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"

    plan = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "plan",
        "--config",
        str(config),
        "--output",
        str(plan_dir),
    )
    run = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "run",
        "--config",
        str(config),
        "--output",
        str(run_dir),
    )
    verify = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "verify",
        "--input",
        str(run_dir),
    )
    report = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "report",
        "--input",
        str(run_dir),
        "--output",
        str(report_dir),
    )

    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["status"] == "planned"
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "completed"
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["status"] == "completed"
    assert report.returncode == 0, report.stderr
    assert json.loads(report.stdout)["status"] == "completed"
    assert (plan_dir / "plan.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "samples.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (report_dir / "report.json").is_file()


def test_qualify_is_physical_only_and_cli_level_opt_in_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "physical.yml"
    config.write_text(_physical_config(), encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_qualify(
        config_path: str, output: str, *, allow_physical: bool
    ) -> dict[str, object]:
        observed.update(
            config_path=config_path, output=output, allow_physical=allow_physical
        )
        return {"status": "completed"}

    monkeypatch.setattr(cli, "qualify_command", fake_qualify)
    assert (
        cli.main(
            [
                "qualify",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "qualify"),
                "--allow-physical",
            ]
        )
        == 0
    )
    assert observed["allow_physical"] is True
    assert "physical" in str(observed["config_path"])
