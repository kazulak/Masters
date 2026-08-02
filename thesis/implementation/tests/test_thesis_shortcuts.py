from __future__ import annotations

import os
from pathlib import Path
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
