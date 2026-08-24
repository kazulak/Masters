"""Regression tests for the intentionally small public Makefile surface."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_is_short_and_points_to_the_active_commands() -> None:
    result = _make("help")

    assert result.returncode == 0, result.stderr
    assert "Thesis tensor-network benchmark" in result.stdout
    assert "make plan" in result.stdout
    assert "make run" in result.stdout
    assert "make report" in result.stdout
    assert "make verify" in result.stdout
    assert "make qualify" in result.stdout
    assert "make build-upmem-runtime" in result.stdout
    assert "make thesis-run" not in result.stdout


def test_active_targets_have_expected_dry_run_shapes() -> None:
    cases = {
        "plan": (("CONFIG=example.yml", "OUTPUT=/tmp/plan"), "cli plan"),
        "run": (("CONFIG=example.yml", "OUTPUT=/tmp/run"), "cli run"),
        "report": (("INPUT=/tmp/run", "REPORT_OUTPUT=/tmp/report"), "cli report"),
        "verify": (("INPUT=/tmp/run",), "cli verify"),
        "qualify": (("CONFIG=example.yml", "OUTPUT=/tmp/qualify"), "cli qualify"),
    }

    for target, (arguments, fragment) in cases.items():
        result = _make("-n", target, *arguments)
        assert result.returncode == 0, (target, result.stderr)
        assert fragment in result.stdout


def test_old_milestone_targets_are_not_public() -> None:
    for target in ("bench-cpu", "thesis-run", "upmem-hw-m5-4", "upmem-provider-plan"):
        result = _make("-n", target)
        assert result.returncode != 0
