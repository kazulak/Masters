from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_native_host_boundary.py"
SPEC = importlib.util.spec_from_file_location("benchmark_native_host_boundary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_packet_and_canonical_bytes_are_deterministic() -> None:
    first = benchmark.build_fixture("fixture", 3, 17)
    second = benchmark.build_fixture("fixture", 3, 17)

    assert first["packet"] == second["packet"]
    assert first["canonical"] == second["canonical"]
    assert len(first["canonical"]) == 3 * benchmark.RECORD.size + 17
    assert first["packet_sha256"] == second["packet_sha256"]


def test_python_arm_matches_canonical_hash() -> None:
    fixture = benchmark.build_fixture("fixture", 5, 31)
    facts = benchmark._python_arm(fixture, 3)

    assert facts["canonical_sha256"] == fixture["canonical_sha256"]
    assert facts["canonical_bytes"] == len(fixture["canonical"])


def test_invalid_iteration_count_is_rejected_by_cli_parser() -> None:
    with pytest.raises(ValueError):
        benchmark.run(Path("/does/not/exist"), Path("/tmp/native-host-test"), 0)

