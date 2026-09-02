from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUALIFY = _module("qualify_quantized_upmem_execution", "qualify_quantized_upmem_execution.py")
ANALYZE = _module("analyze_quantized_upmem_execution", "analyze_quantized_upmem_execution.py")


@pytest.mark.parametrize(
    ("name", "routes", "cells", "attempts"),
    [
        ("tn_benchmark_quantized_upmem_simulator.yml", 3, 4, 4),
        ("tn_benchmark_quantized_upmem_pilot.yml", 3, 4, 4),
        ("tn_benchmark_quantized_upmem_diagnostic.yml", 10, 30, 180),
    ],
)
def test_fixed_configs_have_exact_matrices(
    name: str, routes: int, cells: int, attempts: int
) -> None:
    config = load_experiment_config(ROOT / "configs" / name)
    assert len(config["routes"]) == routes
    assert sum(len(item["route_ids"]) for item in config["matrix"]) == cells
    collection = config["collection"]
    assert cells * (
        collection["warmup_blocks"] + collection["measurement_blocks"]
    ) == attempts
    assert collection["claim_policy"] == "diagnostic_v1"
    assert collection["session_policy"] == "fresh_session_per_attempt_v1"
    assert {
        route["numeric_policy"] for route in config["routes"].values()
    } <= {QUALIFY.FLOAT32, QUALIFY.POLICY}


def test_diagnostic_is_exact_route_policy_resource_product() -> None:
    config = load_experiment_config(
        ROOT / "configs/tn_benchmark_quantized_upmem_diagnostic.yml"
    )
    assert set(config["cases"]) == {"ghz18", "hs18", "stress18"}
    observed = {
        (
            route["numeric_policy"],
            route["options"]["dpu_count"],
            route["options"]["tasklets_per_dpu"],
        )
        for route in config["routes"].values()
    }
    assert observed == {
        (policy, dpus, tasklets)
        for policy in (QUALIFY.FLOAT32, QUALIFY.POLICY)
        for dpus, tasklets in ((1, 1), (1, 4), (1, 8), (2, 8), (4, 8))
    }


def test_checksums_are_sorted_exact_and_detect_mutation(tmp_path: Path) -> None:
    (tmp_path / "z").write_text("z", encoding="ascii")
    (tmp_path / "a").write_text("a", encoding="ascii")
    path = QUALIFY.write_checksums(tmp_path)
    lines = path.read_text(encoding="ascii").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["a", "z"]
    QUALIFY.verify_checksums(tmp_path)
    (tmp_path / "a").write_text("changed", encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        QUALIFY.verify_checksums(tmp_path)


def test_prepare_preserves_label_and_resolves_machine_paths(tmp_path: Path) -> None:
    output = tmp_path / "prepared.yml"
    QUALIFY.prepare(
        ROOT / "configs/tn_benchmark_quantized_upmem_pilot.yml",
        output,
        rank_path="/dev/dpu_rank1",
        session_root=tmp_path / "sessions",
        expected_cpu=0,
        binary_root=tmp_path / "bin",
    )
    config = load_experiment_config(output)
    assert config["experiment_identity_payload"]["label"] == (
        "quantized-upmem-physical-pilot-v1"
    )
    for route_id, route in config["routes"].items():
        assert route["options"]["rank_paths"] == ("/dev/dpu_rank1",)
        assert route["options"]["session_root"] == str(
            (tmp_path / "sessions" / route_id).resolve()
        )


def _synthetic_evidence() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads(
        json.dumps(
            load_experiment_config(
                ROOT / "configs/tn_benchmark_quantized_upmem_diagnostic.yml"
            ),
            default=lambda value: dict(value) if hasattr(value, "items") else list(value),
        )
    )
    manifest = {
        "source_commit": "a" * 40,
        "experiment_id": "b" * 64,
        "configuration": {"experiment": config},
    }
    samples: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for matrix in config["matrix"]:
        for route_id in matrix["route_ids"]:
            route = config["routes"][route_id]
            policy_factor = 1.0 if route["numeric_policy"] == QUALIFY.POLICY else 2.0
            resource_factor = 1.0 / float(route["options"]["dpu_count"])
            for block_id in range(6):
                session_id = f"{matrix['case_id']}:{route_id}:{block_id}"
                sessions.append(
                    {
                        "session_instance_id": session_id,
                        "open_s": 0.2,
                        "session_close_s": 0.1,
                    }
                )
                total = policy_factor * resource_factor + block_id * 0.001
                timing = {field: 0.001 for field in ANALYZE.OPERATION_FIELDS}
                samples.append(
                    {
                        "case_id": matrix["case_id"],
                        "route_id": route_id,
                        "block_id": block_id,
                        "attempt_kind": "warmup" if block_id == 0 else "measurement",
                        "session_instance_id": session_id,
                        "status": "success",
                        "measurement": {
                            "total_wall_s": total,
                            "encode_s": 0.05,
                            "preparation_s": 0.01,
                            "h2d_s": 0.1 * policy_factor,
                            "kernel_s": 0.5 * policy_factor * resource_factor,
                            "host_reduce_s": 0.01,
                            "d2h_s": 0.03,
                            "decode_s": 0.02,
                            "h2d_bytes": 100 if policy_factor == 1 else 400,
                            "d2h_bytes": 80,
                        },
                        "backend_facts": {
                            "operation_facts": ({"timing": timing},),
                            "arithmetic_weighted_tasklet_utilization": 1.0,
                            "arithmetic_weighted_dpu_slot_utilization": 1.0,
                            "dominant_work_wave_utilization": 1.0,
                        },
                        "numeric_facts": {
                            "operand_records": (
                                {"shape": (10,)},
                                {"shape": (20,)},
                            ),
                            "saturation_real": 0,
                            "saturation_imag": 0,
                        },
                        "validation": {
                            "policy_reference_passed": True,
                            "max_abs_error": 0.01,
                            "relative_l2_error": 0.02,
                            "norm_drift": 0.03,
                        },
                    }
                )
    return manifest, samples, sessions


def test_analysis_is_complete_deterministic_and_computes_ratios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ANALYZE,
        "_software_metrics",
        lambda case_id: {
            "int8_max_abs_error_vs_float32_same_dag": 0.1,
            "int8_relative_l2_vs_float32_same_dag": 0.2,
            "int8_norm_drift_vs_float32_same_dag": 0.3,
            "int8_max_abs_error_vs_complex128": 0.4,
            "int8_relative_l2_vs_complex128": 0.5,
            "int8_norm_drift_vs_complex128": 0.6,
            "float32_max_abs_error_vs_complex128": 0.001,
            "float32_relative_l2_vs_complex128": 0.002,
            "float32_norm_drift_vs_complex128": 0.003,
        },
    )
    manifest, samples, sessions = _synthetic_evidence()
    first = ANALYZE.derive(manifest, samples, sessions)
    second = ANALYZE.derive(manifest, samples, sessions)
    assert first == second
    assert len(first["routes"]) == 30
    assert len(first["error_runtime"]) == 15
    assert len(first["optima"]) == 6
    int8 = next(
        row
        for row in first["routes"]
        if row["case_id"] == "ghz18" and row["route_id"] == "int8_1dpu_t1"
    )
    assert int8["float32_over_int8_kernel_s"] == pytest.approx(2.0)
    assert int8["actual_h2d_byte_reduction_ratio"] == pytest.approx(4.0)
    assert int8["nominal_logical_compression_ratio"] == pytest.approx(240 / 76)
    ANALYZE.write_outputs(first, tmp_path)
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == tuple(
        sorted(ANALYZE.OUTPUTS)
    )


def test_analysis_rejects_incomplete_or_mixed_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANALYZE, "_software_metrics", lambda case_id: {})
    manifest, samples, sessions = _synthetic_evidence()
    with pytest.raises(ValueError, match="exactly 180"):
        ANALYZE.derive(manifest, samples[:-1], sessions)
    manifest["configuration"]["experiment"]["experiment_identity_payload"][
        "label"
    ] = "wrong"
    with pytest.raises(ValueError, match="fixed physical diagnostic"):
        ANALYZE.derive(manifest, samples, sessions)
