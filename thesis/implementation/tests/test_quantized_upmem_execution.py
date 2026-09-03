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


def _synthetic_operation_timing() -> dict[str, float]:
    """One internally consistent operation with nested raw envelopes."""

    return {
        "total_wall_s": 0.1,
        "preparation_s": 0.005,
        "encode_s": 0.005,
        "rank_response_h2d_max_sum_s": 0.01,
        "rank_response_kernel_max_sum_s": 0.02,
        "rank_response_d2h_max_sum_s": 0.01,
        "rank_response_total_route_max_sum_s": 0.05,
        "request_wave_wall_sum_s": 0.08,
        "request_build_sum_s": 0.01,
        "request_work_unit_materialization_sum_s": 0.002,
        "request_artifact_build_sum_s": 0.008,
        "request_payload_record_staging_sum_s": 0.004,
        "request_manifest_sidecar_staging_sum_s": 0.002,
        "request_payload_materialization_sum_s": 0.001,
        "request_payload_file_write_sum_s": 0.001,
        "request_payload_hashing_sum_s": 0.001,
        "request_payload_record_construction_sum_s": 0.001,
        "rank_submit_parallel_wall_sum_s": 0.05,
        "rank_submit_total_max_sum_s": 0.04,
        "rank_submit_artifact_validation_max_sum_s": 0.001,
        "rank_submit_protocol_write_max_sum_s": 0.01,
        "rank_submit_response_wait_max_sum_s": 0.025,
        "rank_submit_response_validation_max_sum_s": 0.004,
        "coordinator_response_processing_sum_s": 0.01,
        "assembly_s": 0.002,
        "decode_s": 0.003,
    }


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
                timing = _synthetic_operation_timing()
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
                            "operation_facts": ({"rank_count": 1, "timing": timing},),
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
    assert first["summary"]["schema_version"] == (
        "quantized_upmem_execution_analysis_v2"
    )
    assert len(first["sample_components"]) == 150
    assert len(first["timing_envelopes"]) == 30
    assert len(first["fixed_route_comparisons"]) == 15
    assert len(first["summary"]["best_observed_comparisons"]) == 3
    assert all(
        row["grid_route_count"] == 5
        for row in first["summary"]["best_observed_comparisons"]
    )
    assert all(
        row["selection_scope"] == "best_observed_within_tested_route_grid"
        for row in first["optima"]
    )
    assert all("raw_mad_total_wall_s" in row for row in first["optima"])
    int8 = next(
        row
        for row in first["routes"]
        if row["case_id"] == "ghz18" and row["route_id"] == "int8_1dpu_t1"
    )
    assert int8["float32_over_int8_kernel_s"] == pytest.approx(2.0)
    assert int8["actual_h2d_byte_reduction_ratio"] == pytest.approx(4.0)
    assert int8["nominal_logical_compression_ratio"] == pytest.approx(240 / 76)
    assert int8["dominant_host_component"] == "coordinator_other_s"
    assert "optimal" not in ANALYZE._markdown(first).lower()
    output_a = tmp_path / "first"
    output_b = tmp_path / "second"
    ANALYZE.write_outputs(first, output_a)
    ANALYZE.write_outputs(second, output_b)
    assert tuple(sorted(path.name for path in output_a.iterdir())) == tuple(
        sorted(ANALYZE.OUTPUTS)
    )
    for name in ANALYZE.OUTPUTS:
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()


def test_operation_attribution_is_disjoint_and_keeps_nested_envelopes() -> None:
    values, components = ANALYZE._operation_attribution(
        {"rank_count": 1, "timing": _synthetic_operation_timing()}, index=0
    )
    assert components == pytest.approx({
        "preparation_s": 0.005,
        "encode_s": 0.005,
        "host_request_overhead_s": 0.03,
        "native_request_overhead_s": 0.01,
        "h2d_s": 0.01,
        "kernel_s": 0.02,
        "d2h_s": 0.01,
        "assembly_s": 0.002,
        "decode_s": 0.003,
        "operation_other_s": 0.005,
    })
    assert sum(components.values()) == pytest.approx(values["total_wall_s"])
    assert values["rank_submit_response_wait_max_sum_s"] == 0.025
    assert "rank_submit_response_wait_max_sum_s" not in components


def test_sample_attribution_closes_and_uses_known_no_reduce_policy() -> None:
    manifest, samples, sessions = _synthetic_evidence()
    config = manifest["configuration"]["experiment"]
    sample = samples[0]
    session = sessions[0]
    row = ANALYZE._sample_row(sample, session, config["routes"][sample["route_id"]])
    disjoint = sum(
        row[f"component_{field}"]
        for field in ANALYZE.DISJOINT_COMPONENT_FIELDS
    )
    assert disjoint == pytest.approx(row["total_wall_s"])
    assert row["accounting_residual_s"] == pytest.approx(0.0)
    assert row["component_host_reduce_s"] == pytest.approx(0.01)
    assert row["component_coordinator_other_s"] == pytest.approx(1.89)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "rank_response_total_route_max_sum_s",
            0.039,
            "native request overhead is materially negative",
        ),
        ("request_wave_wall_sum_s", 0.049, "host request overhead is materially negative"),
        ("total_wall_s", 0.09, "operation other is materially negative"),
    ],
)
def test_attribution_rejects_materially_negative_residuals(
    field: str, value: float, message: str
) -> None:
    timing = _synthetic_operation_timing()
    timing[field] = value
    with pytest.raises(ValueError, match=message):
        ANALYZE._operation_attribution(
            {"rank_count": 1, "timing": timing}, index=0
        )


def test_attribution_rejects_missing_fields_and_missing_host_reduce() -> None:
    timing = _synthetic_operation_timing()
    timing.pop("rank_submit_response_wait_max_sum_s")
    with pytest.raises(ValueError, match="timing is missing rank_submit_response_wait"):
        ANALYZE._operation_attribution(
            {"rank_count": 1, "timing": timing}, index=0
        )

    manifest, samples, sessions = _synthetic_evidence()
    config = manifest["configuration"]["experiment"]
    sample = json.loads(json.dumps(samples[0]))
    sample["measurement"].pop("host_reduce_s")
    with pytest.raises(ValueError, match="measurement is missing host_reduce_s"):
        ANALYZE._sample_row(
            sample, sessions[0], config["routes"][sample["route_id"]]
        )


def test_tiny_negative_residual_is_clamped_to_zero() -> None:
    timing = _synthetic_operation_timing()
    timing["rank_response_total_route_max_sum_s"] = 0.04 - 0.5e-6
    _, components = ANALYZE._operation_attribution(
        {"rank_count": 1, "timing": timing}, index=0
    )
    assert components["native_request_overhead_s"] == 0.0


def test_residuals_are_paired_before_median() -> None:
    base = _synthetic_operation_timing()
    timings = []
    for preparation_s, request_wave_s in ((0.0, 0.0), (0.09, 0.0), (0.0, 0.09)):
        timing = dict(base)
        timing["preparation_s"] = preparation_s
        timing["request_wave_wall_sum_s"] = request_wave_s
        timing["total_wall_s"] = 0.1
        for field in (
            "rank_response_h2d_max_sum_s",
            "rank_response_kernel_max_sum_s",
            "rank_response_d2h_max_sum_s",
            "rank_response_total_route_max_sum_s",
        ):
            timing[field] = 0.0
        timings.append(timing)
    attributed = [
        ANALYZE._operation_attribution(
            {"rank_count": 1, "timing": timing}, index=index
        )
        for index, timing in enumerate(timings)
    ]
    observed = float(
        __import__("statistics").median(
            components["operation_other_s"]
            for _, components in attributed
        )
    )
    median_of_children = float(
        __import__("statistics").median(timing["total_wall_s"] for timing in timings)
        - __import__("statistics").median(timing["preparation_s"] for timing in timings)
        - __import__("statistics").median(timing["encode_s"] for timing in timings)
        - __import__("statistics").median(timing["request_wave_wall_sum_s"] for timing in timings)
        - __import__("statistics").median(timing["assembly_s"] for timing in timings)
        - __import__("statistics").median(timing["decode_s"] for timing in timings)
    )
    assert observed == pytest.approx(0.0)
    assert median_of_children == pytest.approx(0.09)
    assert observed != pytest.approx(median_of_children)


def test_envelopes_are_separate_from_disjoint_component_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ANALYZE, "_software_metrics", lambda case_id: {
        "int8_max_abs_error_vs_float32_same_dag": 0.0,
        "int8_relative_l2_vs_float32_same_dag": 0.0,
        "int8_norm_drift_vs_float32_same_dag": 0.0,
        "int8_max_abs_error_vs_complex128": 0.0,
        "int8_relative_l2_vs_complex128": 0.0,
        "int8_norm_drift_vs_complex128": 0.0,
        "float32_max_abs_error_vs_complex128": 0.0,
        "float32_relative_l2_vs_complex128": 0.0,
        "float32_norm_drift_vs_complex128": 0.0,
    })
    manifest, samples, sessions = _synthetic_evidence()
    result = ANALYZE.derive(manifest, samples, sessions)
    component = result["components"][0]
    envelope = result["timing_envelopes"][0]
    assert "median_request_wave_wall_sum_s" not in component
    assert "median_rank_submit_response_wait_max_sum_s" in envelope
    assert envelope["envelope_semantics"] == "inclusive_non_additive"


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
