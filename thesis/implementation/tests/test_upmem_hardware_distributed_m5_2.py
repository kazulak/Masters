from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pytest

from quantum_bench.bench import upmem_hardware_distributed_m5_1 as m51
from quantum_bench.bench import upmem_hardware_distributed_m5_2 as m52
from quantum_bench.bench.upmem_simplepim_taskgraph import _lower_real_float32
from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload
from quantum_bench.tn.contract import contract_binary_task


ROOT = Path(__file__).resolve().parents[1]


def _output() -> np.ndarray:
    workload = build_generic_boundary_workload()
    graph, network = _lower_real_float32(workload.graph, workload.network)
    task = graph.tasks[0]
    return np.asarray(
        contract_binary_task(
            task,
            network.tensors[0].array,
            network.tensors[1].array,
            dtype=np.float32,
        ),
        dtype="<f4",
    ).ravel()


def _response(request: Mapping[str, Any]) -> dict[str, Any]:
    dpu_count = int(request["dpu_count"])
    units = list(request["work_units"])
    output_elements = int(units[0]["output_elements"])
    descriptor = 64 * dpu_count
    operand = 96 * dpu_count
    reset = 128 * dpu_count
    active = 8 * dpu_count
    completion = 64 * dpu_count
    reduction_bytes = output_elements * 4 * dpu_count
    additions = output_elements * (dpu_count - 1)
    h2d = descriptor + operand + reset + active
    d2h = completion + reduction_bytes
    assignments = [
        {
            "package_operation_index": 0,
            "operation_id": 0,
            "partition_mode": "contracted",
            **{
                key: unit[key]
                for key in (
                    "dpu_id",
                    "output_offset",
                    "output_elements",
                    "contracted_offset",
                    "contracted_elements",
                )
            },
        }
        for unit in units
    ]
    completed_units = [
        {
            **{
                key: unit[key]
                for key in (
                    "dpu_id",
                    "output_offset",
                    "output_elements",
                    "contracted_offset",
                    "contracted_elements",
                )
            },
            "partition_mode": "contracted",
            "runtime_cycles": 100,
            "processed_elements": unit["output_elements"],
            "output_checksum_fnv1a64": "0123456789abcdef",
            "completion_count": 1,
        }
        for unit in units
    ]
    return {
        "schema_version": m51.NATIVE_RESPONSE_SCHEMA,
        "status": "completed",
        "failure_stage": None,
        "error": None,
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": dpu_count,
        "tasklets_per_dpu": 1,
        "requested_warmups": 0,
        "requested_repetitions": 1,
        "validation_status": "native_completion_verified",
        "hardware_allocation_verified": True,
        "allocation_succeeded": True,
        "allocation_was_confirmed": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "hardware_functionality_evidence": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "provider_identities": {"communication": "host_mediated_sum_v1"},
        "allocation": {"attempted": True, "confirmed": True, "release_confirmed": True},
        "metrics": {
            "descriptor_h2d_bytes": descriptor,
            "operand_h2d_bytes": operand,
            "reset_h2d_bytes": reset,
            "active_operation_h2d_bytes": active,
            "completion_d2h_bytes": completion,
            "cross_d2h_bytes": 0,
            "cross_h2d_bytes": 0,
            "final_d2h_bytes": 0,
            "reduction_d2h_bytes": reduction_bytes,
            "reduction_partial_reads": dpu_count,
            "reduction_element_additions": additions,
            "reduction_accumulator_resets": 1,
            "reduction_participant_count": dpu_count,
            "actual_h2d_bytes": h2d,
            "actual_d2h_bytes": d2h,
            "actual_transfer_bytes": h2d + d2h,
            "launch_count": dpu_count,
            "synchronize_count": dpu_count,
            "completion_reads": dpu_count,
            "cross_dpu_edge_count": 0,
        },
        "timing": {
            "allocation_time_s": 0.001,
            "binary_load_time_s": 0.001,
            "descriptor_h2d_time_s": 0.001,
            "operand_h2d_time_s": 0.001,
            "cross_dpu_transfer_time_s": 0.0,
            "launch_sync_time_s": 0.001,
            "final_d2h_time_s": 0.0,
            "reduction_time_s": 0.001,
            "output_write_time_s": 0.001,
            "release_time_s": 0.001,
        },
        "operation_assignments": assignments,
        "completed_per_dpu": [1] * dpu_count,
        "distributed_work_units": completed_units,
        "cross_dpu_transfers": [],
        "reduction": {
            "provider": "host_mediated_sum_v1",
            "order": "ascending_dpu_id",
            "participant_count": dpu_count,
            "partial_buffers_read": dpu_count,
            "d2h_bytes": reduction_bytes,
            "element_additions": additions,
            "accumulator_reset_per_repetition": True,
        },
    }


class FakeM52NativeTarget:
    def __init__(self) -> None:
        self.environment: dict[str, str] | None = None
        self.requests: list[Mapping[str, Any]] = []

    def set_environment(self, environment: Mapping[str, str]) -> None:
        self.environment = dict(environment)

    def build(self, build_dir: Path) -> Mapping[str, Any]:
        build_dir.mkdir(parents=True)
        host = build_dir / "host_upmem_execution_plan_v2"
        dpu = build_dir / "dpu_resident_v2"
        host.write_bytes(b"fake-host")
        dpu.write_bytes(b"fake-dpu")
        return {"host_binary_v2": str(host), "dpu_binary_v2": str(dpu)}

    def validate(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        assert request["partition_kind"] == "contracted_partial_sum"
        return {
            "schema_version": m51.NATIVE_RESPONSE_SCHEMA,
            "status": "validated",
            "target_requested": "hardware",
            "target_observed": "not_allocated",
            "requested_dpu_count": request["dpu_count"],
            "allocated_dpu_count": 0,
        }

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        self.requests.append(request)
        output_path = Path(request["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _output().tofile(output_path)
        return _response(request)


@pytest.mark.parametrize("dpu_count", (1, 2, 4))
def test_m52_fake_workflow_covers_explicit_contracted_1_2_4(tmp_path: Path, dpu_count: int) -> None:
    target = FakeM52NativeTarget()
    result = m52.execute(
        tmp_path,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        native_target=target,
    )

    assert result["status"] == "completed"
    assert [request["dpu_count"] for request in target.requests] == [1, 2, 4]
    rows = [json.loads(line) for line in (Path(result["run_dir"]) / "normalized_records.jsonl").read_text().splitlines()]
    assert [row["requested_dpu_count"] for row in rows] == [1, 2, 4]
    assert all(row["partition_kind"] == "contracted_partial_sum" for row in rows)
    assert all(row["communication_provider"] == "host_mediated_sum_v1" for row in rows)


def test_m52_prepare_result_and_artifact_are_prepared(tmp_path: Path) -> None:
    result = m52.prepare(tmp_path, build=True, native_target=FakeM52NativeTarget())

    assert result["status"] == "prepared"
    assert json.loads(Path(result["artifact"]).read_text())["status"] == "prepared"


def test_m52_cpu_reduction_is_same_plan_and_deterministic() -> None:
    low = np.asarray([1.0, 2.0], dtype="<f4")
    high = np.asarray([2.0, 3.0], dtype="<f4")

    assert np.array_equal(
        m52.reduce_float32_partials({1: high, 0: low}),
        m52.reduce_float32_partials({0: low, 1: high}),
    )
    assert np.array_equal(m52.reduce_float32_partials({0: low}), low)


def test_cli_and_make_plan_targets_are_wired() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "upmem-hw-m5-1-plan:" in makefile
    assert "upmem-hw-m5-2-plan:" in makefile
    completed = subprocess.run(
        [sys.executable, "-m", "quantum_bench.bench", "--help"],
        cwd=ROOT,
        env={**dict(PYTHONPATH="src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "upmem-hardware-distributed-m5-2" in completed.stdout
