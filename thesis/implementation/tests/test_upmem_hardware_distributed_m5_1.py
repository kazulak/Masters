from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np
import pytest

from quantum_bench.bench import upmem_hardware_distributed_m5_1 as m51
from quantum_bench.bench.upmem_simplepim_taskgraph import _lower_real_float32
from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload
from quantum_bench.tn.contract import contract_binary_task


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
    work_units = request["work_units"]
    descriptor = 64 * dpu_count
    operand = 96 * dpu_count
    reset = 128 * dpu_count
    active = 8 * dpu_count
    completion = 64 * dpu_count
    final = 64
    h2d = descriptor + operand + reset + active
    d2h = completion + final
    assignments = [
        {
            "package_operation_index": 0,
            "operation_id": 0,
            "partition_mode": "output",
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
        for unit in work_units
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
            "runtime_cycles": 100,
            "processed_elements": unit["output_elements"],
            "output_checksum_fnv1a64": "0123456789abcdef",
            "completion_count": 1,
        }
        for unit in work_units
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
        "allocation": {
            "attempted": True,
            "confirmed": True,
            "release_confirmed": True,
        },
        "metrics": {
            "descriptor_h2d_bytes": descriptor,
            "operand_h2d_bytes": operand,
            "reset_h2d_bytes": reset,
            "active_operation_h2d_bytes": active,
            "completion_d2h_bytes": completion,
            "cross_d2h_bytes": 0,
            "cross_h2d_bytes": 0,
            "final_d2h_bytes": final,
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
            "final_d2h_time_s": 0.001,
            "output_write_time_s": 0.001,
            "release_time_s": 0.001,
        },
        "operation_assignments": assignments,
        "completed_per_dpu": [1] * dpu_count,
        "distributed_work_units": completed_units,
        "cross_dpu_transfers": [],
    }


class FakeM51NativeTarget:
    def __init__(self, *, write_output: bool = True, write_stale: bool = False) -> None:
        self.write_output = write_output
        self.write_stale = write_stale
        self.environment: dict[str, str] | None = None
        self.output_paths: list[Path] = []

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
        if self.write_stale:
            output_path = Path(request["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _output().tofile(output_path)
        return {
            "schema_version": m51.NATIVE_RESPONSE_SCHEMA,
            "status": "validated",
            "target_requested": "hardware",
            "target_observed": "not_allocated",
            "requested_dpu_count": request["dpu_count"],
            "allocated_dpu_count": 0,
        }

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        output_path = Path(request["output_path"])
        self.output_paths.append(output_path)
        if self.write_output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _output().tofile(output_path)
        return _response(request)


def _item(dpu_count: int = 2) -> dict[str, Any]:
    output_elements = 8
    width = output_elements // dpu_count
    return {
        "dpu_count": dpu_count,
        "work_units": [
            {
                "dpu_id": dpu_id,
                "output_offset": dpu_id * width,
                "output_elements": width,
                "contracted_offset": 0,
                "contracted_elements": 4,
            }
            for dpu_id in range(dpu_count)
        ],
    }


def _request_for_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"dpu_count": item["dpu_count"], "work_units": item["work_units"]}


def test_m51_fake_native_response_writes_strict_normalized_rows(tmp_path: Path) -> None:
    target = FakeM51NativeTarget()
    environment = {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_HOME": "/fake/upmem"}

    result = m51.execute(tmp_path, environment=environment, native_target=target)

    assert result["status"] == "completed"
    assert target.environment == environment
    assert len(set(target.output_paths)) == 3
    run_dir = Path(result["run_dir"])
    records = [
        json.loads(line)
        for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()
    ]
    assert [row["requested_dpu_count"] for row in records] == [1, 2, 4]
    assert all(row["actual_transfer_bytes"] > 0 for row in records)
    assert all(row["timing_is_bringup_only"] is True for row in records)
    captured = json.loads((run_dir / "environment.json").read_text())
    assert captured["m5_1_native_environment"]["UPMEM_HOME"] == "/fake/upmem"


def test_m51_does_not_accept_stale_or_response_synthesized_output(tmp_path: Path) -> None:
    target = FakeM51NativeTarget(write_output=False, write_stale=True)

    result = m51.execute(
        tmp_path,
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        native_target=target,
    )

    assert result["status"] == "failed"
    run_dir = Path(result["run_dir"])
    rows = [json.loads(line) for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()]
    assert all(row["failure_stage"] == "result_transfer_failed" for row in rows)


def _delete(path: tuple[Any, ...]) -> Callable[[dict[str, Any]], None]:
    def mutate(response: dict[str, Any]) -> None:
        parent: Any = response
        for part in path[:-1]:
            parent = parent[part]
        del parent[path[-1]]

    return mutate


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(response: dict[str, Any]) -> None:
        parent: Any = response
        for part in path[:-1]:
            parent = parent[part]
        parent[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _delete(("schema_version",)),
        _delete(("failure_stage",)),
        _delete(("allocated_dpu_count",)),
        _set(("requested_dpu_count",), 1),
        _set(("allocated_dpu_count",), True),
        _set(("hardware_allocation_verified",), False),
        _set(("native_kernel_executed",), False),
        _set(("hardware_kernel_executed",), False),
        _set(("simulator_kernel_executed",), True),
        _set(("cpu_fallback_used",), True),
        _set(("allocation", "confirmed"), False),
        _set(("allocation", "release_confirmed"), False),
        _set(("operation_assignments",), []),
        _set(("completed_per_dpu", 1), 0),
        _set(("metrics", "launch_count"), 0),
        _set(("metrics", "actual_h2d_bytes"), 1),
        _set(("distributed_work_units", 0, "runtime_cycles"), 0),
        _delete(("timing", "release_time_s")),
    ],
)
def test_m51_success_response_is_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    item = _item()
    response = deepcopy(_response(_request_for_item(item)))
    mutate(response)

    with pytest.raises(ValueError, match="output_manifest_failed"):
        m51._validate_execute_response(response, item)


def test_m51_prepare_only_requires_build(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--prepare-only requires --build"):
        m51.prepare(tmp_path, native_target=FakeM51NativeTarget())


def test_m51_prepare_result_is_explicitly_prepared(tmp_path: Path) -> None:
    result = m51.prepare(tmp_path, build=True, native_target=FakeM51NativeTarget())

    assert result["status"] == "prepared"
    assert json.loads(Path(result["artifact"]).read_text())["status"] == "prepared"


def test_default_target_passes_injected_environment_to_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = m51._DefaultNativeTarget()
    environment = {"PATH": "/fake/bin", "M5_1_ENV_MARKER": "present"}
    target.set_environment(environment)
    response_path = tmp_path / "response.json"
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        response_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(m51.subprocess, "run", fake_run)
    target.execute(
        {
            "host_binary": str(tmp_path / "bin" / "host"),
            "resident_manifest": str(tmp_path / "manifest.json"),
            "distributed_plan": str(tmp_path / "plan.bin"),
            "response_path": str(response_path),
        },
        timeout_s=1.0,
    )

    assert observed["env"] == environment


def test_native_metrics_json_argument_order_matches_field_order() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "native/upmem/simplepim/upmem_sdk_execution_plan/host.c"
    ).read_text(encoding="utf-8")
    assert source.index("metrics->reset_h2d_bytes", source.index("timing->release_time_s")) < source.index(
        "metrics->active_operation_h2d_bytes", source.index("timing->release_time_s")
    )
