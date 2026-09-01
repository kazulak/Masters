from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytest

from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorSpec, TensorView
from quantum_bench.results import ExecutionFailed, UnsupportedExecution
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemResources,
    UpmemTopology as FinalTopology,
    plan_upmem,
)
from quantum_bench.upmem.runtime import (
    UpmemV4Executor,
    UpmemV4Session,
    open_upmem,
    open_upmem_simulator,
)
from quantum_bench.upmem.protocol import (
    COMPLETION_BYTES,
    CONTROL_BYTES,
    EXECUTION_TARGET_SIMULATOR,
    FLAG_ZERO_WORK,
    STATUS_COMPLETED,
    V4ProtocolError,
    native_execution_identity,
)
import quantum_bench.upmem.runtime as runtime
from quantum_bench.cpu import replay_upmem_plan_once


# Private runtime fixtures


def _task(k: int = 5, *, m: int = 3, n: int = 4) -> ContractNode:
    return ContractNode(
        node_id="fixture",
        left=TensorView(tensor_id="left", labels=(0, 1), shape=(m, k)),
        right=TensorView(tensor_id="right", labels=(1, 2), shape=(k, n)),
        output=TensorSpec(id="out", labels=(0, 2), shape=(m, n), structure="dense"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


def _binaries(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(root / name for name in ("host", "dpu", "init"))
    for path in paths:
        path.write_bytes(path.name.encode("ascii"))
    paths[0].chmod(paths[0].stat().st_mode | 0o100)
    return paths


@dataclass
class _Release:
    release_confirmed: bool = True
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_total_bytes: int = 0
    stderr_total_bytes: int = 0
    stdout_limit_exceeded: bool = False
    stderr_limit_exceeded: bool = False
    event: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeSession:
    profile: Any
    binary_provenance: dict[str, str]
    delay_s: float = 0.0
    closed: bool = False
    startup: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        simulator = self.profile.execution_target == EXECUTION_TARGET_SIMULATOR
        self.startup = {
            "event": "READY",
            "status": "ready",
            "target_requested": "simulator" if simulator else "hardware",
            "target_observed": "sdk_simulator" if simulator else "physical_hardware",
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "rank_path": None if simulator else self.profile.rank_path,
            "hardware_allocation_verified": not simulator,
            "allocation_verified": True,
            **native_execution_identity(self.profile.execution_target),
            **self.binary_provenance,
        }

    def submit(
        self, artifact: Any, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        simulator = self.profile.execution_target == EXECUTION_TARGET_SIMULATOR
        if timeout_s is not None and self.delay_s > timeout_s:
            time.sleep(timeout_s)
            raise V4ProtocolError("kernel_timeout", "fake request exceeded deadline")
        time.sleep(self.delay_s)
        packed = self.profile.numeric_mode_name == "host_packed_int8"
        dtype = np.int8 if packed else np.dtype("<f4")
        payload_cursor = 0
        per_dpu: list[dict[str, Any]] = []
        total_h2d = 0
        total_d2h = 0
        for record in artifact.work_units:
            if hasattr(artifact, "payload_bytes") and isinstance(
                artifact.payload_bytes, bytes
            ):
                a_payload = artifact.payload_bytes[
                    payload_cursor : payload_cursor + record.a_transfer_bytes
                ]
                payload_cursor += record.a_transfer_bytes
                b_payload = artifact.payload_bytes[
                    payload_cursor : payload_cursor + record.b_transfer_bytes
                ]
                payload_cursor += record.b_transfer_bytes
                if not record.flags:
                    left = np.frombuffer(
                        a_payload, dtype=dtype, count=record.m_elements * record.k_elements
                    ).reshape(record.m_elements, record.k_elements)
                    right = np.frombuffer(
                        b_payload, dtype=dtype, count=record.k_elements * record.n_elements
                    ).reshape(record.k_elements, record.n_elements)
                    output = (
                        left.astype(np.int64) @ right.astype(np.int64)
                        if packed
                        else left @ right
                    )
                    output_path = artifact.root / record.c_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(
                        np.asarray(output, dtype="<i4" if packed else "<f4").tobytes()
                    )
            else:
                if not record.flags:
                    left = np.fromfile(
                        artifact.root / record.a_path,
                        dtype=dtype,
                        count=record.m_elements * record.k_elements,
                    ).reshape(record.m_elements, record.k_elements)
                    right = np.fromfile(
                        artifact.root / record.b_path,
                        dtype=dtype,
                        count=record.k_elements * record.n_elements,
                    ).reshape(record.k_elements, record.n_elements)
                    output = (
                        left.astype(np.int64) @ right.astype(np.int64)
                        if packed
                        else left @ right
                    )
                    (artifact.root / record.c_path).write_bytes(
                        np.asarray(output, dtype="<i4" if packed else "<f4").tobytes()
                    )
            h2d = record.a_transfer_bytes + record.b_transfer_bytes + CONTROL_BYTES
            d2h = record.c_transfer_bytes + COMPLETION_BYTES
            total_h2d += h2d
            total_d2h += d2h
            per_dpu.append(
                {
                    "dpu_id": record.local_dpu_id,
                    "tile_id": record.tile_id,
                    "completion_status": STATUS_COMPLETED,
                    "processed_elements": (
                        0
                        if record.flags & FLAG_ZERO_WORK
                        else record.m_elements * record.n_elements
                    ),
                    "h2d_bytes": h2d,
                    "d2h_bytes": d2h,
                    "cycles": 1,
                }
            )
        return {
            "status": "completed",
            "target_requested": "simulator" if simulator else "hardware",
            "target_observed": "sdk_simulator" if simulator else "physical_hardware",
            "native_kernel_executed": True,
            "hardware_kernel_executed": not simulator,
            "simulator_kernel_executed": simulator,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": not simulator,
            "allocation_verified": True,
            "allocated_dpu_count": self.profile.dpu_count,
            "requested_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "request_sequence": artifact.request_sequence,
            "bulk_set_launch_verified": True,
            **native_execution_identity(self.profile.execution_target),
            "transfer": {
                "h2d_bytes": total_h2d,
                "d2h_bytes": total_d2h,
                "total_bytes": total_h2d + total_d2h,
            },
            "per_dpu": per_dpu,
            "timing": {
                "h2d_time_s": 0.01,
                "launch_time_s": 0.02,
                "d2h_time_s": 0.01,
                "total_route_time_s": 0.04,
            },
            "host_submit_timing": {
                "artifact_validation_s": 0.005,
                "protocol_write_s": 0.005,
                "response_wait_s": 0.025,
                "response_validation_s": 0.005,
                "total_submit_s": 0.04,
            },
        }

    def submit_packed(
        self, operation: Any, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        responses = tuple(
            self.submit(request, timeout_s=timeout_s)
            for request in operation.requests
        )
        elapsed = time.perf_counter() - started
        return {
            "event": "OPERATION_RESPONSE",
            "status": "completed",
            "operation_sequence": operation.operation_sequence,
            "response_count": len(responses),
            "responses": responses,
            "host_submit_timing": {
                "artifact_validation_s": 0.0,
                "protocol_write_s": 0.0,
                "response_wait_s": elapsed,
                "response_validation_s": 0.0,
                "total_submit_s": elapsed,
            },
        }

    def close(self, *, timeout_s: float | None = None) -> _Release:
        del timeout_s
        self.closed = True
        return _Release(event={"returncode": 0})


@dataclass(frozen=True)
class _PlannedSession:
    """Expose the active session while retaining a stable test patch seam."""

    session: UpmemV4Session

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


@dataclass(frozen=True)
class _Engine:
    engine: UpmemV4Executor

    def open_session(self, numeric_policy: str, topology: FinalTopology) -> Any:
        return _PlannedSession(self.engine.open_session(numeric_policy, topology))


def _engine(
    root: Path,
    *,
    dpu_count: int = 1,
    rank_count: int = 1,
    tasklets_per_dpu: int = 1,
    execution_target: str = "physical_hardware",
    delay_s: float = 0.0,
) -> _Engine:
    host, dpu, initialization = _binaries(root / "binaries")
    provenance = {
        "host_binary_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "dpu_binary_sha256": hashlib.sha256(dpu.read_bytes()).hexdigest(),
        "initialization_binary_sha256": hashlib.sha256(initialization.read_bytes()).hexdigest(),
    }

    def factory(_command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        del session_root
        return _FakeSession(
            profile=profile,
            binary_provenance=provenance,
            delay_s=delay_s,
        )

    return _Engine(
        UpmemV4Executor(
            session_root=root,
            host_binary=host,
            dpu_binary=dpu,
            initialization_binary=initialization,
            rank_paths=(
                ()
                if execution_target == EXECUTION_TARGET_SIMULATOR
                else tuple(f"/dev/dpu_rank{index}" for index in range(rank_count))
            ),
            dpu_count=dpu_count,
            tasklets_per_dpu=tasklets_per_dpu,
            timeout_s=60.0,
            execution_target=execution_target,
            session_factory=factory,
        )
    )


def _final_plan_for_node(
    node: ContractNode,
    *,
    policy: str,
    dpu_count: int = 1,
    rank_count: int = 1,
    tasklets_per_dpu: int = 1,
) -> tuple[ContractionDAG, UpmemPlan]:
    dag = ContractionDAG(
        tensors=(
            TensorSpec(node.left.tensor_id, node.left.labels, node.left.shape, "dense"),
            TensorSpec(node.right.tensor_id, node.right.labels, node.right.shape, "dense"),
        ),
        nodes=(node,),
        output=TensorView(
            tensor_id=node.output.id,
            labels=node.output.labels,
            shape=node.output.shape,
        ),
    )
    return dag, plan_upmem(
        dag,
        numeric_policy=policy,
        topology=FinalTopology(
            dpu_count=dpu_count,
            tasklets_per_dpu=tasklets_per_dpu,
            rank_count=rank_count,
        ),
    )


# Private test helpers


class _ControlledTerminalSession:
    """Small mock used only to exercise high-level close admission."""

    def __init__(self, terminal: Mapping[str, object]) -> None:
        self._deadline = time.monotonic() + 1.0
        self._terminal = dict(terminal)
        self.close_calls = 0

    def _execute_complex_core(self, *_args, **_kwargs):
        raise AssertionError("close-only mock must not execute work")

    def close(self):
        self.close_calls += 1
        return dict(self._terminal)


def _verified_terminal(plan) -> dict[str, object]:
    return {
        "target_observed": "physical_hardware",
        "allocation_verified": True,
        "hardware_allocation_verified": True,
        "ready_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "physical_target_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "hardware_release_verified": True,
        "hardware_release_confirmed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "test_double_execution": False,
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "observed_rank_count": plan.topology.rank_count,
        "observed_tasklets_per_dpu": plan.topology.tasklets_per_dpu,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }


def _close_mock_session(session) -> None:
    """Release the test-native session without treating it as evidence."""

    try:
        session.close()
    except ExecutionFailed as failure:
        assert failure.stage == "session_close"


def _resources(tmp_path: Path, opener, *, rank_count: int = 1) -> UpmemResources:
    host, dpu, initialization = _binaries(tmp_path / "resources")
    return UpmemResources(
        session_root=str(tmp_path / "session"),
        host_binary=str(host),
        dpu_binary=str(dpu),
        initialization_binary=str(initialization),
        rank_paths=tuple(f"/dev/dpu_rank{index}" for index in range(rank_count)),
        session_opener=opener,
    )


def _simulator_resources(tmp_path: Path, opener=None) -> UpmemResources:
    host, dpu, initialization = _binaries(tmp_path / "simulator-resources")
    return UpmemResources(
        session_root=str(tmp_path / "simulator-session"),
        host_binary=str(host),
        dpu_binary=str(dpu),
        initialization_binary=str(initialization),
        rank_paths=(),
        session_opener=opener,
    )


def _opened(tmp_path: Path, *, policy: str, k: int = 5):
    node = _task(k=k, m=1, n=1)
    dag, plan = _final_plan_for_node(node, policy=policy)
    engine = _engine(tmp_path / "engine", dpu_count=1)
    calls: list[float] = []

    def opener(_dag, final_plan, _resources, timeout_s):
        calls.append(float(timeout_s))
        return engine.open_session(final_plan.numeric_policy, final_plan.topology)

    return node, dag, plan, _resources(tmp_path, opener), calls, engine


def _inputs(
    node: ContractNode, *, k: int, m: int = 1, n: int = 1
) -> dict[str, np.ndarray]:
    left_values = np.arange(m * k, dtype=np.float64)
    right_values = np.arange(k * n, dtype=np.float64)
    left = (
        ((left_values % 11) - 5.0) + 1j * ((left_values % 7) - 3.0)
    ).reshape(m, k)
    right = (
        ((right_values % 13) - 6.0) + 1j * ((right_values % 5) - 2.0)
    ).reshape(k, n)
    return {node.left.tensor_id: left, node.right.tensor_id: right}


def _grouped_slice_fixture() -> tuple[
    ContractionDAG, dict[str, np.ndarray], np.ndarray
]:
    left = TensorSpec("slice_left", (0, 1), (2, 4), "dense", dtype="complex128")
    right = TensorSpec("slice_right", (1, 2), (4, 2), "dense", dtype="complex128")
    node = ContractNode(
        node_id="sliced",
        left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id=right.id, labels=right.labels, shape=right.shape),
        output=TensorSpec(
            "slice_out",
            (0, 2),
            (2, 2),
            "dense",
            dtype="complex128",
            produced_by="sliced",
        ),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = slice_contraction(
        ContractionDAG(
            tensors=(left, right),
            nodes=(node,),
            output=TensorView(tensor_id="slice_out", labels=(0, 2), shape=(2, 2)),
        ),
        node_id="sliced",
        labels=(1,),
    )
    left_value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.complex128)
    right_value = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.complex128)
    expected_real = np.zeros((2, 2), dtype=np.float32)
    for index in range(4):
        branch = np.multiply(
            np.asarray(left_value.real[:, index, None], dtype=np.float32),
            np.asarray(right_value.real[index, None, :], dtype=np.float32),
            dtype=np.float32,
        )
        expected_real = np.add(expected_real, branch, dtype=np.float32)
    return (
        dag,
        {"slice_left": left_value, "slice_right": right_value},
        np.asarray(expected_real, dtype=np.complex64),
    )


def test_open_preflight_rejects_dag_plan_mismatch_before_opener(tmp_path: Path) -> None:
    node, dag, plan, resources, calls, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    del node
    other = replace(plan, logical_plan_id="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        open_upmem(dag, other, resources)
    assert calls == []


def test_open_static_resource_validation_precedes_opener(tmp_path: Path) -> None:
    _, dag, plan, resources, calls, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    invalid = replace(resources, host_binary=str(tmp_path / "missing-host"))
    with pytest.raises(UnsupportedExecution, match="not a regular file"):
        open_upmem(dag, plan, invalid)
    assert calls == []


def test_opener_unsupported_execution_is_preserved(tmp_path: Path) -> None:
    _, dag, plan, resources, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")
    expected = UnsupportedExecution("preflight", "no rank available", "rank_access")

    def opener(*_args):
        raise expected

    with pytest.raises(UnsupportedExecution) as caught:
        open_upmem(dag, plan, replace(resources, session_opener=opener))
    assert caught.value is expected


def test_invalid_opened_object_is_closed_once_before_failure(tmp_path: Path) -> None:
    _, dag, plan, resources, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")

    class InvalidOpened:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    opened = InvalidOpened()
    with pytest.raises(ExecutionFailed) as caught:
        open_upmem(dag, plan, replace(resources, session_opener=lambda *_: opened))
    assert caught.value.stage == "session_open"
    assert opened.close_calls == 1
    assert caught.value.backend_facts["nonconforming_cleanup_succeeded"] is True


@pytest.mark.parametrize(
    "policy",
    ["split_complex_float32_v1", "split_complex_int8_shared_scale_v1"],
)
def test_persistent_session_matches_replay_and_renews_deadline(
    tmp_path: Path, policy: str
) -> None:
    node, dag, plan, resources, calls, _ = _opened(tmp_path, policy=policy, k=257)
    inputs = _inputs(node, k=257)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    session = open_upmem(dag, plan, resources, timeout_s=10.0)
    low_level = session._low_level.session
    first = session.run_once(inputs)
    first_renewed_deadline = getattr(low_level, "_deadline", 0.0)
    second = session.run_once(inputs)
    second_deadline = getattr(low_level, "_deadline", 0.0)
    assert calls == [10.0]
    assert first_renewed_deadline > time.monotonic()
    assert second_deadline > first_renewed_deadline
    np.testing.assert_array_equal(first.output, expected.output)
    np.testing.assert_array_equal(second.output, expected.output)
    assert first.measurement.scope_id == "steady_execution_v1"
    assert first.measurement.h2d_s == pytest.approx(0.08)
    assert first.measurement.kernel_s == pytest.approx(0.16)
    assert first.measurement.d2h_s == pytest.approx(0.08)
    timing = first.backend_facts["operation_facts"][0]["timing"]
    assert timing["rank_response_total_route_max_sum_s"] == pytest.approx(0.32)
    assert timing["request_wave_wall_sum_s"] >= 0.0
    assert timing["total_wall_s"] >= timing["request_wave_wall_sum_s"]
    assert timing["request_build_sum_s"] >= 0.0
    assert timing["request_work_unit_materialization_sum_s"] >= 0.0
    assert timing["request_artifact_build_sum_s"] >= 0.0
    assert timing["request_payload_record_staging_sum_s"] >= 0.0
    assert timing["request_manifest_sidecar_staging_sum_s"] >= 0.0
    assert timing["request_payload_materialization_sum_s"] >= 0.0
    assert timing["request_payload_file_write_sum_s"] >= 0.0
    assert timing["request_payload_hashing_sum_s"] >= 0.0
    assert timing["request_payload_record_construction_sum_s"] >= 0.0
    assert timing["request_payload_record_count"] == 8
    assert timing["request_payload_files_created"] == 16
    assert timing["request_payload_bytes_staged"] == timing["request_payload_bytes_hashed"]
    assert timing["request_build_sum_s"] >= (
        timing["request_work_unit_materialization_sum_s"]
        + timing["request_artifact_build_sum_s"]
    )
    assert timing["request_artifact_build_sum_s"] >= (
        timing["request_payload_record_staging_sum_s"]
        + timing["request_manifest_sidecar_staging_sum_s"]
    )
    assert timing["rank_submit_parallel_wall_sum_s"] >= 0.0
    assert timing["rank_submit_total_max_sum_s"] == pytest.approx(0.32)
    assert timing["rank_submit_artifact_validation_max_sum_s"] == pytest.approx(0.04)
    assert timing["rank_submit_protocol_write_max_sum_s"] == pytest.approx(0.04)
    assert timing["rank_submit_response_wait_max_sum_s"] == pytest.approx(0.2)
    assert timing["rank_submit_response_validation_max_sum_s"] == pytest.approx(0.04)
    assert timing["coordinator_response_processing_sum_s"] >= 0.0
    assert first.backend_facts["physical_plan_id"]
    assert first.backend_facts["startup_resource_admission_passed"] is True
    assert first.backend_facts["execution_resource_admission_passed"] is True
    assert first.backend_facts["execution_active_dpu_count"] == 1
    assert first.backend_facts["operation_facts"][0]["timing_scope"] == (
        "sum_of_per_request_max_rank_response_counters_v1"
    )
    assert first.measurement.h2d_bytes is not None
    assert first.measurement.d2h_bytes is not None
    _close_mock_session(session)


def test_complex_lanes_reuse_only_the_session_record_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1", k=17
    )
    observed: list[object] = []
    original = runtime.build_v4_request

    def wrapped(*args: object, **kwargs: object) -> object:
        observed.append(kwargs.get("record_templates"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "build_v4_request", wrapped)
    session = open_upmem(dag, plan, resources, timeout_s=10.0)
    expected = replay_upmem_plan_once(dag, plan, _inputs(node, k=17))
    actual = session.run_once(_inputs(node, k=17))
    first_run_count = len(observed)
    second = session.run_once(_inputs(node, k=17))

    np.testing.assert_array_equal(actual.output, expected.output)
    np.testing.assert_array_equal(second.output, expected.output)
    assert observed
    assert observed[0] is None
    assert observed[first_run_count] is None
    assert any(isinstance(value, Mapping) for value in observed[:first_run_count])
    assert any(isinstance(value, Mapping) for value in observed[first_run_count:])
    _close_mock_session(session)


@pytest.mark.parametrize("dpu_count", (1, 2, 4))
def test_one_rank_t8_dispatch_replays_on_every_requested_dpu(
    tmp_path: Path, dpu_count: int
) -> None:
    node = _task(k=8, m=300, n=300)
    dag, plan = _final_plan_for_node(
        node,
        policy="split_complex_float32_v1",
        dpu_count=dpu_count,
        tasklets_per_dpu=8,
    )
    engine = _engine(
        tmp_path / "engine",
        dpu_count=dpu_count,
        tasklets_per_dpu=8,
    )

    def opener(_dag, final_plan, _resources, _timeout_s):
        return engine.open_session(final_plan.numeric_policy, final_plan.topology)

    inputs = _inputs(node, k=8, m=300, n=300)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    session = open_upmem(dag, plan, _resources(tmp_path, opener), timeout_s=10.0)
    actual = session.run_once(inputs)
    terminal = session.close()

    np.testing.assert_array_equal(actual.output, expected.output)
    assert actual.backend_facts["physical_plan_id"] == runtime.physical_plan_id(plan)
    assert actual.backend_facts["physical_plan_consumed"] is True
    assert actual.backend_facts["requested_dpus"] == dpu_count
    assert actual.backend_facts["allocated_dpus"] == dpu_count
    assert actual.backend_facts["active_dpus"] == dpu_count
    assert actual.backend_facts["execution_active_dpu_count"] == dpu_count
    assert actual.backend_facts["rank_count"] == 1
    assert actual.backend_facts["execution_active_rank_count"] == 1
    assert actual.backend_facts["tasklets_per_dpu"] == 8
    assert terminal["requested_dpu_count"] == dpu_count
    assert terminal["allocated_dpu_count"] == dpu_count
    assert terminal["observed_rank_count"] == 1
    assert terminal["observed_tasklets_per_dpu"] == 8


def test_upmem_backend_kernel_provenance(tmp_path: Path) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1", k=257
    )
    session = open_upmem(dag, plan, resources)

    sample = session.run_once(_inputs(node, k=257))

    assert sample.backend_facts["kernel_policy"] == plan.kernel_policy
    assert sample.backend_facts["kernel_implementation_id"] == (
        "upmem_sdk_hardware_v4_wram_panel_kernel"
    )
    _close_mock_session(session)


def test_multi_rank_plan_does_not_infer_global_phase_timings(tmp_path: Path) -> None:
    node = _task(k=17, m=1, n=1)
    dag, plan = _final_plan_for_node(
        node,
        policy="split_complex_float32_v1",
        dpu_count=2,
        rank_count=2,
    )
    engine = _engine(tmp_path / "engine", dpu_count=2, rank_count=2)

    def opener(_dag, final_plan, _resources, _timeout_s):
        return engine.open_session(final_plan.numeric_policy, final_plan.topology)

    session = open_upmem(
        dag,
        plan,
        _resources(tmp_path, opener, rank_count=2),
        timeout_s=10.0,
    )
    sample = session.run_once(_inputs(node, k=17))

    assert sample.measurement.h2d_s is None
    assert sample.measurement.kernel_s is None
    assert sample.measurement.d2h_s is None
    _close_mock_session(session)


def test_request_response_requires_total_route_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources)
    rank_session = session._low_level.session.ranks[0].session
    original_submit = rank_session.submit

    def missing_total_route_time(*args, **kwargs):
        response = original_submit(*args, **kwargs)
        del response["timing"]["total_route_time_s"]
        return response

    monkeypatch.setattr(rank_session, "submit", missing_total_route_time)
    with pytest.raises(ExecutionFailed, match="total_route_time_s"):
        session.run_once(_inputs(node, k=5))
    _close_mock_session(session)


def test_request_response_requires_host_submit_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources)
    rank_session = session._low_level.session.ranks[0].session
    original_submit = rank_session.submit

    def missing_host_submit_timing(*args, **kwargs):
        response = original_submit(*args, **kwargs)
        del response["host_submit_timing"]
        return response

    monkeypatch.setattr(rank_session, "submit", missing_host_submit_timing)
    with pytest.raises(ExecutionFailed, match="host_submit_timing"):
        session.run_once(_inputs(node, k=5))
    _close_mock_session(session)


@pytest.mark.parametrize(
    "policy",
    ["split_complex_float32_v1", "split_complex_int8_shared_scale_v1"],
)
def test_open_upmem_simulator_matches_replay_and_rejects_physical_claims(
    tmp_path: Path, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _task(k=17, m=1, n=1)
    dag, plan = _final_plan_for_node(node, policy=policy)
    engine = _engine(
        tmp_path / "simulator-engine",
        dpu_count=1,
        execution_target="sdk_simulator",
    )
    monkeypatch.setattr(runtime, "UpmemV4Executor", lambda **_kwargs: engine.engine)
    resources = _simulator_resources(tmp_path)
    inputs = _inputs(node, k=17)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    with open_upmem_simulator(dag, plan, resources, timeout_s=10.0) as session:
        sample = session.run_once(inputs)
        terminal = session.close()
    np.testing.assert_array_equal(sample.output, expected.output)
    assert sample.backend_facts["target_observed"] == "sdk_simulator"
    assert sample.backend_facts["physical_target_verified"] is False
    assert sample.backend_facts["hardware_kernel_executed"] is False
    assert sample.backend_facts["simulator_kernel_executed"] is True
    assert terminal["target_observed"] == "sdk_simulator"
    assert terminal["physical_target_verified"] is False
    assert terminal["hardware_kernel_executed"] is False
    assert terminal["simulator_kernel_executed"] is True
    assert terminal["hardware_allocation_verified"] is False
    assert terminal["hardware_release_attempted"] is True
    assert terminal["hardware_release_succeeded"] is True
    assert terminal["hardware_release_verified"] is True
    assert terminal["startup_resource_admission_passed"] is True
    for key in (
        "timing_claim_applicable",
        "scaling_claim_applicable",
        "speedup_claim_applicable",
        "energy_claim_applicable",
    ):
        assert terminal[key] is False


def test_open_rejects_ready_resource_admission_mismatch(tmp_path: Path) -> None:
    _, dag, plan, resources, _, engine = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )

    def opener(_dag, final_plan, _resources, _timeout_s):
        opened = engine.open_session(final_plan.numeric_policy, final_plan.topology)
        opened.ranks[0].session.startup["tasklets_per_dpu"] = 2
        return opened

    with pytest.raises(ExecutionFailed) as caught:
        open_upmem(dag, plan, replace(resources, session_opener=opener))

    assert caught.value.stage == "resource_admission"
    assert caught.value.backend_facts["startup_resource_admission_passed"] is False
    assert "ready_tasklet_count_mismatch" in caught.value.backend_facts[
        "startup_resource_admission_reasons"
    ]


def test_open_upmem_simulator_rejects_injected_session_opener(
    tmp_path: Path,
) -> None:
    node = _task(k=5, m=1, n=1)
    dag, plan = _final_plan_for_node(node, policy="split_complex_float32_v1")
    resources = _simulator_resources(tmp_path, lambda *_args: object())

    with pytest.raises(UnsupportedExecution, match="injected session openers"):
        open_upmem_simulator(dag, plan, resources)


@pytest.mark.parametrize(
    "policy",
    ["split_complex_float32_v1", "split_complex_int8_shared_scale_v1"],
)
def test_raw_lanes_and_operands_match_cpu_physical_plan_replay(
    tmp_path: Path,
    policy: str,
) -> None:
    node, dag, plan, resources, _, _ = _opened(tmp_path, policy=policy, k=257)
    inputs = _inputs(node, k=257)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    session = open_upmem(dag, plan, resources)
    actual = session.run_once(inputs)
    _close_mock_session(session)
    np.testing.assert_array_equal(actual.output, expected.output)
    assert (
        actual.numeric_facts["raw_lane_records"]
        == expected.numeric_facts["raw_lane_records"]
    )
    assert (
        actual.numeric_facts["operand_records"]
        == expected.numeric_facts["operand_records"]
    )


def test_output_hash_includes_dtype_and_shape() -> None:
    assert runtime._array_hash(np.array([1], dtype=np.uint8)) != runtime._array_hash(
        np.array([[1]], dtype=np.uint8)
    )
    assert runtime._array_hash(np.array([1], dtype=np.uint8)) != runtime._array_hash(
        np.array([1], dtype=np.int8)
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("cpu_fallback_used", True),
        ("hardware_kernel_executed", False),
        ("simulator_kernel_executed", True),
        ("test_double_execution", True),
    ],
)
def test_operation_observations_reject_unadmitted_execution_facts(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _, _, plan, _, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")
    operation = {
        "target_observed": "physical_hardware",
        "test_double_execution": False,
        "cpu_fallback_used": False,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "physical_stage_consumed": True,
        "bulk_set_launch_verified": True,
        "lane_pass_count": 4,
        "active_rank_count": 1,
        "active_dpu_count": 1,
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "rank_count": plan.topology.rank_count,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }
    operation[field] = value
    with pytest.raises(ValueError):
        runtime._derive_operation_observations([operation], plan)


@pytest.mark.parametrize(
    "field, value",
    [
        ("physical_stage_consumed", False),
        ("bulk_set_launch_verified", False),
        ("lane_pass_count", 3),
        ("active_rank_count", 0),
        ("active_dpu_count", 0),
        ("target_observed", "not_verified"),
    ],
)
def test_operation_observations_require_positive_stage_facts(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _, _, plan, _, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")
    operation = {
        "target_observed": "physical_hardware",
        "test_double_execution": False,
        "cpu_fallback_used": False,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "physical_stage_consumed": True,
        "bulk_set_launch_verified": True,
        "lane_pass_count": 4,
        "active_rank_count": 1,
        "active_dpu_count": 1,
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "rank_count": plan.topology.rank_count,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }
    operation[field] = value
    with pytest.raises(ValueError):
        runtime._derive_operation_observations([operation], plan)


def test_one_sided_complex_and_reduce_stage_are_deterministic(tmp_path: Path) -> None:
    def contract(prefix: str) -> ContractNode:
        return ContractNode(
            node_id=prefix,
            left=TensorView(tensor_id=f"{prefix}_a", labels=(0, 1), shape=(1, 3)),
            right=TensorView(tensor_id=f"{prefix}_b", labels=(1, 2), shape=(3, 1)),
            output=TensorSpec(
                id=f"{prefix}_o", labels=(0, 2), shape=(1, 1), structure="dense"
            ),
            contracted_labels=(1,),
            output_labels=(0, 2),
        )

    first = contract("a")
    second = contract("b")
    reduce = ReduceNode(
        node_id="reduce",
        inputs=(
            TensorView(tensor_id="a_o", labels=(0, 2), shape=(1, 1)),
            TensorView(tensor_id="b_o", labels=(0, 2), shape=(1, 1)),
        ),
        output=TensorSpec(id="out", labels=(0, 2), shape=(1, 1), structure="dense"),
        dependencies=("a", "b"),
    )
    dag = ContractionDAG(
        tensors=tuple(
            TensorSpec(id=name, labels=labels, shape=shape, structure="dense")
            for name, labels, shape in (
                ("a_a", (0, 1), (1, 3)),
                ("a_b", (1, 2), (3, 1)),
                ("b_a", (0, 1), (1, 3)),
                ("b_b", (1, 2), (3, 1)),
            )
        ),
        nodes=(first, second, reduce),
        output=TensorView(tensor_id="out", labels=(0, 2), shape=(1, 1)),
    )
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=FinalTopology(dpu_count=1, tasklets_per_dpu=1, rank_count=1),
    )
    engine = _engine(tmp_path / "engine", dpu_count=1)

    def opener(_dag, final_plan, _resources, _timeout_s):
        return engine.open_session(final_plan.numeric_policy, final_plan.topology)

    resources = _resources(tmp_path, opener)
    inputs = {
        "a_a": np.array([[1 + 0j, 2 + 0j, 3 + 0j]], dtype=np.complex128),
        "a_b": np.array([[1j], [2j], [3j]], dtype=np.complex128),
        "b_a": np.array([[2 + 0j, 1 + 0j, 0 + 0j]], dtype=np.complex128),
        "b_b": np.array([[1j], [0j], [2j]], dtype=np.complex128),
    }
    expected = replay_upmem_plan_once(dag, plan, inputs)
    session = open_upmem(dag, plan, resources)
    actual = session.run_once(inputs)
    _close_mock_session(session)
    np.testing.assert_array_equal(actual.output, expected.output)
    assert actual.output.dtype == np.dtype(np.complex64)
    assert any(stage.kind == "host_reduce" for stage in plan.stages)


def test_grouped_sliced_session_executes_each_branch_once(tmp_path: Path) -> None:
    dag, inputs, expected = _grouped_slice_fixture()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=FinalTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    engine = _engine(tmp_path / "engine", dpu_count=1)
    resources = _resources(
        tmp_path,
        lambda _dag, final_plan, _resources, _timeout: engine.open_session(
            final_plan.numeric_policy, final_plan.topology
        ),
    )
    session = open_upmem(dag, plan, resources)
    actual = session.run_once(inputs)
    _close_mock_session(session)
    np.testing.assert_array_equal(actual.output, expected)
    branch_ids = tuple(sorted(plan.stages[0].node_ids))
    assert branch_ids == (
        "sliced__slice_1_0",
        "sliced__slice_1_1",
        "sliced__slice_1_2",
        "sliced__slice_1_3",
    )
    assert tuple(item["node_id"] for item in actual.numeric_facts["operations"]) == (
        branch_ids
    )
    declared_stage_id = plan.stages[0].stage_id
    operation_facts = actual.backend_facts["operation_facts"]
    assert tuple(item["node_id"] for item in operation_facts) == branch_ids
    assert (
        tuple(item["declared_stage_id"] for item in operation_facts)
        == (declared_stage_id,) * 4
    )
    assert tuple(item["stage_id"] for item in operation_facts) == tuple(
        f"{declared_stage_id}:node:{node_id}" for node_id in branch_ids
    )
    raw_records = actual.numeric_facts["raw_lane_records"]
    assert all(
        sum(record["node_id"] == node_id for record in raw_records) == 4
        for node_id in branch_ids
    )
    units = tuple(unit for stage in plan.stages for unit in stage.work_units)
    assert len({unit.stable_tile_id for unit in units}) == len(units)
    assert all(unit.stable_tile_id.startswith(f"{unit.node_id}:") for unit in units)


def test_grouped_int8_preserves_per_branch_numeric_facts(tmp_path: Path) -> None:
    dag, inputs, _ = _grouped_slice_fixture()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_int8_shared_scale_v1",
        topology=FinalTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    engine = _engine(tmp_path / "engine", dpu_count=1)
    resources = _resources(
        tmp_path,
        lambda _dag, final_plan, _resources, _timeout: engine.open_session(
            final_plan.numeric_policy, final_plan.topology
        ),
    )
    session = open_upmem(dag, plan, resources)
    actual = session.run_once(inputs)
    _close_mock_session(session)

    branch_ids = tuple(sorted(plan.stages[0].node_ids))
    operations = actual.numeric_facts["operations"]
    assert tuple(item["node_id"] for item in operations) == branch_ids
    assert tuple(item["left_scale"] for item in operations) == pytest.approx(
        tuple(value / 127 for value in (5, 6, 7, 8))
    )
    assert tuple(item["right_scale"] for item in operations) == pytest.approx(
        tuple(value / 127 for value in (2, 4, 6, 8))
    )

    records = actual.numeric_facts["operand_records"]
    by_branch = {
        node_id: {
            record["side"]: record for record in records if record["node_id"] == node_id
        }
        for node_id in branch_ids
    }
    assert all(set(items) == {"left", "right"} for items in by_branch.values())
    for operation in operations:
        operands = by_branch[operation["node_id"]]
        assert operands["left"]["scale"] == operation["left_scale"]
        assert operands["right"]["scale"] == operation["right_scale"]
        assert operation["saturation_real"] == sum(
            item["saturation_real"] for item in operands.values()
        )
        assert operation["saturation_imag"] == sum(
            item["saturation_imag"] for item in operands.values()
        )


def test_grouped_branch_failure_identifies_branch_and_declared_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag, inputs, _ = _grouped_slice_fixture()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=FinalTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    engine = _engine(tmp_path / "engine", dpu_count=1)
    resources = _resources(
        tmp_path,
        lambda _dag, final_plan, _resources, _timeout: engine.open_session(
            final_plan.numeric_policy, final_plan.topology
        ),
    )
    session = open_upmem(dag, plan, resources)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fail(node, _left, _right, *, stage, **_kwargs):
        calls.append(
            (
                node.node_id,
                stage.stage_id,
                tuple(unit.node_id for unit in stage.work_units),
            )
        )
        raise RuntimeError("controlled grouped branch failure")

    monkeypatch.setattr(session._low_level.session, "_execute_complex_core", fail)
    with pytest.raises(ExecutionFailed) as caught:
        session.run_once(inputs)
    _close_mock_session(session)
    declared_stage_id = plan.stages[0].stage_id
    branch_node_id = plan.stages[0].node_ids[0]
    branch_stage_id = f"{declared_stage_id}:node:{branch_node_id}"
    assert calls == [(branch_node_id, branch_stage_id, (branch_node_id,))]
    assert caught.value.stage == branch_stage_id
    assert caught.value.backend_facts["plan_stage_id"] == declared_stage_id
    assert caught.value.backend_facts["branch_stage_id"] == branch_stage_id
    assert caught.value.backend_facts["branch_node_id"] == branch_node_id


@pytest.mark.parametrize(
    "helper_name",
    ["_array_hash", "_raw_lane_fact", "_complex_operand_facts", "_json_safe"],
)
def test_evidence_helpers_are_outside_session_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    inputs = _inputs(node, k=5)
    session = open_upmem(dag, plan, resources)
    baseline = session.run_once(inputs)
    original = getattr(runtime, helper_name)

    def slow_helper(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, helper_name, slow_helper)
    slowed = session.run_once(inputs)
    _close_mock_session(session)
    assert slowed.measurement.total_wall_s - baseline.measurement.total_wall_s < 0.03


def test_test_double_close_is_rejected_and_mock_verified_close_is_idempotent(
    tmp_path: Path,
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path / "good", policy="split_complex_float32_v1"
    )
    del node
    test_double = open_upmem(dag, plan, resources)
    _close_mock_session(test_double)

    rejected_terminal = _verified_terminal(plan)
    rejected_terminal["test_double_execution"] = True
    rejected = runtime.UpmemSession(
        dag,
        plan,
        resources,
        _ControlledTerminalSession(rejected_terminal),
        timeout_s=5.0,
    )
    with pytest.raises(ExecutionFailed, match="session_close"):
        rejected.close()

    low_level = _ControlledTerminalSession(_verified_terminal(plan))
    session = runtime.UpmemSession(dag, plan, resources, low_level, timeout_s=5.0)
    first = session.close()
    second = session.close()
    assert first == second
    assert low_level.close_calls == 1
    # This controlled terminal fixture exercises admission only; no sample is
    # produced and it is never used as a physical-execution assertion.

    def failing_opener(*_args):
        raise RuntimeError("open failed")

    failing = _resources(tmp_path / "open-fail", failing_opener)
    with pytest.raises(ExecutionFailed, match="session_open"):
        open_upmem(dag, plan, failing)


@pytest.mark.parametrize(
    "field, value",
    [
        ("hardware_allocation_verified", None),
        ("binary_identity_verified", False),
        ("hardware_release_verified", False),
        ("test_double_execution", True),
        ("target_observed", "not_verified"),
    ],
)
def test_terminal_admission_requires_all_positive_verification_facts(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _, dag, plan, resources, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")
    terminal = _verified_terminal(plan)
    if value is None:
        del terminal[field]
    else:
        terminal[field] = value
    session = runtime.UpmemSession(
        dag,
        plan,
        resources,
        _ControlledTerminalSession(terminal),
        timeout_s=5.0,
    )
    with pytest.raises(ExecutionFailed) as caught:
        session.close()
    assert caught.value.stage == "session_close"
    assert caught.value.backend_facts.get(field) == value


def test_context_body_error_is_not_masked_by_close_failure(tmp_path: Path) -> None:
    _, dag, plan, resources, _, _ = _opened(tmp_path, policy="split_complex_float32_v1")
    terminal = _verified_terminal(plan)
    terminal["hardware_release_verified"] = False
    session = runtime.UpmemSession(
        dag,
        plan,
        resources,
        _ControlledTerminalSession(terminal),
        timeout_s=5.0,
    )
    with pytest.raises(RuntimeError, match="body failure") as caught:
        with session:
            raise RuntimeError("body failure")
    assert any("close also failed" in note for note in caught.value.__notes__)


def test_close_failure_is_execution_failed_and_not_repeated(tmp_path: Path) -> None:
    node, dag, plan, resources, _, engine = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources)
    low_level = session._low_level.session
    for rank in low_level.ranks:
        rank.session.close = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("close failed")
        )
    with pytest.raises(ExecutionFailed, match="session_close"):
        session.close()
    with pytest.raises(ExecutionFailed, match="session_close"):
        session.close()


def test_run_failure_is_execution_failed_with_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources)

    def fail(*_args, **_kwargs):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(session._low_level.session, "_execute_complex_core", fail)
    with pytest.raises(ExecutionFailed) as caught:
        session.run_once(_inputs(node, k=5))
    assert caught.value.stage == "contract_batch:fixture"
    _close_mock_session(session)


def test_native_failure_stage_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources)

    def fail(*_args, **_kwargs):
        raise V4ProtocolError("kernel_timeout", "deadline expired")

    monkeypatch.setattr(session._low_level.session, "_execute_complex_core", fail)
    with pytest.raises(ExecutionFailed) as caught:
        session.run_once(_inputs(node, k=5))
    assert caught.value.stage == "kernel_timeout"
    assert caught.value.backend_facts["plan_stage_id"] == "contract_batch:fixture"
    _close_mock_session(session)


def test_run_timeout_is_renewed_and_reported_as_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_float32_v1"
    )
    session = open_upmem(dag, plan, resources, timeout_s=0.001)
    native_session = session._low_level.session
    original = native_session._execute_complex_core

    def slow_core(*args, **kwargs):
        time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(native_session, "_execute_complex_core", slow_core)
    with pytest.raises(ExecutionFailed) as caught:
        session.run_once(_inputs(node, k=5))
    assert caught.value.stage == "kernel_timeout"
    assert caught.value.backend_facts["plan_stage_id"] == "contract_batch:fixture"
    _close_mock_session(session)


def test_output_and_facts_are_immutable_and_json_safe(tmp_path: Path) -> None:
    node, dag, plan, resources, _, _ = _opened(
        tmp_path, policy="split_complex_int8_shared_scale_v1", k=257
    )
    session = open_upmem(dag, plan, resources)
    sample = session.run_once(_inputs(node, k=257))
    _close_mock_session(session)
    with pytest.raises(ValueError):
        sample.output[0, 0] = 99
    json.dumps(_plain(sample.backend_facts))
    json.dumps(_plain(sample.numeric_facts))


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
