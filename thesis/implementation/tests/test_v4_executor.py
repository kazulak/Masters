from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pytest

from quantum_bench.core.records import TensorSpec
from quantum_bench.model import ContractNode, ContractionDAG, TensorView
from quantum_bench.execution.compiler import compile_execution
from quantum_bench.execution.contracts import (
    ExecutionPlan,
    NumericMode,
    UpmemCompileRequest,
    UpmemTopology,
)
from quantum_bench.formats.fixed_point import FixedPointSpec, quantize_fixed_point
from quantum_bench.lowering import (
    contraction_dag_hash,
)
from quantum_bench.upmem.runtime import (
    UpmemV4Executor,
    UpmemV4Session,
    _task_structure_hash,
)
from quantum_bench.upmem.plan import UpmemTopology as FinalUpmemTopology
from quantum_bench.upmem.plan import plan_upmem
from quantum_bench.cpu import replay_upmem_plan_once
import quantum_bench.upmem.runtime as engine_module
from quantum_bench.upmem.protocol import (
    NATIVE_EXECUTION_IDENTITY,
    V4ProtocolError,
)


def _output(result: tuple[np.ndarray, dict[str, Any]]) -> np.ndarray:
    return result[0]


def _metadata(result: tuple[np.ndarray, dict[str, Any]]) -> dict[str, Any]:
    return result[1]


def _task(k: int = 5, *, m: int = 3, n: int = 4) -> ContractNode:
    return ContractNode(
        node_id="fixture",
        left=TensorView(tensor_id="left", labels=(0, 1), shape=(m, k)),
        right=TensorView(tensor_id="right", labels=(1, 2), shape=(k, n)),
        output=TensorSpec(id="out", labels=(0, 2), shape=(m, n), structure="dense"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


def _packed_expected(
    node: ContractNode, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_q = quantize_fixed_point(left, FixedPointSpec(route_dtype="int8"))
    right_q = quantize_fixed_point(right, FixedPointSpec(route_dtype="int8"))
    return np.asarray(
        np.einsum(
            left_q.array,
            list(node.left.labels),
            right_q.array,
            list(node.right.labels),
            list(node.output_labels),
            dtype=np.int32,
        ),
        dtype=np.float32,
    ) * np.float32(left_q.record.scale * right_q.record.scale)


def _compiled_node_plan(
    node: ContractNode,
    dpu_count: int,
    rank_count: int = 1,
    numeric_mode: NumericMode = NumericMode.FLOAT32_REAL,
) -> Any:
    dag = ContractionDAG(
        tensors=(
            TensorSpec(
                node.left.tensor_id,
                node.left.labels,
                node.left.shape,
                "dense",
                dtype=node.output.dtype,
            ),
            TensorSpec(
                node.right.tensor_id,
                node.right.labels,
                node.right.shape,
                "dense",
                dtype=node.output.dtype,
            ),
        ),
        nodes=(node,),
        output=TensorView(
            tensor_id=node.output.id,
            labels=node.output.labels,
            shape=node.output.shape,
        ),
    )
    compiled = compile_execution(
        dag,
        UpmemCompileRequest(
            contraction_dag_hash=contraction_dag_hash(dag),
            numeric_mode=numeric_mode,
            topology=UpmemTopology(
                dpu_count=dpu_count,
                tasklets_per_dpu=1,
                rank_count=rank_count,
            ),
        ),
    )
    assert isinstance(compiled, ExecutionPlan)
    return compiled.payload.node_plans[0]


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
    binary_provenance: dict[str, str] = field(default_factory=dict)
    startup: dict[str, Any] = field(default_factory=dict)
    delay_s: float = 0.0
    fail_submit: bool = False
    release_confirmed: bool = True
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_total_bytes: int = 0
    stderr_total_bytes: int = 0
    stdout_limit_exceeded: bool = False
    stderr_limit_exceeded: bool = False
    returncode: int | None = 0
    closed: bool = False
    submissions: list[Any] = field(default_factory=list)
    submitted_operand_values: list[tuple[float, float]] = field(default_factory=list)
    submitted_timeouts: list[float] = field(default_factory=list)
    barrier: Any = None
    response_identity_overrides: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.startup = {
            "event": "READY",
            "status": "ready",
            "target_observed": "physical_hardware",
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "hardware_allocation_verified": True,
            **NATIVE_EXECUTION_IDENTITY,
            **self.binary_provenance,
        }

    def submit(
        self, artifact: Any, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        if timeout_s is not None:
            self.submitted_timeouts.append(float(timeout_s))
        self.submissions.append(artifact)
        if self.fail_submit:
            raise RuntimeError("submission failed")
        for record in artifact.work_units:
            if not record.flags:
                dtype = (
                    np.int8
                    if self.profile.numeric_mode_name == "host_packed_int8"
                    else np.dtype("<f4")
                )
                self.submitted_operand_values.append(
                    (
                        float(
                            np.fromfile(
                                artifact.root / record.a_path,
                                dtype=dtype,
                                count=1,
                            )[0]
                        ),
                        float(
                            np.fromfile(
                                artifact.root / record.b_path,
                                dtype=dtype,
                                count=1,
                            )[0]
                        ),
                    )
                )
                break
        if self.barrier is not None:
            self.barrier.enter()
        if timeout_s is not None and self.delay_s > timeout_s:
            time.sleep(max(0.0, timeout_s))
            raise V4ProtocolError(
                "kernel_timeout", "fake request exceeded the graph deadline"
            )
        time.sleep(self.delay_s)
        for record in artifact.work_units:
            if record.flags:
                continue
            a_dtype = (
                np.int8
                if self.profile.numeric_mode_name == "host_packed_int8"
                else np.dtype("<f4")
            )
            a = np.fromfile(
                artifact.root / record.a_path,
                dtype=a_dtype,
                count=record.m_elements * record.k_elements,
            ).reshape(record.m_elements, record.k_elements)
            b = np.fromfile(
                artifact.root / record.b_path,
                dtype=a_dtype,
                count=record.k_elements * record.n_elements,
            ).reshape(record.k_elements, record.n_elements)
            output = (
                a.astype(np.int64) @ b.astype(np.int64) if a_dtype == np.int8 else a @ b
            )
            (artifact.root / record.c_path).write_bytes(
                np.asarray(
                    output, dtype="<i4" if a_dtype == np.int8 else "<f4"
                ).tobytes()
            )
        return {
            "status": "completed",
            "target_observed": "physical_hardware",
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": True,
            "allocated_dpu_count": self.profile.dpu_count,
            "requested_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "request_sequence": artifact.request_sequence,
            "bulk_set_launch_verified": True,
            **NATIVE_EXECUTION_IDENTITY,
            **self.response_identity_overrides,
            "transfer": {"h2d_bytes": 10, "d2h_bytes": 5, "total_bytes": 15},
            "timing": {"h2d_time_s": 0.01, "launch_time_s": 0.02, "d2h_time_s": 0.01},
        }

    def close(self, *, timeout_s: float | None = None) -> _Release:
        del timeout_s
        self.closed = True
        return _Release(
            release_confirmed=self.release_confirmed,
            stdout=self.stdout,
            stderr=self.stderr,
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=self.stderr_truncated,
            stdout_total_bytes=self.stdout_total_bytes,
            stderr_total_bytes=self.stderr_total_bytes,
            stdout_limit_exceeded=self.stdout_limit_exceeded,
            stderr_limit_exceeded=self.stderr_limit_exceeded,
            event={"returncode": self.returncode},
        )


def _engine(
    tmp_path: Path,
    *,
    ranks: int = 1,
    dpu_count: int = 2,
    sessions: list[_FakeSession] | None = None,
    delay_s: float = 0.0,
    barrier: Any = None,
    timeout_s: float = 60.0,
    startup_delay_s: float = 0.0,
    startup_delays_s: tuple[float, ...] = (),
    fail_submit: bool = False,
) -> "_PlannedEngine":
    created: list[_FakeSession] = sessions if sessions is not None else []
    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    binary_provenance = {
        "host_binary_sha256": hashlib.sha256(host_binary.read_bytes()).hexdigest(),
        "dpu_binary_sha256": hashlib.sha256(dpu_binary.read_bytes()).hexdigest(),
        "initialization_binary_sha256": hashlib.sha256(
            initialization_binary.read_bytes()
        ).hexdigest(),
    }

    def factory(command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        del command, session_root
        session_index = len(created)
        session = _FakeSession(
            profile=profile,
            binary_provenance=binary_provenance,
            delay_s=delay_s,
            barrier=barrier,
            fail_submit=fail_submit,
        )
        created.append(session)
        delay = (
            startup_delays_s[session_index]
            if session_index < len(startup_delays_s)
            else startup_delay_s
        )
        time.sleep(delay)
        return session

    return _PlannedEngine(
        UpmemV4Executor(
            session_root=tmp_path,
            host_binary=host_binary,
            dpu_binary=dpu_binary,
            initialization_binary=initialization_binary,
            rank_paths=tuple(f"/dev/dpu_rank{i}" for i in range(ranks)),
            dpu_count=dpu_count,
            timeout_s=timeout_s,
            session_factory=factory,
        )
    )


def _topology(count: int, *, rank_count: int = 1) -> UpmemTopology:
    return UpmemTopology(dpu_count=count, tasklets_per_dpu=1, rank_count=rank_count)


def _final_plan_for_node(
    node: ContractNode,
    *,
    policy: str,
    dpu_count: int = 1,
    rank_count: int = 1,
) -> tuple[ContractionDAG, Any]:
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
        topology=FinalUpmemTopology(
            dpu_count=dpu_count,
            tasklets_per_dpu=1,
            rank_count=rank_count,
        ),
    )


@dataclass
class _SubmissionBarrier:
    participants: int
    entered: int = 0
    both_entered: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self) -> None:
        with self.lock:
            self.entered += 1
            if self.entered == self.participants:
                self.both_entered.set()
        if not self.both_entered.wait(timeout=2.0):
            raise AssertionError("rank submissions were serialized")


@dataclass(frozen=True)
class _PlannedSession:
    """Test helper that supplies a compiled placement to the real session."""

    session: UpmemV4Session

    def execute(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        node_plan: Any = None,
    ) -> Any:
        if node_plan is None:
            mode = (
                NumericMode.HOST_PACKED_INT8
                if self.session.numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                else NumericMode.FLOAT32_REAL
            )
            node_plan = _compiled_node_plan(
                node,
                sum(rank.local_dpus for rank in self.session.ranks),
                len(self.session.ranks),
                numeric_mode=mode,
            )
        return self.session.execute(
            node,
            left,
            right,
            node_plan=node_plan,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


@dataclass(frozen=True)
class _PlannedEngine:
    engine: UpmemV4Executor

    def open_session(self, *args: Any, **kwargs: Any) -> _PlannedSession:
        return _PlannedSession(self.engine.open_session(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)


def test_float_and_packed_int8_match_reference_across_k_chunks(tmp_path: Path) -> None:
    left = np.arange(900, dtype=np.float32).reshape(3, 300) / 7
    right = np.arange(1200, dtype=np.float32).reshape(300, 4) / 5
    engine = _engine(tmp_path)
    task = _task(300)
    float_session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(2))
    float_result = float_session.execute(task, left, right)
    np.testing.assert_allclose(
        _output(float_result), left @ right, rtol=1e-6, atol=1e-6
    )
    assert _metadata(float_result)["k_chunk_count"] == 2
    assert len(_metadata(float_result)["request_manifest_hashes"]) == 2
    assert _metadata(float_result)["application_visible_transfer_bytes"] == 30
    assert _metadata(float_result)["host_quantization_time_s"] == 0.0
    assert _metadata(float_result)["preparation_time_s"] >= 0.0
    assert (
        _metadata(float_result)["application_visible_transfer_bytes"]
        == _metadata(float_result)["application_visible_h2d_bytes"]
        + _metadata(float_result)["application_visible_d2h_bytes"]
    )
    assert float_session.close()["hardware_release_confirmed"]
    packed_session = engine.open_session(
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1, _topology(2)
    )
    packed_result = packed_session.execute(task, left, right)
    expected = _packed_expected(task, left, right)
    np.testing.assert_allclose(_output(packed_result), expected, rtol=0, atol=1e-5)
    assert _metadata(packed_result)["packed_int8_transport"]
    assert _metadata(packed_result)["host_quantization_time_s"] >= 0.0
    assert _metadata(packed_result)["host_dequantization_time_s"] >= 0.0
    assert _metadata(packed_result)["preparation_time_s"] == 0.0
    assert "host_dequantization_timing_overlap" not in _metadata(packed_result)
    assert _metadata(packed_result)["graph_intermediate_placement"] == "host_managed"
    assert _metadata(packed_result)["profile"] == "m5_whole_circuit_v4_v1"
    assert _metadata(packed_result)["abi"] == "execution_plan_v4"
    assert _metadata(packed_result)["session"] == "persistent_rank_session_v1"
    assert _metadata(packed_result)["dispatch"] == "bulk_set_synchronous_v1"
    assert _metadata(packed_result)["kernel"] == "dpu_gemm_tile_v4"
    assert (
        _metadata(packed_result)["transfer_accounting_scope"]
        == "application_visible_sdk_recorded"
    )
    assert (
        _metadata(float_result)["task_structure_sha256"]
        == _metadata(packed_result)["task_structure_sha256"]
    )
    assert (
        _metadata(float_result)["request_contract_sha256"]
        != _metadata(packed_result)["request_contract_sha256"]
    )


def test_packed_request_contract_uses_float32_canonical_inputs(
    tmp_path: Path,
) -> None:
    node = _task(k=3, m=1, n=1)
    left64 = np.array([[1.00000006, -0.33333334, 0.12500001]], dtype=np.float64)
    right64 = np.array([[0.75], [-0.5], [0.25000003]], dtype=np.float64)

    session64 = _engine(tmp_path / "float64", dpu_count=1).open_session(
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1, _topology(1)
    )
    result64 = session64.execute(node, left64, right64)
    session64.close()

    session32 = _engine(tmp_path / "float32", dpu_count=1).open_session(
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1, _topology(1)
    )
    result32 = session32.execute(
        node, left64.astype(np.float32), right64.astype(np.float32)
    )
    session32.close()

    assert (
        _metadata(result64)["request_contract_sha256"]
        == _metadata(result32)["request_contract_sha256"]
    )
    assert _metadata(result64)["left_scale"] == _metadata(result32)["left_scale"]
    assert _metadata(result64)["right_scale"] == _metadata(result32)["right_scale"]
    np.testing.assert_array_equal(_output(result64), _output(result32))


def test_compiled_work_units_reach_native_request_artifact(tmp_path: Path) -> None:
    task = _task(k=5, m=300, n=4)
    left = np.ones((300, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)
    session = _engine(tmp_path, dpu_count=2).open_session(
        NumericMode.FLOAT32_REAL, _topology(2)
    )
    node_plan = _compiled_node_plan(task, 2)

    result = session.execute(task, left, right, node_plan=node_plan)

    submissions = session.ranks[0].session.submissions
    local_dpu_ids = [
        record.local_dpu_id
        for artifact in submissions
        for record in artifact.work_units
        if not record.flags
    ]
    assert local_dpu_ids == [0, 1]
    assert _metadata(result)["physical_plan_consumed"] is True
    np.testing.assert_allclose(_output(result), left @ right, rtol=1e-6, atol=1e-6)
    session.close()


def test_fixed_strategy_identity_binds_active_engine_source(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1).engine
    expected_source_hash = hashlib.sha256(
        Path(engine_module.__file__).read_bytes()
    ).hexdigest()
    identity = engine.strategy_identity
    records = identity["strategies"]

    assert len(expected_source_hash) == 64
    assert expected_source_hash.islower()
    assert all(
        record["implementation_type"] == "fixed_direct_mechanism"
        and record["module_source_sha256"] == expected_source_hash
        for record in records
    )
    expected_config_hash = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    assert engine.strategy_config_hash == expected_config_hash


def test_node_structure_hash_binds_all_contraction_fields() -> None:
    node = _task()
    variants = (
        replace(node, node_id="other"),
        replace(node, left=replace(node.left, tensor_id="other_left")),
        replace(node, right=replace(node.right, tensor_id="other_right")),
        replace(node, output=replace(node.output, id="other_output")),
        replace(node, left=replace(node.left, shape=(1, 5))),
        replace(node, right=replace(node.right, shape=(5, 1))),
        replace(node, output=replace(node.output, shape=(1, 1))),
        replace(node, left=replace(node.left, labels=(3, 1))),
        replace(node, right=replace(node.right, labels=(1, 4))),
        replace(node, contracted_labels=(4,)),
        replace(node, output_labels=(0, 4)),
    )

    hashes = {
        _task_structure_hash(node),
        *(_task_structure_hash(value) for value in variants),
    }
    assert len(hashes) == len(variants) + 1


def test_request_contract_is_deterministic_and_binds_int8_data(tmp_path: Path) -> None:
    task = _task(5)
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1, _topology(1)
    )
    first = session.execute(task, left, right)
    second = session.execute(task, left, right)
    changed = left.copy()
    changed[0, 0] = 100.0
    third = session.execute(task, changed, right)
    assert (
        _metadata(first)["task_structure_sha256"]
        == _metadata(second)["task_structure_sha256"]
        == _metadata(third)["task_structure_sha256"]
    )
    assert (
        _metadata(first)["request_contract_sha256"]
        == _metadata(second)["request_contract_sha256"]
    )
    assert (
        _metadata(first)["request_contract_sha256"]
        != _metadata(third)["request_contract_sha256"]
    )
    assert _metadata(first)["left_scale"] != _metadata(third)["left_scale"]
    session.close()


def test_binary_hashes_and_roots_are_provenance_only(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    result = session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    terminal = session.close()
    for metadata in (_metadata(result), terminal):
        assert metadata["source_root"].endswith("/thesis/implementation")
        assert metadata["session_root"] == str(tmp_path.resolve())
        for label in ("host_binary", "dpu_binary", "initialization_binary"):
            path = Path(metadata[f"{label}_path"])
            assert path.is_absolute()
            assert (
                metadata[f"{label}_sha256"]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
    assert "source_root" not in _metadata(result).get("executor_config", {})
    assert "session_root" not in _metadata(result).get("executor_config", {})


def test_ready_binary_hash_mismatch_denies_admission(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    session.ranks[0].session.startup["dpu_binary_sha256"] = "0" * 64
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["hardware_allocation_verified"] is True
    assert terminal["binary_identity_verified"] is False
    assert terminal["native_kernel_executed"] is True
    assert terminal["hardware_kernel_executed"] is True


def test_close_retains_bounded_rank_diagnostics_on_success(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    native = session.ranks[0].session
    native.stdout = "native stdout\n"
    native.stderr = "native stderr\n"
    native.stdout_truncated = True
    native.stdout_total_bytes = 100_000
    native.returncode = 0
    terminal = session.close()
    assert terminal["native_diagnostics"] == [
        {
            "rank_index": 0,
            "rank_path": "/dev/dpu_rank0",
            "stdout": "native stdout\n",
            "stderr": "native stderr\n",
            "stdout_truncated": True,
            "stderr_truncated": False,
            "stdout_total_bytes": 100_000,
            "stderr_total_bytes": 0,
            "stdout_limit_exceeded": False,
            "stderr_limit_exceeded": False,
            "returncode": 0,
            "release_confirmed": True,
        }
    ]
    json.dumps(terminal)


def test_close_retains_rank_diagnostics_on_release_failure(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    native = session.ranks[0].session
    native.release_confirmed = False
    native.stdout = "release stdout\n"
    native.stderr = "release stderr\n"
    native.returncode = 7
    terminal = session.close()
    assert terminal["native_diagnostics"][0] == {
        "rank_index": 0,
        "rank_path": "/dev/dpu_rank0",
        "stdout": "release stdout\n",
        "stderr": "release stderr\n",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_total_bytes": 0,
        "stderr_total_bytes": 0,
        "stdout_limit_exceeded": False,
        "stderr_limit_exceeded": False,
        "returncode": 7,
        "release_confirmed": False,
    }
    assert terminal["failure_stage"] == "hardware_release_failed"


def test_release_failure_preserves_execution_event_semantics(tmp_path: Path) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    session.ranks[0].session.release_confirmed = False
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_kernel_executed"] is True
    assert terminal["hardware_kernel_executed"] is True
    assert terminal["failure_stage"] == "hardware_release_failed"


def test_missing_or_non_executable_binary_fails_before_session_allocation(
    tmp_path: Path,
) -> None:
    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    host_binary.chmod(host_binary.stat().st_mode & ~0o111)
    with pytest.raises(ValueError, match="not executable"):
        UpmemV4Executor(
            session_root=tmp_path / "session",
            host_binary=host_binary,
            dpu_binary=dpu_binary,
            initialization_binary=initialization_binary,
            rank_paths=("/dev/dpu_rank0",),
            dpu_count=1,
        )


def test_int8_uses_one_full_operand_scale_not_tile_scales(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1, _topology(1)
    )
    task = _task(300)
    left = np.ones((3, 300), dtype=np.float32)
    left[:, 299] = 100
    result = session.execute(task, left, np.ones((300, 4), dtype=np.float32))
    assert _metadata(result)["left_scale"] == pytest.approx(100 / 127)
    assert _metadata(result)["k_chunk_count"] == 2


def test_topology_and_startup_failures_are_closed(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ranks=2, dpu_count=2)
    with pytest.raises(ValueError, match="device count"):
        engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    with pytest.raises(ValueError, match="rank count"):
        engine.open_session(NumericMode.FLOAT32_REAL, _topology(2))
    closed: list[_FakeSession] = []
    calls = 0

    def factory(command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        nonlocal calls
        del command, session_root
        calls += 1
        if calls == 2:
            raise RuntimeError("second rank failed")
        session = _FakeSession(profile=profile)
        closed.append(session)
        return session

    host_binary, dpu_binary, initialization_binary = _binaries(
        tmp_path / "failure" / "binaries"
    )
    failing = UpmemV4Executor(
        session_root=tmp_path / "failure",
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=initialization_binary,
        rank_paths=("/dev/dpu_rank0", "/dev/dpu_rank1"),
        dpu_count=2,
        session_factory=factory,
    )
    with pytest.raises(RuntimeError, match="second rank"):
        failing.open_session(NumericMode.FLOAT32_REAL, _topology(2, rank_count=2))
    assert closed[0].closed


def test_rank_submissions_are_concurrent_and_fail_closed(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        ranks=2,
        dpu_count=2,
        barrier=_SubmissionBarrier(participants=2),
    )
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(2, rank_count=2))
    session.execute(
        _task(m=300),
        np.ones((300, 5), dtype=np.float32),
        np.ones((5, 4), dtype=np.float32),
    )
    assert session.close()["hardware_release_confirmed"]
    sessions: list[_FakeSession] = []
    bad = _engine(tmp_path / "bad", dpu_count=1, sessions=sessions)
    failing_session = bad.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    failing_session.ranks[0].session.fail_submit = True
    with pytest.raises(RuntimeError, match="submission failed"):
        failing_session.execute(
            _task(),
            np.ones((3, 5), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
        )
    terminal = failing_session.close()
    assert terminal["cpu_fallback_used"] is False
    assert terminal["hardware_kernel_executed"] is False
    assert terminal["failure_stage"] == "hardware_task_execution_failed"


def test_session_uses_one_deadline_across_multiple_task_requests(
    tmp_path: Path,
) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        dpu_count=1,
        sessions=sessions,
        delay_s=0.03,
        timeout_s=0.05,
    )
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    left = np.ones((3, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)
    session.execute(_task(), left, right)
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        session.execute(_task(), left, right)
    assert len(sessions[0].submitted_timeouts) == 2
    assert sessions[0].submitted_timeouts[1] < sessions[0].submitted_timeouts[0]
    sessions[0].release_confirmed = False
    terminal = session.close()
    assert terminal["failure_stage"] == "kernel_timeout"
    assert terminal["primary_failure_stage"] == "kernel_timeout"
    assert terminal["release_failure_stage"] == "hardware_release_failed"
    assert terminal["native_kernel_executed"] is True


def test_rank_startup_is_inside_the_whole_graph_deadline(tmp_path: Path) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        dpu_count=1,
        sessions=sessions,
        timeout_s=0.01,
        startup_delay_s=0.02,
    )
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    assert sessions and sessions[0].closed


def test_late_second_rank_releases_every_opened_rank(tmp_path: Path) -> None:
    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        ranks=2,
        dpu_count=2,
        sessions=sessions,
        timeout_s=0.02,
        startup_delays_s=(0.0, 0.03),
    )
    with pytest.raises(V4ProtocolError, match="kernel_timeout"):
        engine.open_session(NumericMode.FLOAT32_REAL, _topology(2, rank_count=2))
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)


def test_request_cleanup_and_release_are_required(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    assert not list((tmp_path / "rank_00" / "requests").iterdir())
    terminal = session.close()
    assert terminal["target_observed"] == "physical_hardware"
    assert terminal["requested_dpu_count"] == 1
    failed_release = _engine(tmp_path / "failed-release", dpu_count=1)
    failed_session = failed_release.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    failed_session.ranks[0].session.release_confirmed = False
    assert failed_session.close()["target_observed"] == "not_verified"


def test_close_without_submitted_request_cannot_claim_hardware_execution(
    tmp_path: Path,
) -> None:
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_kernel_executed"] is False
    assert terminal["hardware_kernel_executed"] is False
    assert terminal["successful_request_count"] == 0
    assert terminal["allocated_dpu_count"] == 1
    assert terminal["requested_dpu_count"] == 1
    assert terminal["hardware_release_verified"] is True
    assert terminal["observed_rank_count"] == 1
    assert terminal["observed_tasklets_per_dpu"] == 1


def test_conflicting_native_ready_identity_fails_before_execution(
    tmp_path: Path,
) -> None:
    calls = 0

    def factory(command: Any, *, session_root: Path, profile: Any) -> _FakeSession:
        nonlocal calls
        del command, session_root
        calls += 1
        session = _FakeSession(profile=profile)
        if calls == 2:
            session.startup["kernel_identity"] = "wrong-native-kernel"
        return session

    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    engine = UpmemV4Executor(
        session_root=tmp_path / "session",
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=initialization_binary,
        rank_paths=("/dev/dpu_rank0", "/dev/dpu_rank1"),
        dpu_count=2,
        session_factory=factory,
    )
    with pytest.raises(RuntimeError, match="native identity kernel_identity"):
        engine.open_session(NumericMode.FLOAT32_REAL, _topology(2, rank_count=2))


def test_conflicting_native_response_identity_fails_closed(tmp_path: Path) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    session.ranks[0].session.response_identity_overrides["profile"] = (
        "wrong-native-profile"
    )
    with pytest.raises(RuntimeError, match="native identity profile"):
        session.execute(
            _task(),
            np.ones((3, 5), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
        )
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_identity_verified"] is True


def test_terminal_never_uses_python_identity_when_native_ready_conflicts(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, dpu_count=1)
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    session.execute(
        _task(), np.ones((3, 5), dtype=np.float32), np.ones((5, 4), dtype=np.float32)
    )
    session.ranks[0].session.startup["profile"] = "wrong-native-profile"
    terminal = session.close()
    assert terminal["target_observed"] == "not_verified"
    assert terminal["native_identity_verified"] is False
    assert "profile" not in terminal


def test_cleanup_rejects_requests_root(tmp_path: Path) -> None:
    requests_root = tmp_path / "requests"
    requests_root.mkdir()
    artifact = type(
        "Artifact",
        (),
        {
            "root": tmp_path,
            "request_dir": requests_root,
            "manifest_path": requests_root / "manifest.json",
            "sidecar_path": requests_root / "sidecar.bin",
        },
    )()
    with pytest.raises(RuntimeError, match="requests root"):
        UpmemV4Session._delete_request_dir(artifact)


def test_engine_supports_batched_permuted_output_labels(tmp_path: Path) -> None:
    task = ContractNode(
        node_id="batched",
        left=TensorView(tensor_id="left", labels=(5, 0, 1), shape=(2, 3, 4)),
        right=TensorView(tensor_id="right", labels=(5, 1, 2), shape=(2, 4, 5)),
        output=TensorSpec(
            id="out", labels=(2, 5, 0), shape=(5, 2, 3), structure="dense"
        ),
        contracted_labels=(1,),
        output_labels=(2, 5, 0),
    )
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    result = (
        _engine(tmp_path, dpu_count=2)
        .open_session(NumericMode.FLOAT32_REAL, _topology(2))
        .execute(task, left, right)
    )
    expected = np.einsum("bmk,bkn->nbm", left, right)
    np.testing.assert_allclose(_output(result), expected, rtol=1e-6, atol=1e-6)


def test_missing_node_plan_rejected(tmp_path: Path) -> None:
    host_binary, dpu_binary, initialization_binary = _binaries(tmp_path / "binaries")
    engine = UpmemV4Executor(
        session_root=tmp_path / "session",
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=initialization_binary,
        rank_paths=("/dev/dpu_rank0",),
        dpu_count=1,
        session_factory=lambda cmd, *, session_root, profile: _FakeSession(
            profile=profile
        ),
    )
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(1))
    task = _task()
    left = np.ones((3, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)

    with pytest.raises(TypeError):
        session.execute(task, left, right)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="requires a valid UpmemNodePlan"):
        session.execute(task, left, right, node_plan=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires a valid UpmemNodePlan"):
        session.execute(task, left, right, node_plan="not_a_plan")  # type: ignore[arg-type]

    plan = _compiled_node_plan(task, 1)
    wrong_id_plan = replace(plan, node_id="other_node")
    with pytest.raises(ValueError, match="does not match contract node"):
        session.execute(task, left, right, node_plan=wrong_id_plan)

    wrong_shape_plan = replace(plan, canonical_shape=(1, 999, 5, 4))
    with pytest.raises(ValueError, match="canonical B/M/K/N shape differs"):
        session.execute(task, left, right, node_plan=wrong_shape_plan)

    session.close()


def test_no_regenerated_placement_and_direct_plan_consumption(tmp_path: Path) -> None:
    task = _task(k=5, m=300, n=4)
    left = np.ones((300, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)

    plan = _compiled_node_plan(task, 4)
    assert len(plan.work_units) > 1

    first_unit = plan.work_units[0]
    modified_unit = replace(first_unit, logical_dpu=3)
    modified_units = (modified_unit, *plan.work_units[1:])
    custom_plan = replace(plan, work_units=modified_units)

    created_sessions: list[_FakeSession] = []
    engine_wrapper = _engine(tmp_path, dpu_count=4, sessions=created_sessions)
    real_session = engine_wrapper.engine.open_session(
        NumericMode.FLOAT32_REAL, _topology(4)
    )

    result = real_session.execute(task, left, right, node_plan=custom_plan)
    assert _metadata(result)["physical_plan_consumed"] is True
    np.testing.assert_allclose(_output(result), left @ right, rtol=1e-5)

    first_submission = created_sessions[0].submissions[0]
    submitted_dpu_ids = [
        u.local_dpu_id for u in first_submission.work_units if not u.flags
    ]
    assert 3 in submitted_dpu_ids
    real_session.close()


def test_direct_compiled_plan_controls_local_ids_on_two_ranks(tmp_path: Path) -> None:
    task = _task(k=5, m=600, n=4)
    left = np.ones((600, 5), dtype=np.float32)
    right = np.ones((5, 4), dtype=np.float32)
    compiled_plan = _compiled_node_plan(task, dpu_count=4, rank_count=2)
    assert {unit.logical_rank for unit in compiled_plan.work_units} == {0, 1}

    # Keep the compiled geometry and wave intact, but select a different valid
    # local slot on rank 1 so a regenerated default placement would be visible.
    explicit_units = tuple(
        replace(unit, logical_dpu=1) if unit.logical_rank == 1 else unit
        for unit in compiled_plan.work_units
    )
    explicit_plan = replace(compiled_plan, work_units=explicit_units)
    expected_by_rank = {
        rank: [
            unit.logical_dpu
            for unit in explicit_plan.work_units
            if unit.logical_rank == rank
        ]
        for rank in (0, 1)
    }

    sessions: list[_FakeSession] = []
    engine = _engine(
        tmp_path,
        ranks=2,
        dpu_count=4,
        sessions=sessions,
    ).engine
    session = engine.open_session(NumericMode.FLOAT32_REAL, _topology(4, rank_count=2))
    result = session.execute(task, left, right, node_plan=explicit_plan)

    assert _metadata(result)["physical_plan_consumed"] is True
    for rank_index, native in enumerate(sessions):
        actual_ids = [
            record.local_dpu_id
            for artifact in native.submissions
            for record in artifact.work_units
            if not record.flags
        ]
        assert actual_ids == expected_by_rank[rank_index]
    np.testing.assert_allclose(_output(result), left @ right, rtol=1e-6, atol=1e-6)
    session.close()


def test_request_work_units_and_payload_bytes() -> None:
    from quantum_bench.upmem.runtime import _build_work_unit
    from quantum_bench.upmem.tiling import (
        lower_binary_contraction,
    )

    task = _task(k=8, m=4, n=4)
    left_f32 = np.arange(32, dtype=np.float32).reshape(4, 8)
    right_f32 = np.arange(32, dtype=np.float32).reshape(8, 4)
    lowering = lower_binary_contraction(task, left_f32, right_f32)
    tile = lowering.tiles[0]

    unit_f32 = _build_work_unit(
        tile, 2, lowering.canonical.left, lowering.canonical.right, packed=False
    )
    expected_left_f32 = np.ascontiguousarray(
        lowering.canonical.left[0, : tile.m_size, : tile.k_size]
    )
    expected_right_f32 = np.ascontiguousarray(
        lowering.canonical.right[0, : tile.k_size, : tile.n_size]
    )
    assert unit_f32.a_payload == expected_left_f32.astype("<f4").tobytes()
    assert unit_f32.b_payload == expected_right_f32.astype("<f4").tobytes()
    assert unit_f32.local_dpu_id == 2
    assert unit_f32.m_offset == tile.m_start
    assert unit_f32.n_offset == tile.n_start
    assert unit_f32.k_offset == tile.k_start
    assert unit_f32.m_elements == tile.m_size
    assert unit_f32.n_elements == tile.n_size
    assert unit_f32.k_elements == tile.k_size

    left_i8 = np.arange(32, dtype=np.int8).reshape(1, 4, 8)
    right_i8 = np.arange(32, dtype=np.int8).reshape(1, 8, 4)
    unit_i8 = _build_work_unit(tile, 3, left_i8, right_i8, packed=True)
    assert unit_i8.a_payload == left_i8[0, : tile.m_size, : tile.k_size].tobytes()
    assert unit_i8.b_payload == right_i8[0, : tile.k_size, : tile.n_size].tobytes()
    assert unit_i8.local_dpu_id == 3


@pytest.mark.parametrize(
    "policy,numeric_mode",
    [
        ("split_complex_float32_v1", NumericMode.FLOAT32_REAL),
        ("split_complex_int8_shared_scale_v1", NumericMode.HOST_PACKED_INT8),
    ],
)
def test_execute_complex_matches_physical_plan_replay(
    tmp_path: Path, policy: str, numeric_mode: NumericMode
) -> None:
    node = _task(k=3, m=1, n=1)
    left = np.array([[1.0 + 2.0j, -0.5 + 0.25j, 0.75 - 1.5j]], dtype=np.complex128)
    right = np.array([[0.5 - 1.0j], [1.25 + 0.5j], [-0.25 + 0.75j]], dtype=np.complex128)
    dag, plan = _final_plan_for_node(node, policy=policy)
    replay = replay_upmem_plan_once(
        dag,
        plan,
        {"left": left, "right": right},
    )
    session = _engine(tmp_path, dpu_count=1).open_session(
        numeric_mode, _topology(1)
    )
    result, metadata = session.execute_complex(
        node,
        left,
        right,
        stage=plan.stages[0],
        numeric_policy=policy,
    )
    session.close()
    np.testing.assert_array_equal(result, replay.output)
    assert metadata["physical_stage_consumed"] is True
    assert metadata["lane_order"] == ("rr", "ii", "ri", "ir")
    assert metadata["lane_pass_count"] == 4
    assert metadata["timing_scope"] == (
        "sum_of_per_request_max_rank_response_counters_v1"
    )
    assert not {"h2d_s", "kernel_s", "d2h_s"} & set(metadata["timing"])
    if policy == "split_complex_float32_v1":
        expected = {
            f"{record['node_id']}/{record['stable_tile_id']}/{record['lane']}": record
            for record in replay.numeric_facts["raw_lane_records"]
        }
        actual = metadata["raw_lane_records"]
        assert set(actual) == set(expected)
        for key, record in expected.items():
            assert actual[key]["sha256"] == record["sha256"]
            assert actual[key]["dtype"] == "<f4"
            assert actual[key]["exact"] is False


def test_execute_complex_submits_four_lanes_in_order(tmp_path: Path) -> None:
    node = _task(k=2, m=1, n=1)
    left = np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex128)
    right = np.array([[2.0 - 1.0j], [-1.0 + 0.5j]], dtype=np.complex128)
    _, plan = _final_plan_for_node(
        node,
        policy="split_complex_float32_v1",
    )
    created: list[_FakeSession] = []
    session = _engine(tmp_path, dpu_count=1, sessions=created).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    session.execute_complex(
        node,
        left,
        right,
        stage=plan.stages[0],
        numeric_policy="split_complex_float32_v1",
    )
    terminal = session.close()
    assert len(created) == 1
    assert len(created[0].submissions) == 4
    assert created[0].submitted_operand_values == [
        (1.0, 2.0),
        (2.0, -1.0),
        (1.0, -1.0),
        (2.0, 2.0),
    ]
    assert terminal["active_rank_indices"] == (0,)
    assert terminal["active_dpu_ids"] == ((0, 0),)


def test_execute_complex_int8_facts_match_replay_hashes(tmp_path: Path) -> None:
    policy = "split_complex_int8_shared_scale_v1"
    node = _task(k=4, m=1, n=1)
    left = np.array([[2.0 + 4.0j, -1.0 + 0.5j, 0.25 - 3.0j, 1.5 + 0.75j]], dtype=np.complex128)
    right = np.array([[0.5 - 2.0j], [1.25 + 0.5j], [-0.75 + 1.0j], [2.0 - 0.25j]], dtype=np.complex128)
    dag, plan = _final_plan_for_node(node, policy=policy)
    replay = replay_upmem_plan_once(dag, plan, {"left": left, "right": right})
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.HOST_PACKED_INT8, _topology(1)
    )
    result, metadata = session.execute_complex(
        node,
        left,
        right,
        stage=plan.stages[0],
        numeric_policy=policy,
    )
    session.close()
    np.testing.assert_array_equal(result, replay.output)
    assert metadata["left_scale"] > 0.0
    assert metadata["right_scale"] > 0.0
    assert metadata["saturation_real"] == sum(
        int(record["saturation_real"])
        for record in metadata["operand_records"]
    )
    assert metadata["saturation_imag"] == sum(
        int(record["saturation_imag"])
        for record in metadata["operand_records"]
    )
    expected = {
        f"{record['node_id']}/{record['stable_tile_id']}/{record['lane']}": record
        for record in replay.numeric_facts["raw_lane_records"]
    }
    actual = metadata["raw_lane_records"]
    assert set(actual) == set(expected)
    for key, record in expected.items():
        assert actual[key]["sha256"] == record["sha256"]
        assert actual[key]["dtype"] == "<i4"
        assert actual[key]["exact"] is True


def test_execute_complex_int8_multi_k_chunk_matches_replay(tmp_path: Path) -> None:
    policy = "split_complex_int8_shared_scale_v1"
    k = 257
    node = _task(k=k, m=1, n=1)
    values = np.arange(k, dtype=np.float64)
    left = (
        ((values % 11) - 5.0) + 1j * ((values % 7) - 3.0)
    ).reshape(1, k)
    right = (
        ((values % 13) - 6.0) + 1j * ((values % 5) - 2.0)
    ).reshape(k, 1)
    dag, plan = _final_plan_for_node(node, policy=policy)
    stage = plan.stages[0]
    assert len(stage.work_units) == 2
    assert sorted(unit.k_size for unit in stage.work_units) == [1, 256]
    replay = replay_upmem_plan_once(dag, plan, {"left": left, "right": right})
    session = _engine(tmp_path, dpu_count=1).open_session(
        NumericMode.HOST_PACKED_INT8, _topology(1)
    )
    result, metadata = session.execute_complex(
        node,
        left,
        right,
        stage=stage,
        numeric_policy=policy,
    )
    session.close()
    np.testing.assert_array_equal(result, replay.output)
    expected = {
        f"{record['node_id']}/{record['stable_tile_id']}/{record['lane']}": record
        for record in replay.numeric_facts["raw_lane_records"]
    }
    actual = metadata["raw_lane_records"]
    assert set(actual) == set(expected)
    for key, record in expected.items():
        assert actual[key]["sha256"] == record["sha256"]
        assert actual[key]["dtype"] == "<i4"


def test_execute_complex_submit_failure_has_no_fallback(tmp_path: Path) -> None:
    node = _task(k=2, m=1, n=1)
    left = np.array([[1.0 + 1.0j, 2.0 - 2.0j]], dtype=np.complex128)
    right = np.array([[1.0 - 1.0j], [2.0 + 2.0j]], dtype=np.complex128)
    _, plan = _final_plan_for_node(node, policy="split_complex_float32_v1")
    session = _engine(tmp_path, dpu_count=1, fail_submit=True).open_session(
        NumericMode.FLOAT32_REAL, _topology(1)
    )
    with pytest.raises(RuntimeError, match="submission failed"):
        session.execute_complex(
            node,
            left,
            right,
            stage=plan.stages[0],
            numeric_policy="split_complex_float32_v1",
        )
    assert session._failed is True
    assert not hasattr(session, "fallback_result")
    session.close()


def test_decoding_and_host_reduction(tmp_path: Path) -> None:
    from quantum_bench.upmem.runtime import (
        _assemble_output,
        _read_output,
    )
    from quantum_bench.upmem.tiling import (
        M5TileLimits,
        lower_binary_contraction,
    )

    task = _task(k=16, m=4, n=4)
    left = np.arange(64, dtype=np.float32).reshape(4, 16)
    right = np.arange(64, dtype=np.float32).reshape(16, 4)
    limits = M5TileLimits(max_tile_dim=2, max_elements=4, max_packed_k=8)
    lowering = lower_binary_contraction(task, left, right, limits=limits)
    tile = lowering.tiles[0]

    f32_data = np.arange(tile.output_element_count, dtype="<f4")
    f32_file = tmp_path / "f32_out.bin"
    f32_file.write_bytes(f32_data.tobytes())
    decoded_f32 = _read_output(f32_file, tile, packed=False)
    np.testing.assert_array_equal(
        decoded_f32, f32_data.reshape(tile.m_size, tile.n_size)
    )
    assert decoded_f32.dtype == np.float64

    i32_data = np.arange(tile.output_element_count, dtype="<i4") * 10
    i32_file = tmp_path / "i32_out.bin"
    i32_file.write_bytes(i32_data.tobytes())
    decoded_i8 = _read_output(i32_file, tile, packed=True)
    np.testing.assert_array_equal(
        decoded_i8, i32_data.reshape(tile.m_size, tile.n_size)
    )
    assert decoded_i8.dtype == np.int64

    trunc_file = tmp_path / "trunc.bin"
    trunc_file.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="truncated"):
        _read_output(trunc_file, tile, packed=False)

    partials: dict[str, np.ndarray] = {}
    for t in lowering.tiles:
        left_sub, right_sub = lowering.extract_tile_operands(t)
        partials[t.id] = left_sub @ right_sub

    expected_f32 = left @ right
    res_f32 = _assemble_output(lowering, partials, packed=False, scale=1.0)
    np.testing.assert_allclose(res_f32, expected_f32, rtol=1e-5)

    scale = 0.125
    res_packed = _assemble_output(lowering, partials, packed=True, scale=scale)
    np.testing.assert_allclose(res_packed, expected_f32 * scale, rtol=1e-5)
