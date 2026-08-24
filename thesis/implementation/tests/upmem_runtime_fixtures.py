"""Private ABI-v4 test fixtures for the active UPMEM runtime tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.plan import UpmemPlan, UpmemTopology, plan_upmem
from quantum_bench.upmem.protocol import (
    EXECUTION_TARGET_SIMULATOR,
    V4ProtocolError,
    native_execution_identity,
)
from quantum_bench.upmem.runtime import UpmemV4Executor, UpmemV4Session


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
        for record in artifact.work_units:
            if record.flags:
                continue
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
            "transfer": {"h2d_bytes": 10, "d2h_bytes": 5, "total_bytes": 15},
            "timing": {
                "h2d_time_s": 0.01,
                "launch_time_s": 0.02,
                "d2h_time_s": 0.01,
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

    def open_session(self, numeric_policy: str, topology: UpmemTopology) -> Any:
        return _PlannedSession(self.engine.open_session(numeric_policy, topology))


def _engine(
    root: Path,
    *,
    dpu_count: int = 1,
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
                else ("/dev/dpu_rank0",)
            ),
            dpu_count=dpu_count,
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
        topology=UpmemTopology(
            dpu_count=dpu_count,
            tasklets_per_dpu=1,
            rank_count=rank_count,
        ),
    )
