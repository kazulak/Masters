"""Bounded physical v4 execution engine for contraction DAG nodes.

The engine deliberately owns only one binary contraction at a time. The caller
owns DAG dependencies and the host tensor store; this module lowers one
``ContractNode`` to bounded v4 output/K tiles, submits those tiles to persistent
physical rank sessions, and reconstructs the output. It does not claim
DPU-resident graph intermediates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from quantum_bench.execution.contracts import (
    BackendFacts,
    ExecutionPlan,
    ExecutionResult,
    NumericMode,
    RunContext,
    Target,
    TimingBreakdown,
    UpmemNodePlan,
    UpmemPlan,
    UpmemRuntimeResources,
    UpmemTopology,
    canonical_serialize,
    validate_execution_plan,
    validate_execution_result,
    validate_transfer_bytes,
    validate_upmem_runtime_resources,
)
from quantum_bench.execution.numeric import (
    decode_contraction_output,
    encode_tensor,
)
from quantum_bench.tn.graph import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorView,
    contraction_dag_hash,
    validate_contraction_dag,
)
from quantum_bench.tn.network import validate_dag_inputs
from quantum_bench.upmem.plan import (
    validate_active_upmem_plan,
    validate_upmem_plan_for_dag,
)
from quantum_bench.upmem.native_session import V4Session
from quantum_bench.upmem.protocol import (
    MAX_INT32_SAFE_K,
    NATIVE_EXECUTION_IDENTITY,
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4ProtocolError,
    V4WorkUnit,
    build_v4_request,
)
from quantum_bench.upmem.tiling import (
    M5Tile,
    M5TileLimits,
    M5TileLowering,
    lower_binary_contraction,
)

_INT64_MAX = (1 << 63) - 1

_EXPECTED_NATIVE_IDENTITY = dict(NATIVE_EXECUTION_IDENTITY)

_COORDINATOR_PROVENANCE = {
    "transfer_accounting_scope": "application_visible_sdk_recorded",
    "graph_intermediate_placement": "host_managed",
    "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
    "request_level_speedup_applicable": False,
    "energy_claim_applicable": False,
}

# The active runtime has one fixed lowering, placement, kernel, and reduction
# mechanism.  Keep the historical metadata fields for evidence compatibility,
# but do not retain a strategy registry for a single implementation.
_ACTIVE_MECHANISM_IDS = {
    "decomposition": "m5_v4_tile_decomposition",
    "placement": "m5_rank_wave_placement",
    "kernel": "upmem_sdk_hardware_v4_tile_kernel",
    "reduction": "m5_tile_host_reduction",
}
_ACTIVE_ENGINE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ACTIVE_STRATEGY_IDENTITY = {
    "schema_version": "strategy_configuration_v2",
    "strategies": [
        {
            "role": "decomposition",
            "implementation_id": _ACTIVE_MECHANISM_IDS["decomposition"],
            "version": "1",
            "provider": "quantum_bench_host",
            "transport": "host_control",
            "config": {"limits_source": "numeric_policy"},
            "implementation_type": "fixed_direct_mechanism",
            "module_source_sha256": _ACTIVE_ENGINE_SOURCE_SHA256,
        },
        {
            "role": "kernel",
            "implementation_id": _ACTIVE_MECHANISM_IDS["kernel"],
            "version": "1",
            "provider": "raw_upmem_sdk_v4",
            "transport": "application_visible_sdk_transfer",
            "config": {"abi": "execution_plan_v4", "kernel": "dpu_gemm_tile_v4"},
            "implementation_type": "fixed_direct_mechanism",
            "module_source_sha256": _ACTIVE_ENGINE_SOURCE_SHA256,
        },
        {
            "role": "placement",
            "implementation_id": _ACTIVE_MECHANISM_IDS["placement"],
            "version": "1",
            "provider": "quantum_bench_host",
            "transport": "host_control",
            "config": {
                "local_dpu_order": "compiled_plan",
                "wave_partition": "compiled_plan",
            },
            "implementation_type": "fixed_direct_mechanism",
            "module_source_sha256": _ACTIVE_ENGINE_SOURCE_SHA256,
        },
        {
            "role": "reduction",
            "implementation_id": _ACTIVE_MECHANISM_IDS["reduction"],
            "version": "1",
            "provider": "quantum_bench_host",
            "transport": "host_memory",
            "config": {
                "accumulator": "int64_packed_or_float64_float32",
                "location": "host",
            },
            "implementation_type": "fixed_direct_mechanism",
            "module_source_sha256": _ACTIVE_ENGINE_SOURCE_SHA256,
        },
    ],
}
_ACTIVE_STRATEGY_CONFIG_HASH = hashlib.sha256(
    json.dumps(
        _ACTIVE_STRATEGY_IDENTITY,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
).hexdigest()


class _V4SessionLike(Protocol):
    startup: Mapping[str, Any]

    def submit(
        self, artifact: Any, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]: ...

    def close(self, *, timeout_s: float | None = None) -> Any: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_structure_hash(node: ContractNode) -> str:
    payload = repr(
        (
            node.node_id,
            (node.left.tensor_id, node.right.tensor_id),
            node.output.id,
            (node.left.shape, node.right.shape),
            node.output.shape,
            node.left.labels,
            node.right.labels,
            node.contracted_labels,
            node.output_labels,
        )
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_binary_provenance(
    path: Path, *, label: str, executable: bool
) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"M5 binary is missing or not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"M5 host binary is not executable: {resolved}")
    return {
        f"{label}_path": str(resolved),
        f"{label}_sha256": _file_sha256(resolved),
    }


def _request_contract_hash(
    task_structure_sha256: str,
    *,
    numeric_transport: str,
    left_scale: float,
    right_scale: float,
    left_payload: np.ndarray,
    right_payload: np.ndarray,
    strategy_config_hash: str,
) -> str:
    """Bind the native request to structure, numeric mode, scales, and data.

    The v4 ABI calls this digest ``task_contract_sha256``. The engine keeps
    the structural digest separately and uses this request digest for every
    request so host dequantization metadata cannot be changed independently
    of the staged operands.
    """

    mode = numeric_transport.encode("utf-8")
    payload = b"m5_request_contract_v2\0" + bytes.fromhex(task_structure_sha256)
    payload += bytes.fromhex(strategy_config_hash)
    payload += struct.pack("<I", len(mode)) + mode
    payload += struct.pack("<dd", float(left_scale), float(right_scale))
    payload += bytes.fromhex(_sha256_bytes(np.asarray(left_payload).tobytes(order="C")))
    payload += bytes.fromhex(
        _sha256_bytes(np.asarray(right_payload).tobytes(order="C"))
    )
    return _sha256_bytes(payload)


def _build_work_unit(
    tile: M5Tile,
    local_dpu_id: int,
    left: np.ndarray,
    right: np.ndarray,
    *,
    packed: bool,
) -> V4WorkUnit:
    """Serialize one compiled tile without changing its compiled placement."""

    left_array = left if left.ndim == 3 else left.reshape((1, *left.shape))
    right_array = right if right.ndim == 3 else right.reshape((1, *right.shape))
    left_tile = np.ascontiguousarray(
        left_array[
            tile.batch_index,
            tile.m_start : tile.m_start + tile.m_size,
            tile.k_start : tile.k_start + tile.k_size,
        ]
    )
    right_tile = np.ascontiguousarray(
        right_array[
            tile.batch_index,
            tile.k_start : tile.k_start + tile.k_size,
            tile.n_start : tile.n_start + tile.n_size,
        ]
    )
    dtype = np.int8 if packed else np.dtype("<f4")
    return V4WorkUnit(
        local_dpu_id=local_dpu_id,
        tile_id=int.from_bytes(
            hashlib.sha256(tile.id.encode("utf-8")).digest()[:8], "little"
        ),
        batch_index=tile.batch_index,
        m_offset=tile.m_start,
        n_offset=tile.n_start,
        k_offset=tile.k_start,
        m_elements=tile.m_size,
        n_elements=tile.n_size,
        k_elements=tile.k_size,
        a_payload=np.asarray(left_tile, dtype=dtype).tobytes(order="C"),
        b_payload=np.asarray(right_tile, dtype=dtype).tobytes(order="C"),
    )


def _read_output(path: Path, tile: M5Tile, *, packed: bool) -> np.ndarray:
    dtype = np.dtype("<i4") if packed else np.dtype("<f4")
    expected_bytes = tile.output_element_count * dtype.itemsize
    raw = path.read_bytes()
    if len(raw) < expected_bytes:
        raise RuntimeError(f"v4 output is truncated: {path}")
    values = np.frombuffer(raw[:expected_bytes], dtype=dtype)
    return np.asarray(
        values.reshape(tile.m_size, tile.n_size),
        dtype=np.int64 if packed else np.float64,
    )


def _assemble_output(
    lowering: M5TileLowering,
    partials: Mapping[str, np.ndarray],
    *,
    packed: bool,
    scale: float,
) -> np.ndarray:
    accumulator = _assemble_accumulator(lowering, partials, packed=packed)
    mode = (
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1 if packed else NumericMode.FLOAT32_REAL
    )
    return decode_contraction_output(accumulator, mode, scale)


def _assemble_accumulator(
    lowering: M5TileLowering,
    partials: Mapping[str, np.ndarray],
    *,
    packed: bool,
) -> np.ndarray:
    """Assemble tile outputs without applying numeric output decoding."""

    return lowering.assemble(partials, dtype=np.int64 if packed else np.float64)


@dataclass(frozen=True)
class _RankSession:
    index: int
    root: Path
    session: _V4SessionLike
    local_dpus: int


def _native_identity(event: Mapping[str, Any], *, source: str) -> dict[str, str]:
    """Read the identity emitted by native v4 code, never Python provenance."""

    observed: dict[str, str] = {}
    for field, expected in _EXPECTED_NATIVE_IDENTITY.items():
        value = event.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{source} is missing native identity {field}")
        if value != expected:
            raise RuntimeError(
                f"{source} native identity {field}={value!r} does not match "
                f"the compiled v4 contract"
            )
        observed[field] = value
    return observed


def _agreed_native_identity(
    observations: tuple[tuple[str, Mapping[str, Any]], ...],
) -> dict[str, str]:
    """Return one identity only when every rank/event reported the same contract."""

    if not observations:
        raise RuntimeError("no native identity observations were recorded")
    identities = [
        _native_identity(event, source=source) for source, event in observations
    ]
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise RuntimeError("native v4 identity observations disagree across ranks")
    return first


def _native_identity_metadata(identity: Mapping[str, str]) -> dict[str, str]:
    """Expose compatibility aliases derived from observed native fields."""

    return {
        **identity,
        "physical_profile": identity["profile"],
        "hardware_profile": identity["profile"],
        "hardware_profile_version": identity["profile"],
        "abi_version": identity["abi"],
        "session": identity["session_protocol"],
        "dispatch": identity["dispatch_mode"],
        "kernel": identity["kernel_identity"],
        "kernel_strategy": identity["kernel_identity"],
    }


def _close_rank_before_deadline(rank: _RankSession, deadline: float) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise V4ProtocolError(
            "kernel_timeout", "whole-circuit release deadline expired"
        )
    return rank.session.close(timeout_s=remaining)


class UpmemV4Executor:
    """Physical v4 tile engine with explicit ranks and no fallback route."""

    name = "upmem_execution_plan_v4_whole_circuit"

    def __init__(
        self,
        *,
        session_root: Path,
        host_binary: Path,
        dpu_binary: Path,
        initialization_binary: Path,
        rank_paths: tuple[str, ...],
        dpu_count: int,
        tasklets_per_dpu: int = 1,
        timeout_s: float = 60.0,
        session_factory: Callable[..., _V4SessionLike] = V4Session.start,
    ) -> None:
        self.session_root = Path(session_root)
        self.host_binary = Path(host_binary)
        self.dpu_binary = Path(dpu_binary)
        self.initialization_binary = Path(initialization_binary)
        self.rank_paths = tuple(rank_paths)
        self.dpu_count = int(dpu_count)
        self.tasklets_per_dpu = int(tasklets_per_dpu)
        self.timeout_s = float(timeout_s)
        self.session_factory = session_factory
        self._binary_provenance = {
            **_validated_binary_provenance(
                self.host_binary, label="host_binary", executable=True
            ),
            **_validated_binary_provenance(
                self.dpu_binary, label="dpu_binary", executable=False
            ),
            **_validated_binary_provenance(
                self.initialization_binary,
                label="initialization_binary",
                executable=False,
            ),
        }
        self._source_root = str(Path(__file__).resolve().parents[3])
        self._provenance = {
            "source_root": self._source_root,
            "session_root": str(self.session_root.resolve()),
            **self._binary_provenance,
        }
        if not self.rank_paths:
            raise ValueError("M5 whole-circuit engine requires explicit rank_paths")
        if self.dpu_count < 1 or self.dpu_count % len(self.rank_paths):
            raise ValueError("dpu_count must be positive and divisible by rank count")
        if self.tasklets_per_dpu < 1 or self.timeout_s <= 0:
            raise ValueError("tasklets_per_dpu and timeout_s must be positive")
        if self.dpu_count // len(self.rank_paths) > 64:
            raise ValueError("v4 supports at most 64 local DPUs per rank")

    @property
    def strategy_identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(_ACTIVE_STRATEGY_IDENTITY))

    @property
    def strategy_config_hash(self) -> str:
        return _ACTIVE_STRATEGY_CONFIG_HASH

    def open_session(
        self,
        numeric_mode: NumericMode,
        topology: UpmemTopology,
    ) -> UpmemV4Session:
        if topology.dpu_count != self.dpu_count:
            raise ValueError("topology device count must match engine dpu_count")
        if topology.tasklets_per_dpu != self.tasklets_per_dpu:
            raise ValueError(
                "topology tasklet count must match engine tasklets_per_dpu"
            )
        if topology.rank_count != len(self.rank_paths):
            raise ValueError("topology rank count must match engine rank_paths")
        if numeric_mode not in {
            NumericMode.FLOAT32_REAL,
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        }:
            raise ValueError(f"unsupported M5 numeric mode: {numeric_mode}")

        deadline = time.monotonic() + self.timeout_s
        self.session_root.mkdir(parents=True, exist_ok=True)
        local_dpus = self.dpu_count // len(self.rank_paths)
        ranks: list[_RankSession] = []
        try:
            for index, rank_path in enumerate(self.rank_paths):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise V4ProtocolError(
                        "kernel_timeout",
                        "whole-circuit deadline expired while opening rank sessions",
                    )
                root = self.session_root / f"rank_{index:02d}"
                root.mkdir(parents=True, exist_ok=True)
                profile = V4Profile(
                    dpu_count=local_dpus,
                    tasklets_per_dpu=self.tasklets_per_dpu,
                    numeric_mode=(
                        NUMERIC_HOST_PACKED_INT8
                        if numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                        else NUMERIC_FLOAT32
                    ),
                    rank_path=rank_path,
                    timeout_s=remaining,
                )
                command = (
                    str(self.host_binary),
                    "--session-root",
                    str(root.resolve()),
                    "--rank-path",
                    rank_path,
                    "--dpus",
                    str(local_dpus),
                    "--tasklets",
                    str(self.tasklets_per_dpu),
                    "--initialization-binary",
                    str(self.initialization_binary.resolve()),
                    "--dpu-binary",
                    str(self.dpu_binary.resolve()),
                    "--timeout-s",
                    str(max(1, int(remaining))),
                )
                session = self.session_factory(
                    command, session_root=root, profile=profile
                )
                ranks.append(_RankSession(index, root, session, local_dpus))
                _native_identity(session.startup, source=f"READY rank {index}")
                if time.monotonic() >= deadline:
                    raise V4ProtocolError(
                        "kernel_timeout",
                        "whole-circuit deadline expired while opening rank sessions",
                    )
        except BaseException:
            if ranks:
                remaining = deadline - time.monotonic()
                cleanup_deadline = (
                    deadline
                    if remaining > 0
                    else time.monotonic() + min(1.0, self.timeout_s)
                )
                with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
                    futures = [
                        pool.submit(_close_rank_before_deadline, rank, cleanup_deadline)
                        for rank in ranks
                    ]
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
            raise
        return UpmemV4Session(
            numeric_mode=numeric_mode,
            ranks=tuple(ranks),
            engine=self,
            deadline=deadline,
        )


class UpmemV4Session:
    """Persistent rank sessions used by one whole-graph measurement."""

    def __init__(
        self,
        *,
        numeric_mode: NumericMode,
        ranks: tuple[_RankSession, ...],
        engine: UpmemV4Executor,
        deadline: float,
    ) -> None:
        self.numeric_mode = numeric_mode
        self.ranks = ranks
        self.engine = engine
        self._deadline = float(deadline)
        self._closed = False
        self._failed = False
        self._failure_stage: str | None = None
        self._sequence = 0
        self._successful_request_count = 0
        self._active_rank_indices: set[int] = set()
        self._active_dpu_ids: set[tuple[int, int]] = set()
        self._startup_native_identity = _agreed_native_identity(
            tuple(
                (f"READY rank {rank.index}", rank.session.startup)
                for rank in self.ranks
            )
        )
        self._response_native_identity_events: list[tuple[str, Mapping[str, Any]]] = []
        self._test_double_execution = any(
            rank.session.startup.get("test_double_execution") is True
            for rank in self.ranks
        )
        self._terminal_metadata: dict[str, Any] = {}

    @property
    def strategy_identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(_ACTIVE_STRATEGY_IDENTITY))

    @property
    def strategy_config_hash(self) -> str:
        return _ACTIVE_STRATEGY_CONFIG_HASH

    def execute(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        node_plan: UpmemNodePlan,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        if self._closed:
            raise RuntimeError("UPMEM v4 session is closed")
        if node_plan is None or not isinstance(node_plan, UpmemNodePlan):
            raise ValueError("UpmemV4Session.execute requires a valid UpmemNodePlan")
        self._remaining_timeout()
        started = time.perf_counter()
        packed = self.numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        limits = M5TileLimits.host_packed_int8() if packed else M5TileLimits.float32()
        lowering = lower_binary_contraction(node, left, right, limits=limits)
        canonical_left = lowering.canonical.left
        canonical_right = lowering.canonical.right
        quantization_metadata: dict[str, Any] = {}
        left_scale = right_scale = 1.0
        numeric_mode = self.numeric_mode
        quantization_started = time.perf_counter()
        left_payload, left_scale, left_saturation = encode_tensor(
            canonical_left, numeric_mode
        )
        right_payload, right_scale, right_saturation = encode_tensor(
            canonical_right, numeric_mode
        )
        preparation_time_s = time.perf_counter() - quantization_started
        canonical_left = left_payload
        canonical_right = right_payload
        if packed:
            host_quantization_time_s = preparation_time_s
            preparation_time_s = 0.0
            scale = float(left_scale * right_scale)
            if (
                max((chunk.k_size for chunk in lowering.k_chunks), default=0)
                > MAX_INT32_SAFE_K
            ):
                raise ValueError(
                    "packed int8 K chunk exceeds int32 accumulation safety bound"
                )
            if lowering.canonical.k * 128 * 128 > _INT64_MAX:
                raise ValueError(
                    "packed int8 aggregate exceeds int64 accumulation safety bound"
                )
            quantization_metadata = {
                "packed_int8_transport": True,
                "left_scale": left_scale,
                "right_scale": right_scale,
                "saturation_count": int(left_saturation + right_saturation),
                "host_quantization_time_s": float(host_quantization_time_s),
            }
        else:
            scale = 1.0
            host_quantization_time_s = 0.0
            quantization_metadata = {
                "packed_int8_transport": False,
                "preparation_time_s": float(preparation_time_s),
            }

        partials: dict[str, np.ndarray] = {}
        bytes_h2d = bytes_d2h = 0
        timing = {"h2d_time_s": 0.0, "kernel_time_s": 0.0, "d2h_time_s": 0.0}
        request_hashes: list[str] = []
        parallel_rank_waves = 0
        bulk_verified = True
        total_dpus = sum(rank.local_dpus for rank in self.ranks)
        waves, planned_requests = self._requests_from_plan(node, lowering, node_plan)
        self._validate_waves(lowering.tiles, waves, total_dpus)
        task_structure_sha256 = _task_structure_hash(node)
        numeric_transport = "host_packed_int8_mram" if packed else "float32_mram"
        request_contract = _request_contract_hash(
            task_structure_sha256,
            numeric_transport=numeric_transport,
            left_scale=left_scale,
            right_scale=right_scale,
            left_payload=canonical_left,
            right_payload=canonical_right,
            strategy_config_hash=self.strategy_config_hash,
        )
        try:
            for wave_index, wave in enumerate(waves):
                self._remaining_timeout()
                outcomes, wave_metrics, wave_parallel, wave_bulk_verified = (
                    self._submit_wave(
                        lowering=lowering,
                        canonical_left=canonical_left,
                        canonical_right=canonical_right,
                        packed=packed,
                        request_contract=request_contract,
                        wave=wave,
                        requests=planned_requests[wave_index],
                    )
                )
                parallel_rank_waves += int(wave_parallel)
                bulk_verified = bulk_verified and wave_bulk_verified
                bytes_h2d += int(wave_metrics["h2d_bytes"])
                bytes_d2h += int(wave_metrics["d2h_bytes"])
                for key in timing:
                    timing[key] += float(wave_metrics[key])
                request_hashes.extend(wave_metrics["request_manifest_hashes"])
                self._successful_request_count += int(
                    wave_metrics["successful_request_count"]
                )
                self._active_rank_indices.update(wave_metrics["active_rank_indices"])
                self._active_dpu_ids.update(wave_metrics["active_dpu_ids"])
                for tile, value in outcomes:
                    partials[tile.id] = value
        except BaseException as exc:
            self._failed = True
            self._failure_stage = str(
                getattr(exc, "failure_stage", "hardware_task_execution_failed")
            )
            raise
        self._remaining_timeout()
        assembly_started = time.perf_counter()
        accumulator = _assemble_accumulator(lowering, partials, packed=packed)
        host_tile_assembly_time_s = time.perf_counter() - assembly_started
        decode_started = time.perf_counter()
        output = decode_contraction_output(accumulator, numeric_mode, scale)
        host_dequantization_time_s = (
            time.perf_counter() - decode_started if packed else 0.0
        )
        elapsed = time.perf_counter() - started
        return np.asarray(output), {
            "engine": self.engine.name,
            "execution_time_s": elapsed,
            "timing": {
                **timing,
                "host_quantization_time_s": host_quantization_time_s,
                "preparation_time_s": preparation_time_s,
                "host_dequantization_time_s": host_dequantization_time_s,
                "host_tile_assembly_time_s": host_tile_assembly_time_s,
                "total_route_time_s": elapsed,
            },
            **_native_identity_metadata(self._startup_native_identity),
            **_COORDINATOR_PROVENANCE,
            **self.engine._provenance,
            "strategy_identity": self.strategy_identity,
            "strategy_config_hash": self.strategy_config_hash,
            "decomposition_strategy": _ACTIVE_MECHANISM_IDS["decomposition"],
            "placement_strategy": _ACTIVE_MECHANISM_IDS["placement"],
            "kernel_provider": _ACTIVE_MECHANISM_IDS["kernel"],
            "reduction_provider": _ACTIVE_MECHANISM_IDS["reduction"],
            "reduction_strategy": _ACTIVE_MECHANISM_IDS["reduction"],
            "numeric_transport": numeric_transport,
            "packed_int8_transfer": packed,
            "host_quantization_time_s": host_quantization_time_s,
            "preparation_time_s": preparation_time_s,
            "host_dequantization_time_s": host_dequantization_time_s,
            "host_tile_assembly_time_s": host_tile_assembly_time_s,
            "application_visible_h2d_bytes": bytes_h2d,
            "application_visible_d2h_bytes": bytes_d2h,
            "application_visible_transfer_bytes": bytes_h2d + bytes_d2h,
            "response_transfer_bytes": bytes_h2d + bytes_d2h,
            "tile_count": len(lowering.tiles),
            "output_tile_count": len(lowering.output_tiles),
            "k_chunk_count": len(lowering.k_chunks),
            "wave_count": len(waves),
            "request_manifest_hashes": tuple(request_hashes),
            "task_structure_sha256": task_structure_sha256,
            "request_contract_version": "m5_request_contract_v2",
            "request_contract_sha256": request_contract,
            # ABI compatibility name: v4 carries the request contract.
            "task_contract_sha256": request_contract,
            "bulk_set_launch_verified": bulk_verified,
            "concurrent_rank_submission": parallel_rank_waves > 0,
            "concurrent_rank_wave_count": parallel_rank_waves,
            "whole_graph_deadline_enforced": True,
            "whole_graph_timeout_s": self.engine.timeout_s,
            "target_observed": (
                "not_verified" if self._test_double_execution else "physical_hardware"
            ),
            "test_double_execution": self._test_double_execution,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "physical_plan_consumed": True,
            **quantization_metadata,
        }

    def _requests_from_plan(
        self,
        node: ContractNode,
        lowering: Any,
        node_plan: UpmemNodePlan,
    ) -> tuple[
        tuple[tuple[M5Tile, ...], ...],
        tuple[list[tuple[Any, list[tuple[M5Tile, int]]]], ...],
    ]:
        """Turn the compiled work units into native rank requests.

        The live lowering is used only to obtain payload arrays.  It cannot
        change the compiled geometry or placement: every work unit is checked
        against its tile before a request is constructed.
        """

        if node_plan.node_id != node.node_id or node_plan.node_kind != "contract":
            raise ValueError("compiled UPMEM node plan does not match contract node")
        canonical = lowering.canonical
        if node_plan.canonical_shape != (
            canonical.b,
            canonical.m,
            canonical.k,
            canonical.n,
        ):
            raise ValueError(
                "compiled UPM canonical B/M/K/N shape differs from lowering"
            )
        tiles = {tile.id: tile for tile in lowering.tiles}
        units = node_plan.work_units
        if len(tiles) != len(lowering.tiles) or len(units) != len(tiles):
            raise ValueError("compiled UPM work-unit count differs from lowering")
        if {unit.stable_tile_id for unit in units} != set(tiles):
            raise ValueError("compiled UPM tile IDs differ from lowering")

        for unit in units:
            tile = tiles.get(unit.stable_tile_id)
            if tile is None or unit.node_id != node.node_id:
                raise ValueError("compiled UPM work unit references an unknown tile")
            expected = (
                tile.batch_index,
                1,
                tile.m_start,
                tile.m_size,
                tile.n_start,
                tile.n_size,
                tile.k_start,
                tile.k_size,
                tile.left_bytes + tile.right_bytes,
                tile.output_bytes,
                tile.aligned_mram_bytes,
                tile.m_size * tile.n_size * tile.k_size,
            )
            actual = (
                unit.batch_start,
                unit.batch_size,
                unit.m_start,
                unit.m_size,
                unit.n_start,
                unit.n_size,
                unit.k_start,
                unit.k_size,
                unit.estimated_input_bytes,
                unit.estimated_output_bytes,
                unit.aligned_mram_bytes,
                unit.estimated_arithmetic_work,
            )
            if actual != expected:
                raise ValueError(
                    f"compiled UPM tile {unit.stable_tile_id} extents differ from lowering"
                )

        wave_numbers = sorted({unit.wave for unit in units})
        if wave_numbers != list(range(len(wave_numbers))):
            raise ValueError("compiled UPM work-unit waves are not contiguous")
        waves: list[tuple[M5Tile, ...]] = []
        requests_by_wave: list[list[tuple[Any, list[tuple[M5Tile, int]]]]] = []
        for wave_number in wave_numbers:
            wave_units = [unit for unit in units if unit.wave == wave_number]
            if len(wave_units) > sum(rank.local_dpus for rank in self.ranks):
                raise ValueError("compiled UPM wave exceeds available DPUs")
            seen_slots: set[tuple[int, int]] = set()
            grouped: dict[int, list[tuple[M5Tile, int]]] = {}
            wave_tiles: list[M5Tile] = []
            for unit in wave_units:
                slot = (unit.logical_rank, unit.logical_dpu)
                if slot in seen_slots:
                    raise ValueError("compiled UPM wave reuses a rank/local-DPU slot")
                if not 0 <= unit.logical_rank < len(self.ranks):
                    raise ValueError("compiled UPM logical rank is out of range")
                rank = self.ranks[unit.logical_rank]
                if not 0 <= unit.logical_dpu < rank.local_dpus:
                    raise ValueError("compiled UPM logical DPU is out of range")
                seen_slots.add(slot)
                tile = tiles[unit.stable_tile_id]
                wave_tiles.append(tile)
                grouped.setdefault(unit.logical_rank, []).append(
                    (tile, unit.logical_dpu)
                )
            waves.append(tuple(wave_tiles))
            requests_by_wave.append(
                [
                    (self.ranks[rank_index], grouped[rank_index])
                    for rank_index in sorted(grouped)
                ]
            )
        return tuple(waves), tuple(requests_by_wave)

    def _submit_wave(
        self,
        *,
        lowering: Any,
        canonical_left: np.ndarray,
        canonical_right: np.ndarray,
        packed: bool,
        request_contract: str,
        wave: tuple[M5Tile, ...],
        requests: list[tuple[Any, list[tuple[M5Tile, int]]]],
    ) -> tuple[list[tuple[M5Tile, np.ndarray]], dict[str, Any], bool, bool]:
        self._validate_rank_assignments(wave, requests)
        prepared: list[tuple[_RankSession, list[tuple[M5Tile, int]], Any]] = []
        for rank, assignments in requests:
            units = [
                _build_work_unit(
                    tile,
                    local_id,
                    canonical_left,
                    canonical_right,
                    packed=packed,
                )
                for tile, local_id in assignments
            ]
            artifact = build_v4_request(
                rank.root,
                profile=rank.session.profile,
                canonical_batch_count=lowering.canonical.b,
                canonical_m=lowering.canonical.m,
                canonical_n=lowering.canonical.n,
                canonical_k=lowering.canonical.k,
                work_units=units,
                task_contract_sha256=request_contract,
                request_sequence=self._sequence,
            )
            prepared.append((rank, assignments, artifact))
        self._sequence += 1
        responses: dict[int, Mapping[str, Any]] = {}
        try:
            with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
                future_to_rank = {
                    pool.submit(self._submit_with_deadline, rank, artifact): rank.index
                    for rank, _, artifact in prepared
                }
                for future in as_completed(future_to_rank):
                    responses[future_to_rank[future]] = future.result()
            request_metrics: dict[str, Any] = {
                "h2d_bytes": 0,
                "d2h_bytes": 0,
                "response_transfer_bytes": 0,
                "h2d_time_s": 0.0,
                "kernel_time_s": 0.0,
                "d2h_time_s": 0.0,
                "request_manifest_hashes": tuple(
                    artifact.manifest_sha256 for _, _, artifact in prepared
                ),
                "successful_request_count": 0,
                "active_rank_indices": tuple(),
                "active_dpu_ids": tuple(),
            }
            active_rank_indices: set[int] = set()
            active_dpu_ids: set[tuple[int, int]] = set()
            for rank, assignments, artifact in prepared:
                response = responses[rank.index]
                self._validate_successful_response(response, rank, artifact)
                self._response_native_identity_events.append(
                    (f"RESPONSE rank {rank.index}", response)
                )
                transfer = response.get("transfer", {})
                h2d_bytes = int(transfer.get("h2d_bytes", 0))
                d2h_bytes = int(transfer.get("d2h_bytes", 0))
                total_bytes = int(transfer.get("total_bytes", -1))
                if total_bytes != h2d_bytes + d2h_bytes:
                    raise RuntimeError(
                        "v4 response transfer total does not equal H2D plus D2H"
                    )
                response_timing = response.get("timing", {})
                request_metrics["h2d_bytes"] += h2d_bytes
                request_metrics["d2h_bytes"] += d2h_bytes
                request_metrics["response_transfer_bytes"] += total_bytes
                for metric, response_key in (
                    ("h2d_time_s", "h2d_time_s"),
                    ("kernel_time_s", "launch_time_s"),
                    ("d2h_time_s", "d2h_time_s"),
                ):
                    request_metrics[metric] = max(
                        request_metrics[metric],
                        float(response_timing.get(response_key, 0.0)),
                    )
                active_rank_indices.add(rank.index)
                active_dpu_ids.update(
                    (rank.index, record.local_dpu_id)
                    for record in artifact.work_units
                    if not record.flags
                )
            request_metrics["successful_request_count"] = len(prepared)
            if request_metrics["response_transfer_bytes"] != (
                request_metrics["h2d_bytes"] + request_metrics["d2h_bytes"]
            ):
                raise RuntimeError("v4 response transfer accounting is inconsistent")
            request_metrics["active_rank_indices"] = tuple(sorted(active_rank_indices))
            request_metrics["active_dpu_ids"] = tuple(sorted(active_dpu_ids))
            results: list[tuple[M5Tile, np.ndarray]] = []
            for rank, assignments, artifact in prepared:
                records = {
                    record.local_dpu_id: record for record in artifact.work_units
                }
                for tile, local_id in assignments:
                    record = records[local_id]
                    value = _read_output(
                        artifact.root / record.c_path,
                        tile,
                        packed=packed,
                    )
                    results.append((tile, value))
            bulk_verified = all(
                response.get("bulk_set_launch_verified") is True
                for response in responses.values()
            )
            return results, request_metrics, len(prepared) > 1, bulk_verified
        finally:
            for _, _, artifact in prepared:
                self._delete_request_dir(artifact)

    @staticmethod
    def _validate_waves(
        tiles: tuple[M5Tile, ...],
        waves: tuple[tuple[M5Tile, ...], ...],
        total_dpu_count: int,
    ) -> None:
        if not isinstance(waves, tuple):
            raise TypeError("compiled UPMEM plan must provide a tuple of waves")
        expected = [id(tile) for tile in tiles]
        if len(expected) != len(set(expected)):
            raise ValueError("lowering contains duplicate tile objects")
        actual: list[int] = []
        for wave in waves:
            if not isinstance(wave, tuple) or not wave:
                raise ValueError(
                    "compiled UPMEM plan contains an empty or invalid wave"
                )
            if len(wave) > total_dpu_count:
                raise ValueError("compiled UPMEM wave exceeds available DPU count")
            actual.extend(id(tile) for tile in wave)
        if len(actual) != len(set(actual)):
            raise ValueError("compiled UPMEM plan duplicated a tile across waves")
        if set(actual) != set(expected):
            raise ValueError("compiled UPMEM plan omitted or replaced a tile")

    def _validate_rank_assignments(
        self,
        wave: tuple[M5Tile, ...],
        requests: list[tuple[Any, list[tuple[M5Tile, int]]]],
    ) -> None:
        if not isinstance(requests, list) or not requests:
            raise ValueError("compiled UPMEM plan produced no rank assignments")
        expected_tiles = {id(tile) for tile in wave}
        assigned_tiles: list[int] = []
        assigned_ranks: set[int] = set()
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("rank assignment must be a (rank, assignments) tuple")
            rank, assignments = request
            if not any(rank is candidate for candidate in self.ranks):
                raise ValueError("compiled UPMEM plan referenced a foreign rank")
            rank_identity = id(rank)
            if rank_identity in assigned_ranks:
                raise ValueError("compiled UPMEM plan repeated a rank assignment")
            assigned_ranks.add(rank_identity)
            if not isinstance(assignments, list) or not assignments:
                raise ValueError(
                    "compiled UPMEM plan contains an empty rank assignment"
                )
            local_ids: set[int] = set()
            for assignment in assignments:
                if not isinstance(assignment, tuple) or len(assignment) != 2:
                    raise TypeError("tile assignment must be a (tile, local_id) tuple")
                tile, local_id = assignment
                if id(tile) not in expected_tiles:
                    raise ValueError(
                        "compiled UPMEM plan assigned a tile outside the wave"
                    )
                if not isinstance(local_id, int) or isinstance(local_id, bool):
                    raise TypeError("local DPU ID must be an integer")
                if not 0 <= local_id < rank.local_dpus:
                    raise ValueError("local DPU ID is outside the assigned rank")
                if local_id in local_ids:
                    raise ValueError("compiled UPMEM plan reused a local DPU ID")
                local_ids.add(local_id)
                assigned_tiles.append(id(tile))
        if len(assigned_tiles) != len(set(assigned_tiles)):
            raise ValueError("compiled UPMEM plan assigned a wave tile more than once")
        if set(assigned_tiles) != expected_tiles:
            raise ValueError("compiled UPMEM plan omitted a wave tile")

    @staticmethod
    def _delete_request_dir(artifact: Any) -> None:
        request_dir = Path(artifact.request_dir).resolve()
        root = Path(artifact.root).resolve()
        requests_root = (root / "requests").resolve()
        try:
            request_dir.relative_to(requests_root)
        except ValueError as exc:
            raise RuntimeError("refusing to delete a non-request v4 directory") from exc
        if request_dir == requests_root or request_dir.parent != requests_root:
            raise RuntimeError(
                "refusing to delete the v4 requests root or nested directory"
            )
        if not request_dir.is_dir() or not request_dir.name.isdigit():
            raise RuntimeError(
                "refusing to delete a directory not created as a v4 request"
            )
        if (
            Path(artifact.manifest_path).resolve().parent != request_dir
            or Path(artifact.sidecar_path).resolve().parent != request_dir
        ):
            raise RuntimeError(
                "refusing to delete an artifact outside its request directory"
            )
        shutil.rmtree(request_dir)

    def _validate_successful_response(
        self, response: Mapping[str, Any], rank: _RankSession, artifact: Any
    ) -> None:
        expected = {
            "status": "completed",
            "target_observed": "physical_hardware",
            "bulk_set_launch_verified": True,
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": True,
            "allocated_dpu_count": rank.local_dpus,
            "requested_dpu_count": rank.local_dpus,
            "tasklets_per_dpu": rank.session.profile.tasklets_per_dpu,
            "request_sequence": artifact.request_sequence,
        }
        for key, value in expected.items():
            if response.get(key) != value:
                raise RuntimeError(
                    f"unverified v4 response field {key}: {response.get(key)!r}"
                )
        startup_identity = _native_identity(
            rank.session.startup, source=f"READY rank {rank.index}"
        )
        response_identity = _native_identity(
            response, source=f"RESPONSE rank {rank.index}"
        )
        if response_identity != startup_identity:
            raise RuntimeError(
                f"RESPONSE rank {rank.index} native identity disagrees with READY"
            )

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._terminal_metadata
        self._closed = True
        release_failed = self._failed
        remaining = self._deadline - time.monotonic()
        cleanup_deadline = (
            self._deadline
            if remaining > 0
            else time.monotonic() + min(1.0, self.engine.timeout_s)
        )
        releases: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=len(self.ranks)) as pool:
            future_to_rank = {
                pool.submit(self._close_rank, rank, cleanup_deadline): rank
                for rank in self.ranks
            }
            for future in as_completed(future_to_rank):
                rank = future_to_rank[future]
                try:
                    releases[rank.index] = future.result()
                except BaseException:
                    release_failed = True
                    releases[rank.index] = None
        diagnostics = [
            self._rank_diagnostics(rank, releases.get(rank.index))
            for rank in self.ranks
        ]

        native_identity_error: str | None = None
        try:
            observed_native_identity = _agreed_native_identity(
                tuple(
                    (f"READY rank {rank.index}", rank.session.startup)
                    for rank in self.ranks
                )
                + tuple(self._response_native_identity_events)
            )
        except RuntimeError as exc:
            observed_native_identity = None
            native_identity_error = str(exc)

        confirmed = len(diagnostics) == len(self.ranks) and all(
            diagnostic["release_confirmed"] for diagnostic in diagnostics
        )
        physical_target_verified = not self._test_double_execution and all(
            rank.session.startup.get("target_observed") == "physical_hardware"
            for rank in self.ranks
        )
        allocation_verified = all(
            rank.session.startup.get("event") == "READY"
            and rank.session.startup.get("status") == "ready"
            and rank.session.startup.get("hardware_allocation_verified") is True
            and rank.session.startup.get("requested_dpu_count") == rank.local_dpus
            and rank.session.startup.get("allocated_dpu_count") == rank.local_dpus
            and rank.session.startup.get("tasklets_per_dpu")
            == self.engine.tasklets_per_dpu
            for rank in self.ranks
        )
        binary_identity_verified = all(
            rank.session.startup.get("dpu_binary_sha256")
            == self.engine._binary_provenance["dpu_binary_sha256"]
            and rank.session.startup.get("initialization_binary_sha256")
            == self.engine._binary_provenance["initialization_binary_sha256"]
            for rank in self.ranks
        )
        ready_verified = allocation_verified and binary_identity_verified
        native_execution = (
            self._successful_request_count > 0 and not self._test_double_execution
        )
        verified = (
            physical_target_verified
            and allocation_verified
            and binary_identity_verified
            and observed_native_identity is not None
            and confirmed
            and not release_failed
            and native_execution
        )
        self._terminal_metadata = {
            **(
                _native_identity_metadata(observed_native_identity)
                if observed_native_identity is not None
                else {}
            ),
            **_COORDINATOR_PROVENANCE,
            **self.engine._provenance,
            "strategy_identity": self.strategy_identity,
            "strategy_config_hash": self.strategy_config_hash,
            "decomposition_strategy": _ACTIVE_MECHANISM_IDS["decomposition"],
            "placement_strategy": _ACTIVE_MECHANISM_IDS["placement"],
            "kernel_provider": _ACTIVE_MECHANISM_IDS["kernel"],
            "reduction_provider": _ACTIVE_MECHANISM_IDS["reduction"],
            "reduction_strategy": _ACTIVE_MECHANISM_IDS["reduction"],
            "target_observed": "physical_hardware" if verified else "not_verified",
            "requested_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_rank_count": len(self.ranks),
            "allocated_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_tasklets_per_dpu": self.engine.tasklets_per_dpu,
            "tasklets_per_dpu": self.engine.tasklets_per_dpu,
            "hardware_allocation_verified": allocation_verified,
            "native_kernel_executed": native_execution,
            "hardware_kernel_executed": native_execution,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_release_verified": confirmed,
            "hardware_release_confirmed": confirmed,
            "ready_verified": ready_verified,
            "physical_target_verified": physical_target_verified,
            "binary_identity_verified": binary_identity_verified,
            "native_identity_verified": observed_native_identity is not None,
            "native_identity_failure": native_identity_error,
            "native_identity_observation_count": len(self.ranks)
            + len(self._response_native_identity_events),
            "successful_request_count": self._successful_request_count,
            "active_rank_indices": tuple(sorted(self._active_rank_indices)),
            "active_dpu_ids": tuple(sorted(self._active_dpu_ids)),
            "native_diagnostics": diagnostics,
            "primary_failure_stage": self._failure_stage,
            "release_failure_stage": (
                "hardware_release_failed" if not confirmed else None
            ),
            "failure_stage": self._failure_stage
            or ("hardware_release_failed" if not confirmed else None),
        }
        return self._terminal_metadata

    def _submit_with_deadline(
        self, rank: _RankSession, artifact: Any
    ) -> Mapping[str, Any]:
        return rank.session.submit(
            artifact,
            timeout_s=self._remaining_timeout(),
        )

    @staticmethod
    def _close_rank(rank: _RankSession, deadline: float) -> Any:
        return _close_rank_before_deadline(rank, deadline)

    def _remaining_timeout(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._failed = True
            self._failure_stage = "kernel_timeout"
            raise V4ProtocolError(
                "kernel_timeout", "whole-circuit physical deadline expired"
            )
        return remaining

    @staticmethod
    def _rank_diagnostics(rank: _RankSession, release: Any) -> dict[str, Any]:
        """Return only bounded, JSON-safe diagnostics for one rank session."""

        event = getattr(release, "event", {})
        if not isinstance(event, Mapping):
            event = {}
        returncode = event.get("returncode")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            process = getattr(rank.session, "process", None)
            candidate = getattr(process, "returncode", None)
            returncode = candidate if isinstance(candidate, int) else None
        stdout = getattr(release, "stdout", "")
        stderr = getattr(release, "stderr", "")
        return {
            "rank_index": rank.index,
            "rank_path": str(
                event.get("rank_path")
                or rank.session.startup.get("rank_path")
                or getattr(getattr(rank.session, "profile", None), "rank_path", "")
                or ""
            ),
            "stdout": stdout if isinstance(stdout, str) else "",
            "stderr": stderr if isinstance(stderr, str) else "",
            "stdout_truncated": bool(getattr(release, "stdout_truncated", False)),
            "stderr_truncated": bool(getattr(release, "stderr_truncated", False)),
            "stdout_total_bytes": int(getattr(release, "stdout_total_bytes", 0)),
            "stderr_total_bytes": int(getattr(release, "stderr_total_bytes", 0)),
            "stdout_limit_exceeded": bool(
                getattr(release, "stdout_limit_exceeded", False)
            ),
            "stderr_limit_exceeded": bool(
                getattr(release, "stderr_limit_exceeded", False)
            ),
            "returncode": returncode,
            "release_confirmed": bool(getattr(release, "release_confirmed", False)),
        }


@dataclass
class _Aggregate:
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    host_quantization_s: float = 0.0
    preparation_s: float = 0.0
    h2d_s: float = 0.0
    kernel_s: float = 0.0
    d2h_s: float = 0.0
    host_dequantization_s: float = 0.0
    reduction_s: float = 0.0
    route_total_s: float = 0.0
    physical_plan_consumed: bool = False

    def add(self, result: tuple[np.ndarray, Mapping[str, Any]]) -> None:
        _, metadata = result
        if not isinstance(metadata, Mapping):
            metadata = {}
        timing = metadata.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        if metadata.get("physical_plan_consumed") is not True:
            raise RuntimeError(
                "UPMEM task result did not consume the compiled physical plan"
            )
        self.physical_plan_consumed = True
        self.h2d_bytes += _required_byte_count(
            metadata,
            "application_visible_h2d_bytes",
            "h2d_bytes",
        )
        self.d2h_bytes += _required_byte_count(
            metadata,
            "application_visible_d2h_bytes",
            "d2h_bytes",
        )
        self.host_quantization_s += _seconds(
            timing.get(
                "host_quantization_time_s",
                metadata.get("host_quantization_time_s", 0.0),
            )
        )
        self.preparation_s += _seconds(
            timing.get("preparation_time_s", metadata.get("preparation_time_s", 0.0))
        )
        self.h2d_s += _seconds(timing.get("h2d_time_s", 0.0))
        self.kernel_s += _seconds(timing.get("kernel_time_s", 0.0))
        self.d2h_s += _seconds(timing.get("d2h_time_s", 0.0))
        self.host_dequantization_s += _seconds(
            timing.get(
                "host_dequantization_time_s",
                metadata.get("host_dequantization_time_s", 0.0),
            )
        )
        self.reduction_s += _seconds(
            timing.get(
                "host_tile_assembly_time_s",
                metadata.get("host_tile_assembly_time_s", 0.0),
            )
        )
        if metadata.get("target_observed") == "sdk_simulator" or bool(
            metadata.get("simulator_kernel_executed", False)
        ):
            raise RuntimeError("UPMEM adapter refuses simulator execution")
        if bool(metadata.get("cpu_fallback_used", False)):
            raise RuntimeError("UPMEM adapter refuses CPU fallback execution")


def run_upmem(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> ExecutionResult:
    """Execute a compiled M5 plan with one persistent session and no fallback.

    Deterministic unsupported dispatch is represented by ``ExecutionFailure``
    at the public dispatcher. Malformed inputs and native/session failures
    raise so their original failure stage remains available to the experiment
    orchestrator.
    """

    tensors = {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}
    _validate_invocation(plan, dag, tensors, context)
    upmem_plan = plan.payload
    assert isinstance(upmem_plan, UpmemPlan)
    resources = context.target_resources
    assert resources is not None
    resource_hashes = _validate_resources(resources)
    validate_upmem_runtime_resources(resources, upmem_plan.topology)
    aggregate = _Aggregate()
    output: np.ndarray | None = None
    output_digest: str | None = None
    session: Any | None = None
    terminal_metadata: Mapping[str, Any] | None = None
    completed_node_ids: tuple[str, ...] = ()
    execution_error: BaseException | None = None
    close_error: BaseException | None = None
    session_open_s = 0.0
    session_close_s = 0.0
    try:
        open_started = time.perf_counter()
        session = _open_session(plan, context)
        session_open_s = time.perf_counter() - open_started
        # Warmups run on the persistent session but are deliberately outside
        # route_total_s.  The measured route is the sum of session lifecycle
        # and measured repetitions, including host DAG and reduction work.
        aggregate.route_total_s += session_open_s
        for _ in range(context.warmups):
            _execute_once(
                session,
                dag,
                tensors,
                upmem_plan,
                resources=None,
                aggregate=None,
            )
        for _ in range(context.repetitions):
            route_started = time.perf_counter()
            output, completed_node_ids = _execute_once(
                session,
                dag,
                tensors,
                upmem_plan,
                resources=resources,
                aggregate=aggregate,
            )
            aggregate.route_total_s += time.perf_counter() - route_started
            # This reproducibility check is intentionally outside route timing.
            digest = _array_hash(output)
            if output_digest is None:
                output_digest = digest
            elif digest != output_digest:
                raise RuntimeError("UPMEM execution produced non-deterministic output")
    except BaseException as exc:
        execution_error = exc
    finally:
        if session is not None:
            close_started = time.perf_counter()
            try:
                terminal_metadata = session.close()
            except BaseException as exc:
                close_error = exc
            session_close_s = time.perf_counter() - close_started
            aggregate.route_total_s += session_close_s

    if execution_error is not None and close_error is not None:
        raise RuntimeError(
            f"UPMEM execution failed: {execution_error}; "
            f"session close failed: {close_error}"
        ) from execution_error
    if execution_error is not None:
        raise execution_error
    if close_error is not None:
        raise RuntimeError("UPMEM session close failed") from close_error
    _validate_terminal_metadata(terminal_metadata, upmem_plan)

    if output is None or output_digest is None:
        raise RuntimeError("UPMEM execution did not produce an output")
    h2d = aggregate.h2d_bytes
    d2h = aggregate.d2h_bytes
    transfer = h2d + d2h
    validate_transfer_bytes(h2d, d2h, transfer)
    facts = replace(
        _facts_from_metadata(terminal_metadata),
        **resource_hashes,
        rank_binding_sha256=_rank_binding_sha256(resources.rank_paths),
        physical_plan_consumed=aggregate.physical_plan_consumed,
    )
    result = ExecutionResult(
        contraction_dag_hash=contraction_dag_hash(dag),
        target=Target.UPMEM,
        output=np.array(output, copy=True),
        executed_node_ids=completed_node_ids,
        timing=TimingBreakdown(
            host_quantization_s=aggregate.host_quantization_s or None,
            preparation_s=aggregate.preparation_s or None,
            h2d_s=aggregate.h2d_s or None,
            kernel_s=aggregate.kernel_s or None,
            d2h_s=aggregate.d2h_s or None,
            host_dequantization_s=aggregate.host_dequantization_s or None,
            reduction_s=aggregate.reduction_s or None,
            session_open_s=session_open_s or None,
            session_close_s=session_close_s or None,
            route_total_s=aggregate.route_total_s or None,
        ),
        h2d_bytes=h2d,
        d2h_bytes=d2h,
        transfer_bytes=transfer,
        output_hash=output_digest,
        backend_facts=facts,
    )
    validate_execution_result(result)
    return result


def _open_session(plan: ExecutionPlan, context: RunContext) -> Any:
    """Create the real M5 session; tests replace this single seam."""

    payload = plan.payload
    assert isinstance(payload, UpmemPlan)
    resources = context.target_resources
    assert resources is not None
    if resources.session_opener is not None:
        return resources.session_opener(plan, context)
    timeout_s = 60.0 if context.timeout_s is None else context.timeout_s
    engine = UpmemV4Executor(
        session_root=Path(resources.session_root),
        host_binary=Path(resources.host_binary),
        dpu_binary=Path(resources.dpu_binary),
        initialization_binary=Path(resources.initialization_binary),
        rank_paths=resources.rank_paths,
        dpu_count=payload.topology.dpu_count,
        tasklets_per_dpu=payload.topology.tasklets_per_dpu,
        timeout_s=timeout_s,
    )
    topology = UpmemTopology(
        dpu_count=payload.topology.dpu_count,
        tasklets_per_dpu=payload.topology.tasklets_per_dpu,
        rank_count=payload.topology.rank_count,
    )
    return engine.open_session(payload.numeric_mode, topology)


def _execute_once(
    session: Any,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    plan: UpmemPlan,
    *,
    resources: UpmemRuntimeResources | None,
    aggregate: _Aggregate | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    tensors = dict(inputs)
    nodes = {node.node_id: node for node in dag.nodes}
    remaining_consumers = _remaining_consumers(dag)
    produced_tensor_ids = {node.output.id for node in dag.nodes}
    completed_node_ids: list[str] = []
    for node_plan in plan.node_plans:
        node_id = node_plan.node_id
        node = nodes[node_id]
        if isinstance(node, ContractNode):
            left = _resolve_view(node.left, tensors)
            right = _resolve_view(node.right, tensors)
            value, metadata = session.execute(node, left, right, node_plan=node_plan)
            if aggregate is not None:
                assert resources is not None
                aggregate.add((value, metadata))
            value = np.asarray(value)
        elif isinstance(node, ReduceNode):
            reduction_started = time.perf_counter()
            value = np.sum(
                np.stack(
                    [_resolve_view(view, tensors) for view in node.inputs], axis=0
                ),
                axis=0,
            )
            if aggregate is not None:
                aggregate.reduction_s += time.perf_counter() - reduction_started
        else:  # pragma: no cover - graph validation closes this union
            raise TypeError(f"unsupported UPMEM DAG node: {type(node).__name__}")
        if tuple(value.shape) != node.output.shape:
            raise ValueError(
                f"UPMEM node {node_id} produced shape {value.shape}; expected {node.output.shape}"
            )
        tensors[node.output.id] = value
        if (
            remaining_consumers.get(node.output.id, 0) == 0
            and node.output.id != dag.output.tensor_id
        ):
            tensors.pop(node.output.id, None)
        for tensor_id in _node_input_tensor_ids(node):
            remaining_consumers[tensor_id] -= 1
            if (
                remaining_consumers[tensor_id] == 0
                and tensor_id in produced_tensor_ids
                and tensor_id != dag.output.tensor_id
            ):
                tensors.pop(tensor_id, None)
        # The host coordinator has validated and published this node output
        # for its dependants. This does not claim native-kernel exactly once.
        completed_node_ids.append(node_id)
    return _resolve_view(dag.output, tensors), tuple(completed_node_ids)


def _validate_invocation(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> None:
    validate_contraction_dag(dag)
    validate_execution_plan(plan)
    if plan.target is not Target.UPMEM or not isinstance(plan.payload, UpmemPlan):
        raise ValueError("run_upmem requires an UPMEM execution plan")
    if context.target is not Target.UPMEM:
        raise ValueError("run_upmem requires an UPMEM RunContext")
    if context.warmups < 0 or context.repetitions < 1:
        raise ValueError(
            "warmups must be non-negative and repetitions must be positive"
        )
    if context.timeout_s is not None and (
        context.timeout_s <= 0 or not math.isfinite(context.timeout_s)
    ):
        raise ValueError("timeout_s must be finite and positive when provided")
    actual_hash = contraction_dag_hash(dag)
    if plan.contraction_dag_hash != actual_hash:
        raise ValueError("execution plan hash does not match supplied DAG")
    if plan.payload.numeric_mode in {
        NumericMode.FLOAT32_REAL,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    }:
        for value in inputs.values():
            array = np.asarray(value)
            if np.iscomplexobj(array) and np.any(np.imag(array) != 0):
                raise ValueError(
                    "M5 real-valued UPMEM numeric modes reject nonzero imaginary inputs"
                )
    validate_dag_inputs(dag, inputs)
    validate_upmem_plan_for_dag(dag, plan.payload)
    validate_active_upmem_plan(plan.payload)
    if context.target_resources is None:
        raise ValueError("UPMEM runtime resources are required")
    _validate_upmem_resources(plan.payload, context.target_resources)


def _validate_upmem_resources(
    plan: UpmemPlan, resources: UpmemRuntimeResources | None
) -> None:
    topology = plan.topology
    if resources is None:
        raise ValueError("UPMEM runtime resources are required")
    validate_upmem_runtime_resources(resources, topology)
    if topology.dpu_count // topology.rank_count > 64:
        raise ValueError("UPMEM plan exceeds 64 DPUs per rank")
    if not 1 <= topology.tasklets_per_dpu <= 24:
        raise ValueError("UPMEM plan tasklets_per_dpu must be in [1, 24]")


def _node_input_tensor_ids(node: ContractNode | ReduceNode) -> tuple[str, ...]:
    if isinstance(node, ContractNode):
        return (node.left.tensor_id, node.right.tensor_id)
    return tuple(view.tensor_id for view in node.inputs)


def _remaining_consumers(dag: ContractionDAG) -> dict[str, int]:
    remaining: dict[str, int] = {}
    for node in dag.nodes:
        for tensor_id in _node_input_tensor_ids(node):
            remaining[tensor_id] = remaining.get(tensor_id, 0) + 1
    return remaining


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_resources(resources: UpmemRuntimeResources) -> dict[str, str]:
    paths = {
        "host_binary": Path(resources.host_binary),
        "dpu_binary": Path(resources.dpu_binary),
        "initialization_binary": Path(resources.initialization_binary),
    }
    if not resources.session_root:
        raise ValueError("UPMEM session_root must be non-empty")
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"UPMEM {label} is not a regular file: {path}")
        if label == "host_binary" and not os.access(path, os.X_OK):
            raise ValueError("UPMEM host_binary is not executable")
    return {
        "host_binary_sha256": _file_sha256(paths["host_binary"]),
        "dpu_binary_sha256": _file_sha256(paths["dpu_binary"]),
        "initialization_binary_sha256": _file_sha256(paths["initialization_binary"]),
    }


def _facts_from_metadata(
    metadata: Mapping[str, Any],
) -> BackendFacts:
    return BackendFacts(
        backend_id=_observed_text(metadata, "backend_id"),
        profile_id=_observed_text(metadata, "profile", "physical_profile"),
        abi_id=_observed_text(metadata, "abi", "abi_version"),
        session_id=_observed_text(metadata, "session_protocol"),
        dispatch_id=_observed_text(metadata, "dispatch_mode"),
        kernel_id=_observed_text(metadata, "kernel_identity"),
        execution_class=_observed_text(metadata, "execution_class"),
        intermediate_placement=_observed_text(metadata, "graph_intermediate_placement"),
        intermediate_placement_origin=_observed_text(
            metadata, "graph_intermediate_placement_origin"
        ),
        native_identity_verified=bool(metadata.get("native_identity_verified", False)),
        target_observed=metadata.get("target_observed"),
        hardware_allocation_verified=bool(
            metadata.get("hardware_allocation_verified", False)
        ),
        hardware_release_verified=bool(
            metadata.get("hardware_release_verified", False)
        ),
        hardware_release_confirmed=bool(
            metadata.get("hardware_release_confirmed", False)
        ),
        requested_dpu_count=_optional_int(metadata.get("requested_dpu_count")),
        allocated_dpu_count=_optional_int(metadata.get("allocated_dpu_count")),
        observed_rank_count=_optional_int(metadata.get("observed_rank_count")),
        tasklets_per_dpu=_optional_int(
            metadata.get("observed_tasklets_per_dpu", metadata.get("tasklets_per_dpu"))
        ),
        native_kernel_executed=bool(metadata.get("native_kernel_executed", False)),
        hardware_kernel_executed=bool(metadata.get("hardware_kernel_executed", False)),
        simulator_kernel_executed=bool(
            metadata.get("simulator_kernel_executed", False)
        ),
        cpu_fallback_used=bool(metadata.get("cpu_fallback_used", False)),
        physical_plan_consumed=bool(metadata.get("physical_plan_consumed", False)),
    )


def _validate_terminal_metadata(
    metadata: Mapping[str, Any] | None, plan: UpmemPlan
) -> None:
    """Admit a result only when the terminal M5 close contract is complete."""

    if not isinstance(metadata, Mapping):
        raise RuntimeError("UPMEM session close returned no terminal metadata")
    required = {
        "target_observed": "physical_hardware",
        "hardware_allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_release_verified": True,
        "hardware_release_confirmed": True,
        "native_identity_verified": True,
        "failure_stage": None,
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "observed_rank_count": plan.topology.rank_count,
        "observed_tasklets_per_dpu": plan.topology.tasklets_per_dpu,
        "session_protocol": plan.session_id,
        "dispatch_mode": plan.dispatch_id,
        "kernel_identity": plan.kernel_id,
        "execution_class": "physical_v4_output_tile",
        "graph_intermediate_placement": "host_managed",
        "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"UPMEM terminal metadata is not physically verified: "
                f"{key}={metadata.get(key)!r}"
            )

    aliases = {
        "profile": ("physical_profile",),
        "abi": ("abi_version",),
    }
    for canonical, alternatives in aliases.items():
        observed = _observed_text(metadata, canonical, *alternatives)
        expected = plan.profile_id if canonical == "profile" else plan.abi_id
        if observed != expected:
            raise RuntimeError(
                "UPMEM terminal metadata is not physically verified: "
                f"{canonical}={observed!r}"
            )


def _observed_text(metadata: Mapping[str, Any], *keys: str) -> str:
    """Read one observed terminal value, rejecting absent or conflicting aliases."""

    values = [str(metadata[key]) for key in keys if metadata.get(key) is not None]
    if not values:
        raise RuntimeError(f"UPMEM terminal metadata is missing {keys[0]}")
    if len(set(values)) != 1:
        raise RuntimeError(f"UPMEM terminal metadata has conflicting {keys[0]} values")
    return values[0]


def _required_byte_count(metadata: Mapping[str, Any], *keys: str) -> int:
    """Return an observed application-visible byte count, never a guessed zero."""

    values = [metadata[key] for key in keys if key in metadata]
    if not values:
        raise RuntimeError(f"UPMEM task metadata is missing {keys[0]}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise RuntimeError(f"UPMEM task metadata has invalid {keys[0]}")
    parsed = [int(value) for value in values]
    if len(set(parsed)) != 1:
        raise RuntimeError(f"UPMEM task metadata has conflicting {keys[0]} values")
    return parsed[0]


def _rank_binding_sha256(rank_paths: tuple[str, ...]) -> str:
    """Hash ordered runtime rank bindings without exposing their raw paths."""

    return hashlib.sha256(
        canonical_serialize(tuple(rank_paths)).encode("utf-8")
    ).hexdigest()


def _resolve_view(view: TensorView, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    if view.tensor_id not in tensors:
        raise ValueError(f"UPMEM tensor {view.tensor_id} is not available")
    value = tensors[view.tensor_id]
    if not view.slice_spec:
        return value
    indices: list[slice | int] = [slice(None)] * value.ndim
    for axis, index in view.slice_spec:
        indices[axis] = index
    sliced = value[tuple(indices)]
    if tuple(sliced.shape) != view.shape:
        raise ValueError(f"UPMEM sliced tensor {view.tensor_id} has wrong shape")
    return sliced


def _topological_order(dag: ContractionDAG) -> tuple[str, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    remaining = {node_id: len(node.dependencies) for node_id, node in nodes.items()}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in dag.nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)
    ready = sorted(node_id for node_id, count in remaining.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for dependent in sorted(dependents[node_id]):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
        ready.sort()
    if len(order) != len(nodes):
        raise ValueError("UPMEM DAG cannot be topologically ordered")
    return tuple(order)


def _seconds(value: Any) -> float:
    result = float(value or 0.0)
    if result < 0 or not math.isfinite(result):
        raise ValueError("UPMEM timing values must be finite and non-negative")
    return result


def _nonnegative_int(value: Any) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError("UPMEM byte values must be non-negative")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError("UPMEM count values must be non-negative")
    return result


__all__ = ["UpmemV4Executor", "UpmemV4Session", "run_upmem"]
