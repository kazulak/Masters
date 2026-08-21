"""Bounded physical v4 execution engine for contraction DAG nodes.

The engine deliberately owns only one binary contraction at a time. The caller
owns DAG dependencies and the host tensor store; this module lowers one
``ContractNode`` to bounded v4 output/K tiles, submits those tiles to persistent
physical rank sessions, and reconstructs the output. It does not claim
DPU-resident graph intermediates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from quantum_bench.formats.fixed_point import FixedPointSpec, quantize_fixed_point
from quantum_bench.execution.contracts import UpmemNodePlan
from quantum_bench.tn.graph import ContractNode
from quantum_bench.targets.upmem.execution_plan_v4 import (
    MAX_INT32_SAFE_K,
    NATIVE_EXECUTION_IDENTITY,
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4ProtocolError,
    V4Session,
)
from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
    M5Tile,
    M5TileLimits,
)
from quantum_bench.targets.upmem.m5_strategies import (
    M5DecompositionStrategy,
    M5KernelProvider,
    M5PlacementStrategy,
    M5ReductionProvider,
    M5StrategyBundle,
)
from quantum_bench.whole_circuit.core import (
    DeviceTopology,
    EngineTaskResult,
    NumericPolicy,
)
from quantum_bench.whole_circuit.strategies import (
    DecompositionStrategy,
    KernelProvider,
    PlacementStrategy,
    ReductionProvider,
    StrategyConfiguration,
    StrategyRole,
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


def _real_float32(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(np.imag(array) != 0):
            raise ValueError("M5 whole-circuit engine requires real-valued tensors")
        array = np.real(array)
    result = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("M5 whole-circuit engine requires finite float32 values")
    return result


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


class M5WholeCircuitEngine:
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
        decomposition_strategy: DecompositionStrategy | None = None,
        placement_strategy: PlacementStrategy | None = None,
        kernel_provider: KernelProvider | None = None,
        reduction_provider: ReductionProvider | None = None,
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
        self.decomposition_strategy = (
            decomposition_strategy
            if decomposition_strategy is not None
            else M5DecompositionStrategy()
        )
        self.placement_strategy = (
            placement_strategy
            if placement_strategy is not None
            else M5PlacementStrategy()
        )
        self.kernel_provider = (
            kernel_provider if kernel_provider is not None else M5KernelProvider()
        )
        self.reduction_provider = (
            reduction_provider
            if reduction_provider is not None
            else M5ReductionProvider()
        )
        self._strategy_bundle = M5StrategyBundle(
            decomposition=self.decomposition_strategy,
            placement=self.placement_strategy,
            kernel=self.kernel_provider,
            reduction=self.reduction_provider,
        )
        self._strategy_configuration = self._strategy_bundle.identity_configuration()
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
        self._source_root = str(Path(__file__).resolve().parents[4])
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
        return self._strategy_configuration.to_record()

    @property
    def strategy_config_hash(self) -> str:
        return self._strategy_configuration.sha256

    def open_session(
        self,
        policy: NumericPolicy,
        topology: DeviceTopology = DeviceTopology(),
        *,
        decomposition_strategy: DecompositionStrategy | None = None,
        placement_strategy: PlacementStrategy | None = None,
        kernel_provider: KernelProvider | None = None,
        reduction_provider: ReductionProvider | None = None,
    ) -> M5WholeCircuitSession:
        if topology.backend != "upmem":
            raise ValueError("M5WholeCircuitEngine requires an upmem topology")
        if len(topology.device_ids) != self.dpu_count:
            raise ValueError("topology device count must match engine dpu_count")
        if topology.tasklets_per_device != self.tasklets_per_dpu:
            raise ValueError(
                "topology tasklet count must match engine tasklets_per_dpu"
            )
        if policy.name not in {"float32_real", "host_packed_int8_per_task_v1"}:
            raise ValueError(f"unsupported M5 numeric policy: {policy.name}")

        strategy_bundle = M5StrategyBundle(
            decomposition=(
                decomposition_strategy
                if decomposition_strategy is not None
                else self.decomposition_strategy
            ),
            placement=(
                placement_strategy
                if placement_strategy is not None
                else self.placement_strategy
            ),
            kernel=(
                kernel_provider if kernel_provider is not None else self.kernel_provider
            ),
            reduction=(
                reduction_provider
                if reduction_provider is not None
                else self.reduction_provider
            ),
        )
        strategy_configuration = strategy_bundle.identity_configuration()

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
                        if policy.name == "host_packed_int8_per_task_v1"
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
                _native_identity(
                    session.startup, source=f"READY rank {index}"
                )
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
        return M5WholeCircuitSession(
            policy=policy,
            ranks=tuple(ranks),
            engine=self,
            deadline=deadline,
            strategy_bundle=strategy_bundle,
            strategy_configuration=strategy_configuration,
        )


class M5WholeCircuitSession:
    """Persistent rank sessions used by one whole-graph measurement."""

    def __init__(
        self,
        *,
        policy: NumericPolicy,
        ranks: tuple[_RankSession, ...],
        engine: M5WholeCircuitEngine,
        deadline: float,
        strategy_bundle: M5StrategyBundle,
        strategy_configuration: StrategyConfiguration,
    ) -> None:
        self.policy = policy
        self.ranks = ranks
        self.engine = engine
        self._deadline = float(deadline)
        self.strategy_bundle = strategy_bundle
        self.decomposition_strategy = strategy_bundle.decomposition
        self.placement_strategy = strategy_bundle.placement
        self.kernel_provider = strategy_bundle.kernel
        self.reduction_provider = strategy_bundle.reduction
        self._strategy_configuration = strategy_configuration
        self._strategy_implementation_ids = {
            identity.role: identity.implementation_id
            for identity in strategy_configuration.strategies
        }
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
        return self._strategy_configuration.to_record()

    @property
    def strategy_config_hash(self) -> str:
        return self._strategy_configuration.sha256

    def execute(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        node_plan: UpmemNodePlan | None = None,
    ) -> EngineTaskResult:
        if self._closed:
            raise RuntimeError("M5 whole-circuit session is closed")
        self._remaining_timeout()
        started = time.perf_counter()
        packed = self.policy.name == "host_packed_int8_per_task_v1"
        limits = M5TileLimits.host_packed_int8() if packed else M5TileLimits.float32()
        lowering = self.decomposition_strategy.decompose(
            node, left, right, limits=limits
        )
        canonical_left = lowering.canonical.left
        canonical_right = lowering.canonical.right
        quantization_metadata: dict[str, Any] = {}
        left_scale = right_scale = 1.0
        if packed:
            left_quantized = quantize_fixed_point(
                canonical_left, FixedPointSpec(route_dtype="int8")
            )
            right_quantized = quantize_fixed_point(
                canonical_right, FixedPointSpec(route_dtype="int8")
            )
            canonical_left = np.ascontiguousarray(left_quantized.array, dtype=np.int8)
            canonical_right = np.ascontiguousarray(right_quantized.array, dtype=np.int8)
            left_scale = float(left_quantized.record.scale)
            right_scale = float(right_quantized.record.scale)
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
                "saturation_count": int(
                    left_quantized.record.saturation_count
                    + right_quantized.record.saturation_count
                ),
                "host_quantization_time_s": float(
                    left_quantized.record.conversion_time_s
                    + right_quantized.record.conversion_time_s
                ),
            }
        else:
            canonical_left = np.ascontiguousarray(_real_float32(canonical_left))
            canonical_right = np.ascontiguousarray(_real_float32(canonical_right))
            scale = 1.0
            quantization_metadata = {"packed_int8_transport": False}
        host_quantization_time_s = float(
            quantization_metadata.get("host_quantization_time_s", 0.0)
        )

        partials: dict[str, np.ndarray] = {}
        bytes_h2d = bytes_d2h = 0
        timing = {"h2d_time_s": 0.0, "kernel_time_s": 0.0, "d2h_time_s": 0.0}
        request_hashes: list[str] = []
        parallel_rank_waves = 0
        bulk_verified = True
        total_dpus = sum(rank.local_dpus for rank in self.ranks)
        if node_plan is None:
            # Temporary compatibility for direct engine tests and legacy callers.
            # The active execution adapter always supplies the compiled plan.
            waves = self.placement_strategy.place_waves(lowering.tiles, total_dpus)
            planned_requests = tuple(
                self.placement_strategy.map_wave_to_ranks(wave, self.ranks)
                for wave in waves
            )
            physical_plan_consumed = False
        else:
            waves, planned_requests = self._requests_from_plan(
                node, lowering, node_plan
            )
            physical_plan_consumed = True
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
        output = self.reduction_provider.reduce(
            lowering, partials, packed=packed, scale=scale
        )
        host_tile_assembly_time_s = time.perf_counter() - assembly_started
        host_dequantization_time_s = host_tile_assembly_time_s if packed else 0.0
        elapsed = time.perf_counter() - started
        return EngineTaskResult(
            output=output,
            metadata={
                "engine": self.engine.name,
                "execution_time_s": elapsed,
                "timing": {
                    **timing,
                    "host_quantization_time_s": host_quantization_time_s,
                    "host_dequantization_time_s": host_dequantization_time_s,
                    "host_tile_assembly_time_s": host_tile_assembly_time_s,
                    "total_route_time_s": elapsed,
                },
                **_native_identity_metadata(self._startup_native_identity),
                **_COORDINATOR_PROVENANCE,
                **self.engine._provenance,
                "strategy_identity": self.strategy_identity,
                "strategy_config_hash": self.strategy_config_hash,
                "decomposition_strategy": self._strategy_implementation_ids[
                    StrategyRole.DECOMPOSITION
                ],
                "placement_strategy": self._strategy_implementation_ids[
                    StrategyRole.PLACEMENT
                ],
                "kernel_provider": self._strategy_implementation_ids[
                    StrategyRole.KERNEL
                ],
                "reduction_provider": self._strategy_implementation_ids[
                    StrategyRole.REDUCTION
                ],
                "reduction_strategy": self._strategy_implementation_ids[
                    StrategyRole.REDUCTION
                ],
                "numeric_transport": numeric_transport,
                "packed_int8_transfer": packed,
                "host_quantization_time_s": host_quantization_time_s,
                "host_dequantization_time_s": host_dequantization_time_s,
                "host_tile_assembly_time_s": host_tile_assembly_time_s,
                "host_dequantization_timing_overlap": packed,
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
                    "not_verified"
                    if self._test_double_execution
                    else "physical_hardware"
                ),
                "test_double_execution": self._test_double_execution,
                "cpu_fallback_used": False,
                "simulator_kernel_executed": False,
                "physical_plan_consumed": physical_plan_consumed,
                **quantization_metadata,
            },
        )

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
            raise ValueError("compiled UPM canonical B/M/K/N shape differs from lowering")
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
                self.kernel_provider.build_work_unit(
                    tile, local_id, canonical_left, canonical_right, packed
                )
                for tile, local_id in assignments
            ]
            artifact = self.kernel_provider.prepare_request(
                rank.root,
                profile=rank.session.profile,
                lowering=lowering,
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
                    value = self.kernel_provider.read_output(
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
            raise TypeError("placement strategy must return a tuple of waves")
        expected = [id(tile) for tile in tiles]
        if len(expected) != len(set(expected)):
            raise ValueError("lowering contains duplicate tile objects")
        actual: list[int] = []
        for wave in waves:
            if not isinstance(wave, tuple) or not wave:
                raise ValueError("placement strategy returned an empty or invalid wave")
            if len(wave) > total_dpu_count:
                raise ValueError("placement wave exceeds available DPU count")
            actual.extend(id(tile) for tile in wave)
        if len(actual) != len(set(actual)):
            raise ValueError("placement strategy duplicated a tile across waves")
        if set(actual) != set(expected):
            raise ValueError("placement strategy omitted or replaced a tile")

    def _validate_rank_assignments(
        self,
        wave: tuple[M5Tile, ...],
        requests: list[tuple[Any, list[tuple[M5Tile, int]]]],
    ) -> None:
        if not isinstance(requests, list) or not requests:
            raise ValueError("placement strategy returned no rank assignments")
        expected_tiles = {id(tile) for tile in wave}
        assigned_tiles: list[int] = []
        assigned_ranks: set[int] = set()
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("rank assignment must be a (rank, assignments) tuple")
            rank, assignments = request
            if not any(rank is candidate for candidate in self.ranks):
                raise ValueError("placement strategy returned a foreign rank object")
            rank_identity = id(rank)
            if rank_identity in assigned_ranks:
                raise ValueError("placement strategy returned a rank more than once")
            assigned_ranks.add(rank_identity)
            if not isinstance(assignments, list) or not assignments:
                raise ValueError("placement strategy returned an empty rank assignment")
            local_ids: set[int] = set()
            for assignment in assignments:
                if not isinstance(assignment, tuple) or len(assignment) != 2:
                    raise TypeError("tile assignment must be a (tile, local_id) tuple")
                tile, local_id = assignment
                if id(tile) not in expected_tiles:
                    raise ValueError(
                        "placement strategy assigned a tile outside the wave"
                    )
                if not isinstance(local_id, int) or isinstance(local_id, bool):
                    raise TypeError("local DPU ID must be an integer")
                if not 0 <= local_id < rank.local_dpus:
                    raise ValueError("local DPU ID is outside the assigned rank")
                if local_id in local_ids:
                    raise ValueError("placement strategy reused a local DPU ID")
                local_ids.add(local_id)
                assigned_tiles.append(id(tile))
        if len(assigned_tiles) != len(set(assigned_tiles)):
            raise ValueError("placement strategy assigned a wave tile more than once")
        if set(assigned_tiles) != expected_tiles:
            raise ValueError("placement strategy omitted a wave tile")

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


__all__ = ["M5WholeCircuitEngine", "M5WholeCircuitSession"]
