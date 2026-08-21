"""UPMEM M5 strategy adapters for whole-circuit execution.

These strategies delegate directly to the existing deterministic lowering,
wave placement, work-unit serialization, output decoding, and tile assembly
routines.

Truthful identity: raw SDK hardware v4 tile execution kernel (dpu_gemm_tile_v4).
Does NOT claim SimplePIM, PID-Comm, or ATiM offload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_bench.targets.upmem.execution_plan_v4 import V4WorkUnit, build_v4_request
from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
    M5Tile,
    M5TileLimits,
    M5TileLowering,
    lower_binary_contraction,
    order_tile_waves,
)
from quantum_bench.tn.graph import ContractNode
from quantum_bench.whole_circuit.strategies import (
    DecompositionStrategy,
    KernelProvider,
    PlacementStrategy,
    ReductionProvider,
    StrategyConfiguration,
    StrategyIdentity,
    StrategyRole,
    bind_strategy_identity,
)


@dataclass(frozen=True)
class M5DecompositionStrategy:
    """Decompose one semantic contract node into bounded physical tiles."""

    name: str = "m5_v4_tile_decomposition"

    def identity(self) -> StrategyIdentity:
        return StrategyIdentity(
            role=StrategyRole.DECOMPOSITION,
            implementation_id=self.name,
            version="1",
            provider="quantum_bench_host",
            transport="host_control",
            config=(("limits_source", "numeric_policy"),),
        )

    def decompose(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        limits: M5TileLimits | None = None,
    ) -> M5TileLowering:
        if limits is None:
            limits = M5TileLimits()
        return lower_binary_contraction(node, left, right, limits=limits)


@dataclass(frozen=True)
class M5PlacementStrategy:
    """M5 strategy for partitioning tiles into sequential waves and mapping to rank DPUs."""

    name: str = "m5_rank_wave_placement"

    def identity(self) -> StrategyIdentity:
        return StrategyIdentity(
            role=StrategyRole.PLACEMENT,
            implementation_id=self.name,
            version="1",
            provider="quantum_bench_host",
            transport="host_control",
            config=(
                ("local_dpu_order", "contiguous"),
                ("wave_partition", "k_chunk_then_width"),
            ),
        )

    def place_waves(
        self,
        tiles: tuple[M5Tile, ...],
        total_dpu_count: int,
    ) -> tuple[tuple[M5Tile, ...], ...]:
        """Group tiles into waves such that K-chunks are separated and rank capacity is respected."""
        return order_tile_waves(tiles, total_dpu_count)

    def map_wave_to_ranks(
        self,
        wave: tuple[M5Tile, ...],
        ranks: tuple[Any, ...],
    ) -> list[tuple[Any, list[tuple[M5Tile, int]]]]:
        """Map wave tiles to rank instances and local DPU IDs."""
        requests: list[tuple[Any, list[tuple[M5Tile, int]]]] = []
        local_width = ranks[0].local_dpus
        for global_slot, tile in enumerate(wave):
            rank = ranks[global_slot // local_width]
            local_id = global_slot % local_width
            found = next((item for item in requests if item[0] is rank), None)
            if found is None:
                found = (rank, [])
                requests.append(found)
            found[1].append((tile, local_id))
        return requests


@dataclass(frozen=True)
class M5KernelProvider:
    """M5 strategy for preparing work units and decoding output.

    Identified truthfully as raw SDK hardware v4 tile execution kernel (dpu_gemm_tile_v4).
    Does NOT claim SimplePIM, PID-Comm, or ATiM offload.
    """

    name: str = "upmem_sdk_hardware_v4_tile_kernel"

    def identity(self) -> StrategyIdentity:
        return StrategyIdentity(
            role=StrategyRole.KERNEL,
            implementation_id=self.name,
            version="1",
            provider="raw_upmem_sdk_v4",
            transport="application_visible_sdk_transfer",
            config=(("abi", "execution_plan_v4"), ("kernel", "dpu_gemm_tile_v4")),
        )

    def build_work_unit(
        self,
        tile: M5Tile,
        local_id: int,
        left: np.ndarray,
        right: np.ndarray,
        packed: bool,
    ) -> V4WorkUnit:
        left_arr = (
            left
            if left.ndim == 3
            else np.reshape(left, (1, left.shape[0], left.shape[1]))
        )
        right_arr = (
            right
            if right.ndim == 3
            else np.reshape(right, (1, right.shape[0], right.shape[1]))
        )
        left_tile = np.ascontiguousarray(
            left_arr[
                tile.batch_index,
                tile.m_start : tile.m_start + tile.m_size,
                tile.k_start : tile.k_start + tile.k_size,
            ]
        )
        right_tile = np.ascontiguousarray(
            right_arr[
                tile.batch_index,
                tile.k_start : tile.k_start + tile.k_size,
                tile.n_start : tile.n_start + tile.n_size,
            ]
        )
        dtype = np.int8 if packed else np.dtype("<f4")
        return V4WorkUnit(
            local_dpu_id=local_id,
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

    def read_output(
        self,
        path: Path,
        tile: M5Tile,
        *,
        packed: bool,
    ) -> np.ndarray:
        dtype = np.dtype("<i4") if packed else np.dtype("<f4")
        expected = tile.output_element_count
        raw = Path(path).read_bytes()
        if len(raw) < expected * dtype.itemsize:
            raise RuntimeError(f"v4 output is truncated: {path}")
        values = np.frombuffer(raw[: expected * dtype.itemsize], dtype=dtype)
        return np.asarray(
            values.reshape(tile.m_size, tile.n_size),
            dtype=np.int64 if packed else np.float64,
        )

    def prepare_request(
        self,
        root: Any,
        *,
        profile: Any,
        lowering: M5TileLowering,
        work_units: list[V4WorkUnit],
        task_contract_sha256: str,
        request_sequence: int,
    ) -> Any:
        return build_v4_request(
            root,
            profile=profile,
            canonical_batch_count=lowering.canonical.b,
            canonical_m=lowering.canonical.m,
            canonical_n=lowering.canonical.n,
            canonical_k=lowering.canonical.k,
            work_units=work_units,
            task_contract_sha256=task_contract_sha256,
            request_sequence=request_sequence,
        )


@dataclass(frozen=True)
class M5ReductionProvider:
    """M5 strategy for assembling tile partial outputs into canonical host tensor."""

    name: str = "m5_tile_host_reduction"

    def identity(self) -> StrategyIdentity:
        return StrategyIdentity(
            role=StrategyRole.REDUCTION,
            implementation_id=self.name,
            version="1",
            provider="quantum_bench_host",
            transport="host_memory",
            config=(
                ("accumulator", "int64_packed_or_float64_float32"),
                ("location", "host"),
            ),
        )

    def reduce(
        self,
        lowering: M5TileLowering,
        partials: Mapping[str, np.ndarray],
        *,
        packed: bool = False,
        scale: float = 1.0,
    ) -> np.ndarray:
        output_dtype: np.dtype[Any] = np.int64 if packed else np.float64
        canonical = lowering.assemble(partials, dtype=output_dtype)
        if packed:
            return np.asarray(canonical, dtype=np.float32) * np.float32(scale)
        return np.asarray(canonical, dtype=np.float32)


@dataclass(frozen=True)
class M5StrategyBundle:
    """Typed bundle of M5 strategy adapters."""

    decomposition: DecompositionStrategy
    placement: PlacementStrategy
    kernel: KernelProvider
    reduction: ReductionProvider

    @classmethod
    def default(cls) -> M5StrategyBundle:
        return cls(
            decomposition=M5DecompositionStrategy(),
            placement=M5PlacementStrategy(),
            kernel=M5KernelProvider(),
            reduction=M5ReductionProvider(),
        )

    def identity_configuration(self) -> StrategyConfiguration:
        strategies = (
            (StrategyRole.DECOMPOSITION, self.decomposition),
            (StrategyRole.PLACEMENT, self.placement),
            (StrategyRole.KERNEL, self.kernel),
            (StrategyRole.REDUCTION, self.reduction),
        )
        identities: list[StrategyIdentity] = []
        for expected_role, strategy in strategies:
            try:
                identity_method = strategy.identity
            except AttributeError as exc:
                raise TypeError(
                    f"strategy for {expected_role.value} must define identity()"
                ) from exc
            identity = identity_method()
            if not isinstance(identity, StrategyIdentity):
                raise TypeError("strategy identity() must return StrategyIdentity")
            if identity.role is not expected_role:
                raise ValueError(
                    f"strategy for {expected_role.value} returned role {identity.role.value}"
                )
            identities.append(bind_strategy_identity(strategy, identity))
        return StrategyConfiguration(
            tuple(sorted(identities, key=lambda identity: identity.role.value))
        )

    def to_identity_dict(self) -> dict[str, Any]:
        return self.identity_configuration().to_record()

    @property
    def config_hash(self) -> str:
        return self.identity_configuration().sha256


__all__ = [
    "M5DecompositionStrategy",
    "M5PlacementStrategy",
    "M5KernelProvider",
    "M5ReductionProvider",
    "M5StrategyBundle",
]
