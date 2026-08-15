from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from quantum_bench.core.indices import LABEL_LIST_EINSUM_SENTINEL
from quantum_bench.core.records import ContractionTask
from quantum_bench.targets.upmem.execution_plan_v4 import (
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4WorkUnit,
    build_v4_request,
)
from quantum_bench.targets.upmem.m5_strategies import (
    M5DecompositionStrategy,
    M5KernelProvider,
    M5PlacementStrategy,
    M5ReductionProvider,
    M5StrategyBundle,
)
from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
    M5WholeCircuitEngine,
)
from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
    M5Tile,
    M5TileLimits,
    M5TileLowering,
    lower_binary_contraction,
)
from quantum_bench.whole_circuit.core import DeviceTopology
from quantum_bench.whole_circuit.policies import Float32RealPolicy
from quantum_bench.whole_circuit.strategies import StrategyIdentity, StrategyRole


def _strategy_identity(
    role: StrategyRole,
    implementation_id: str,
    *,
    config: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> StrategyIdentity:
    return StrategyIdentity(
        role=role,
        implementation_id=implementation_id,
        version="1",
        provider="test_double",
        transport="in_process_test",
        config=config,
    )


def _task(k: int = 5, *, m: int = 3, n: int = 4) -> ContractionTask:
    return ContractionTask(
        id="fixture",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression=f"{LABEL_LIST_EINSUM_SENTINEL}:fixture",
        input_shapes=((m, k), (k, n)),
        output_shape=(m, n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=m,
        gemm_k=k,
        gemm_n=n,
        structure="dense",
        estimated_flops=0,
        estimated_bytes=0,
    )


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
    event: dict[str, Any] = field(default_factory=lambda: {"returncode": 0})


@dataclass
class _NumpyV4AbiTestDoubleSession:
    profile: Any
    binary_provenance: dict[str, str]
    startup: dict[str, Any] = field(default_factory=dict)
    submissions: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.startup = {
            "event": "READY",
            "status": "ready",
            "target_observed": "physical_hardware",
            "test_double_execution": True,
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "hardware_allocation_verified": True,
            **self.binary_provenance,
        }

    def submit(
        self, artifact: Any, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        del timeout_s
        self.submissions.append(artifact)
        dtype = (
            np.int8
            if self.profile.numeric_mode_name == "host_packed_int8"
            else np.dtype("<f4")
        )
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
                if dtype == np.int8
                else left @ right
            )
            (artifact.root / record.c_path).write_bytes(
                np.asarray(output, dtype="<i4" if dtype == np.int8 else "<f4").tobytes()
            )
        return {
            "status": "completed",
            "target_observed": "physical_hardware",
            "test_double_execution": True,
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
            "transfer": {"h2d_bytes": 10, "d2h_bytes": 5, "total_bytes": 15},
            "timing": {
                "h2d_time_s": 0.01,
                "launch_time_s": 0.02,
                "d2h_time_s": 0.01,
            },
        }

    def close(self, *, timeout_s: float | None = None) -> _Release:
        del timeout_s
        return _Release()


def _create_abi_test_double_engine(
    tmp_path: Path,
    *,
    placement_strategy: Any = None,
    kernel_provider: Any = None,
) -> M5WholeCircuitEngine:
    binaries_dir = tmp_path / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)
    host_binary = binaries_dir / "host"
    dpu_binary = binaries_dir / "dpu"
    init_binary = binaries_dir / "init"
    for p in (host_binary, dpu_binary, init_binary):
        p.write_bytes(p.name.encode("ascii"))
    host_binary.chmod(host_binary.stat().st_mode | 0o100)

    binary_provenance = {
        "host_binary_sha256": hashlib.sha256(host_binary.read_bytes()).hexdigest(),
        "dpu_binary_sha256": hashlib.sha256(dpu_binary.read_bytes()).hexdigest(),
        "initialization_binary_sha256": hashlib.sha256(
            init_binary.read_bytes()
        ).hexdigest(),
    }

    def factory(
        command: object, *, session_root: Path, profile: object
    ) -> _NumpyV4AbiTestDoubleSession:
        return _NumpyV4AbiTestDoubleSession(
            profile=profile, binary_provenance=binary_provenance
        )

    return M5WholeCircuitEngine(
        session_root=tmp_path / "session",
        host_binary=host_binary,
        dpu_binary=dpu_binary,
        initialization_binary=init_binary,
        rank_paths=("/dev/dpu_rank0", "/dev/dpu_rank1"),
        dpu_count=4,
        session_factory=factory,
        placement_strategy=placement_strategy,
        kernel_provider=kernel_provider,
    )


def test_decomposition_strategy_equivalence() -> None:
    task = _task(k=20, m=6, n=8)
    left = np.arange(120, dtype=np.float32).reshape(6, 20)
    right = np.arange(160, dtype=np.float32).reshape(20, 8)

    strategy = M5DecompositionStrategy()
    assert strategy.name == "m5_v4_tile_decomposition"

    # Float32 limits
    lowering_strat = strategy.decompose(task, left, right)
    lowering_direct = lower_binary_contraction(task, left, right)

    assert len(lowering_strat.tiles) == len(lowering_direct.tiles)
    assert lowering_strat.tiles == lowering_direct.tiles
    assert lowering_strat.canonical.m == lowering_direct.canonical.m
    assert lowering_strat.canonical.n == lowering_direct.canonical.n
    assert lowering_strat.canonical.k == lowering_direct.canonical.k
    assert lowering_strat.output_tiles == lowering_direct.output_tiles
    assert lowering_strat.k_chunks == lowering_direct.k_chunks

    # Packed int8 limits
    limits = M5TileLimits.host_packed_int8()
    lowering_strat_packed = strategy.decompose(task, left, right, limits=limits)
    lowering_direct_packed = lower_binary_contraction(task, left, right, limits=limits)

    assert len(lowering_strat_packed.tiles) == len(lowering_direct_packed.tiles)
    assert lowering_strat_packed.tiles == lowering_direct_packed.tiles


def test_placement_strategy_preserves_wave_and_rank_invariants() -> None:
    task = _task(k=100, m=4, n=4)
    left = np.ones((4, 100), dtype=np.float32)
    right = np.ones((100, 4), dtype=np.float32)
    limits = M5TileLimits(max_tile_dim=2, max_elements=4, max_packed_k=32)
    lowering = lower_binary_contraction(task, left, right, limits=limits)

    strategy = M5PlacementStrategy()
    assert strategy.name == "m5_rank_wave_placement"

    for dpu_count in (1, 2, 4, 8):
        waves = strategy.place_waves(lowering.tiles, total_dpu_count=dpu_count)
        assert sum(len(wave) for wave in waves) == len(lowering.tiles)
        assert sorted(tile.id for wave in waves for tile in wave) == sorted(
            tile.id for tile in lowering.tiles
        )
        assert all(len(wave) <= dpu_count for wave in waves)
        assert all(len({tile.k_chunk_id for tile in wave}) == 1 for wave in waves)
        assert [wave[0].k_chunk_id for wave in waves] == sorted(
            (wave[0].k_chunk_id for wave in waves),
            key=lambda value: int(value.removeprefix("k_")),
        )

    ranks = tuple(type("Rank", (), {"local_dpus": 2})() for _ in range(2))
    assignments = strategy.map_wave_to_ranks(lowering.tiles[:3], ranks)
    assert [tile for _, items in assignments for tile, _ in items] == list(
        lowering.tiles[:3]
    )
    assert [local_id for _, items in assignments for _, local_id in items] == [0, 1, 0]


def test_kernel_provider_work_unit_bytes_equivalence_to_session() -> None:
    provider = M5KernelProvider()
    assert provider.name == "upmem_sdk_hardware_v4_tile_kernel"

    task = _task(k=8, m=4, n=4)
    left_f32 = np.arange(32, dtype=np.float32).reshape(4, 8)
    right_f32 = np.arange(32, dtype=np.float32).reshape(8, 4)
    lowering = lower_binary_contraction(task, left_f32, right_f32)
    tile = lowering.tiles[0]

    # Float32 mode
    unit_strat_f32 = provider.build_work_unit(
        tile, 2, lowering.canonical.left, lowering.canonical.right, packed=False
    )
    expected_left_f32 = np.ascontiguousarray(
        lowering.canonical.left[0, : tile.m_size, : tile.k_size]
    )
    expected_right_f32 = np.ascontiguousarray(
        lowering.canonical.right[0, : tile.k_size, : tile.n_size]
    )
    assert unit_strat_f32.a_payload == expected_left_f32.astype("<f4").tobytes()
    assert unit_strat_f32.b_payload == expected_right_f32.astype("<f4").tobytes()
    assert unit_strat_f32.local_dpu_id == 2
    assert unit_strat_f32.m_offset == tile.m_start
    assert unit_strat_f32.n_offset == tile.n_start
    assert unit_strat_f32.k_offset == tile.k_start
    assert unit_strat_f32.m_elements == tile.m_size
    assert unit_strat_f32.n_elements == tile.n_size
    assert unit_strat_f32.k_elements == tile.k_size

    # Packed int8 mode (3D canonical layout: 1, 4, 8 and 1, 8, 4)
    left_i8 = np.arange(32, dtype=np.int8).reshape(1, 4, 8)
    right_i8 = np.arange(32, dtype=np.int8).reshape(1, 8, 4)
    unit_strat_i8 = provider.build_work_unit(tile, 3, left_i8, right_i8, packed=True)
    assert unit_strat_i8.a_payload == left_i8[0, : tile.m_size, : tile.k_size].tobytes()
    assert (
        unit_strat_i8.b_payload == right_i8[0, : tile.k_size, : tile.n_size].tobytes()
    )
    assert unit_strat_i8.local_dpu_id == 3


@pytest.mark.parametrize(
    ("packed", "numeric_mode"),
    ((False, NUMERIC_FLOAT32), (True, NUMERIC_HOST_PACKED_INT8)),
)
def test_kernel_provider_request_artifact_matches_v4_builder(
    tmp_path: Path, packed: bool, numeric_mode: str
) -> None:
    provider = M5KernelProvider()
    task = _task(k=5, m=3, n=4)
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    lowering = lower_binary_contraction(task, left, right)
    tile = lowering.tiles[0]
    left_payload = lowering.canonical.left
    right_payload = lowering.canonical.right
    if packed:
        left_payload = left_payload.astype(np.int8)
        right_payload = right_payload.astype(np.int8)
    unit = provider.build_work_unit(tile, 0, left_payload, right_payload, packed=packed)
    profile = V4Profile(dpu_count=1, numeric_mode=numeric_mode)
    contract = "ab" * 32
    via_provider = provider.prepare_request(
        tmp_path / "provider",
        profile=profile,
        lowering=lowering,
        work_units=[unit],
        task_contract_sha256=contract,
        request_sequence=7,
    )
    direct = build_v4_request(
        tmp_path / "direct",
        profile=profile,
        canonical_batch_count=lowering.canonical.b,
        canonical_m=lowering.canonical.m,
        canonical_n=lowering.canonical.n,
        canonical_k=lowering.canonical.k,
        work_units=[unit],
        task_contract_sha256=contract,
        request_sequence=7,
    )

    assert via_provider.header.pack() == direct.header.pack()
    assert [record.pack() for record in via_provider.work_units] == [
        record.pack() for record in direct.work_units
    ]
    assert via_provider.manifest_path.read_bytes() == direct.manifest_path.read_bytes()
    assert via_provider.sidecar_path.read_bytes() == direct.sidecar_path.read_bytes()
    for provider_record, direct_record in zip(
        via_provider.work_units, direct.work_units, strict=True
    ):
        assert (via_provider.root / provider_record.a_path).read_bytes() == (
            direct.root / direct_record.a_path
        ).read_bytes()
        assert (via_provider.root / provider_record.b_path).read_bytes() == (
            direct.root / direct_record.b_path
        ).read_bytes()


def test_kernel_provider_output_decode_equivalence_to_session(tmp_path: Path) -> None:
    provider = M5KernelProvider()
    task = _task(k=5, m=3, n=4)
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.arange(20, dtype=np.float32).reshape(5, 4)
    lowering = lower_binary_contraction(task, left, right)
    tile = lowering.tiles[0]

    # Float32 output decode
    f32_data = np.arange(12, dtype="<f4")
    f32_file = tmp_path / "f32_out.bin"
    f32_file.write_bytes(f32_data.tobytes())

    decoded_f32 = provider.read_output(f32_file, tile, packed=False)
    np.testing.assert_array_equal(decoded_f32, f32_data.reshape(3, 4))
    assert decoded_f32.dtype == np.float64

    # Packed int8 (int32 buffer) output decode
    i32_data = np.arange(12, dtype="<i4") * 10
    i32_file = tmp_path / "i32_out.bin"
    i32_file.write_bytes(i32_data.tobytes())

    decoded_i8 = provider.read_output(i32_file, tile, packed=True)
    np.testing.assert_array_equal(decoded_i8, i32_data.reshape(3, 4))
    assert decoded_i8.dtype == np.int64

    # Truncation check
    trunc_file = tmp_path / "trunc.bin"
    trunc_file.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="truncated"):
        provider.read_output(trunc_file, tile, packed=False)


def test_reduction_provider_output_equivalence() -> None:
    provider = M5ReductionProvider()
    assert provider.name == "m5_tile_host_reduction"

    task = _task(k=16, m=4, n=4)
    left = np.arange(64, dtype=np.float32).reshape(4, 16)
    right = np.arange(64, dtype=np.float32).reshape(16, 4)
    limits = M5TileLimits(max_tile_dim=2, max_elements=4, max_packed_k=8)
    lowering = lower_binary_contraction(task, left, right, limits=limits)

    partials: dict[str, np.ndarray] = {}
    for tile in lowering.tiles:
        left_sub, right_sub = lowering.extract_tile_operands(tile)
        partials[tile.id] = left_sub @ right_sub

    # Float32 reduction
    expected_f32 = left @ right
    res_f32 = provider.reduce(lowering, partials, packed=False)
    np.testing.assert_allclose(res_f32, expected_f32, rtol=1e-5)

    # Packed int8 reduction with scale
    scale = 0.125
    res_packed = provider.reduce(lowering, partials, packed=True, scale=scale)
    np.testing.assert_allclose(res_packed, expected_f32 * scale, rtol=1e-5)


def test_default_and_explicit_adapters_preserve_test_double_abi_metadata(
    tmp_path: Path,
) -> None:
    engine = _create_abi_test_double_engine(tmp_path)
    topology = DeviceTopology(backend="upmem", device_ids=("d0", "d1", "d2", "d3"))
    task = _task(k=10, m=3, n=4)
    left = np.arange(30, dtype=np.float32).reshape(3, 10)
    right = np.arange(40, dtype=np.float32).reshape(10, 4)

    # 1. Default session
    session_default = engine.open_session(Float32RealPolicy(), topology)
    res_default = session_default.execute(task, left, right)
    term_default = session_default.close()

    # 2. Session with explicit strategy adapter instances
    decomp_strat = M5DecompositionStrategy()
    place_strat = M5PlacementStrategy()
    kernel_prov = M5KernelProvider()
    reduc_prov = M5ReductionProvider()

    session_explicit = engine.open_session(
        Float32RealPolicy(),
        topology,
        decomposition_strategy=decomp_strat,
        placement_strategy=place_strat,
        kernel_provider=kernel_prov,
        reduction_provider=reduc_prov,
    )
    res_explicit = session_explicit.execute(task, left, right)
    term_explicit = session_explicit.close()

    np.testing.assert_allclose(res_default.output, res_explicit.output)
    assert (
        res_default.metadata["strategy_identity"]
        == res_explicit.metadata["strategy_identity"]
    )
    identity = res_default.metadata["strategy_identity"]
    assert identity["schema_version"] == "strategy_configuration_v2"
    by_role = {item["role"]: item for item in identity["strategies"]}
    assert by_role["decomposition"]["implementation_id"] == "m5_v4_tile_decomposition"
    assert by_role["placement"]["implementation_id"] == "m5_rank_wave_placement"
    assert by_role["kernel"]["implementation_id"] == "upmem_sdk_hardware_v4_tile_kernel"
    assert by_role["kernel"]["provider"] == "raw_upmem_sdk_v4"
    assert by_role["reduction"]["implementation_id"] == "m5_tile_host_reduction"
    assert by_role["reduction"]["provider"] == "quantum_bench_host"
    assert (
        res_default.metadata["decomposition_strategy"]
        == res_explicit.metadata["decomposition_strategy"]
        == "m5_v4_tile_decomposition"
    )
    assert (
        res_default.metadata["placement_strategy"]
        == res_explicit.metadata["placement_strategy"]
        == "m5_rank_wave_placement"
    )
    assert (
        res_default.metadata["kernel_provider"]
        == res_explicit.metadata["kernel_provider"]
        == "upmem_sdk_hardware_v4_tile_kernel"
    )
    assert (
        res_default.metadata["reduction_provider"]
        == res_explicit.metadata["reduction_provider"]
        == "m5_tile_host_reduction"
    )
    assert (
        res_default.metadata["reduction_strategy"]
        == res_explicit.metadata["reduction_strategy"]
        == "m5_tile_host_reduction"
    )
    assert (
        res_default.metadata["strategy_config_hash"]
        == res_explicit.metadata["strategy_config_hash"]
        == engine.strategy_config_hash
    )

    assert term_default["strategy_identity"] == term_explicit["strategy_identity"]
    assert term_default["strategy_config_hash"] == term_explicit["strategy_config_hash"]

    for result, terminal in (
        (res_default, term_default),
        (res_explicit, term_explicit),
    ):
        assert result.metadata["test_double_execution"] is True
        assert result.metadata["target_observed"] == "not_verified"
        assert terminal["physical_target_verified"] is False
        assert terminal["native_kernel_executed"] is False


def test_custom_strategy_injection_direct_dispatch(tmp_path: Path) -> None:
    engine = _create_abi_test_double_engine(tmp_path)
    topology = DeviceTopology(backend="upmem", device_ids=("d0", "d1", "d2", "d3"))
    task = _task(k=6, m=2, n=2)
    left = np.ones((2, 6), dtype=np.float32)
    right = np.ones((6, 2), dtype=np.float32)

    invocations: list[str] = []

    default_session = engine.open_session(Float32RealPolicy(), topology)
    default_result = default_session.execute(task, left, right)
    default_session.close()

    @dataclass(frozen=True)
    class TrackingDecomposition:
        name: str = "custom_tracking_decomp"

        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.DECOMPOSITION, self.name)

        def decompose(
            self,
            task: ContractionTask,
            left: np.ndarray,
            right: np.ndarray,
            *,
            limits: M5TileLimits | None = None,
        ) -> M5TileLowering:
            invocations.append("decompose")
            return lower_binary_contraction(
                task, left, right, limits=limits or M5TileLimits()
            )

    @dataclass(frozen=True)
    class TrackingPlacement:
        name: str = "custom_tracking_placement"

        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.PLACEMENT, self.name)

        def place_waves(
            self,
            tiles: tuple[M5Tile, ...],
            total_dpu_count: int,
        ) -> tuple[tuple[M5Tile, ...], ...]:
            invocations.append("place_waves")
            return M5PlacementStrategy().place_waves(tiles, total_dpu_count)

        def map_wave_to_ranks(
            self,
            wave: tuple[M5Tile, ...],
            ranks: tuple[Any, ...],
        ) -> list[tuple[Any, list[tuple[M5Tile, int]]]]:
            invocations.append("map_wave_to_ranks")
            return M5PlacementStrategy().map_wave_to_ranks(wave, ranks)

    @dataclass(frozen=True)
    class TrackingKernel:
        name: str = "custom_tracking_kernel"

        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.KERNEL, self.name)

        def build_work_unit(
            self,
            tile: M5Tile,
            local_id: int,
            left: np.ndarray,
            right: np.ndarray,
            packed: bool,
        ) -> V4WorkUnit:
            invocations.append("build_work_unit")
            return M5KernelProvider().build_work_unit(
                tile, local_id, left, right, packed
            )

        def prepare_request(
            self,
            root: Path,
            *,
            profile: Any,
            lowering: M5TileLowering,
            work_units: list[V4WorkUnit],
            task_contract_sha256: str,
            request_sequence: int,
        ) -> Any:
            invocations.append("prepare_request")
            return M5KernelProvider().prepare_request(
                root,
                profile=profile,
                lowering=lowering,
                work_units=work_units,
                task_contract_sha256=task_contract_sha256,
                request_sequence=request_sequence,
            )

        def read_output(
            self,
            path: Path,
            tile: M5Tile,
            *,
            packed: bool,
        ) -> np.ndarray:
            invocations.append("read_output")
            return M5KernelProvider().read_output(path, tile, packed=packed)

    @dataclass(frozen=True)
    class TrackingReduction:
        name: str = "custom_tracking_reduc"

        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.REDUCTION, self.name)

        def reduce(
            self,
            lowering: M5TileLowering,
            partials: Mapping[str, np.ndarray],
            *,
            packed: bool = False,
            scale: float = 1.0,
        ) -> np.ndarray:
            invocations.append("reduce")
            return M5ReductionProvider().reduce(
                lowering, partials, packed=packed, scale=scale
            )

    session = engine.open_session(
        Float32RealPolicy(),
        topology,
        decomposition_strategy=TrackingDecomposition(),
        placement_strategy=TrackingPlacement(),
        kernel_provider=TrackingKernel(),
        reduction_provider=TrackingReduction(),
    )
    result = session.execute(task, left, right)
    session.close()

    assert "decompose" in invocations
    assert "place_waves" in invocations
    assert "map_wave_to_ranks" in invocations
    assert "build_work_unit" in invocations
    assert "prepare_request" in invocations
    assert "read_output" in invocations
    assert "reduce" in invocations

    assert result.metadata["decomposition_strategy"] == "custom_tracking_decomp"
    assert result.metadata["placement_strategy"] == "custom_tracking_placement"
    assert result.metadata["kernel_provider"] == "custom_tracking_kernel"
    assert result.metadata["reduction_provider"] == "custom_tracking_reduc"
    assert result.metadata["strategy_config_hash"] != engine.strategy_config_hash
    assert (
        result.metadata["request_contract_sha256"]
        != default_result.metadata["request_contract_sha256"]
    )
    assert (
        result.metadata["task_structure_sha256"]
        == default_result.metadata["task_structure_sha256"]
    )


class _GuardKernelProvider:
    def __init__(self) -> None:
        self.delegate = M5KernelProvider()
        self.prepare_calls = 0

    def identity(self) -> StrategyIdentity:
        return _strategy_identity(StrategyRole.KERNEL, "guard_v4_kernel")

    def build_work_unit(
        self,
        tile: M5Tile,
        local_id: int,
        left: np.ndarray,
        right: np.ndarray,
        packed: bool,
    ) -> V4WorkUnit:
        return self.delegate.build_work_unit(tile, local_id, left, right, packed)

    def prepare_request(
        self,
        root: Path,
        *,
        profile: Any,
        lowering: M5TileLowering,
        work_units: list[V4WorkUnit],
        task_contract_sha256: str,
        request_sequence: int,
    ) -> Any:
        self.prepare_calls += 1
        return self.delegate.prepare_request(
            root,
            profile=profile,
            lowering=lowering,
            work_units=work_units,
            task_contract_sha256=task_contract_sha256,
            request_sequence=request_sequence,
        )

    def read_output(self, path: Path, tile: M5Tile, *, packed: bool) -> np.ndarray:
        return self.delegate.read_output(path, tile, packed=packed)


@dataclass(frozen=True)
class _AdversarialPlacement:
    mode: str

    def identity(self) -> StrategyIdentity:
        return _strategy_identity(
            StrategyRole.PLACEMENT,
            "adversarial_placement",
            config=(("mode", self.mode),),
        )

    def place_waves(
        self, tiles: tuple[M5Tile, ...], total_dpu_count: int
    ) -> tuple[tuple[M5Tile, ...], ...]:
        waves = M5PlacementStrategy().place_waves(tiles, total_dpu_count)
        if self.mode == "omit_wave":
            return waves[:-1]
        if self.mode == "duplicate_wave_tile":
            return ((waves[0][0], waves[0][0], *waves[0][1:]), *waves[1:])
        return waves

    def map_wave_to_ranks(
        self, wave: tuple[M5Tile, ...], ranks: tuple[Any, ...]
    ) -> list[tuple[Any, list[tuple[M5Tile, int]]]]:
        requests = M5PlacementStrategy().map_wave_to_ranks(wave, ranks)
        if self.mode == "foreign_rank":
            foreign = type("ForeignRank", (), {"local_dpus": ranks[0].local_dpus})()
            return [(foreign, requests[0][1]), *requests[1:]]
        rank, assignments = requests[0]
        if self.mode == "duplicate_local_id":
            changed = list(assignments)
            changed[1] = (changed[1][0], changed[0][1])
            return [(rank, changed), *requests[1:]]
        if self.mode == "out_of_range_local_id":
            changed = list(assignments)
            changed[0] = (changed[0][0], rank.local_dpus)
            return [(rank, changed), *requests[1:]]
        if self.mode == "omit_assignment":
            return [(rank, assignments[:-1]), *requests[1:]]
        if self.mode == "duplicate_assignment":
            return [(rank, [assignments[0], assignments[0]]), *requests[1:]]
        return requests


def test_invalid_placement_fails_before_request_creation(tmp_path: Path) -> None:
    modes = (
        "omit_wave",
        "duplicate_wave_tile",
        "foreign_rank",
        "duplicate_local_id",
        "out_of_range_local_id",
        "omit_assignment",
        "duplicate_assignment",
    )
    topology = DeviceTopology(backend="upmem", device_ids=("d0", "d1", "d2", "d3"))
    task = _task(k=2, m=300, n=2)
    left = np.ones((300, 2), dtype=np.float32)
    right = np.ones((2, 2), dtype=np.float32)
    for mode in modes:
        guard = _GuardKernelProvider()
        engine = _create_abi_test_double_engine(
            tmp_path / mode,
            placement_strategy=_AdversarialPlacement(mode),
            kernel_provider=guard,
        )
        session = engine.open_session(Float32RealPolicy(), topology)
        try:
            with pytest.raises(
                (TypeError, ValueError), match="placement|DPU|rank|tile"
            ):
                session.execute(task, left, right)
            assert guard.prepare_calls == 0
            assert all(not rank.session.submissions for rank in session.ranks)
        finally:
            session.close()


def test_strategy_bundle_identity_and_defaults() -> None:
    bundle = M5StrategyBundle.default()
    assert isinstance(bundle.decomposition, M5DecompositionStrategy)
    assert isinstance(bundle.placement, M5PlacementStrategy)
    assert isinstance(bundle.kernel, M5KernelProvider)
    assert isinstance(bundle.reduction, M5ReductionProvider)

    ident = bundle.to_identity_dict()
    assert ident["schema_version"] == "strategy_configuration_v2"
    assert [item["role"] for item in ident["strategies"]] == [
        "decomposition",
        "kernel",
        "placement",
        "reduction",
    ]
    assert len(bundle.config_hash) == 64
    assert bundle.config_hash == M5StrategyBundle.default().config_hash
    with pytest.raises(TypeError, match="decomposition.*identity"):
        M5StrategyBundle(
            decomposition=object(),
            placement=bundle.placement,
            kernel=bundle.kernel,
            reduction=bundle.reduction,
        ).identity_configuration()


def test_same_claimed_identity_uses_distinct_implementation_evidence(
    tmp_path: Path,
) -> None:
    class FirstDecomposition(M5DecompositionStrategy):
        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.DECOMPOSITION, "shared_claim")

    class SecondDecomposition(M5DecompositionStrategy):
        def identity(self) -> StrategyIdentity:
            return _strategy_identity(StrategyRole.DECOMPOSITION, "shared_claim")

    first = FirstDecomposition()
    second = SecondDecomposition()
    assert first.identity() == second.identity()

    default_bundle = M5StrategyBundle.default()
    first_bundle = M5StrategyBundle(
        decomposition=first,
        placement=default_bundle.placement,
        kernel=default_bundle.kernel,
        reduction=default_bundle.reduction,
    )
    second_bundle = M5StrategyBundle(
        decomposition=second,
        placement=default_bundle.placement,
        kernel=default_bundle.kernel,
        reduction=default_bundle.reduction,
    )
    assert first_bundle.config_hash != second_bundle.config_hash

    topology = DeviceTopology(backend="upmem", device_ids=("d0", "d1", "d2", "d3"))
    task = _task(k=6, m=2, n=2)
    left = np.ones((2, 6), dtype=np.float32)
    right = np.ones((6, 2), dtype=np.float32)
    first_engine = _create_abi_test_double_engine(tmp_path / "first")
    second_engine = _create_abi_test_double_engine(tmp_path / "second")
    first_session = first_engine.open_session(
        Float32RealPolicy(), topology, decomposition_strategy=first
    )
    second_session = second_engine.open_session(
        Float32RealPolicy(), topology, decomposition_strategy=second
    )
    try:
        first_result = first_session.execute(task, left, right)
        second_result = second_session.execute(task, left, right)
    finally:
        first_session.close()
        second_session.close()

    first_identity = first_result.metadata["strategy_identity"]
    second_identity = second_result.metadata["strategy_identity"]
    first_decomposition = next(
        item for item in first_identity["strategies"] if item["role"] == "decomposition"
    )
    second_decomposition = next(
        item
        for item in second_identity["strategies"]
        if item["role"] == "decomposition"
    )
    assert (
        first_decomposition["module_source_sha256"]
        == second_decomposition["module_source_sha256"]
    )
    assert (
        first_decomposition["implementation_type"]
        != second_decomposition["implementation_type"]
    )
    assert (
        first_result.metadata["strategy_config_hash"]
        != second_result.metadata["strategy_config_hash"]
    )
    assert (
        first_result.metadata["request_contract_sha256"]
        != second_result.metadata["request_contract_sha256"]
    )
