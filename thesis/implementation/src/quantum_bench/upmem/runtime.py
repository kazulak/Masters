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
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from quantum_bench.lowering import (
    contraction_dag_hash,
    validate_contraction_dag,
    validate_dag_inputs,
)
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode
from quantum_bench.numerics import (
    NumericPolicy,
    decode_complex_products,
    encode_complex_tensor,
)
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    JsonValue,
    Measurement,
    UnsupportedExecution,
)
from quantum_bench.upmem.plan import (
    UpmemStage,
    UpmemPlan as FinalUpmemPlan,
    UpmemResources,
    UpmemTopology as FinalUpmemTopology,
    UpmemWorkUnit,
    collection_resource_admission,
    physical_plan_id,
    validate_upmem_plan,
)
from quantum_bench.upmem.native_session import V4Session
from quantum_bench.upmem.packed_operation import (
    PACKED_OPERATION_TRANSPORT,
    PackedOperation,
    build_packed_v4_request,
    pack_operation,
)
from quantum_bench.upmem.protocol import (
    EXECUTION_TARGET_PHYSICAL,
    EXECUTION_TARGET_SIMULATOR,
    INT8_MAX_PRODUCT,
    MAX_INT32_SAFE_K,
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4ProtocolError,
    V4WorkUnit,
    WRAM_PANEL_DMA_BYTES,
    WRAM_PANEL_KC,
    WRAM_PANEL_NC,
    WRAM_PANEL_UNALIGNED_SCRATCH_BYTES,
    _record_abi_fields,
    native_execution_identity,
)
from quantum_bench.upmem.tiling import (
    M5Tile,
    M5TileLimits,
    M5TileLowering,
    lower_binary_contraction,
)

_INT64_MAX = (1 << 63) - 1

_NUMERIC_POLICY_FLOAT32 = "split_complex_float32_v1"
_NUMERIC_POLICY_INT8 = "complex_int8_shared_scale_v1"
_WRAM_PANEL_LANE_COUNT = 4
_WRAM_PANEL_OUTPUT_BYTES_PER_ELEMENT = 4
_WRAM_PANEL_A_BUFFER_BYTES = WRAM_PANEL_KC * 4
_WRAM_PANEL_OUTPUT_BUFFER_BYTES = WRAM_PANEL_NC * _WRAM_PANEL_OUTPUT_BYTES_PER_ELEMENT
_WRAM_PANEL_SHARED_BUFFER_BYTES = WRAM_PANEL_KC * WRAM_PANEL_NC * 4
_WRAM_PANEL_PRIVATE_BYTES_PER_TASKLET = (
    _WRAM_PANEL_A_BUFFER_BYTES
    + _WRAM_PANEL_OUTPUT_BUFFER_BYTES
    + WRAM_PANEL_UNALIGNED_SCRATCH_BYTES
)


def _validate_numeric_policy(policy: NumericPolicy) -> NumericPolicy:
    if policy not in {_NUMERIC_POLICY_FLOAT32, _NUMERIC_POLICY_INT8}:
        raise ValueError(f"unsupported final UPMEM numeric policy: {policy!r}")
    return policy


def _is_packed_policy(policy: NumericPolicy) -> bool:
    return _validate_numeric_policy(policy) == _NUMERIC_POLICY_INT8


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _aligned_transfer_span(offset: int, payload_bytes: int) -> int:
    """Return the aligned MRAM span implied by one source-level helper call."""

    return _align8((offset & 7) + payload_bytes)


def _wram_panel_operation_facts(
    work_units: tuple[UpmemWorkUnit, ...],
    *,
    numeric_policy: NumericPolicy,
    tasklets_per_dpu: int,
) -> dict[str, int | str]:
    """Derive source-level movement facts for four sequential real products.

    The values count calls made by the frozen WRAM-panel algorithm. They are
    not device counters: unaligned SDK helpers may issue additional internal
    transactions, so their aligned byte total is deliberately an estimate.
    """

    packed = _is_packed_policy(numeric_policy)
    input_bytes = 1 if packed else 4
    b_read_calls = b_read_payload_bytes = b_read_aligned_bytes = 0
    a_read_calls = a_read_payload_bytes = a_read_aligned_bytes = 0
    partial_c_read_calls = partial_c_read_payload_bytes = partial_c_read_aligned_bytes = 0
    c_write_calls = c_write_payload_bytes = c_write_aligned_bytes = 0
    barrier_events = 0
    real_macs = 0

    for unit in work_units:
        m_size = unit.m_size
        n_size = unit.n_size
        k_size = unit.k_size
        a_bytes = _align8(m_size * k_size * input_bytes)
        b_offset = a_bytes
        b_bytes = _align8(k_size * n_size * input_bytes)
        c_offset = b_offset + b_bytes
        n_panel_count = (n_size + WRAM_PANEL_NC - 1) // WRAM_PANEL_NC
        k_panel_count = (k_size + WRAM_PANEL_KC - 1) // WRAM_PANEL_KC
        barrier_events += 4 + 2 * n_panel_count * k_panel_count
        real_macs += m_size * n_size * k_size

        for n_start in range(0, n_size, WRAM_PANEL_NC):
            actual_n = min(WRAM_PANEL_NC, n_size - n_start)
            for k_start in range(0, k_size, WRAM_PANEL_KC):
                actual_k = min(WRAM_PANEL_KC, k_size - k_start)
                b_panel_bytes = actual_k * actual_n * input_bytes
                full_b_panel = (
                    actual_k == WRAM_PANEL_KC
                    and actual_n == WRAM_PANEL_NC
                    and n_size == WRAM_PANEL_NC
                )
                if full_b_panel:
                    calls = (
                        b_panel_bytes + WRAM_PANEL_DMA_BYTES - 1
                    ) // WRAM_PANEL_DMA_BYTES
                    b_read_calls += calls
                    b_read_payload_bytes += b_panel_bytes
                    b_read_aligned_bytes += b_panel_bytes
                else:
                    for k_index in range(actual_k):
                        offset = b_offset + (
                            (k_start + k_index) * n_size + n_start
                        ) * input_bytes
                        b_read_calls += 1
                        b_read_payload_bytes += actual_n * input_bytes
                        b_read_aligned_bytes += _aligned_transfer_span(
                            offset, actual_n * input_bytes
                        )

                for row in range(m_size):
                    a_offset = (row * k_size + k_start) * input_bytes
                    a_payload = actual_k * input_bytes
                    a_read_calls += 1
                    a_read_payload_bytes += a_payload
                    a_read_aligned_bytes += _aligned_transfer_span(a_offset, a_payload)

                    c_row_offset = c_offset + (
                        row * n_size + n_start
                    ) * _WRAM_PANEL_OUTPUT_BYTES_PER_ELEMENT
                    c_payload = actual_n * _WRAM_PANEL_OUTPUT_BYTES_PER_ELEMENT
                    if k_start:
                        partial_c_read_calls += 1
                        partial_c_read_payload_bytes += c_payload
                        partial_c_read_aligned_bytes += _aligned_transfer_span(
                            c_row_offset, c_payload
                        )
                    c_write_calls += 1
                    c_write_payload_bytes += c_payload
                    c_write_aligned_bytes += _aligned_transfer_span(c_row_offset, c_payload)

    lane_count = _WRAM_PANEL_LANE_COUNT
    a_read_helper_calls = lane_count * a_read_calls
    b_read_helper_calls = lane_count * b_read_calls
    partial_c_read_helper_calls = lane_count * partial_c_read_calls
    c_write_helper_calls = lane_count * c_write_calls
    a_read_payload = lane_count * a_read_payload_bytes
    b_read_payload = lane_count * b_read_payload_bytes
    partial_c_read_payload = lane_count * partial_c_read_payload_bytes
    c_write_payload = lane_count * c_write_payload_bytes
    a_read_aligned = lane_count * a_read_aligned_bytes
    b_read_aligned = lane_count * b_read_aligned_bytes
    partial_c_read_aligned = lane_count * partial_c_read_aligned_bytes
    c_write_aligned = lane_count * c_write_aligned_bytes
    return {
        "origin": "wram_panel_algorithm_v1",
        "lane_count": lane_count,
        "a_read_helper_calls_exact": a_read_helper_calls,
        "b_read_helper_calls_exact": b_read_helper_calls,
        "partial_c_read_helper_calls_exact": partial_c_read_helper_calls,
        "c_write_helper_calls_exact": c_write_helper_calls,
        "a_read_payload_bytes_exact": a_read_payload,
        "b_read_payload_bytes_exact": b_read_payload,
        "partial_c_read_payload_bytes_exact": partial_c_read_payload,
        "c_write_payload_bytes_exact": c_write_payload,
        "a_read_aligned_span_bytes_estimate": a_read_aligned,
        "b_read_aligned_span_bytes_estimate": b_read_aligned,
        "partial_c_read_aligned_span_bytes_estimate": partial_c_read_aligned,
        "c_write_aligned_span_bytes_estimate": c_write_aligned,
        "operand_read_helper_calls_exact": a_read_helper_calls + b_read_helper_calls,
        "output_partial_read_helper_calls_exact": partial_c_read_helper_calls,
        "output_write_helper_calls_exact": c_write_helper_calls,
        "mram_requested_payload_bytes_exact": (
            a_read_payload + b_read_payload + partial_c_read_payload + c_write_payload
        ),
        "mram_aligned_transfer_bytes_estimate": (
            a_read_aligned
            + b_read_aligned
            + partial_c_read_aligned
            + c_write_aligned
        ),
        "barrier_events_exact": lane_count * barrier_events,
        "barrier_tasklet_calls_exact": lane_count * barrier_events * tasklets_per_dpu,
        "real_mac_count_exact": lane_count * real_macs,
        "wram_shared_bytes_exact": _WRAM_PANEL_SHARED_BUFFER_BYTES,
        "wram_private_bytes_per_tasklet_exact": _WRAM_PANEL_PRIVATE_BYTES_PER_TASKLET,
        "wram_kernel_buffers_allocated_bytes_exact": (
            _WRAM_PANEL_SHARED_BUFFER_BYTES
            + tasklets_per_dpu * _WRAM_PANEL_PRIVATE_BYTES_PER_TASKLET
        ),
        "mram_helper_count_scope": "source_level_helper_calls",
        "mram_aligned_bytes_scope": "geometric_aligned_span_estimate",
    }


def _encode_real_plane(
    array: np.ndarray, policy: NumericPolicy
) -> tuple[np.ndarray, float, int]:
    value = np.asarray(array)
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
        raise ValueError("numeric policy requires real-valued tensors")
    real = np.ascontiguousarray(
        np.real(value), dtype=np.float64 if _is_packed_policy(policy) else np.float32
    )
    if not np.all(np.isfinite(real)):
        raise ValueError("numeric policy requires finite tensors")
    if not _is_packed_policy(policy):
        return real, 1.0, 0
    converted = encode_complex_tensor(real, _NUMERIC_POLICY_INT8)
    return (
        np.ascontiguousarray(converted.real, dtype=np.int8),
        float(converted.scale),
        int(converted.saturation_real),
    )


def _decode_real_accumulator(
    accumulator: np.ndarray, policy: NumericPolicy, scale: float
) -> np.ndarray:
    if not _is_packed_policy(policy):
        return np.asarray(accumulator, dtype=np.float32)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("contraction output scale must be finite and positive")
    return np.asarray(accumulator, dtype=np.float64) * float(scale)


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
    "kernel": "upmem_sdk_hardware_v4_wram_panel_kernel",
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
            "config": {
                "abi": "execution_plan_v4",
                "kernel": "dpu_real_tile_v4_wram_panel_v1",
            },
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

    def submit_packed(
        self, operation: PackedOperation, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]: ...

    def close(self, *, timeout_s: float | None = None) -> Any: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _payload_hash(value: np.ndarray, *, dtype: np.dtype | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return _sha256_bytes(array.tobytes(order="C"))


def _resolve_view(view: Any, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
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


def _required_byte_count(metadata: Mapping[str, Any], *keys: str) -> int:
    values = [metadata[key] for key in keys if key in metadata]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError(f"UPMEM task metadata is missing or invalid {keys[0]}")
    if len(set(values)) != 1:
        raise RuntimeError(f"UPMEM task metadata has conflicting {keys[0]} values")
    return int(values[0])


def _seconds(value: Any) -> float:
    result = float(value or 0.0)
    if result < 0 or not math.isfinite(result):
        raise ValueError("UPMEM timing values must be finite and non-negative")
    return result


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


def _read_raw_output(path: Path, tile: M5Tile, *, packed: bool) -> np.ndarray:
    """Read one native tile without widening its ABI representation."""

    dtype = np.dtype("<i4") if packed else np.dtype("<f4")
    expected_bytes = tile.output_element_count * dtype.itemsize
    raw = path.read_bytes()
    if len(raw) < expected_bytes:
        raise RuntimeError(f"v4 output is truncated: {path}")
    return (
        np.frombuffer(raw[:expected_bytes], dtype=dtype)
        .reshape(tile.m_size, tile.n_size)
        .copy()
    )


def _complex_canonical_planes(
    real_lowering: M5TileLowering,
    imag_lowering: M5TileLowering,
    *,
    left: bool,
    dtype: np.dtype | type = np.complex64,
) -> np.ndarray:
    real = real_lowering.canonical.left if left else real_lowering.canonical.right
    imag = imag_lowering.canonical.left if left else imag_lowering.canonical.right
    if real.shape != imag.shape:
        raise ValueError("real and imaginary canonical planes have different shapes")
    result = np.empty(real.shape, dtype=dtype)
    component_dtype = np.float64 if np.dtype(dtype) == np.dtype(np.complex128) else np.float32
    result.real = np.asarray(real, dtype=component_dtype)
    result.imag = np.asarray(imag, dtype=component_dtype)
    return result


def _validate_complex_lowerings(
    real_lowering: M5TileLowering,
    imag_lowering: M5TileLowering,
    stage: UpmemStage,
    node: ContractNode,
) -> None:
    real = real_lowering.canonical
    imag = imag_lowering.canonical
    if (
        (real.b, real.m, real.k, real.n) != (imag.b, imag.m, imag.k, imag.n)
        or real.batch_labels != imag.batch_labels
        or real.free_left_labels != imag.free_left_labels
        or real.contracted_labels != imag.contracted_labels
        or real.free_right_labels != imag.free_right_labels
        or real.canonical_output_labels != imag.canonical_output_labels
        or real.label_dimensions != imag.label_dimensions
        or real_lowering.output_tiles != imag_lowering.output_tiles
        or real_lowering.k_chunks != imag_lowering.k_chunks
        or real_lowering.tiles != imag_lowering.tiles
    ):
        raise ValueError("real and imaginary canonical tile metadata differ")
    tiles = {f"{node.node_id}:{tile.id}": tile for tile in real_lowering.tiles}
    units = stage.work_units
    unit_ids = {unit.stable_tile_id for unit in units}
    if len(units) != len(tiles) or unit_ids != set(tiles):
        raise ValueError("UPMEM stage tile IDs do not match live lowering")
    for unit in units:
        tile = tiles.get(unit.stable_tile_id)
        if tile is None or unit.node_id != node.node_id:
            raise ValueError("UPMEM stage references an unknown live tile")
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
                f"UPMEM stage work unit {unit.stable_tile_id} differs from lowering"
            )


def _raw_lane_fact(
    node_id: str, tile_id: str, lane: str, value: np.ndarray
) -> dict[str, JsonValue]:
    dtype = (
        np.dtype("<i4") if np.issubdtype(value.dtype, np.integer) else np.dtype("<f4")
    )
    canonical = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return {
        "node_id": node_id,
        "stable_tile_id": tile_id,
        "lane": lane,
        "dtype": dtype.str,
        "shape": tuple(int(item) for item in canonical.shape),
        "sha256": _payload_hash(canonical),
        "exact": bool(np.issubdtype(value.dtype, np.integer)),
    }


def _complex_operand_facts(
    node_id: str,
    side: str,
    encoded: Any,
) -> dict[str, JsonValue]:
    return {
        "node_id": node_id,
        "side": side,
        "scale": float(encoded.scale),
        "saturation_real": int(encoded.saturation_real),
        "saturation_imag": int(encoded.saturation_imag),
        "real_dtype": encoded.real.dtype.str,
        "imag_dtype": encoded.imag.dtype.str,
        "shape": tuple(int(value) for value in encoded.real.shape),
        "real_sha256": _payload_hash(encoded.real),
        "imag_sha256": _payload_hash(encoded.imag),
    }


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


def _native_identity(
    event: Mapping[str, Any], *, source: str, execution_target: str
) -> dict[str, str]:
    """Read the identity emitted by native v4 code, never Python provenance."""

    observed: dict[str, str] = {}
    for field, expected in native_execution_identity(execution_target).items():
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
    *,
    execution_target: str,
) -> dict[str, str]:
    """Return one identity only when every rank/event reported the same contract."""

    if not observations:
        raise RuntimeError("no native identity observations were recorded")
    identities = [
        _native_identity(event, source=source, execution_target=execution_target)
        for source, event in observations
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
    """One v4 tile engine for an explicit physical or SDK-simulator target."""

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
        execution_target: str = EXECUTION_TARGET_PHYSICAL,
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
        self.execution_target = execution_target
        self.request_transport = PACKED_OPERATION_TRANSPORT
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
            "request_transport": self.request_transport,
            **self._binary_provenance,
        }
        if self.execution_target not in {
            EXECUTION_TARGET_PHYSICAL,
            EXECUTION_TARGET_SIMULATOR,
        }:
            raise ValueError("unsupported v4 execution target")
        if self.execution_target == EXECUTION_TARGET_PHYSICAL:
            if not self.rank_paths:
                raise ValueError("physical v4 engine requires explicit rank_paths")
            if self.dpu_count < 1 or self.dpu_count % len(self.rank_paths):
                raise ValueError(
                    "dpu_count must be positive and divisible by rank count"
                )
            if self.request_transport == PACKED_OPERATION_TRANSPORT and len(self.rank_paths) != 1:
                raise ValueError("packed operation transport currently requires one rank")
        elif self.rank_paths or not 1 <= self.dpu_count <= 64:
            raise ValueError(
                "v4 simulator engine requires 1..64 DPUs and no rank paths"
            )
        if self.tasklets_per_dpu < 1 or self.timeout_s <= 0:
            raise ValueError("tasklets_per_dpu and timeout_s must be positive")
        session_count = len(self.rank_paths) if self.rank_paths else 1
        if self.dpu_count // session_count > 64:
            raise ValueError("v4 supports at most 64 local DPUs per rank")

    @property
    def strategy_identity(self) -> dict[str, Any]:
        return json.loads(json.dumps(_ACTIVE_STRATEGY_IDENTITY))

    @property
    def strategy_config_hash(self) -> str:
        return _ACTIVE_STRATEGY_CONFIG_HASH

    def open_session(
        self,
        numeric_policy: NumericPolicy,
        topology: FinalUpmemTopology,
    ) -> UpmemV4Session:
        policy = _validate_numeric_policy(numeric_policy)
        if not isinstance(topology, FinalUpmemTopology):
            raise TypeError("open_session requires the final UpmemTopology record")
        if topology.dpu_count != self.dpu_count:
            raise ValueError("topology device count must match engine dpu_count")
        if topology.tasklets_per_dpu != self.tasklets_per_dpu:
            raise ValueError(
                "topology tasklet count must match engine tasklets_per_dpu"
            )
        if self.execution_target == EXECUTION_TARGET_PHYSICAL:
            if topology.rank_count != len(self.rank_paths):
                raise ValueError("topology rank count must match engine rank_paths")
        elif topology.rank_count != 1:
            raise ValueError("v4 simulator requires one rank and 1..64 DPUs")

        deadline = time.monotonic() + self.timeout_s
        self.session_root.mkdir(parents=True, exist_ok=True)
        rank_paths = self.rank_paths or (None,)
        local_dpus = self.dpu_count // len(rank_paths)
        ranks: list[_RankSession] = []
        try:
            for index, rank_path in enumerate(rank_paths):
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
                        if policy == _NUMERIC_POLICY_INT8
                        else NUMERIC_FLOAT32
                    ),
                    rank_path=rank_path,
                    execution_target=self.execution_target,
                    timeout_s=remaining,
                )
                command_parts: list[str] = [
                    str(self.host_binary),
                    "--target",
                    "simulator"
                    if self.execution_target == EXECUTION_TARGET_SIMULATOR
                    else "hardware",
                    "--session-root",
                    str(root.resolve()),
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
                ]
                if rank_path is not None:
                    command_parts[5:5] = ["--rank-path", rank_path]
                command = tuple(command_parts)
                session = self.session_factory(
                    command, session_root=root, profile=profile
                )
                ranks.append(_RankSession(index, root, session, local_dpus))
                _native_identity(
                    session.startup,
                    source=f"READY rank {index}",
                    execution_target=self.execution_target,
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
        return UpmemV4Session(
            numeric_policy=policy,
            ranks=tuple(ranks),
            engine=self,
            deadline=deadline,
        )


class UpmemV4Session:
    """Persistent rank sessions used by one whole-graph measurement."""

    def __init__(
        self,
        *,
        numeric_policy: NumericPolicy,
        ranks: tuple[_RankSession, ...],
        engine: UpmemV4Executor,
        deadline: float,
    ) -> None:
        self.numeric_policy = _validate_numeric_policy(numeric_policy)
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
            ),
            execution_target=self.engine.execution_target,
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

    def execute_real(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        stage: UpmemStage,
        numeric_policy: NumericPolicy,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        if self._closed:
            raise RuntimeError("UPMEM v4 session is closed")
        policy = _validate_numeric_policy(numeric_policy)
        if self.numeric_policy != policy:
            raise ValueError("UPMEM session numeric policy does not match execution")
        if not isinstance(stage, UpmemStage) or stage.kind != "contract_batch":
            raise ValueError("execute_real requires a contract_batch UpmemStage")
        if stage.node_ids != (node.node_id,):
            raise ValueError("execute_real stage does not match contract node")
        self._remaining_timeout()
        started = time.perf_counter()
        packed = _is_packed_policy(policy)
        limits = M5TileLimits.host_packed_int8() if packed else M5TileLimits.float32()
        lowering = lower_binary_contraction(node, left, right, limits=limits)
        canonical_left = lowering.canonical.left
        canonical_right = lowering.canonical.right
        quantization_metadata: dict[str, Any] = {}
        left_scale = right_scale = 1.0
        quantization_started = time.perf_counter()
        left_payload, left_scale, left_saturation = _encode_real_plane(
            canonical_left, policy
        )
        right_payload, right_scale, right_saturation = _encode_real_plane(
            canonical_right, policy
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
            if lowering.canonical.k * INT8_MAX_PRODUCT > _INT64_MAX:
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
        packed_operation_count = 0
        packed_operation_bytes = 0
        packed_operation_request_count = 0
        packed_operation_max_descriptor_count = 0
        packed_operation_max_bytes = 0
        packed_operation_max_payload_bytes = 0
        total_dpus = sum(rank.local_dpus for rank in self.ranks)
        waves, planned_requests = self._requests_from_work_units(
            node, lowering, stage.work_units
        )
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
            (
                outcomes,
                wave_metrics,
                wave_parallel,
                wave_bulk_verified,
                _record_templates,
            ) = self._submit_packed_operation(
                lowering=lowering,
                canonical_left=canonical_left,
                canonical_right=canonical_right,
                packed=packed,
                request_contract=request_contract,
                waves=waves,
                requests_by_wave=planned_requests,
            )
            parallel_rank_waves += int(wave_parallel)
            bulk_verified = bulk_verified and wave_bulk_verified
            bytes_h2d += int(wave_metrics["h2d_bytes"])
            bytes_d2h += int(wave_metrics["d2h_bytes"])
            for key in timing:
                timing[key] += float(wave_metrics[key])
            request_hashes.extend(wave_metrics["request_manifest_hashes"])
            packed_operation_count += int(wave_metrics.get("packed_operation_count", 0))
            packed_operation_bytes += int(wave_metrics.get("packed_operation_bytes", 0))
            packed_operation_request_count += int(
                wave_metrics.get("packed_operation_request_count", 0)
            )
            packed_operation_max_descriptor_count = max(
                packed_operation_max_descriptor_count,
                int(wave_metrics.get("packed_operation_max_descriptor_count", 0)),
            )
            packed_operation_max_bytes = max(
                packed_operation_max_bytes,
                int(wave_metrics.get("packed_operation_max_bytes", 0)),
            )
            packed_operation_max_payload_bytes = max(
                packed_operation_max_payload_bytes,
                int(wave_metrics.get("packed_operation_max_payload_bytes", 0)),
            )
            self._successful_request_count += int(wave_metrics["successful_request_count"])
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
        output = _decode_real_accumulator(accumulator, policy, scale)
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
            "request_transport": self.engine.request_transport,
            "packed_operation_count": packed_operation_count,
            "packed_operation_bytes": packed_operation_bytes,
            "packed_operation_request_count": packed_operation_request_count,
            "packed_operation_max_descriptor_count": packed_operation_max_descriptor_count,
            "packed_operation_max_bytes": packed_operation_max_bytes,
            "packed_operation_max_payload_bytes": packed_operation_max_payload_bytes,
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
                else (
                    "sdk_simulator"
                    if self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
                    else "physical_hardware"
                )
            ),
            "test_double_execution": self._test_double_execution,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": (
                self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
            ),
            "physical_plan_consumed": True,
            **quantization_metadata,
        }

    def _execute_complex_core(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        stage: UpmemStage,
        numeric_policy: NumericPolicy,
        include_evidence: bool,
    ) -> tuple[
        np.ndarray,
        Mapping[str, JsonValue],
        Mapping[str, tuple[str, str, str, np.ndarray]],
        tuple[Any, Any],
    ]:
        """Execute one final-plan complex contraction through ABI v4 lanes.

        ``include_evidence`` is intentionally private.  The public low-level
        method keeps its historical metadata behavior, while the graph session
        can stop its execution timer before hashing raw arrays and operands.
        """

        if self._closed:
            raise RuntimeError("UPMEM v4 session is closed")
        if not isinstance(stage, UpmemStage) or stage.kind != "contract_batch":
            raise ValueError("execute_complex requires a contract_batch UpmemStage")
        if stage.node_ids != (node.node_id,):
            raise ValueError("execute_complex stage does not match contract node")
        policy = _validate_numeric_policy(numeric_policy)
        if self.numeric_policy != policy:
            raise ValueError("UPMEM session numeric mode does not match final policy")

        packed = policy == _NUMERIC_POLICY_INT8
        limits = M5TileLimits.host_packed_int8() if packed else M5TileLimits.float32()
        materialized_left = np.array(left, copy=True, order="C")
        materialized_right = np.array(right, copy=True, order="C")
        if materialized_left.shape != node.left.shape:
            raise ValueError("left operand shape does not match contract node")
        if materialized_right.shape != node.right.shape:
            raise ValueError("right operand shape does not match contract node")
        if not np.issubdtype(materialized_left.dtype, np.number) or not np.issubdtype(
            materialized_right.dtype, np.number
        ):
            raise ValueError("complex execution requires numeric operands")

        started = time.perf_counter()
        preparation_started = time.perf_counter()
        real_lowering = lower_binary_contraction(
            node,
            np.asarray(
                materialized_left.real,
                dtype=np.float64 if packed else np.float32,
            ),
            np.asarray(
                materialized_right.real,
                dtype=np.float64 if packed else np.float32,
            ),
            limits=limits,
        )
        imag_lowering = lower_binary_contraction(
            node,
            np.asarray(
                materialized_left.imag,
                dtype=np.float64 if packed else np.float32,
            ),
            np.asarray(
                materialized_right.imag,
                dtype=np.float64 if packed else np.float32,
            ),
            limits=limits,
        )
        _validate_complex_lowerings(real_lowering, imag_lowering, stage, node)
        preparation_s = time.perf_counter() - preparation_started

        encode_started = time.perf_counter()
        canonical_left = _complex_canonical_planes(
            real_lowering,
            imag_lowering,
            left=True,
            dtype=np.complex128 if packed else np.complex64,
        )
        canonical_right = _complex_canonical_planes(
            real_lowering,
            imag_lowering,
            left=False,
            dtype=np.complex128 if packed else np.complex64,
        )
        encoded_left = encode_complex_tensor(canonical_left, numeric_policy)
        encoded_right = encode_complex_tensor(canonical_right, numeric_policy)
        if packed:
            max_chunk = max((chunk.k_size for chunk in real_lowering.k_chunks), default=0)
            if max_chunk > MAX_INT32_SAFE_K:
                raise ValueError(
                    "packed int8 K chunk exceeds full-component int32 accumulation safety bound"
                )
            if 2 * real_lowering.canonical.k * INT8_MAX_PRODUCT > _INT64_MAX:
                raise ValueError(
                    "packed int8 aggregate exceeds int64 full-component accumulation safety bound"
                )
        encode_s = time.perf_counter() - encode_started

        waves, planned_requests = self._requests_from_work_units(
            node, real_lowering, stage.work_units
        )
        lane_names = ("rr", "ii", "ri", "ir")
        lane_operands = (
            (encoded_left.real, encoded_right.real),
            (encoded_left.imag, encoded_right.imag),
            (encoded_left.real, encoded_right.imag),
            (encoded_left.imag, encoded_right.real),
        )
        lane_partials: dict[str, dict[str, np.ndarray]] = {}
        lane_request_hashes: dict[str, tuple[str, ...]] = {}
        raw_lane_values: dict[str, tuple[str, str, str, np.ndarray]] = {}
        lane_request_contract_hashes: dict[str, str] = {}
        h2d_bytes = 0
        d2h_bytes = 0
        rank_response_h2d_max_sum_s = 0.0
        rank_response_kernel_max_sum_s = 0.0
        rank_response_d2h_max_sum_s = 0.0
        rank_response_total_route_max_sum_s = 0.0
        request_wave_wall_sum_s = 0.0
        request_build_sum_s = 0.0
        request_work_unit_materialization_sum_s = 0.0
        request_artifact_build_sum_s = 0.0
        request_payload_record_staging_sum_s = 0.0
        request_manifest_sidecar_staging_sum_s = 0.0
        request_payload_materialization_sum_s = 0.0
        request_payload_file_write_sum_s = 0.0
        request_payload_hashing_sum_s = 0.0
        request_payload_record_construction_sum_s = 0.0
        request_payload_record_count = 0
        request_payload_files_created = 0
        request_payload_bytes_staged = 0
        request_payload_bytes_hashed = 0
        rank_submit_parallel_wall_sum_s = 0.0
        rank_submit_total_max_sum_s = 0.0
        rank_submit_artifact_validation_max_sum_s = 0.0
        rank_submit_protocol_write_max_sum_s = 0.0
        rank_submit_response_wait_max_sum_s = 0.0
        rank_submit_response_validation_max_sum_s = 0.0
        coordinator_response_processing_sum_s = 0.0
        parallel_rank_waves = 0
        bulk_verified = True
        packed_operation_count = 0
        packed_operation_bytes = 0
        packed_operation_request_count = 0
        packed_operation_max_descriptor_count = 0
        packed_operation_max_bytes = 0
        packed_operation_max_payload_bytes = 0
        active_rank_indices: set[int] = set()
        active_dpu_ids: set[tuple[int, int]] = set()
        numeric_transport = "host_packed_int8_mram" if packed else "float32_mram"
        record_templates_by_wave: dict[
            int, Mapping[tuple[int, int], tuple[int, ...]]
        ] = {}

        try:
            for lane, (lane_left, lane_right) in zip(
                lane_names, lane_operands, strict=True
            ):
                request_contract = _request_contract_hash(
                    _task_structure_hash(node),
                    numeric_transport=numeric_transport,
                    left_scale=encoded_left.scale,
                    right_scale=encoded_right.scale,
                    left_payload=lane_left,
                    right_payload=lane_right,
                    strategy_config_hash=self.strategy_config_hash,
                )
                lane_request_contract_hashes[lane] = request_contract
                partials: dict[str, np.ndarray] = {}
                request_hashes: list[str] = []
                if self.engine.request_transport == PACKED_OPERATION_TRANSPORT:
                    self._remaining_timeout()
                    operation_started = time.perf_counter()
                    (
                        outcomes,
                        metrics,
                        wave_parallel,
                        wave_bulk_verified,
                        packed_templates,
                    ) = self._submit_packed_operation(
                        lowering=real_lowering,
                        canonical_left=lane_left,
                        canonical_right=lane_right,
                        packed=packed,
                        request_contract=request_contract,
                        waves=waves,
                        requests_by_wave=planned_requests,
                        preserve_native=True,
                        record_templates=(
                            record_templates_by_wave or None
                        ),
                    )
                    record_templates_by_wave = packed_templates
                    request_wave_wall_sum_s += time.perf_counter() - operation_started
                    parallel_rank_waves += int(wave_parallel)
                    bulk_verified = bulk_verified and wave_bulk_verified
                    h2d_bytes += int(metrics["h2d_bytes"])
                    d2h_bytes += int(metrics["d2h_bytes"])
                    rank_response_h2d_max_sum_s += float(metrics["h2d_time_s"])
                    rank_response_kernel_max_sum_s += float(metrics["kernel_time_s"])
                    rank_response_d2h_max_sum_s += float(metrics["d2h_time_s"])
                    rank_response_total_route_max_sum_s += float(
                        metrics["total_route_time_s"]
                    )
                    request_build_sum_s += float(metrics["request_build_s"])
                    request_work_unit_materialization_sum_s += float(
                        metrics["request_work_unit_materialization_s"]
                    )
                    request_artifact_build_sum_s += float(
                        metrics["request_artifact_build_s"]
                    )
                    request_payload_record_staging_sum_s += float(
                        metrics["request_payload_record_staging_s"]
                    )
                    request_manifest_sidecar_staging_sum_s += float(
                        metrics["request_manifest_sidecar_staging_s"]
                    )
                    request_payload_materialization_sum_s += float(
                        metrics["request_payload_materialization_sum_s"]
                    )
                    request_payload_file_write_sum_s += float(
                        metrics["request_payload_file_write_sum_s"]
                    )
                    request_payload_hashing_sum_s += float(
                        metrics["request_payload_hashing_sum_s"]
                    )
                    request_payload_record_construction_sum_s += float(
                        metrics["request_payload_record_construction_sum_s"]
                    )
                    request_payload_record_count += int(
                        metrics["request_payload_record_count"]
                    )
                    request_payload_files_created += int(
                        metrics["request_payload_files_created"]
                    )
                    request_payload_bytes_staged += int(
                        metrics["request_payload_bytes_staged"]
                    )
                    request_payload_bytes_hashed += int(
                        metrics["request_payload_bytes_hashed"]
                    )
                    rank_submit_parallel_wall_sum_s += float(
                        metrics["rank_submit_parallel_wall_s"]
                    )
                    rank_submit_total_max_sum_s += float(
                        metrics["rank_submit_total_max_s"]
                    )
                    rank_submit_artifact_validation_max_sum_s += float(
                        metrics["rank_submit_artifact_validation_max_s"]
                    )
                    rank_submit_protocol_write_max_sum_s += float(
                        metrics["rank_submit_protocol_write_max_s"]
                    )
                    rank_submit_response_wait_max_sum_s += float(
                        metrics["rank_submit_response_wait_max_s"]
                    )
                    rank_submit_response_validation_max_sum_s += float(
                        metrics["rank_submit_response_validation_max_s"]
                    )
                    coordinator_response_processing_sum_s += float(
                        metrics["coordinator_response_processing_s"]
                    )
                    request_hashes.extend(metrics["request_manifest_hashes"])
                    packed_operation_count += int(
                        metrics.get("packed_operation_count", 0)
                    )
                    packed_operation_bytes += int(
                        metrics.get("packed_operation_bytes", 0)
                    )
                    packed_operation_request_count += int(
                        metrics.get("packed_operation_request_count", 0)
                    )
                    packed_operation_max_descriptor_count = max(
                        packed_operation_max_descriptor_count,
                        int(metrics.get("packed_operation_max_descriptor_count", 0)),
                    )
                    packed_operation_max_bytes = max(
                        packed_operation_max_bytes,
                        int(metrics.get("packed_operation_max_bytes", 0)),
                    )
                    packed_operation_max_payload_bytes = max(
                        packed_operation_max_payload_bytes,
                        int(metrics.get("packed_operation_max_payload_bytes", 0)),
                    )
                    self._successful_request_count += int(
                        metrics["successful_request_count"]
                    )
                    active_rank_indices.update(metrics["active_rank_indices"])
                    active_dpu_ids.update(metrics["active_dpu_ids"])
                    self._active_rank_indices.update(metrics["active_rank_indices"])
                    self._active_dpu_ids.update(metrics["active_dpu_ids"])
                    partials.update({tile.id: value for tile, value in outcomes})
                lane_request_hashes[lane] = tuple(request_hashes)
                lane_partials[lane] = partials
                for tile_id, value in partials.items():
                    stable_tile_id = f"{node.node_id}:{tile_id}"
                    key = f"{stable_tile_id}/{lane}"
                    raw_lane_values[key] = (
                        node.node_id,
                        stable_tile_id,
                        lane,
                        value,
                    )
        except BaseException as exc:
            self._failed = True
            self._failure_stage = str(
                getattr(exc, "failure_stage", "hardware_task_execution_failed")
            )
            raise

        assembly_started = time.perf_counter()
        lane_outputs = tuple(
            real_lowering.assemble(
                lane_partials[lane],
                dtype=np.int64 if packed else np.float32,
            )
            for lane in lane_names
        )
        assembled_s = time.perf_counter() - assembly_started
        decode_started = time.perf_counter()
        output = decode_complex_products(
            lane_outputs,
            encoded_left.scale,
            encoded_right.scale,
            numeric_policy,
        )
        output = np.array(output, dtype=np.complex64, copy=True, order="C")
        output.setflags(write=False)
        decode_s = time.perf_counter() - decode_started
        total_wall_s = time.perf_counter() - started
        operational_metadata: dict[str, Any] = {
            "numeric_policy": numeric_policy,
            "numeric_transport": numeric_transport,
            "lane_order": lane_names,
            "lane_pass_count": 4,
            "left_scale": float(encoded_left.scale),
            "right_scale": float(encoded_right.scale),
            "saturation_real": int(
                encoded_left.saturation_real + encoded_right.saturation_real
            ),
            "saturation_imag": int(
                encoded_left.saturation_imag + encoded_right.saturation_imag
            ),
            "active_rank_indices": tuple(sorted(active_rank_indices)),
            "active_dpu_ids": tuple(sorted(active_dpu_ids)),
            "requested_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "allocated_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "active_rank_count": len(active_rank_indices),
            "active_dpu_count": len(active_dpu_ids),
            "rank_count": len(self.ranks),
            "tasklets_per_dpu": self.engine.tasklets_per_dpu,
            "request_transport": self.engine.request_transport,
            "packed_operation_count": packed_operation_count,
            "packed_operation_bytes": packed_operation_bytes,
            "packed_operation_request_count": packed_operation_request_count,
            "packed_operation_max_descriptor_count": packed_operation_max_descriptor_count,
            "packed_operation_max_bytes": packed_operation_max_bytes,
            "packed_operation_max_payload_bytes": packed_operation_max_payload_bytes,
            "parallel_rank_wave_count": parallel_rank_waves,
            "bulk_set_launch_verified": bulk_verified,
            "application_visible_h2d_bytes": h2d_bytes,
            "application_visible_d2h_bytes": d2h_bytes,
            "application_visible_transfer_bytes": h2d_bytes + d2h_bytes,
            "wram_panel_facts": _wram_panel_operation_facts(
                stage.work_units,
                numeric_policy=numeric_policy,
                tasklets_per_dpu=self.engine.tasklets_per_dpu,
            ),
            "physical_stage_consumed": True,
            "stage_id": stage.stage_id,
            "cpu_fallback_used": False,
            "hardware_kernel_executed": (
                self.engine.execution_target == EXECUTION_TARGET_PHYSICAL
            ),
            "simulator_kernel_executed": (
                self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
            ),
            "test_double_execution": self._test_double_execution,
            "target_observed": (
                "not_verified"
                if self._test_double_execution
                else (
                    "sdk_simulator"
                    if self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
                    else "physical_hardware"
                )
            ),
            "timing_scope": "sum_of_per_request_max_rank_response_counters_v1",
            "timing": {
                "total_wall_s": float(total_wall_s),
                "preparation_s": float(preparation_s),
                "encode_s": float(encode_s),
                "rank_response_h2d_max_sum_s": float(rank_response_h2d_max_sum_s),
                "rank_response_kernel_max_sum_s": float(rank_response_kernel_max_sum_s),
                "rank_response_d2h_max_sum_s": float(rank_response_d2h_max_sum_s),
                "rank_response_total_route_max_sum_s": float(
                    rank_response_total_route_max_sum_s
                ),
                "request_wave_wall_sum_s": float(request_wave_wall_sum_s),
                "request_build_sum_s": float(request_build_sum_s),
                "request_work_unit_materialization_sum_s": float(
                    request_work_unit_materialization_sum_s
                ),
                "request_artifact_build_sum_s": float(request_artifact_build_sum_s),
                "request_payload_record_staging_sum_s": float(
                    request_payload_record_staging_sum_s
                ),
                "request_manifest_sidecar_staging_sum_s": float(
                    request_manifest_sidecar_staging_sum_s
                ),
                "request_payload_materialization_sum_s": float(
                    request_payload_materialization_sum_s
                ),
                "request_payload_file_write_sum_s": float(
                    request_payload_file_write_sum_s
                ),
                "request_payload_hashing_sum_s": float(
                    request_payload_hashing_sum_s
                ),
                "request_payload_record_construction_sum_s": float(
                    request_payload_record_construction_sum_s
                ),
                "request_payload_record_count": request_payload_record_count,
                "request_payload_files_created": request_payload_files_created,
                "request_payload_bytes_staged": request_payload_bytes_staged,
                "request_payload_bytes_hashed": request_payload_bytes_hashed,
                "rank_submit_parallel_wall_sum_s": float(
                    rank_submit_parallel_wall_sum_s
                ),
                "rank_submit_total_max_sum_s": float(rank_submit_total_max_sum_s),
                "rank_submit_artifact_validation_max_sum_s": float(
                    rank_submit_artifact_validation_max_sum_s
                ),
                "rank_submit_protocol_write_max_sum_s": float(
                    rank_submit_protocol_write_max_sum_s
                ),
                "rank_submit_response_wait_max_sum_s": float(
                    rank_submit_response_wait_max_sum_s
                ),
                "rank_submit_response_validation_max_sum_s": float(
                    rank_submit_response_validation_max_sum_s
                ),
                "coordinator_response_processing_sum_s": float(
                    coordinator_response_processing_sum_s
                ),
                "assembly_s": float(assembled_s),
                "decode_s": float(decode_s),
            },
        }
        if not include_evidence:
            return (
                output,
                operational_metadata,
                raw_lane_values,
                (encoded_left, encoded_right),
            )

        evidence_started = time.perf_counter()
        raw_lane_records = {
            key: _raw_lane_fact(node_id, tile_id, lane, value)
            for key, (node_id, tile_id, lane, value) in raw_lane_values.items()
        }
        operand_records = (
            _complex_operand_facts(node.node_id, "left", encoded_left),
            _complex_operand_facts(node.node_id, "right", encoded_right),
        )
        evidence_hash_s = time.perf_counter() - evidence_started
        metadata: dict[str, JsonValue] = {
            **operational_metadata,
            "operand_records": operand_records,
            "raw_lane_records": raw_lane_records,
            "evidence_hash_s": float(evidence_hash_s),
            "lane_request_contract_hashes": lane_request_contract_hashes,
            "lane_request_manifest_hashes": lane_request_hashes,
            "request_manifest_hashes": tuple(
                hash_value
                for lane in lane_names
                for hash_value in lane_request_hashes[lane]
            ),
        }
        return output, metadata, raw_lane_values, (encoded_left, encoded_right)

    def execute_complex(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        stage: UpmemStage,
        numeric_policy: NumericPolicy,
    ) -> tuple[np.ndarray, Mapping[str, JsonValue]]:
        """Execute one final-plan complex contraction through ABI v4 lanes."""

        output, metadata, _, _ = self._execute_complex_core(
            node,
            left,
            right,
            stage=stage,
            numeric_policy=numeric_policy,
            include_evidence=True,
        )
        return output, metadata

    def _requests_from_work_units(
        self,
        node: ContractNode,
        lowering: Any,
        units: tuple[UpmemWorkUnit, ...],
    ) -> tuple[
        tuple[tuple[M5Tile, ...], ...],
        tuple[list[tuple[Any, list[tuple[M5Tile, int]]]], ...],
    ]:
        """Group final work units without changing compiled placement."""

        if not all(isinstance(unit, UpmemWorkUnit) for unit in units):
            raise TypeError("UPMEM stage work units must use final UpmemWorkUnit")
        tiles = {f"{node.node_id}:{tile.id}": tile for tile in lowering.tiles}
        if len(tiles) != len(lowering.tiles) or len(units) != len(tiles):
            raise ValueError("compiled UPM work-unit count differs from lowering")
        tile_ids = {unit.stable_tile_id for unit in units}
        if tile_ids != set(tiles):
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

    def _submit_packed_operation(
        self,
        *,
        lowering: Any,
        canonical_left: np.ndarray,
        canonical_right: np.ndarray,
        packed: bool,
        request_contract: str,
        waves: tuple[tuple[M5Tile, ...], ...],
        requests_by_wave: tuple[list[tuple[Any, list[tuple[M5Tile, int]]]], ...],
        preserve_native: bool = False,
        record_templates: Mapping[
            int, Mapping[tuple[int, int], tuple[int, ...]]
        ] | None = None,
    ) -> tuple[
        list[tuple[M5Tile, np.ndarray]],
        dict[str, Any],
        bool,
        bool,
        dict[int, Mapping[tuple[int, int], tuple[int, ...]]],
    ]:
        """Submit all waves for one real lane in one packed operation."""

        if len(self.ranks) != 1:
            raise UnsupportedExecution(
                stage="request_transport",
                reason="packed operation transport currently supports one rank",
                capability="packed_operation_one_rank",
            )
        if not callable(getattr(self.ranks[0].session, "submit_packed", None)):
            raise RuntimeError("packed v4 session lacks submit_packed")

        request_build_started = time.perf_counter()
        prepared: list[tuple[_RankSession, list[tuple[M5Tile, int]], Any]] = []
        built_record_templates: dict[
            int, dict[tuple[int, int], tuple[int, ...]]
        ] = {}
        request_work_unit_materialization_s = 0.0
        request_artifact_build_s = 0.0
        request_payload_record_staging_s = 0.0
        request_manifest_sidecar_staging_s = 0.0
        request_payload_materialization_sum_s = 0.0
        request_payload_file_write_sum_s = 0.0
        request_payload_hashing_sum_s = 0.0
        request_payload_record_construction_sum_s = 0.0
        request_payload_record_count = 0
        request_payload_files_created = 0
        request_payload_bytes_staged = 0
        request_payload_bytes_hashed = 0
        for wave_index, (wave, requests) in enumerate(
            zip(waves, requests_by_wave, strict=True)
        ):
            self._validate_rank_assignments(wave, requests)
            if len(requests) != 1:
                raise UnsupportedExecution(
                    stage="request_transport",
                    reason="packed operation transport requires one rank per wave",
                    capability="packed_operation_one_rank",
                )
            rank, assignments = requests[0]
            materialization_started = time.perf_counter()
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
            request_work_unit_materialization_s += (
                time.perf_counter() - materialization_started
            )
            artifact_build_started = time.perf_counter()
            artifact = build_packed_v4_request(
                rank.root,
                profile=rank.session.profile,
                canonical_batch_count=lowering.canonical.b,
                canonical_m=lowering.canonical.m,
                canonical_n=lowering.canonical.n,
                canonical_k=lowering.canonical.k,
                work_units=units,
                task_contract_sha256=request_contract,
                request_sequence=self._sequence,
                record_templates=(
                    None
                    if record_templates is None
                    else {
                        local_id: record_templates[wave_index][(rank.index, local_id)]
                        for _, local_id in assignments
                    }
                ),
            )
            request_artifact_build_s += time.perf_counter() - artifact_build_started
            request_payload_record_staging_s += artifact.payload_record_staging_s
            request_manifest_sidecar_staging_s += artifact.manifest_sidecar_staging_s
            request_payload_materialization_sum_s += artifact.payload_materialization_s
            request_payload_file_write_sum_s += artifact.payload_file_write_s
            request_payload_hashing_sum_s += artifact.payload_hashing_s
            request_payload_record_construction_sum_s += (
                artifact.payload_record_construction_s
            )
            request_payload_record_count += artifact.payload_record_count
            request_payload_files_created += artifact.payload_files_created
            request_payload_bytes_staged += artifact.payload_bytes_staged
            request_payload_bytes_hashed += artifact.payload_bytes_hashed
            if record_templates is None:
                wave_templates: dict[tuple[int, int], tuple[int, ...]] = {}
                for unit in units:
                    wave_templates[(rank.index, unit.local_dpu_id)] = (
                        _record_abi_fields(
                            unit,
                            profile=rank.session.profile,
                            canonical_batch_count=lowering.canonical.b,
                            canonical_m=lowering.canonical.m,
                            canonical_n=lowering.canonical.n,
                            canonical_k=lowering.canonical.k,
                            validate_payload=False,
                            validate_geometry=False,
                        )
                    )
                built_record_templates[wave_index] = wave_templates
            prepared.append((rank, assignments, artifact))
            self._sequence += 1

        operation_sequence = prepared[0][2].request_sequence
        operation = pack_operation(
            self.ranks[0].root,
            requests=tuple(artifact for _, _, artifact in prepared),
            operation_sequence=operation_sequence,
            filename=f"packed/operation_{operation_sequence:016d}.bin",
        )
        response_path: Path | None = (
            self.ranks[0].root
            / "results"
            / f"operation_{operation_sequence:016x}.jsonl"
        )

        def cleanup_operation() -> None:
            try:
                operation.path.unlink()
            except FileNotFoundError:
                pass
            if response_path is not None:
                try:
                    response_path.unlink()
                except FileNotFoundError:
                    pass
            for _, _, artifact in prepared:
                self._delete_packed_request_dir(artifact)

        try:
            operation.path.parent.mkdir(parents=True, exist_ok=True)
            operation.path.write_bytes(operation.data)
        except BaseException:
            cleanup_operation()
            raise
        request_build_s = time.perf_counter() - request_build_started
        try:
            responses = self._submit_with_deadline_packed(self.ranks[0], operation)
        except BaseException:
            cleanup_operation()
            raise
        reported_response_path = self._packed_response_path(
            self.ranks[0].root, responses.get("response_path")
        )
        if reported_response_path is not None:
            response_path = reported_response_path
        response_records = responses.get("responses")
        if not isinstance(response_records, (list, tuple)):
            cleanup_operation()
            raise RuntimeError("packed response is missing per-request records")
        if len(response_records) != len(prepared):
            cleanup_operation()
            raise RuntimeError("packed response count does not match requests")
        response_processing_started = time.perf_counter()
        request_metrics: dict[str, Any] = {
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "response_transfer_bytes": 0,
            "h2d_time_s": 0.0,
            "kernel_time_s": 0.0,
            "d2h_time_s": 0.0,
            "total_route_time_s": 0.0,
            "request_build_s": request_build_s,
            "request_work_unit_materialization_s": request_work_unit_materialization_s,
            "request_artifact_build_s": request_artifact_build_s,
            "request_payload_record_staging_s": request_payload_record_staging_s,
            "request_manifest_sidecar_staging_s": request_manifest_sidecar_staging_s,
            "request_payload_materialization_sum_s": request_payload_materialization_sum_s,
            "request_payload_file_write_sum_s": request_payload_file_write_sum_s,
            "request_payload_hashing_sum_s": request_payload_hashing_sum_s,
            "request_payload_record_construction_sum_s": request_payload_record_construction_sum_s,
            "request_payload_record_count": request_payload_record_count,
            "request_payload_files_created": request_payload_files_created,
            "request_payload_bytes_staged": request_payload_bytes_staged,
            "request_payload_bytes_hashed": request_payload_bytes_hashed,
            "request_manifest_hashes": tuple(
                artifact.manifest_sha256 for _, _, artifact in prepared
            ),
            "successful_request_count": 0,
            "active_rank_indices": (0,),
            "active_dpu_ids": tuple(),
            "packed_operation_count": 1,
            "packed_operation_bytes": len(operation.data),
            "packed_operation_request_count": len(prepared),
            "packed_operation_max_descriptor_count": len(operation.requests),
            "packed_operation_max_bytes": len(operation.data),
            "packed_operation_max_payload_bytes": operation.payload_bytes,
            "rank_submit_parallel_wall_s": 0.0,
            "rank_submit_total_max_s": 0.0,
            "rank_submit_artifact_validation_max_s": 0.0,
            "rank_submit_protocol_write_max_s": 0.0,
            "rank_submit_response_wait_max_s": 0.0,
            "rank_submit_response_validation_max_s": 0.0,
            "coordinator_response_processing_s": 0.0,
        }
        submit_timing = responses.get("host_submit_timing")
        if not isinstance(submit_timing, Mapping):
            cleanup_operation()
            raise RuntimeError("packed response is missing host_submit_timing")
        try:
            submit_values: dict[str, float] = {}
            for field in (
                "artifact_validation_s",
                "protocol_write_s",
                "response_wait_s",
                "response_validation_s",
                "total_submit_s",
            ):
                value = submit_timing.get(field)
                if type(value) not in (int, float):
                    raise RuntimeError(
                        f"packed response is missing host_submit_timing.{field}"
                    )
                submit_values[field] = _seconds(value)
        except BaseException:
            cleanup_operation()
            raise
        request_metrics["rank_submit_parallel_wall_s"] = submit_values["total_submit_s"]
        request_metrics["rank_submit_total_max_s"] = submit_values["total_submit_s"]
        request_metrics["rank_submit_artifact_validation_max_s"] = submit_values[
            "artifact_validation_s"
        ]
        request_metrics["rank_submit_protocol_write_max_s"] = submit_values[
            "protocol_write_s"
        ]
        request_metrics["rank_submit_response_wait_max_s"] = submit_values[
            "response_wait_s"
        ]
        request_metrics["rank_submit_response_validation_max_s"] = submit_values[
            "response_validation_s"
        ]
        results: list[tuple[M5Tile, np.ndarray]] = []
        active_dpu_ids: set[tuple[int, int]] = set()
        try:
            for response, (rank, assignments, artifact) in zip(
                response_records, prepared, strict=True
            ):
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
                        "packed response transfer total does not equal H2D plus D2H"
                    )
                response_timing = response.get("timing", {})
                if not isinstance(response_timing, Mapping):
                    raise RuntimeError("packed response timing is not a mapping")
                response_times: dict[str, float] = {}
                for field in (
                    "h2d_time_s",
                    "launch_time_s",
                    "d2h_time_s",
                    "total_route_time_s",
                ):
                    value = response_timing.get(field)
                    if type(value) not in (int, float):
                        raise RuntimeError(
                            f"packed response is missing timing.{field}"
                        )
                    response_times[field] = _seconds(value)
                request_metrics["h2d_bytes"] += h2d_bytes
                request_metrics["d2h_bytes"] += d2h_bytes
                request_metrics["response_transfer_bytes"] += total_bytes
                request_metrics["h2d_time_s"] += response_times["h2d_time_s"]
                request_metrics["kernel_time_s"] += response_times["launch_time_s"]
                request_metrics["d2h_time_s"] += response_times["d2h_time_s"]
                request_metrics["total_route_time_s"] += response_times[
                    "total_route_time_s"
                ]
                active_dpu_ids.update(
                    (rank.index, record.local_dpu_id)
                    for record in artifact.work_units
                    if not record.flags
                )
                records = {
                    record.local_dpu_id: record for record in artifact.work_units
                }
                for tile, local_id in assignments:
                    record = records[local_id]
                    output_path = artifact.root / record.c_path
                    value = (
                        _read_raw_output(output_path, tile, packed=packed)
                        if preserve_native
                        else _read_output(output_path, tile, packed=packed)
                    )
                    results.append((tile, value))
            request_metrics["successful_request_count"] = len(prepared)
            request_metrics["active_dpu_ids"] = tuple(sorted(active_dpu_ids))
            request_metrics["coordinator_response_processing_s"] = (
                time.perf_counter() - response_processing_started
            )
            return (
                results,
                request_metrics,
                False,
                all(
                    response.get("bulk_set_launch_verified") is True
                    for response in response_records
                ),
                built_record_templates
                if record_templates is None
                else dict(record_templates),
            )
        finally:
            cleanup_operation()

    def _submit_with_deadline_packed(
        self, rank: _RankSession, operation: PackedOperation
    ) -> Mapping[str, Any]:
        submit_packed = getattr(rank.session, "submit_packed", None)
        if not callable(submit_packed):
            raise RuntimeError("packed v4 session lacks submit_packed")
        return submit_packed(operation, timeout_s=self._remaining_timeout())

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
    def _delete_packed_request_dir(artifact: Any) -> None:
        """Remove only the generated request directory for one packed request."""

        request_dir = Path(artifact.request_dir).resolve()
        root = Path(artifact.root).resolve()
        requests_root = (root / "requests").resolve()
        try:
            request_dir.relative_to(requests_root)
        except ValueError as exc:
            raise RuntimeError(
                "refusing to delete a non-packed-request directory"
            ) from exc
        if request_dir == requests_root or request_dir.parent != requests_root:
            raise RuntimeError(
                "refusing to delete the packed requests root or nested directory"
            )
        if not request_dir.name.isdigit():
            raise RuntimeError(
                "refusing to delete a directory not created as a packed request"
            )
        if request_dir.is_dir():
            shutil.rmtree(request_dir)

    @staticmethod
    def _packed_response_path(root: Path, relative: Any) -> Path | None:
        """Return a deletable native response path, or None for malformed data."""

        if not isinstance(relative, str):
            return None
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 2
            or candidate.parts[0] != "results"
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            return None
        resolved_root = Path(root).resolve()
        resolved = (resolved_root / candidate).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            return None
        if resolved.parent != (resolved_root / "results").resolve():
            return None
        return resolved

    def _validate_successful_response(
        self, response: Mapping[str, Any], rank: _RankSession, artifact: Any
    ) -> None:
        simulator = self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
        expected = {
            "status": "completed",
            "target_observed": "sdk_simulator" if simulator else "physical_hardware",
            "bulk_set_launch_verified": True,
            "native_kernel_executed": True,
            "hardware_kernel_executed": not simulator,
            "simulator_kernel_executed": simulator,
            "cpu_fallback_used": False,
            "allocation_verified": True,
            "hardware_allocation_verified": not simulator,
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
            rank.session.startup,
            source=f"READY rank {rank.index}",
            execution_target=self.engine.execution_target,
        )
        response_identity = _native_identity(
            response,
            source=f"RESPONSE rank {rank.index}",
            execution_target=self.engine.execution_target,
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
                + tuple(self._response_native_identity_events),
                execution_target=self.engine.execution_target,
            )
        except RuntimeError as exc:
            observed_native_identity = None
            native_identity_error = str(exc)

        confirmed = len(diagnostics) == len(self.ranks) and all(
            diagnostic["release_confirmed"] for diagnostic in diagnostics
        )
        simulator = self.engine.execution_target == EXECUTION_TARGET_SIMULATOR
        target_verified = not self._test_double_execution and all(
            rank.session.startup.get("target_observed") == "physical_hardware"
            if not simulator
            else rank.session.startup.get("target_observed") == "sdk_simulator"
            for rank in self.ranks
        )
        allocation_verified = all(
            rank.session.startup.get("event") == "READY"
            and rank.session.startup.get("status") == "ready"
            and rank.session.startup.get("allocation_verified") is True
            and rank.session.startup.get("hardware_allocation_verified")
            is (not simulator)
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
            target_verified
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
            "target_observed": (
                ("sdk_simulator" if simulator else "physical_hardware")
                if verified
                else "not_verified"
            ),
            "requested_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_rank_count": len(self.ranks),
            "allocated_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_dpu_count": sum(rank.local_dpus for rank in self.ranks),
            "observed_tasklets_per_dpu": self.engine.tasklets_per_dpu,
            "tasklets_per_dpu": self.engine.tasklets_per_dpu,
            "hardware_allocation_verified": allocation_verified and not simulator,
            "allocation_verified": allocation_verified,
            "native_kernel_executed": native_execution,
            "hardware_kernel_executed": native_execution and not simulator,
            "simulator_kernel_executed": native_execution and simulator,
            "cpu_fallback_used": False,
            "test_double_execution": self._test_double_execution,
            "hardware_release_attempted": bool(self.ranks),
            "hardware_release_succeeded": confirmed,
            "hardware_release_verified": confirmed,
            "hardware_release_confirmed": confirmed,
            "ready_verified": ready_verified,
            "physical_target_verified": target_verified and not simulator,
            "simulator_target_verified": target_verified and simulator,
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
            **(
                {
                    "timing_claim_applicable": False,
                    "scaling_claim_applicable": False,
                    "speedup_claim_applicable": False,
                    "energy_claim_applicable": False,
                }
                if simulator
                else {}
            ),
        }
        return self._terminal_metadata

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


class UpmemSession:
    """Persistent host coordinator for one final UPMEM physical plan."""

    def __init__(
        self,
        dag: ContractionDAG,
        plan: FinalUpmemPlan,
        resources: UpmemResources,
        low_level: Any,
        timeout_s: float,
        startup_resource_admission: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self._dag = dag
        self._plan = plan
        self._resources = resources
        self._low_level = low_level
        self._timeout_s = float(timeout_s)
        self._startup_resource_admission = (
            {} if startup_resource_admission is None else startup_resource_admission
        )
        self._closed = False
        self._terminal_facts: Mapping[str, JsonValue] | None = None
        self._close_failure: ExecutionFailed | None = None

    def __enter__(self) -> "UpmemSession":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, traceback
        try:
            self.close()
        except ExecutionFailed as close_failure:
            if exc_value is None:
                raise
            _add_exception_note(
                exc_value,
                f"UPMEM session close also failed: {close_failure}",
            )

    def run_once(self, inputs: Mapping[str, np.ndarray]) -> ExecutionSample:
        """Execute one complete graph sample on the already-open session."""

        if self._closed:
            raise ValueError("UPMEM session is closed")
        validate_dag_inputs(self._dag, inputs)
        self._renew_low_level_deadline()
        started = time.perf_counter()
        working = {tensor_id: np.asarray(value) for tensor_id, value in inputs.items()}
        nodes = {node.node_id: node for node in self._dag.nodes}
        operation_metadata: list[Mapping[str, Any]] = []
        numeric_descriptors: list[tuple[str, Mapping[str, Any], tuple[Any, ...]]] = []
        active_rank_indices: set[int] = set()
        active_dpu_ids: set[tuple[int, int]] = set()
        total_h2d = 0
        total_d2h = 0
        preparation_s = 0.0
        encode_s = 0.0
        decode_s = 0.0
        host_reduce_s = 0.0
        host_reduce_executed = False
        raw_values_all: list[tuple[str, str, str, np.ndarray]] = []
        encoded_operands: list[tuple[str, str, Any]] = []
        stage_id = "run"
        declared_stage_id = "run"
        branch_node_id: str | None = None
        try:
            for stage, declared_stage_id in _session_stage_nodes(self._plan, nodes):
                stage_id = stage.stage_id
                self._check_operation_deadline(started)
                node = nodes[stage.node_ids[0]]
                branch_node_id = node.node_id
                if isinstance(node, ContractNode):
                    left = _resolve_view(node.left, working)
                    right = _resolve_view(node.right, working)
                    core = getattr(self._low_level, "_execute_complex_core", None)
                    if not callable(core):
                        raise RuntimeError(
                            "UPMEM session lacks the no-evidence complex core"
                        )
                    output, metadata, raw_values, encoded = core(
                        node,
                        left,
                        right,
                        stage=stage,
                        numeric_policy=self._plan.numeric_policy,
                        include_evidence=False,
                    )
                    if not isinstance(metadata, Mapping):
                        raise ValueError(
                            "UPMEM operation returned non-mapping metadata"
                        )
                    metadata = {
                        **metadata,
                        "node_id": node.node_id,
                        "declared_stage_id": declared_stage_id,
                    }
                    output = np.asarray(output, dtype=np.complex64)
                    if tuple(output.shape) != node.output.shape:
                        raise ValueError(
                            f"UPMEM node {node.node_id} produced shape {output.shape}; "
                            f"expected {node.output.shape}"
                        )
                    timing = metadata.get("timing", {})
                    if not isinstance(timing, Mapping):
                        raise ValueError("UPMEM operation timing is not a mapping")
                    preparation_s += _seconds(timing.get("preparation_s", 0.0))
                    encode_s += _seconds(timing.get("encode_s", 0.0))
                    decode_s += _seconds(timing.get("decode_s", 0.0))
                    total_h2d += _required_byte_count(
                        metadata,
                        "application_visible_h2d_bytes",
                        "h2d_bytes",
                    )
                    total_d2h += _required_byte_count(
                        metadata,
                        "application_visible_d2h_bytes",
                        "d2h_bytes",
                    )
                    operation_metadata.append(metadata)
                    raw_tuple = tuple(raw_values.values())
                    numeric_descriptors.append((node.node_id, metadata, raw_tuple))
                    raw_values_all.extend(raw_tuple)
                    encoded_operands.extend(
                        (
                            (node.node_id, "left", encoded[0]),
                            (node.node_id, "right", encoded[1]),
                        )
                    )
                    active_rank_indices.update(
                        int(value) for value in metadata.get("active_rank_indices", ())
                    )
                    active_dpu_ids.update(
                        tuple(int(part) for part in value)
                        for value in metadata.get("active_dpu_ids", ())
                    )
                    working[node.output.id] = output
                elif isinstance(node, ReduceNode):
                    host_reduce_executed = True
                    reduce_started = time.perf_counter()
                    producer_ids = {
                        candidate.output.id: candidate.node_id
                        for candidate in self._dag.nodes
                    }
                    ordered = sorted(
                        node.inputs,
                        key=lambda view: (
                            producer_ids.get(view.tensor_id, ""),
                            view.tensor_id,
                            view.slice_spec,
                        ),
                    )
                    values = [
                        np.asarray(_resolve_view(view, working), dtype=np.complex64)
                        for view in ordered
                    ]
                    result = np.array(values[0], dtype=np.complex64, copy=True)
                    for value in values[1:]:
                        result = np.add(result, value, dtype=np.complex64)
                    host_reduce_s += time.perf_counter() - reduce_started
                    working[node.output.id] = result
                else:  # pragma: no cover - model validation closes this union.
                    raise TypeError(
                        f"unsupported UPMEM DAG node: {type(node).__name__}"
                    )
            output = np.array(
                _resolve_view(self._dag.output, working),
                dtype=np.complex64,
                copy=True,
                order="C",
            )
            output.setflags(write=False)
            self._check_operation_deadline(started)
        except ExecutionFailed as exc:
            failure_facts = dict(exc.backend_facts)
            failure_facts.update(
                {
                    "plan_stage_id": declared_stage_id,
                    "branch_stage_id": stage_id,
                    "branch_node_id": branch_node_id,
                }
            )
            raise ExecutionFailed(
                stage=exc.stage,
                reason=exc.reason,
                backend_facts=failure_facts,
            ) from exc
        except Exception as exc:
            failure_stage = getattr(exc, "failure_stage", None)
            if not isinstance(failure_stage, str) or not failure_stage:
                failure_stage = stage_id
            failure_facts = dict(
                self._failure_facts(active_rank_indices, active_dpu_ids)
            )
            backend_facts = getattr(exc, "backend_facts", None)
            if isinstance(backend_facts, Mapping):
                failure_facts.update(backend_facts)
            failure_facts.update(
                {
                    "plan_stage_id": declared_stage_id,
                    "branch_stage_id": stage_id,
                    "branch_node_id": branch_node_id,
                }
            )
            raise ExecutionFailed(
                stage=failure_stage,
                reason=str(exc).strip() or type(exc).__name__,
                backend_facts=failure_facts,
            ) from exc

        total_wall_s = time.perf_counter() - started
        if self._plan.topology.rank_count == 1:
            def phase_sum(field: str) -> float:
                total = 0.0
                for metadata in operation_metadata:
                    timing = metadata.get("timing")
                    if not isinstance(timing, Mapping):
                        raise ValueError("UPMEM operation timing is not a mapping")
                    total += _seconds(timing.get(field, 0.0))
                return total

            h2d_s = phase_sum("rank_response_h2d_max_sum_s")
            kernel_s = phase_sum("rank_response_kernel_max_sum_s")
            d2h_s = phase_sum("rank_response_d2h_max_sum_s")
        else:
            h2d_s = None
            kernel_s = None
            d2h_s = None
        try:
            observations = _derive_operation_observations(
                operation_metadata,
                self._plan,
            )
            operation_facts = tuple(
                _operation_summary(metadata) for metadata in operation_metadata
            )
            raw_lane_records = tuple(
                _raw_lane_fact(node_id, tile_id, lane, value)
                for node_id, tile_id, lane, value in sorted(
                    raw_values_all,
                    key=_raw_value_sort_key,
                )
            )
            operand_records = tuple(
                _complex_operand_facts(node_id, side, encoded)
                for node_id, side, encoded in sorted(
                    encoded_operands,
                    key=lambda item: (item[0], item[1]),
                )
            )
            numeric_operations = tuple(
                _json_safe(
                    {
                        "node_id": node_id,
                        "numeric_policy": self._plan.numeric_policy,
                        "left_scale": metadata.get("left_scale"),
                        "right_scale": metadata.get("right_scale"),
                        "saturation_real": metadata.get("saturation_real", 0),
                        "saturation_imag": metadata.get("saturation_imag", 0),
                        "raw_lane_count": len(raw_values),
                    }
                )
                for node_id, metadata, raw_values in numeric_descriptors
            )
            backend_facts = self._backend_facts(
                active_rank_indices=active_rank_indices,
                active_dpu_ids=active_dpu_ids,
                observations=observations,
                output_hash=_array_hash(output),
                operation_facts=operation_facts,
            )
            normalized_numeric_facts = _json_safe(
                {
                    "numeric_policy": self._plan.numeric_policy,
                    "operations": numeric_operations,
                    "saturation_real": sum(
                        int(item.get("saturation_real", 0))
                        for item in operation_metadata
                    ),
                    "saturation_imag": sum(
                        int(item.get("saturation_imag", 0))
                        for item in operation_metadata
                    ),
                    "operand_records": operand_records,
                    "raw_lane_records": raw_lane_records,
                }
            )
        except Exception as exc:
            raise ExecutionFailed(
                stage="execution_facts",
                reason=str(exc),
                backend_facts=self._failure_facts(
                    active_rank_indices,
                    active_dpu_ids,
                ),
            ) from exc

        return ExecutionSample(
            output=output,
            measurement=Measurement(
                scope_id="steady_execution_v1",
                total_wall_s=total_wall_s,
                preparation_s=preparation_s,
                encode_s=encode_s,
                h2d_s=h2d_s,
                kernel_s=kernel_s,
                host_reduce_s=host_reduce_s if host_reduce_executed else None,
                d2h_s=d2h_s,
                decode_s=decode_s,
                h2d_bytes=total_h2d,
                d2h_bytes=total_d2h,
            ),
            backend_facts=backend_facts,
            numeric_facts=normalized_numeric_facts,
        )

    def close(self) -> Mapping[str, JsonValue]:
        """Close once and admit only fully verified physical terminal facts."""

        if self._closed:
            if self._close_failure is not None:
                raise self._close_failure
            assert self._terminal_facts is not None
            return self._terminal_facts
        self._closed = True
        terminal_facts: Mapping[str, JsonValue] | None = None
        try:
            self._renew_low_level_deadline()
            result = self._low_level.close()
            if not isinstance(result, Mapping):
                raise ValueError("UPMEM session close returned non-mapping facts")
            normalized = _json_safe(dict(result))
            if not isinstance(normalized, Mapping):  # pragma: no cover
                raise TypeError("normalized terminal facts are not a mapping")
            terminal_facts = normalized
            _validate_terminal_admission(terminal_facts, self._plan)
            self._terminal_facts = {
                **terminal_facts,
                **self._startup_resource_admission,
            }
            return self._terminal_facts
        except Exception as exc:
            failure = ExecutionFailed(
                stage="session_close",
                reason=str(exc),
                backend_facts=(
                    terminal_facts
                    if terminal_facts is not None
                    else self._failure_facts(set(), set())
                ),
            )
            self._close_failure = failure
            raise failure from exc

    def _renew_low_level_deadline(self) -> None:
        _renew_session_deadline(self._low_level, self._timeout_s)

    def _check_operation_deadline(self, started: float) -> None:
        if time.perf_counter() - started >= self._timeout_s:
            raise TimeoutError("UPMEM operation deadline expired")

    def _failure_facts(
        self,
        active_rank_indices: set[int],
        active_dpu_ids: set[tuple[int, int]],
    ) -> Mapping[str, JsonValue]:
        return {
            "backend_id": "upmem_final_plan_v1",
            "physical_plan_id": physical_plan_id(self._plan),
            "kernel_policy": self._plan.kernel_policy,
            "kernel_implementation_id": _ACTIVE_MECHANISM_IDS["kernel"],
            "requested_dpus": self._plan.topology.dpu_count,
            "active_dpus": len(active_dpu_ids),
            "active_ranks": tuple(sorted(active_rank_indices)),
            "rank_count": self._plan.topology.rank_count,
            "tasklets_per_dpu": self._plan.topology.tasklets_per_dpu,
            **self._startup_resource_admission,
        }

    def _backend_facts(
        self,
        *,
        active_rank_indices: set[int],
        active_dpu_ids: set[tuple[int, int]],
        observations: Mapping[str, JsonValue],
        output_hash: str,
        operation_facts: tuple[Mapping[str, JsonValue], ...],
    ) -> Mapping[str, JsonValue]:
        simulator = observations["target_observed"] == "sdk_simulator"
        execution_admission = _execution_resource_admission(
            self._plan,
            active_rank_indices=active_rank_indices,
            active_dpu_ids=active_dpu_ids,
        )
        return _json_safe(
            {
                "backend_id": "upmem_final_plan_v1",
                "physical_plan_id": physical_plan_id(self._plan),
                "logical_plan_id": self._plan.logical_plan_id,
                "kernel_policy": self._plan.kernel_policy,
                "kernel_implementation_id": _ACTIVE_MECHANISM_IDS["kernel"],
                "execution_class": (
                    "sdk_simulator" if simulator else "upmem_v4_real_tile"
                ),
                "request_transport": PACKED_OPERATION_TRANSPORT,
                "target_observed": observations["target_observed"],
                "test_double_execution": observations["test_double_execution"],
                "cpu_fallback_used": observations["cpu_fallback_used"],
                "simulator_kernel_executed": observations["simulator_kernel_executed"],
                "requested_dpus": observations["requested_dpu_count"],
                "allocated_dpus": observations["allocated_dpu_count"],
                "active_dpus": len(active_dpu_ids),
                "active_ranks": tuple(sorted(active_rank_indices)),
                "rank_count": observations["rank_count"],
                "tasklets_per_dpu": observations["tasklets_per_dpu"],
                "intermediate_policy": self._plan.intermediate_policy,
                "physical_plan_consumed": observations["physical_stage_consumed"],
                "output_hash": output_hash,
                "packed_operation_count": sum(
                    int(fact.get("packed_operation_count", 0))
                    for fact in operation_facts
                ),
                "packed_operation_bytes": sum(
                    int(fact.get("packed_operation_bytes", 0))
                    for fact in operation_facts
                ),
                "packed_operation_request_count": sum(
                    int(fact.get("packed_operation_request_count", 0))
                    for fact in operation_facts
                ),
                "packed_operation_max_descriptor_count": max(
                    int(fact.get("packed_operation_max_descriptor_count", 0))
                    for fact in operation_facts
                ),
                "packed_operation_max_bytes": max(
                    int(fact.get("packed_operation_max_bytes", 0))
                    for fact in operation_facts
                ),
                "packed_operation_max_payload_bytes": max(
                    int(fact.get("packed_operation_max_payload_bytes", 0))
                    for fact in operation_facts
                ),
                "operation_facts": operation_facts,
                **self._startup_resource_admission,
                **execution_admission,
                "release_verified": None,
                **(
                    {
                        "physical_target_verified": False,
                        "hardware_kernel_executed": False,
                        "timing_claim_applicable": False,
                        "scaling_claim_applicable": False,
                        "speedup_claim_applicable": False,
                        "energy_claim_applicable": False,
                    }
                    if simulator
                    else {}
                ),
                "rank_response_timing_scope": (
                    "sum_of_per_request_max_rank_response_counters_v1"
                ),
            }
        )


_OPERATION_BOOL_FIELDS = (
    "test_double_execution",
    "cpu_fallback_used",
    "hardware_kernel_executed",
    "simulator_kernel_executed",
    "physical_stage_consumed",
    "bulk_set_launch_verified",
)
_OPERATION_COUNT_FIELDS = (
    "requested_dpu_count",
    "allocated_dpu_count",
    "rank_count",
    "tasklets_per_dpu",
)
_TERMINAL_TRUE_FIELDS = (
    "hardware_allocation_verified",
    "ready_verified",
    "binary_identity_verified",
    "native_identity_verified",
    "physical_target_verified",
    "native_kernel_executed",
    "hardware_kernel_executed",
    "hardware_release_verified",
    "hardware_release_confirmed",
)
_TERMINAL_FALSE_FIELDS = (
    "simulator_kernel_executed",
    "cpu_fallback_used",
    "test_double_execution",
)


def _derive_operation_observations(
    operations: list[Mapping[str, Any]],
    plan: FinalUpmemPlan,
) -> Mapping[str, JsonValue]:
    if not operations:
        raise ValueError("UPMEM run produced no operation observations")

    field_names = (
        "target_observed",
        *_OPERATION_BOOL_FIELDS,
        *_OPERATION_COUNT_FIELDS,
    )
    observations: dict[str, JsonValue] = {}
    for field in field_names:
        values: list[object] = []
        for index, operation in enumerate(operations):
            if field not in operation:
                raise ValueError(
                    f"UPMEM operation {index} omitted required field {field!r}"
                )
            value = operation[field]
            if field == "target_observed":
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"UPMEM operation {index} has invalid target_observed"
                    )
            elif field in _OPERATION_BOOL_FIELDS:
                if type(value) is not bool:
                    raise ValueError(
                        f"UPMEM operation {index} field {field!r} must be boolean"
                    )
            elif type(value) is not int or int(value) <= 0:
                raise ValueError(
                    f"UPMEM operation {index} field {field!r} "
                    "must be a positive integer"
                )
            values.append(value)
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"UPMEM operation observations disagree on {field!r}")
        observations[field] = first  # type: ignore[assignment]

    if observations["cpu_fallback_used"] is True:
        raise ValueError("UPMEM operation reported CPU fallback")
    observed_pair = (
        observations["target_observed"],
        observations["test_double_execution"],
        observations["simulator_kernel_executed"],
    )
    if observed_pair not in {
        ("physical_hardware", False, False),
        ("sdk_simulator", False, True),
        ("not_verified", True, False),
    }:
        raise ValueError(
            "UPMEM target observation and test-double flag are inconsistent"
        )
    expected_hardware_kernel = observations["target_observed"] == "physical_hardware"
    if observations["hardware_kernel_executed"] is not expected_hardware_kernel:
        raise ValueError("UPMEM operation kernel target label is inconsistent")
    for field in ("physical_stage_consumed", "bulk_set_launch_verified"):
        if observations[field] is not True:
            raise ValueError(f"UPMEM operation field {field!r} must be true")

    for index, operation in enumerate(operations):
        if operation.get("lane_pass_count") != 4:
            raise ValueError(f"UPMEM operation {index} did not execute four lanes")
        active_ranks = operation.get("active_rank_count")
        active_dpus = operation.get("active_dpu_count")
        if (
            type(active_ranks) is not int
            or not 1 <= active_ranks <= plan.topology.rank_count
        ):
            raise ValueError(f"UPMEM operation {index} has invalid active rank count")
        if (
            type(active_dpus) is not int
            or not 1 <= active_dpus <= plan.topology.dpu_count
        ):
            raise ValueError(f"UPMEM operation {index} has invalid active DPU count")

    expected_counts = {
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "rank_count": plan.topology.rank_count,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }
    for field, expected in expected_counts.items():
        if observations[field] != expected:
            raise ValueError(
                f"UPMEM operation field {field!r} is {observations[field]!r}; "
                f"expected {expected}"
            )
    return observations


def _operation_summary(metadata: Mapping[str, Any]) -> Mapping[str, JsonValue]:
    fields = (
        "stage_id",
        "declared_stage_id",
        "node_id",
        "request_transport",
        "numeric_transport",
        "lane_pass_count",
        "active_rank_indices",
        "active_dpu_ids",
        "active_rank_count",
        "active_dpu_count",
        "requested_dpu_count",
        "allocated_dpu_count",
        "rank_count",
        "tasklets_per_dpu",
        "parallel_rank_wave_count",
        "bulk_set_launch_verified",
        "application_visible_h2d_bytes",
        "application_visible_d2h_bytes",
        "application_visible_transfer_bytes",
        "wram_panel_facts",
        "physical_stage_consumed",
        "target_observed",
        "test_double_execution",
        "cpu_fallback_used",
        "hardware_kernel_executed",
        "simulator_kernel_executed",
        "timing_scope",
        "packed_operation_count",
        "packed_operation_bytes",
        "packed_operation_request_count",
        "packed_operation_max_descriptor_count",
        "packed_operation_max_bytes",
        "packed_operation_max_payload_bytes",
        "timing",
    )
    summary = {field: metadata[field] for field in fields if field in metadata}
    normalized = _json_safe(summary)
    if not isinstance(normalized, Mapping):  # pragma: no cover - fixed input shape.
        raise TypeError("normalized operation summary is not a mapping")
    return normalized


def _session_stage_nodes(
    plan: FinalUpmemPlan,
    nodes: Mapping[str, ContractNode | ReduceNode],
) -> tuple[tuple[UpmemStage, str], ...]:
    records: list[tuple[UpmemStage, str]] = []
    for declared_stage in plan.stages:
        for node_id in declared_stage.node_ids:
            if node_id not in nodes:  # pragma: no cover - plan validation closes this.
                raise ValueError(f"UPMEM stage references unknown node {node_id!r}")
            stage_id = declared_stage.stage_id
            if len(declared_stage.node_ids) > 1:
                stage_id = f"{declared_stage.stage_id}:node:{node_id}"
            records.append(
                (
                    UpmemStage(
                        stage_id=stage_id,
                        kind=declared_stage.kind,
                        node_ids=(node_id,),
                        work_units=tuple(
                            unit
                            for unit in declared_stage.work_units
                            if unit.node_id == node_id
                        ),
                    ),
                    declared_stage.stage_id,
                )
            )
    return tuple(records)


def _raw_value_sort_key(
    value: tuple[str, str, str, np.ndarray],
) -> tuple[str, str, int]:
    lane_order = {"rr": 0, "ii": 1, "ri": 2, "ir": 3}
    return value[0], value[1], lane_order.get(value[2], len(lane_order))


def _validate_terminal_admission(
    terminal_facts: Mapping[str, JsonValue],
    plan: FinalUpmemPlan,
) -> None:
    simulator = terminal_facts.get("target_observed") == "sdk_simulator"
    if terminal_facts.get("target_observed") not in {
        "physical_hardware",
        "sdk_simulator",
    }:
        raise ValueError("terminal target_observed is not a supported v4 target")
    true_fields = _TERMINAL_TRUE_FIELDS
    false_fields = _TERMINAL_FALSE_FIELDS
    if simulator:
        true_fields = tuple(
            field
            for field in _TERMINAL_TRUE_FIELDS
            if field
            not in {
                "hardware_allocation_verified",
                "physical_target_verified",
                "hardware_kernel_executed",
            }
        )
        false_fields = tuple(
            field
            for field in _TERMINAL_FALSE_FIELDS
            if field != "simulator_kernel_executed"
        )
        if terminal_facts.get("simulator_target_verified") is not True:
            raise ValueError(
                "simulator terminal facts require simulator_target_verified"
            )
        if terminal_facts.get("simulator_kernel_executed") is not True:
            raise ValueError("simulator terminal facts require simulator execution")
        if terminal_facts.get("hardware_allocation_verified") is not False:
            raise ValueError(
                "simulator terminal facts cannot verify hardware allocation"
            )
        if terminal_facts.get("physical_target_verified") is not False:
            raise ValueError("simulator terminal facts cannot verify physical target")
        if terminal_facts.get("hardware_kernel_executed") is not False:
            raise ValueError("simulator terminal facts cannot verify hardware kernel")
        for field in (
            "timing_claim_applicable",
            "scaling_claim_applicable",
            "speedup_claim_applicable",
            "energy_claim_applicable",
        ):
            if terminal_facts.get(field) is not False:
                raise ValueError(f"simulator terminal field {field!r} must be false")
    for field in true_fields:
        if terminal_facts.get(field) is not True:
            raise ValueError(f"terminal field {field!r} must be exactly true")
    for field in false_fields:
        if terminal_facts.get(field) is not False:
            raise ValueError(f"terminal field {field!r} must be exactly false")

    expected_counts = {
        "requested_dpu_count": plan.topology.dpu_count,
        "allocated_dpu_count": plan.topology.dpu_count,
        "observed_rank_count": plan.topology.rank_count,
        "observed_tasklets_per_dpu": plan.topology.tasklets_per_dpu,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }
    for field, expected in expected_counts.items():
        observed = terminal_facts.get(field)
        if type(observed) is not int or observed != expected:
            raise ValueError(
                f"terminal field {field!r} is {observed!r}; expected {expected}"
            )


def _deadline_owner(low_level: object) -> object:
    current = low_level
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        namespace = getattr(current, "__dict__", {})
        if isinstance(namespace, Mapping) and "_deadline" in namespace:
            return current
        try:
            current = object.__getattribute__(current, "session")
        except AttributeError:
            break
    raise TypeError("UPMEM low-level session has no renewable deadline")


def _renew_session_deadline(low_level: object, timeout_s: float) -> None:
    owner = _deadline_owner(low_level)
    setattr(owner, "_deadline", time.monotonic() + float(timeout_s))


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    try:  # Python 3.10 compatibility.
        notes = list(getattr(error, "__notes__", ()))
        notes.append(note)
        setattr(error, "__notes__", notes)
    except Exception:  # pragma: no cover - unusual immutable exception type.
        pass


def _session_contract_error(low_level: object) -> str | None:
    if not callable(getattr(low_level, "close", None)):
        return "UPMEM session does not provide close()"
    if not callable(getattr(low_level, "_execute_complex_core", None)):
        return "UPMEM session does not provide the no-evidence complex core"
    try:
        _deadline_owner(low_level)
    except TypeError as exc:
        return str(exc)
    return None


def _cleanup_nonconforming_session(
    low_level: object,
) -> Mapping[str, JsonValue]:
    close = getattr(low_level, "close", None)
    if not callable(close):
        return {
            "nonconforming_cleanup_attempted": False,
            "nonconforming_cleanup_succeeded": False,
        }
    try:
        close()
    except Exception as exc:
        return {
            "nonconforming_cleanup_attempted": True,
            "nonconforming_cleanup_succeeded": False,
            "nonconforming_cleanup_error": str(exc) or type(exc).__name__,
        }
    return {
        "nonconforming_cleanup_attempted": True,
        "nonconforming_cleanup_succeeded": True,
    }


def _open_failure_facts(plan: FinalUpmemPlan) -> dict[str, JsonValue]:
    return {
        "backend_id": "upmem_final_plan_v1",
        "physical_plan_id": physical_plan_id(plan),
        "logical_plan_id": plan.logical_plan_id,
        "kernel_policy": plan.kernel_policy,
        "kernel_implementation_id": _ACTIVE_MECHANISM_IDS["kernel"],
        "requested_dpus": plan.topology.dpu_count,
        "rank_count": plan.topology.rank_count,
        "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }


def _startup_resource_admission(
    low_level: object,
    plan: FinalUpmemPlan,
    *,
    expected_target: str,
) -> Mapping[str, JsonValue]:
    """Check READY facts after opening and before the first timed operation."""

    ranks = getattr(low_level, "ranks", None)
    engine = getattr(low_level, "engine", None)
    if not isinstance(ranks, (list, tuple)) or engine is None:
        return {
            "startup_resource_admission_passed": True,
            "startup_resource_admission_reasons": (),
            "startup_resource_admission_source": "injected_session",
        }
    reasons: list[str] = []
    expected_target_observed = (
        "sdk_simulator" if expected_target == EXECUTION_TARGET_SIMULATOR else "physical_hardware"
    )
    if len(ranks) != plan.topology.rank_count:
        reasons.append("rank_count_mismatch")
    allocated_dpus = 0
    ready_records: list[Mapping[str, JsonValue]] = []
    provenance = getattr(engine, "_binary_provenance", {})
    for rank in ranks:
        startup = getattr(getattr(rank, "session", None), "startup", {})
        local_dpus = getattr(rank, "local_dpus", None)
        if not isinstance(startup, Mapping) or type(local_dpus) is not int:
            reasons.append("ready_facts_missing")
            continue
        allocated = startup.get("allocated_dpu_count")
        requested = startup.get("requested_dpu_count")
        tasklets = startup.get("tasklets_per_dpu")
        if requested != local_dpus or allocated != local_dpus:
            reasons.append("ready_dpu_count_mismatch")
        if tasklets != plan.topology.tasklets_per_dpu:
            reasons.append("ready_tasklet_count_mismatch")
        if startup.get("target_observed") != expected_target_observed:
            reasons.append("ready_target_mismatch")
        for key in ("dpu_binary_sha256", "initialization_binary_sha256"):
            if startup.get(key) != provenance.get(key):
                reasons.append("ready_binary_identity_mismatch")
                break
        if type(allocated) is int:
            allocated_dpus += allocated
        ready_records.append(
            {
                "rank_index": int(getattr(rank, "index", -1)),
                "requested_dpu_count": requested if type(requested) is int else None,
                "allocated_dpu_count": allocated if type(allocated) is int else None,
                "tasklets_per_dpu": tasklets if type(tasklets) is int else None,
                "target_observed": (
                    startup.get("target_observed")
                    if isinstance(startup.get("target_observed"), str)
                    else None
                ),
            }
        )
    if allocated_dpus != plan.topology.dpu_count:
        reasons.append("allocated_dpu_total_mismatch")
    return {
        "startup_resource_admission_passed": not reasons,
        "startup_resource_admission_reasons": tuple(sorted(set(reasons))),
        "startup_resource_admission_source": "native_ready",
        "startup_ready_records": tuple(ready_records),
        "startup_requested_dpu_count": plan.topology.dpu_count,
        "startup_allocated_dpu_count": allocated_dpus,
        "startup_requested_tasklets_per_dpu": plan.topology.tasklets_per_dpu,
    }


def _execution_resource_admission(
    plan: FinalUpmemPlan,
    *,
    active_rank_indices: set[int],
    active_dpu_ids: set[tuple[int, int]],
) -> Mapping[str, JsonValue]:
    """Persist post-execution resource facts without inferring concurrent timing."""

    planned = collection_resource_admission(plan)
    reasons: list[str] = []
    if len(active_dpu_ids) != plan.topology.dpu_count:
        reasons.append("active_dpu_count_mismatch")
    if len(active_rank_indices) != plan.topology.rank_count:
        reasons.append("active_rank_count_mismatch")
    return {
        **planned,
        "execution_resource_admission_passed": not reasons,
        "execution_resource_admission_reasons": tuple(reasons),
        "execution_active_dpu_count": len(active_dpu_ids),
        "execution_active_rank_count": len(active_rank_indices),
    }


def open_upmem(
    dag: ContractionDAG,
    plan: FinalUpmemPlan,
    resources: UpmemResources,
    *,
    timeout_s: float = 120.0,
) -> UpmemSession:
    """Open one persistent session for a validated final UPMEM plan."""

    if not isinstance(dag, ContractionDAG):
        raise ValueError("open_upmem requires a ContractionDAG")
    if not isinstance(plan, FinalUpmemPlan):
        raise ValueError("open_upmem requires the final UpmemPlan record")
    if not isinstance(resources, UpmemResources):
        raise ValueError("open_upmem requires the final UpmemResources record")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be finite and positive")
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")

    validate_contraction_dag(dag)
    if plan.logical_plan_id != contraction_dag_hash(dag):
        raise ValueError("UPMEM physical plan does not match supplied DAG")
    validate_upmem_plan(dag, plan)
    if len(resources.rank_paths) != plan.topology.rank_count:
        raise UnsupportedExecution(
            stage="preflight",
            reason="resource rank paths do not match the final topology",
            capability="rank_topology",
        )
    if not resources.rank_paths:
        raise UnsupportedExecution(
            stage="preflight",
            reason="explicit UPMEM rank paths are required",
            capability="rank_paths",
        )
    _validate_final_resources(resources)

    try:
        if resources.session_opener is not None:
            low_level = resources.session_opener(
                dag,
                plan,
                resources,
                timeout_s,
            )
        else:
            engine = UpmemV4Executor(
                session_root=Path(resources.session_root),
                host_binary=Path(resources.host_binary),
                dpu_binary=Path(resources.dpu_binary),
                initialization_binary=Path(resources.initialization_binary),
                rank_paths=resources.rank_paths,
                dpu_count=plan.topology.dpu_count,
                tasklets_per_dpu=plan.topology.tasklets_per_dpu,
                timeout_s=timeout_s,
            )
            low_level = engine.open_session(
                plan.numeric_policy,
                plan.topology,
            )
    except UnsupportedExecution:
        raise
    except Exception as exc:
        raise ExecutionFailed(
            stage="session_open",
            reason=str(exc),
            backend_facts=_open_failure_facts(plan),
        ) from exc

    contract_error = _session_contract_error(low_level)
    if contract_error is not None:
        facts = _open_failure_facts(plan)
        facts.update(_cleanup_nonconforming_session(low_level))
        failure = ExecutionFailed(
            stage="session_open",
            reason=contract_error,
            backend_facts=facts,
        )
        raise failure
    startup_admission = _startup_resource_admission(
        low_level,
        plan,
        expected_target=EXECUTION_TARGET_PHYSICAL,
    )
    if startup_admission["startup_resource_admission_passed"] is False:
        facts = _open_failure_facts(plan)
        facts.update(startup_admission)
        facts.update(_cleanup_nonconforming_session(low_level))
        raise ExecutionFailed(
            stage="resource_admission",
            reason="native READY facts did not satisfy resource admission",
            backend_facts=facts,
        )
    return UpmemSession(
        dag, plan, resources, low_level, timeout_s, startup_admission
    )


def open_upmem_simulator(
    dag: ContractionDAG,
    plan: FinalUpmemPlan,
    resources: UpmemResources,
    *,
    timeout_s: float = 120.0,
) -> UpmemSession:
    """Open the active ABI-v4 route through the SDK simulator only.

    This path is limited to one simulated rank with 1..64 DPUs and is
    admitted solely for protocol and numerical correctness.  It deliberately
    has no physical opt-in or rank-path requirement.
    """

    if not isinstance(dag, ContractionDAG):
        raise ValueError("open_upmem_simulator requires a ContractionDAG")
    if not isinstance(plan, FinalUpmemPlan):
        raise ValueError("open_upmem_simulator requires the final UpmemPlan record")
    if not isinstance(resources, UpmemResources):
        raise ValueError(
            "open_upmem_simulator requires the final UpmemResources record"
        )
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError("timeout_s must be finite and positive")
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    validate_contraction_dag(dag)
    if plan.logical_plan_id != contraction_dag_hash(dag):
        raise ValueError("UPMEM physical plan does not match supplied DAG")
    validate_upmem_plan(dag, plan)
    if not 1 <= plan.topology.dpu_count <= 64 or plan.topology.rank_count != 1:
        raise UnsupportedExecution(
            stage="preflight",
            reason="SDK simulator v4 route requires one rank and 1..64 DPUs",
            capability="simulator_topology",
        )
    if resources.rank_paths:
        raise UnsupportedExecution(
            stage="preflight",
            reason="SDK simulator v4 route forbids physical rank paths",
            capability="simulator_rank_paths",
        )
    if resources.session_opener is not None:
        raise UnsupportedExecution(
            stage="preflight",
            reason="SDK simulator v4 route does not accept injected session openers",
            capability="simulator_target_boundary",
        )
    _validate_final_resources(resources)
    try:
        engine = UpmemV4Executor(
            session_root=Path(resources.session_root),
            host_binary=Path(resources.host_binary),
            dpu_binary=Path(resources.dpu_binary),
            initialization_binary=Path(resources.initialization_binary),
            rank_paths=(),
            dpu_count=plan.topology.dpu_count,
            tasklets_per_dpu=plan.topology.tasklets_per_dpu,
            timeout_s=timeout_s,
            execution_target=EXECUTION_TARGET_SIMULATOR,
        )
        low_level = engine.open_session(plan.numeric_policy, plan.topology)
    except UnsupportedExecution:
        raise
    except Exception as exc:
        raise ExecutionFailed(
            stage="session_open",
            reason=str(exc),
            backend_facts=_open_failure_facts(plan),
        ) from exc
    contract_error = _session_contract_error(low_level)
    if contract_error is not None:
        facts = _open_failure_facts(plan)
        facts.update(_cleanup_nonconforming_session(low_level))
        raise ExecutionFailed(
            stage="session_open",
            reason=contract_error,
            backend_facts=facts,
        )
    startup_admission = _startup_resource_admission(
        low_level,
        plan,
        expected_target=EXECUTION_TARGET_SIMULATOR,
    )
    if startup_admission["startup_resource_admission_passed"] is False:
        facts = _open_failure_facts(plan)
        facts.update(startup_admission)
        facts.update(_cleanup_nonconforming_session(low_level))
        raise ExecutionFailed(
            stage="resource_admission",
            reason="native READY facts did not satisfy resource admission",
            backend_facts=facts,
        )
    return UpmemSession(
        dag, plan, resources, low_level, timeout_s, startup_admission
    )


def _json_safe(value: object) -> JsonValue:
    """Convert runtime metadata to the narrow evidence JSON value contract."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("runtime facts must contain finite floats")
        return result
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_safe(item) for item in value)
    raise TypeError(f"runtime facts contain unsupported value {type(value).__name__}")


def _validate_final_resources(resources: UpmemResources) -> None:
    """Reject malformed production binary paths before opening a session.

    The injected opener used by tests is still given real files.  Keeping this
    validation outside the opener means a production run cannot create native
    side effects before its static executable inputs are known to be usable.
    """

    paths = {
        "host_binary": Path(resources.host_binary),
        "dpu_binary": Path(resources.dpu_binary),
        "initialization_binary": Path(resources.initialization_binary),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise UnsupportedExecution(
                stage="preflight",
                reason=f"UPMEM {name} is not a regular file: {path}",
                capability="native_binaries",
            )
    if not os.access(paths["host_binary"], os.X_OK):
        raise UnsupportedExecution(
            stage="preflight",
            reason="UPMEM host_binary is not executable",
            capability="native_binaries",
        )


__all__ = [
    "UpmemSession",
    "UpmemV4Executor",
    "UpmemV4Session",
    "open_upmem",
    "open_upmem_simulator",
]
