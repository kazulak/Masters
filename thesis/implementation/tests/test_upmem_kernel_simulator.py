from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

from quantum_bench.cpu import replay_upmem_plan_once
from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.plan import (
    UpmemResources,
    UpmemTopology,
    UpmemWorkUnit,
    plan_upmem,
)
from quantum_bench.upmem.protocol import (
    EXECUTION_TARGET_SIMULATOR,
    MAX_CONTRACTED,
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4WorkUnit,
    build_v4_request,
)
from quantum_bench.upmem.packed_operation import (
    build_packed_v4_request,
    pack_operation,
)
from quantum_bench.upmem.runtime import (
    UpmemV4Executor,
    _wram_panel_operation_facts,
    open_upmem_simulator,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "upmem" / "runtime"
SOURCE = NATIVE / "dpu.c"
MAKEFILE = NATIVE / "Makefile"

_LEGACY_SDK_CASES = (
    ("scalar_1x1x1", "float32", 1, 1, 1, 1),
    ("tail_m3_n35_k65", "float32", 1, 3, 35, 65),
    ("tasklet_t1_m17_n32_k65", "float32", 1, 17, 32, 65),
    ("tasklet_t2_m17_n32_k65", "float32", 2, 17, 32, 65),
    ("tasklet_t4_m17_n32_k65", "float32", 4, 17, 32, 65),
    ("tasklet_t8_m17_n32_k65", "float32", 8, 17, 32, 65),
    ("t8_k65", "float32", 8, 8, 32, 65),
    ("t8_k130", "float32", 8, 8, 32, 130),
    ("direct_k257", "float32", 8, 8, 32, 257),
    ("planned_k257", "planned", 8, 8, 32, 257),
    ("int8_tail", "int8", 8, 8, 35, 130),
    ("t24_functional", "float32", 24, 24, 32, 65),
)
_RESOURCE_GENERAL_SDK_CASES = (
    ("resource_general_t3_m2", "float32", 3, 2, 32, 65),
    ("resource_general_t5_m5", "float32", 5, 5, 32, 65),
    ("resource_general_t7_m8", "float32", 7, 8, 32, 65),
    ("resource_general_t12_m17", "float32", 12, 17, 32, 65),
    ("resource_general_t16_m33", "float32", 16, 33, 32, 65),
    ("resource_general_t24_m16", "float32", 24, 16, 32, 65),
)
_SDK_CASES = _LEGACY_SDK_CASES + _RESOURCE_GENERAL_SDK_CASES
_SDK_CASE_IDS = tuple(case[0] for case in _SDK_CASES)
_LEGACY_SDK_CASE_IDS = tuple(case[0] for case in _LEGACY_SDK_CASES)
_SDK_CASE_RESULTS: dict[str, str] = {}


@pytest.fixture(scope="session", autouse=True)
def _write_sdk_case_summary_at_session_end() -> None:
    """Emit direct-case coverage for the external SDK qualification script."""

    yield
    destination = os.environ.get("UPMEM_SDK_SIMULATOR_CASE_SUMMARY")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = list(_LEGACY_SDK_CASE_IDS)
    executed = sorted(
        case_id
        for case_id, result in _SDK_CASE_RESULTS.items()
        if case_id in _LEGACY_SDK_CASE_IDS and result in {"passed", "failed"}
    )
    passed = sorted(
        case_id
        for case_id, result in _SDK_CASE_RESULTS.items()
        if case_id in _LEGACY_SDK_CASE_IDS and result == "passed"
    )
    failed = sorted(
        case_id
        for case_id, result in _SDK_CASE_RESULTS.items()
        if case_id in _LEGACY_SDK_CASE_IDS and result == "failed"
    )
    skipped = sorted(
        case_id
        for case_id, result in _SDK_CASE_RESULTS.items()
        if case_id in _LEGACY_SDK_CASE_IDS and result == "skipped"
    )
    path.write_text(
        json.dumps(
            {
                "required_case_ids": required,
                "executed_case_ids": executed,
                "passed_case_ids": passed,
                "failed_case_ids": failed,
                "skipped_case_ids": skipped,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _require_sdk_simulator(case_id: str) -> None:
    missing: list[str] = []
    for command, label in (
        ("make", "make"),
        ("dpu-upmem-dpurte-clang", "dpu-upmem-dpurte-clang"),
        ("dpu-pkg-config", "UPMEM host libraries"),
    ):
        if shutil.which(command) is None:
            missing.append(label)
    if not missing:
        probe = subprocess.run(
            ["dpu-pkg-config", "--cflags", "--libs", "dpu"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            missing.append("UPMEM host libraries")
    if not missing:
        return
    _SDK_CASE_RESULTS[case_id] = "skipped"
    reason = "SDK simulator prerequisites unavailable: " + ", ".join(missing)
    if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_build_tool(command: str, label: str) -> None:
    if shutil.which(command) is not None:
        return
    reason = f"native build prerequisite unavailable: {label}"
    if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@lru_cache(maxsize=None)
def _sdk_binaries(tasklets: int) -> tuple[Path, Path, Path]:
    result = subprocess.run(
        ["make", "-C", str(NATIVE), "v4", f"NR_TASKLETS={tasklets}"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    paths = (
        NATIVE / "bin" / f"host_upmem_execution_plan_v4_t{tasklets}",
        NATIVE / "bin" / f"dpu_gemm_tile_v4_t{tasklets}",
        NATIVE / "bin" / f"dpu_simplepim_management_init_t{tasklets}",
    )
    assert all(path.is_file() for path in paths)
    return paths


def _direct_values(
    *, m_size: int, n_size: int, k_size: int, numeric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if numeric == "int8":
        left = ((np.arange(m_size * k_size) % 17) - 8).astype(np.int8).reshape(
            m_size, k_size
        )
        right = ((np.arange(k_size * n_size) % 19) - 9).astype(np.int8).reshape(
            k_size, n_size
        )
        expected = left.astype(np.int64) @ right.astype(np.int64)
        return left, right, expected, NUMERIC_HOST_PACKED_INT8
    left = ((np.arange(m_size * k_size) % 17) - 8).astype(np.float32).reshape(
        m_size, k_size
    )
    right = ((np.arange(k_size * n_size) % 19) - 9).astype(np.float32).reshape(
        k_size, n_size
    )
    return left, right, left @ right, NUMERIC_FLOAT32


def _run_direct_sdk_case(
    tmp_path: Path,
    *,
    case_id: str,
    numeric: str,
    tasklets: int,
    m_size: int,
    n_size: int,
    k_size: int,
) -> None:
    _require_sdk_simulator(case_id)
    host, dpu, initialization = _sdk_binaries(tasklets)
    left, right, expected, numeric_mode = _direct_values(
        m_size=m_size, n_size=n_size, k_size=k_size, numeric=numeric
    )
    engine = UpmemV4Executor(
        session_root=tmp_path / "session",
        host_binary=host,
        dpu_binary=dpu,
        initialization_binary=initialization,
        rank_paths=(),
        dpu_count=1,
        tasklets_per_dpu=tasklets,
        timeout_s=120.0,
        execution_target=EXECUTION_TARGET_SIMULATOR,
    )
    topology = UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=tasklets)
    session = engine.open_session(
        "complex_int8_shared_scale_v1"
        if numeric == "int8"
        else "split_complex_float32_v1",
        topology,
    )
    try:
        rank = session.ranks[0]
        request = build_packed_v4_request(
            rank.root,
            profile=rank.session.profile,
            canonical_batch_count=1,
            canonical_m=m_size,
            canonical_n=n_size,
            canonical_k=k_size,
            work_units=(
                V4WorkUnit(
                    local_dpu_id=0,
                    tile_id=1,
                    batch_index=0,
                    m_offset=0,
                    n_offset=0,
                    k_offset=0,
                    m_elements=m_size,
                    n_elements=n_size,
                    k_elements=k_size,
                    a_payload=np.ascontiguousarray(left).tobytes(),
                    b_payload=np.ascontiguousarray(right).tobytes(),
                ),
            ),
            task_contract_sha256="ab" * 32,
            request_sequence=1,
        )
        operation = pack_operation(
            rank.root,
            requests=(request,),
            operation_sequence=1,
            filename="packed/operation_0000000000000001.bin",
        )
        operation.path.parent.mkdir(parents=True, exist_ok=True)
        operation.path.write_bytes(operation.data)
        response = rank.session.submit_packed(operation, timeout_s=120.0)
        assert response["responses"][0]["target_observed"] == "sdk_simulator"
        assert response["responses"][0]["simulator_kernel_executed"] is True
        assert response["responses"][0]["cpu_fallback_used"] is False
        dtype = (
            np.dtype("<i4")
            if numeric_mode == NUMERIC_HOST_PACKED_INT8
            else np.dtype("<f4")
        )
        actual = np.fromfile(
            request.output_paths[0], dtype=dtype, count=m_size * n_size
        ).reshape(m_size, n_size)
        if numeric == "int8":
            np.testing.assert_array_equal(actual.astype(np.int64), expected)
        else:
            np.testing.assert_array_equal(actual, expected)
    finally:
        operation.path.unlink(missing_ok=True)
        session.close()


def test_packed_operation_sdk_simulator_case(tmp_path: Path) -> None:
    """Exercise the native UPOENV2 dispatch without physical hardware."""

    case_id = "packed_operation_v2"
    _require_sdk_simulator(case_id)
    host, dpu, initialization = _sdk_binaries(8)
    left, right, expected, _numeric_mode = _direct_values(
        m_size=3, n_size=35, k_size=65, numeric="float32"
    )
    engine = UpmemV4Executor(
        session_root=tmp_path / "session",
        host_binary=host,
        dpu_binary=dpu,
        initialization_binary=initialization,
        rank_paths=(),
        dpu_count=1,
        tasklets_per_dpu=8,
        timeout_s=120.0,
        execution_target=EXECUTION_TARGET_SIMULATOR,
    )
    topology = UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=8)
    session = engine.open_session("split_complex_float32_v1", topology)
    try:
        rank = session.ranks[0]
        def make_request(sequence: int, tile_id: int) -> Any:
            return build_packed_v4_request(
                rank.root,
                profile=rank.session.profile,
                canonical_batch_count=1,
                canonical_m=3,
                canonical_n=35,
                canonical_k=65,
                work_units=(
                    V4WorkUnit(
                        local_dpu_id=0,
                        tile_id=tile_id,
                        batch_index=0,
                        m_offset=0,
                        n_offset=0,
                        k_offset=0,
                        m_elements=3,
                        n_elements=35,
                        k_elements=65,
                        a_payload=np.ascontiguousarray(left).tobytes(),
                        b_payload=np.ascontiguousarray(right).tobytes(),
                    ),
                ),
                task_contract_sha256="ab" * 32,
                request_sequence=sequence,
            )

        request = make_request(1, 1)
        second_request = make_request(2, 2)
        operation = pack_operation(
            rank.root,
            requests=(request, second_request),
            operation_sequence=1,
            filename="packed/operation_0000000000000001.bin",
        )
        operation.path.parent.mkdir(parents=True, exist_ok=True)
        operation.path.write_bytes(operation.data)
        response = rank.session.submit_packed(operation, timeout_s=120.0)
        assert response["event"] == "OPERATION_RESPONSE"
        assert response["status"] == "completed"
        assert response["response_count"] == 2
        assert response["responses"][0]["target_observed"] == "sdk_simulator"
        assert response["responses"][1]["target_observed"] == "sdk_simulator"
        for packed_request in (request, second_request):
            output = packed_request.root / packed_request.work_units[0].c_path
            actual = np.fromfile(
                output, dtype=np.dtype("<f4"), count=3 * 35
            ).reshape(3, 35)
            np.testing.assert_array_equal(actual, expected)
    except Exception:
        _SDK_CASE_RESULTS[case_id] = "failed"
        raise
    finally:
        session.close()
    _SDK_CASE_RESULTS[case_id] = "passed"


def _planned_k257_node() -> tuple[ContractionDAG, ContractNode, dict[str, np.ndarray]]:
    left = TensorSpec("left", (0, 1), (8, 257), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (257, 32), "dense", dtype="complex128")
    node = ContractNode(
        node_id="planned_k257",
        left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id=right.id, labels=right.labels, shape=right.shape),
        output=TensorSpec("out", (0, 2), (8, 32), "dense", dtype="complex128"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=(0, 2), shape=(8, 32)),
    )
    values = np.arange(8 * 257, dtype=np.float64).reshape(8, 257)
    inputs = {
        "left": values + 1j * (values % 5),
        "right": (
            np.arange(257 * 32, dtype=np.float64).reshape(257, 32) % 11
        )
        + 1j * 2.0,
    }
    return dag, node, inputs


def _work_unit(*, m_size: int, n_size: int, k_size: int) -> UpmemWorkUnit:
    return UpmemWorkUnit(
        node_id="contract",
        stable_tile_id="contract:tile",
        wave=0,
        logical_rank=0,
        logical_dpu=0,
        batch_start=0,
        batch_size=1,
        m_start=0,
        m_size=m_size,
        n_start=0,
        n_size=n_size,
        k_start=0,
        k_size=k_size,
        estimated_input_bytes=0,
        estimated_output_bytes=0,
        aligned_mram_bytes=0,
        estimated_arithmetic_work=m_size * n_size * k_size,
    )


def test_active_wram_panel_target_uses_the_v4_binary() -> None:
    makefile = MAKEFILE.read_text(encoding="ascii")

    active_v4_rule = makefile.split("v4:", 1)[1].split("bin:", 1)[0]
    assert "bin/dpu_gemm_tile_v4_t$(NR_TASKLETS)" in active_v4_rule
    assert "dpu_gemm_tile_v4_wram_panel_internal" not in makefile
    assert not (NATIVE / "dpu_wram_panel_internal.c").exists()


def test_active_wram_panel_source_uses_global_staging_and_bounded_dma() -> None:
    source = SOURCE.read_text(encoding="ascii")

    assert "#define KC EXECUTION_PLAN_V4_WRAM_PANEL_KC" in source
    assert "#define NC EXECUTION_PLAN_V4_WRAM_PANEL_NC" in source
    assert "#define B_PANEL_DATA_BYTES (KC * NC * sizeof(float))" in source
    assert "#define B_PANEL_ROW_STRIDE_BYTES (NC * sizeof(float))" in source
    assert "#define A_BUFFER_DATA_BYTES (KC * sizeof(float))" in source
    assert "#define OUTPUT_BUFFER_DATA_BYTES (NC * sizeof(uint32_t))" in source
    assert "#define UNALIGNED_SCRATCH_BYTES EXECUTION_PLAN_V4_WRAM_PANEL_UNALIGNED_SCRATCH_BYTES" in source
    assert "__dma_aligned uint8_t shared_b_panel" in source
    assert "__dma_aligned uint8_t tasklet_a_buffer" in source
    assert "__dma_aligned v4_output_slot_t tasklet_output_buffer" in source
    assert "__dma_aligned uint8_t tasklet_unaligned_scratch" in source
    assert "mram_read_unaligned" in source
    assert "mram_write_unaligned" in source
    assert "const uint32_t dst_align = mram_c_offset & 7u;" in source
    assert "tasklet_unaligned_scratch[tid] + dst_align" in source
    assert "B_CONTIGUOUS_CHUNK_BYTES <= 2048u" in source
    assert "barrier_wait(&v4_barrier);" in source

    b_stage = source.split("if (actual_kc == KC", 1)[1].split(
        "barrier_wait(&v4_barrier);", 1
    )[0]
    arithmetic = source.split("if (is_int8)", 1)[1].split(
        "barrier_wait(&v4_barrier);", 1
    )[0]
    assert "mram_read(" in b_stage
    assert "mram_read(" not in arithmetic
    assert "mram_read_unaligned" not in arithmetic


def test_wram_panel_facts_count_four_real_products_for_a_full_panel() -> None:
    facts = _wram_panel_operation_facts(
        (_work_unit(m_size=2, n_size=32, k_size=64),),
        numeric_policy="split_complex_float32_v1",
        tasklets_per_dpu=1,
    )

    assert facts == {
        "origin": "wram_panel_algorithm_v1",
        "lane_count": 4,
        "a_read_helper_calls_exact": 8,
        "b_read_helper_calls_exact": 16,
        "partial_c_read_helper_calls_exact": 0,
        "c_write_helper_calls_exact": 8,
        "a_read_payload_bytes_exact": 2_048,
        "b_read_payload_bytes_exact": 32_768,
        "partial_c_read_payload_bytes_exact": 0,
        "c_write_payload_bytes_exact": 1_024,
        "a_read_aligned_span_bytes_estimate": 2_048,
        "b_read_aligned_span_bytes_estimate": 32_768,
        "partial_c_read_aligned_span_bytes_estimate": 0,
        "c_write_aligned_span_bytes_estimate": 1_024,
        "operand_read_helper_calls_exact": 24,
        "output_partial_read_helper_calls_exact": 0,
        "output_write_helper_calls_exact": 8,
        "mram_requested_payload_bytes_exact": 35_840,
        "mram_aligned_transfer_bytes_estimate": 35_840,
        "barrier_events_exact": 24,
        "barrier_tasklet_calls_exact": 24,
        "real_mac_count_exact": 16_384,
        "wram_shared_bytes_exact": 8_192,
        "wram_private_bytes_per_tasklet_exact": 672,
        "wram_kernel_buffers_allocated_bytes_exact": 8_864,
        "mram_helper_count_scope": "source_level_helper_calls",
        "mram_aligned_bytes_scope": "geometric_aligned_span_estimate",
    }


def test_wram_panel_facts_account_for_tail_helpers_and_tasklets() -> None:
    facts = _wram_panel_operation_facts(
        (_work_unit(m_size=3, n_size=35, k_size=65),),
        numeric_policy="complex_int8_shared_scale_v1",
        tasklets_per_dpu=8,
    )

    assert facts["operand_read_helper_calls_exact"] > 0
    assert facts["output_partial_read_helper_calls_exact"] > 0
    assert facts["mram_aligned_transfer_bytes_estimate"] >= facts[
        "mram_requested_payload_bytes_exact"
    ]
    assert facts["barrier_tasklet_calls_exact"] == 8 * facts["barrier_events_exact"]
    assert facts["real_mac_count_exact"] == 4 * 3 * 35 * 65
    assert facts["wram_kernel_buffers_allocated_bytes_exact"] == 8_192 + 8 * 672
    assert facts["operand_read_helper_calls_exact"] == (
        facts["a_read_helper_calls_exact"] + facts["b_read_helper_calls_exact"]
    )
    assert facts["mram_requested_payload_bytes_exact"] == sum(
        facts[field]
        for field in (
            "a_read_payload_bytes_exact",
            "b_read_payload_bytes_exact",
            "partial_c_read_payload_bytes_exact",
            "c_write_payload_bytes_exact",
        )
    )


@pytest.mark.parametrize("tasklets", [1, 2, 4, 8, 24])
def test_active_wram_panel_binary_builds_when_sdk_compiler_is_available(tasklets: int) -> None:
    _require_native_build_tool("make", "make")
    _require_native_build_tool("dpu-upmem-dpurte-clang", "dpu-upmem-dpurte-clang")

    result = subprocess.run(
        [
            "make",
            "-C",
            str(NATIVE),
            f"bin/dpu_gemm_tile_v4_t{tasklets}",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (NATIVE / "bin" / f"dpu_gemm_tile_v4_t{tasklets}").is_file()


@pytest.mark.parametrize(
    ("case_id", "numeric", "tasklets", "m_size", "n_size", "k_size"),
    _SDK_CASES,
    ids=_SDK_CASE_IDS,
)
def test_direct_sdk_simulator_case_matrix(
    tmp_path: Path,
    case_id: str,
    numeric: str,
    tasklets: int,
    m_size: int,
    n_size: int,
    k_size: int,
) -> None:
    """Execute the exact M7B direct ABI and planned-K boundary matrix."""

    try:
        if case_id == "planned_k257":
            _require_sdk_simulator(case_id)
            host, dpu, initialization = _sdk_binaries(tasklets)
            dag, _, inputs = _planned_k257_node()
            plan = plan_upmem(
                dag,
                numeric_policy="split_complex_float32_v1",
                topology=UpmemTopology(
                    dpu_count=1, rank_count=1, tasklets_per_dpu=tasklets
                ),
            )
            units = tuple(
                unit
                for stage in plan.stages
                if stage.kind == "contract_batch"
                for unit in stage.work_units
            )
            assert len(units) > 1
            assert {unit.k_size for unit in units} == {1, 256}
            resources = UpmemResources(
                session_root=str(tmp_path / "planned-session"),
                host_binary=str(host),
                dpu_binary=str(dpu),
                initialization_binary=str(initialization),
            )
            expected = replay_upmem_plan_once(dag, plan, inputs)
            with open_upmem_simulator(
                dag, plan, resources, timeout_s=120.0
            ) as session:
                actual = session.run_once(inputs)
            np.testing.assert_allclose(actual.output, expected.output, atol=1.0e-5, rtol=1.0e-5)

            packed_root = tmp_path / "packed-planned-session"
            packed_resources = UpmemResources(
                session_root=str(packed_root),
                host_binary=str(host),
                dpu_binary=str(dpu),
                initialization_binary=str(initialization),
                request_transport="packed_operation_v1",
            )
            with open_upmem_simulator(
                dag, plan, packed_resources, timeout_s=120.0
            ) as packed_session:
                packed_actual = packed_session.run_once(inputs)
            np.testing.assert_allclose(
                packed_actual.output, expected.output, atol=1.0e-5, rtol=1.0e-5
            )
            assert packed_actual.backend_facts["request_transport"] == (
                "packed_operation_v1"
            )
            assert packed_actual.backend_facts["packed_operation_count"] > 0
            assert packed_actual.backend_facts["packed_operation_request_count"] > 0
            for directory_name in ("requests", "results", "packed"):
                directory = packed_root / "rank_00" / directory_name
                assert not directory.exists() or not any(directory.iterdir())
        else:
            _run_direct_sdk_case(
                tmp_path,
                case_id=case_id,
                numeric=numeric,
                tasklets=tasklets,
                m_size=m_size,
                n_size=n_size,
                k_size=k_size,
            )
    except pytest.skip.Exception:
        raise
    except Exception:
        _SDK_CASE_RESULTS[case_id] = "failed"
        raise
    _SDK_CASE_RESULTS[case_id] = "passed"


def test_protocol_accepts_last_int8_k_and_rejects_first_excess(tmp_path: Path) -> None:
    profile = V4Profile(
        dpu_count=1,
        numeric_mode=NUMERIC_HOST_PACKED_INT8,
        execution_target=EXECUTION_TARGET_SIMULATOR,
    )
    payload = np.ones(MAX_CONTRACTED, dtype=np.int8).tobytes()
    artifact = build_v4_request(
        tmp_path / "accepted",
        profile=profile,
        canonical_batch_count=1,
        canonical_m=1,
        canonical_n=1,
        canonical_k=MAX_CONTRACTED,
        work_units=(
            V4WorkUnit(
                local_dpu_id=0,
                tile_id=1,
                batch_index=0,
                m_offset=0,
                n_offset=0,
                k_offset=0,
                m_elements=1,
                n_elements=1,
                k_elements=MAX_CONTRACTED,
                a_payload=payload,
                b_payload=payload,
            ),
        ),
        task_contract_sha256="ab" * 32,
        request_sequence=1,
    )
    assert artifact.header.canonical_k == MAX_CONTRACTED
    with pytest.raises(ValueError, match="canonical dimensions exceed native bounds"):
        build_v4_request(
            tmp_path / "rejected",
            profile=profile,
            canonical_batch_count=1,
            canonical_m=1,
            canonical_n=1,
            canonical_k=MAX_CONTRACTED + 1,
            work_units=(
                V4WorkUnit(
                    local_dpu_id=0,
                    tile_id=1,
                    batch_index=0,
                    m_offset=0,
                    n_offset=0,
                    k_offset=0,
                    m_elements=1,
                    n_elements=1,
                    k_elements=MAX_CONTRACTED + 1,
                ),
            ),
            task_contract_sha256="ab" * 32,
            request_sequence=1,
        )


def test_zero_work_is_not_a_public_v4_request(tmp_path: Path) -> None:
    profile = V4Profile(dpu_count=1, execution_target=EXECUTION_TARGET_SIMULATOR)
    with pytest.raises(ValueError, match="cannot contain only zero-work"):
        build_v4_request(
            tmp_path,
            profile=profile,
            canonical_batch_count=1,
            canonical_m=1,
            canonical_n=1,
            canonical_k=1,
            work_units=(
                V4WorkUnit(
                    local_dpu_id=0,
                    tile_id=1,
                    batch_index=0,
                    m_offset=0,
                    n_offset=0,
                    k_offset=0,
                    m_elements=0,
                    n_elements=0,
                    k_elements=0,
                    flags=1,
                ),
            ),
            task_contract_sha256="ab" * 32,
            request_sequence=1,
        )
