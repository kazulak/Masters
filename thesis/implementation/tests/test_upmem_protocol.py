from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import Any

import pytest

import quantum_bench.upmem.native_session as native_session
import quantum_bench.upmem.protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "upmem" / "runtime"
TASK_HASH = "ab" * 32


def _payload(count: int, numeric_mode: str) -> bytes:
    if numeric_mode == protocol.NUMERIC_FLOAT32:
        return struct.pack("<" + "f" * count, *range(1, count + 1))
    return bytes((index % 127 for index in range(1, count + 1)))


def _request(
    tmp_path: Path,
    *,
    numeric_mode: str = protocol.NUMERIC_FLOAT32,
    dpu_count: int = 2,
    sequence: int = 0,
) -> protocol.V4RequestArtifact:
    profile = protocol.V4Profile(dpu_count=dpu_count, numeric_mode=numeric_mode)
    payload = _payload(3, numeric_mode)
    return protocol.build_v4_request(
        tmp_path,
        profile=profile,
        canonical_batch_count=1,
        canonical_m=1,
        canonical_n=1,
        canonical_k=3,
        work_units=[
            protocol.V4WorkUnit(
                local_dpu_id=0,
                tile_id=1,
                batch_index=0,
                m_offset=0,
                n_offset=0,
                k_offset=0,
                m_elements=1,
                n_elements=1,
                k_elements=3,
                a_payload=payload,
                b_payload=payload,
            )
        ],
        task_contract_sha256=TASK_HASH,
        request_sequence=sequence,
    )


def _response(
    artifact: protocol.V4RequestArtifact,
    profile: protocol.V4Profile,
) -> dict[str, Any]:
    simulator = profile.execution_target == protocol.EXECUTION_TARGET_SIMULATOR
    per_dpu: list[dict[str, int]] = []
    h2d_bytes = 0
    d2h_bytes = 0
    for work_unit in artifact.work_units:
        unit_h2d = (
            work_unit.a_transfer_bytes
            + work_unit.b_transfer_bytes
            + protocol.CONTROL_BYTES
        )
        unit_d2h = work_unit.c_transfer_bytes + protocol.COMPLETION_BYTES
        h2d_bytes += unit_h2d
        d2h_bytes += unit_d2h
        per_dpu.append(
            {
                "dpu_id": work_unit.local_dpu_id,
                "tile_id": work_unit.tile_id,
                "completion_status": protocol.STATUS_COMPLETED,
                "processed_elements": (
                    0
                    if work_unit.flags & protocol.FLAG_ZERO_WORK
                    else work_unit.m_elements * work_unit.n_elements
                ),
                "h2d_bytes": unit_h2d,
                "d2h_bytes": unit_d2h,
            }
        )
    return {
        "event": "RESPONSE",
        "status": "completed",
        "target_requested": "simulator" if simulator else "hardware",
        "target_observed": "sdk_simulator" if simulator else "physical_hardware",
        "rank_path": None if simulator else profile.rank_path,
        "request_sequence": artifact.request_sequence,
        "request_output_elements": artifact.request_output_elements,
        "global_output_elements": artifact.global_output_elements,
        "global_completeness": False,
        "task_contract_sha256": artifact.task_contract_sha256,
        "request_sha256": artifact.manifest_sha256,
        "request_manifest_sha256": artifact.manifest_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "dispatch_mode": "bulk_set_synchronous_v1",
        "bulk_set_launch_verified": True,
        "requested_dpu_count": profile.dpu_count,
        "allocated_dpu_count": profile.dpu_count,
        "tasklets_per_dpu": profile.tasklets_per_dpu,
        "hardware_allocation_verified": not simulator,
        "allocation_verified": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": not simulator,
        "simulator_kernel_executed": simulator,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": not simulator,
        "simulator_functionality_evidence": simulator,
        **protocol.native_execution_identity(profile.execution_target),
        "transfer": {
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "total_bytes": h2d_bytes + d2h_bytes,
        },
        "per_dpu": per_dpu,
    }


def _response_validator(profile: protocol.V4Profile) -> native_session.V4Session:
    """Create only the object state used by response validation, not a session."""

    validator = object.__new__(native_session.V4Session)
    validator.profile = profile
    return validator


def test_v4_struct_layout_and_field_order_match_native_header() -> None:
    header = (NATIVE / "protocol.h").read_text(encoding="ascii")

    assert protocol.VERSION == 4
    assert protocol.HEADER_FORMAT == "<8s10I7Q32s32s"
    assert protocol.WORK_UNIT_FORMAT == "<2I5Q9I"
    assert protocol.CONTROL_FORMAT == "<18I"
    assert protocol.COMPLETION_FORMAT == "<4I3Q"
    assert (
        protocol.HEADER_BYTES,
        protocol.WORK_UNIT_BYTES,
        protocol.CONTROL_BYTES,
        protocol.COMPLETION_BYTES,
    ) == (168, 84, 72, 40)

    for declaration in (
        "char magic[8];",
        "uint32_t version;",
        "uint32_t work_unit_count;",
        "uint64_t canonical_k;",
        "uint64_t request_sequence;",
        "unsigned char task_contract_sha256[32];",
        "unsigned char request_sha256[32];",
    ):
        assert declaration in header
    assert header.index("char magic[8];") < header.index("uint32_t version;")
    assert header.index("uint64_t canonical_k;") < header.index(
        "uint64_t request_sequence;"
    )
    assert "_Static_assert(sizeof(execution_plan_v4_header_t) == 168u" in header
    assert "_Static_assert(sizeof(execution_plan_v4_work_unit_t) == 84u" in header
    assert "_Static_assert(sizeof(execution_plan_v4_control_t) == 72u" in header
    assert "_Static_assert(sizeof(execution_plan_v4_completion_t) == 40u" in header


def test_v4_request_is_deterministic_and_binds_contract_and_manifest_hash(
    tmp_path: Path,
) -> None:
    first = _request(tmp_path / "first", sequence=7)
    second = _request(tmp_path / "second", sequence=7)

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.sidecar_path.read_bytes() == second.sidecar_path.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.sidecar_sha256 == second.sidecar_sha256
    assert first.task_contract_sha256 == TASK_HASH
    assert first.header.task_contract_sha256 == bytes.fromhex(TASK_HASH)
    assert first.header.request_sha256 == bytes.fromhex(first.manifest_sha256)
    assert hashlib.sha256(first.manifest_path.read_bytes()).hexdigest() == (
        first.manifest_sha256
    )
    assert hashlib.sha256(first.sidecar_path.read_bytes()).hexdigest() == (
        first.sidecar_sha256
    )
    assert protocol.unpack_v4_header(first.sidecar_path.read_bytes()[: protocol.HEADER_BYTES]) == (
        first.header
    )


@pytest.mark.parametrize(
    ("numeric_mode", "expected_operand_bytes"),
    [
        (protocol.NUMERIC_FLOAT32, 16),
        (protocol.NUMERIC_HOST_PACKED_INT8, 8),
    ],
)
def test_v4_payloads_are_padded_and_mram_aligned(
    tmp_path: Path,
    numeric_mode: str,
    expected_operand_bytes: int,
) -> None:
    artifact = _request(tmp_path, numeric_mode=numeric_mode)
    active, zero = artifact.work_units

    assert active.a_transfer_bytes == expected_operand_bytes
    assert active.b_transfer_bytes == expected_operand_bytes
    assert active.c_transfer_bytes == 8
    assert all(
        value % protocol.MRAM_ALIGNMENT == 0
        for value in (
            active.a_transfer_bytes,
            active.b_transfer_bytes,
            active.c_transfer_bytes,
            active.a_offset_bytes,
            active.b_offset_bytes,
            active.c_offset_bytes,
        )
    )
    assert len((artifact.root / active.a_path).read_bytes()) == expected_operand_bytes
    assert len((artifact.root / active.b_path).read_bytes()) == expected_operand_bytes
    assert (artifact.root / active.a_path).read_bytes()[-1] == 0
    assert zero.flags == protocol.FLAG_ZERO_WORK
    assert (zero.a_transfer_bytes, zero.b_transfer_bytes, zero.c_transfer_bytes) == (
        0,
        0,
        0,
    )


def test_v4_builder_rejects_unsafe_paths_and_k_bounds(tmp_path: Path) -> None:
    for path in ("../escape", "/absolute", "nested\\escape"):
        with pytest.raises(ValueError, match="unsafe"):
            protocol._safe_relative(path)

    with pytest.raises(ValueError, match="canonical dimensions exceed native bounds"):
        protocol.build_v4_request(
            tmp_path,
            profile=protocol.V4Profile(
                dpu_count=1,
                numeric_mode=protocol.NUMERIC_HOST_PACKED_INT8,
            ),
            canonical_batch_count=1,
            canonical_m=1,
            canonical_n=1,
            canonical_k=protocol.MAX_CONTRACTED + 1,
            work_units=[
                protocol.V4WorkUnit(
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    protocol.MAX_CONTRACTED + 1,
                    b"x",
                    b"x",
                )
            ],
            task_contract_sha256=TASK_HASH,
            request_sequence=0,
        )

    header = (NATIVE / "protocol.h").read_text(encoding="ascii")
    assert protocol.MAX_CONTRACTED * 128 * 128 <= 2**31 - 1
    assert "EXECUTION_PLAN_V4_MAX_CONTRACTED * 128u * 128u <= 2147483647u" in header


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda event: event.__setitem__("request_sha256", "00" * 32),
            "request_sha256",
        ),
        (
            lambda event: event.__setitem__("per_dpu", "not-a-list"),
            "lacks one result",
        ),
        (
            lambda event: event["transfer"].__setitem__("total_bytes", 1),
            "transfer total",
        ),
    ],
)
def test_v4_response_validator_rejects_tampered_or_malformed_evidence(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    profile = protocol.V4Profile(dpu_count=2, rank_path="/dev/dpu_rank0")
    artifact = _request(tmp_path)
    response = _response(artifact, profile)
    mutate(response)

    with pytest.raises(protocol.V4ProtocolError, match=message):
        _response_validator(profile)._validate_response(response, artifact)


def test_v4_target_identities_keep_simulator_and_physical_facts_distinct(
    tmp_path: Path,
) -> None:
    physical = protocol.V4Profile(dpu_count=1, rank_path="/dev/dpu_rank0")
    simulator = protocol.V4Profile(
        dpu_count=1,
        execution_target=protocol.EXECUTION_TARGET_SIMULATOR,
    )
    artifact = _request(tmp_path, dpu_count=1)

    assert protocol.native_execution_identity(physical.execution_target)[
        "execution_class"
    ] == "physical_v4_output_tile"
    assert protocol.native_execution_identity(simulator.execution_target)[
        "execution_class"
    ] == "sdk_simulator_v4_output_tile"
    _response_validator(physical)._validate_response(_response(artifact, physical), artifact)
    _response_validator(simulator)._validate_response(
        _response(artifact, simulator), artifact
    )
    with pytest.raises(ValueError, match="exactly one DPU"):
        protocol.V4Profile(
            dpu_count=2,
            execution_target=protocol.EXECUTION_TARGET_SIMULATOR,
        )


def test_v4_native_sources_preserve_the_abi_and_build_contract() -> None:
    protocol_header = (NATIVE / "protocol.h").read_text(encoding="ascii")
    host = (NATIVE / "host.c").read_text(encoding="ascii")
    dpu = (NATIVE / "dpu.c").read_text(encoding="ascii")
    makefile = (NATIVE / "Makefile").read_text(encoding="ascii")

    assert '#define EXECUTION_PLAN_V4_MAGIC "UPXDPV4"' in protocol_header
    assert "#define EXECUTION_PLAN_V4_VERSION 4u" in protocol_header
    assert "EXECUTION_PLAN_V4_MAX_TASKLETS 24u" in protocol_header
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in host
    assert "--target hardware|simulator" in host
    assert "dpu_launch(v4_provider.set, DPU_SYNCHRONOUS)" in host
    assert "request_manifest_sha256" in host
    assert "cpu_fallback_used\\\":false" in host
    assert "__mram_noinit uint8_t V4_MRAM" in dpu
    assert "__dma_aligned uint8_t v4_input_window" in dpu
    assert "mram_read" in dpu and "mram_write" in dpu
    assert "v4 requires NR_TASKLETS in [1,24]" in dpu
    assert "MAX_TASKLETS := 24" in makefile
    assert "bin/host_upmem_execution_plan_v4_t%" in makefile
    assert "bin/dpu_gemm_tile_v4_t%" in makefile
    assert "NR_TASKLETS=$*" in makefile
