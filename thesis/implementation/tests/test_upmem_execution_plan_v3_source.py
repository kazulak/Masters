from __future__ import annotations

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/simplepim"
PLAN = NATIVE / "upmem_sdk_execution_plan"


def test_v3_binary_layout_is_additive_and_frozen() -> None:
    assert struct.calcsize("<8s16I32s32s") == 136
    assert struct.calcsize("<8I") == 32
    common = (PLAN / "execution_plan_v3_common.h").read_text(encoding="ascii")
    assert '#define EXECUTION_PLAN_V3_MAGIC "UPXDPV3"' in common
    assert "EXECUTION_PLAN_V3_VERSION 3u" in common
    assert "EXECUTION_PLAN_V3_MAX_DPUS 64u" in common
    assert "EXECUTION_PLAN_V3_MAX_TASKLETS 24u" in common
    assert "sizeof(execution_plan_v3_header_t) == EXECUTION_PLAN_V3_HEADER_BYTES" in common


def test_v3_runtime_uses_dynamic_dpu_metadata_and_safe_reduction() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    sidecar = (PLAN / "distributed_plan_v3.c").read_text(encoding="ascii")
    assert "struct dpu_set_t handles[EXECUTION_PLAN_V3_MAX_DPUS]" in source
    assert "dpu_launch(handles[dpu_id], DPU_ASYNCHRONOUS)" in source
    assert "metrics->launch_attempted = 1" in source
    assert source.index("DPU_ASYNCHRONOUS") < source.index("dpu_sync(handles[dpu_id])")
    assert "double *accumulator" in source
    assert "result[index] = (float)accumulator[index]" in source
    assert "output_offset % 2u" in sidecar
    assert "EXECUTION_PLAN_V3_MAX_DPUS" in sidecar
    assert "EXECUTION_PLAN_V2_MAX_DPUS" not in sidecar


def test_v3_numeric_and_target_claims_are_explicit() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    runner = (NATIVE / "upmem_sdk_execution_plan_runner.py").read_text(encoding="ascii")
    assert r'\"numeric_transport\":\"float32_mram\"' in source
    assert r'\"numeric_arithmetic\":\"%s\"' in source
    assert r'\"requantization_scope\":\"%s\"' in source
    assert r'\"packed_int8_transfer\":false' in source
    assert 'int8_requantization ? "per_task_on_dpu" : "none"' in source
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in source
    assert "DPU_BACKEND" in runner
    assert "no simulator or CPU fallback" in runner


def test_v3_loader_identity_and_limits_are_separate_from_v2() -> None:
    host = (PLAN / "host_v3.c").read_text(encoding="ascii")
    protocol = (NATIVE / "upmem_sdk_generic_loop_resident/session_protocol.c").read_text(encoding="ascii")
    header = (NATIVE / "upmem_sdk_generic_loop_resident/session_protocol.h").read_text(encoding="ascii")
    assert "resident_request_load_execution_plan_v3" in header
    assert "resident_request_load_execution_plan_v3(manifest_path" in host
    assert "resident_request_load_execution_plan_v2(manifest_path" not in host
    assert "resident_request_load_profile(manifest_path, 64u, 1" in protocol
    assert "resident_binary_matches_v3_tasklets" in protocol
    assert '"dpu_resident_v3_t%llu"' in protocol
    assert "manifest_tasklets < 1u || manifest_tasklets > 24u" in protocol
    assert "resident_request_load_profile(manifest_path, 2u, 0" in protocol
    assert "resident_request_load_profile(manifest_path, 4u, 0" in protocol
    assert '"dpu_resident_v2"' in protocol


def test_v3_requires_exact_resident_and_sidecar_dpu_counts() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    assert "request.resident.requested_dpus != plan.header.dpu_count" in source
    assert "request.resident.requested_dpus != 1u &&" not in source


def test_v3_response_and_build_contract_are_canonical() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    common = (NATIVE / "upmem_sdk_generic_loop_resident/common.h").read_text(encoding="ascii")
    makefile = (PLAN / "Makefile").read_text(encoding="ascii")
    assert r'\"repetitions\":[' in source
    for field in ("repeat_id", "warmup", "total_time_s", "launch_sync_time_s", "assembly_time_s", "per_dpu"):
        assert rf'\"{field}\"' in source
    assert "repeat < request->warmup_repetitions" in source
    assert "warmups > 4u" in source
    assert r'\"hardware_kernel_executed\"' in source
    assert r'\"timing_scope\"' in source
    assert r'\"application_visible_transfer_totals\"' in source
    assert r'\"run_total_transfers\"' in source
    assert r'\"policy_reference_validation\"' in source
    assert r'\"passed\":%s' in source
    assert "hash_verified" in source
    assert "isfinite(validation->max_abs_error)" in source
    assert "v3_validate_policy_reference" in source
    assert "UPMEM_HW_RANK_PATH" in source
    assert "dpu_get_nr_ranks" in (PLAN / "execution_plan_provider.c").read_text(encoding="ascii")
    assert "rankPath=%s" in (PLAN / "execution_plan_provider.c").read_text(encoding="ascii")
    assert r'\"allocation_provider\":\"upmem_sdk_rank_profile_v1\"' in source
    assert r'\"simplepim_role\":\"initialization_binary_and_management_state_only\"' in source
    assert r'\"kernel_provider\":\"thesis_resident_generic_c_v3\"' in source
    assert r'\"transfer_provider\":\"upmem_sdk_synchronous_v1\"' in source
    assert r'\"collective_provider\":\"%s\"' in source
    assert r'\"reconstruction_provider\":\"%s\"' in source
    assert r'\"load_balance\"' in source
    assert "UPMEM_GENERIC_MAX_ELEMS 65536" in common
    assert "RESIDENT_OUTPUT_TILE_ELEMS 2" in common
    assert "V3_MAX_ELEMS := 65536" in makefile
    assert "V3_MRAM_POOL_BYTES := 524288" in makefile
    assert "V3_OUTPUT_TILE_ELEMS := 2" in makefile


def test_completion_v3_has_capacity_for_all_tasklets() -> None:
    common = (NATIVE / "upmem_sdk_generic_loop_resident/common.h").read_text(encoding="ascii")
    assert "tasklet_processed_elements[24]" in common
    assert "sizeof(resident_completion_t) == 152u" in common
    makefile = (PLAN / "Makefile").read_text(encoding="ascii")
    assert "dpu_resident_v3_t%" in makefile
    assert "host_upmem_execution_plan_v3_t%" in makefile
