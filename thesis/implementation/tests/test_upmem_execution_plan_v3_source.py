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
    assert "dpu_launch(set, DPU_SYNCHRONOUS)" in source
    assert "dpu_launch(handles[dpu_id], DPU_ASYNCHRONOUS)" not in source
    assert "dpu_sync(handles[dpu_id])" not in source
    assert "v3_json_string(file, RESIDENT_ROUTE_ID)" in source
    assert "v3_json_string(file, RESIDENT_BACKEND_ID)" in source
    assert "v3_json_string(file, RESIDENT_PROFILE_VERSION)" in source
    assert "upmem_sdk_hardware_execution_plan_v3" not in source
    assert 'dpu_copy_to(handles[dpu_id], "RESIDENT_ACTIVE_OPERATION"' not in source
    setup_body = source.split("static int v3_copy_package_to_dpu(", 1)[1].split(
        "static int v3_validate_completion(", 1
    )[0]
    assert 'dpu_copy_to(dpu, "RESIDENT_ACTIVE_OPERATION"' in setup_body
    assert "metrics->descriptor_h2d_bytes += sizeof(active_operation)" in setup_body
    execute_body = source.split("static int v3_execute_repetition(", 1)[1].split(
        "static void v3_write_response(", 1
    )[0]
    assert "RESIDENT_ACTIVE_OPERATION" not in execute_body
    assert "v3_reset_dpu" not in source
    assert "metrics->launch_attempted = 1" in source
    assert source.index("const double launch_started") < source.index("dpu_launch(set, DPU_SYNCHRONOUS)")
    assert source.index("completion_started = now_s()") > source.index("dpu_launch(set, DPU_SYNCHRONOUS)")
    assert r'\"dispatch_mode\":\"bulk_set_synchronous_v1\"' in source
    assert r'\"kernel_launch_api_calls\"' in source
    assert r'\"dpu_program_instances\"' in source
    assert r'\"explicit_sync_api_calls\"' in source
    assert r'\"completion_read_and_validation_time_s\"' in source
    assert r'\"launch_count_semantics\":\"set_launch_api_calls\"' in source
    assert r'\"synchronize_count_semantics\":\"explicit_dpu_sync_api_calls\"' in source
    assert "metrics->repeats[repeat_index].reset_h2d_bytes" not in execute_body
    assert "double *accumulator" in source
    assert "result[index] = (float)accumulator[index]" in source
    assert "output_offset % 2u" in sidecar
    assert "EXECUTION_PLAN_V3_MAX_DPUS" in sidecar
    assert "EXECUTION_PLAN_V2_MAX_DPUS" not in sidecar


def test_v3_numeric_and_target_claims_are_explicit() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    runner = (NATIVE / "upmem_sdk_execution_plan_runner.py").read_text(encoding="ascii")
    assert r'\"numeric_transport\":\"%s\"' in source
    assert r'\"numeric_arithmetic\":\"%s\"' in source
    assert r'\"requantization_scope\":\"%s\"' in source
    assert r'\"packed_int8_transfer\":%s' in source
    assert 'packed_int8 ? "host_packed_int8_mram" : "float32_mram"' in source
    assert '"int8_multiply_int32_accumulate"' in source
    assert "v3_validate_integer_reference" in source
    assert "int64_t *accumulator" in source
    assert "INT32_MIN" in source and "INT32_MAX" in source
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


def test_v3_observes_physical_hardware_after_confirmed_allocation() -> None:
    source = (PLAN / "host_v3.c").read_text(encoding="ascii")
    target_line = next(line for line in source.splitlines() if '\\"target_observed\\"' in line)
    assert 'provider != NULL && provider->allocation_used ? "physical_hardware" : "not_allocated"' in target_line
    assert 'status != NULL && strcmp(status, "completed")' not in target_line


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
    assert "RESIDENT_CONTROL_FLAG_CONTRACTED_FINAL_REFERENCE_VALIDATION_ONLY" in common
    assert "RESIDENT_CHECKSUM_FNV1A64_OFFSET_BASIS" in common
    assert "output_checksum_policy" in source
    assert 'contracted_partition ? "final_reference_validation_only" : "output_slice_per_dpu"' in source
    assert r'\"load_balance\"' in source
    assert "UPMEM_GENERIC_MAX_ELEMS 65536" in common
    assert "RESIDENT_OUTPUT_TILE_ELEMS 2" in common
    assert "V3_MAX_ELEMS := 65536" in makefile
    assert "V3_MRAM_POOL_BYTES := 524288" in makefile
    assert "V3_OUTPUT_TILE_ELEMS := 2" in makefile
    assert "bin/dpu_simplepim_management_init" in makefile.split("v3:", 1)[1]
    assert "initialization_binary_sha256" in source
    assert "execution_plan_sha256_file(initialization_binary" in source


def test_v3_checksum_policy_preserves_output_checksums_and_skips_contracted_checksums() -> None:
    host = (PLAN / "host_v3.c").read_text(encoding="ascii")
    dpu = (NATIVE / "upmem_sdk_generic_loop_resident/dpu.c").read_text(encoding="ascii")
    common = (NATIVE / "upmem_sdk_generic_loop_resident/common.h").read_text(encoding="ascii")

    assert "RESIDENT_CONTROL_FLAG_CONTRACTED_FINAL_REFERENCE_VALIDATION_ONLY" in common
    assert "control.reserved" not in host
    assert "? RESIDENT_CONTROL_FLAG_CONTRACTED_FINAL_REFERENCE_VALIDATION_ONLY : 0u" in host

    execute_body = host.split("static int v3_execute_repetition(", 1)[1].split(
        "static void v3_write_response(", 1
    )[0]
    contracted_body = execute_body.split("if (contracted_partition) {", 1)[1].split(
        "\n        } else {", 1
    )[0]
    assert "checksum_f32_bytes" not in contracted_body
    assert "int64_t *accumulator" in contracted_body
    assert "double *accumulator" in contracted_body

    output_body = execute_body.split("\n        } else {", 1)[1].split(
        "metrics->repeats[repeat_index].assembly_time_s", 1
    )[0]
    assert "checksum_f32_bytes" in output_body

    checksum_body = dpu.split("const int final_reference_validation_only =", 1)[1].split(
        "RESIDENT_COMPLETION.output_checksum_fnv1a64", 1
    )[0]
    assert "RESIDENT_CONTROL_FLAG_CONTRACTED_FINAL_REFERENCE_VALIDATION_ONLY" in checksum_body
    assert "if (!final_reference_validation_only && out_slot != NULL)" in checksum_body
    assert "!final_reference_validation_only && operation->kind == RESIDENT_OPERATION_COMPLEX_COMBINE" in checksum_body
    assert "contracted v3 DPU did not acknowledge final-reference-only checksum policy" in host


def test_completion_v3_has_capacity_for_all_tasklets() -> None:
    common = (NATIVE / "upmem_sdk_generic_loop_resident/common.h").read_text(encoding="ascii")
    assert "tasklet_processed_elements[24]" in common
    assert "sizeof(resident_completion_t) == 152u" in common
    dpu = (NATIVE / "upmem_sdk_generic_loop_resident/dpu.c").read_text(encoding="ascii")
    assert "sizeof(RESIDENT_COMPLETION.tasklet_processed_elements)" in dpu
    assert "index < 16u" not in dpu
    makefile = (PLAN / "Makefile").read_text(encoding="ascii")
    assert "dpu_resident_v3_t%" in makefile
    assert "host_upmem_execution_plan_v3_t%" in makefile
    assert "bin/dpu_simplepim_management_init" in makefile
