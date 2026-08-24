from __future__ import annotations

import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "native" / "upmem" / "runtime"


def test_v4_abi_is_separate_and_supports_bounded_request_batches() -> None:
    common = (PLAN / "protocol.h").read_text(encoding="ascii")
    assert struct.calcsize("<8s10I7Q32s32s") == 168
    assert struct.calcsize("<2I5Q9I") == 84
    assert struct.calcsize("<18I") == 72
    assert struct.calcsize("<4I3Q") == 40
    assert '#define EXECUTION_PLAN_V4_MAGIC "UPXDPV4"' in common
    assert "EXECUTION_PLAN_V4_VERSION 4u" in common
    assert "request_output_elements" in common
    assert "request_sequence" in common
    assert "k_offset" in common
    assert "sizeof(execution_plan_v4_header_t) == 168u" in common
    assert "sizeof(execution_plan_v4_work_unit_t) == 84u" in common
    assert "sizeof(execution_plan_v4_control_t) == 72u" in common
    assert "sizeof(execution_plan_v4_completion_t) == 40u" in common
    assert "MAX_TASKLETS 24u" in common


def test_v4_validation_is_bounded_not_global_output_claim() -> None:
    sidecar = (PLAN / "plan.c").read_text(encoding="ascii")
    request = (PLAN / "request.c").read_text(encoding="ascii")
    common = (PLAN / "protocol.h").read_text(encoding="ascii")
    assert "covered != header->request_output_elements" in sidecar
    assert "area != header->request_output_elements" in request
    assert "global_output_elements" in sidecar
    assert "request_output_elements" in sidecar
    assert "request_sequence" in common
    assert (
        "extent_inside(unit->k_offset, unit->k_elements, header->canonical_k)"
        in sidecar
    )
    assert (
        "extent_inside(unit->k_offset, unit->k_elements, header->canonical_k)"
        in request
    )
    assert "unit->local_dpu_id <= work_units[index - 1u].local_dpu_id" in sidecar
    assert "unit->local_dpu_id <= units[index - 1u].local_dpu_id" in request
    assert "unit->k_offset != 0u" not in sidecar
    assert "unit->k_elements != header->canonical_k" not in request
    assert "output tiles overlap" in sidecar
    assert "EXECUTION_PLAN_V4_MRAM_POOL_BYTES" in sidecar
    assert "dpu_free" not in sidecar
    assert "a_sha256" in request
    assert "b_sha256" in request
    assert "%64s %64s" in request
    assert "digest_text(a_sha256" in request
    assert "payload_digest_matches" in request
    assert "A or B payload SHA-256 mismatch" in request


def test_v4_session_contract_is_physical_bulk_and_fail_closed() -> None:
    host = (PLAN / "host.c").read_text(encoding="ascii")
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in host
    assert 'getenv("DPU_BACKEND")' in host
    assert 'getenv("UPMEM_EXECUTION_MODE")' in host
    assert "upmem_v4_provider_init_on_rank" in host
    assert "dpu_load(v4_provider.set, dpu_binary, NULL)" in host
    assert "dpu_launch(v4_provider.set, DPU_SYNCHRONOUS)" in host
    assert "task_contract_sha256" in host
    assert "request_sha256" in host
    assert "request_manifest_sha256" in host
    assert "sidecar_sha256" in host
    assert '\\"rank_path\\":' in host
    assert "total_route_time_s" in host
    assert "dispatch_mode" in host
    assert "bulk_set_launch_verified" in host
    assert "request_level_speedup_applicable" in host
    assert "request_timing_is_bringup_only" in host
    assert "SUBMIT" in host and "CLOSE" in host
    assert "safe-relative-manifest" in host
    assert 'global_completeness\\":false' in host
    assert 'simulator_kernel_executed\\":false' in host
    assert 'cpu_fallback_used\\":false' in host
    assert "v4_release_done" in host
    assert "upmem_v4_provider_release" in host
    assert "dpu_free_called_once = v4_provider.allocation_active" in host
    assert "request_sequence must increase" in host
    assert "release_proof" not in host
    assert 'event\\":\\"RELEASE' in host
    assert "release_succeeded" in host
    assert (
        "if (execute_request(root_real, manifest, digest, dpus, tasklets, timeout_s) != 0) rc = 1;"
        in host
    )
    assert "return rc;" in host
    assert "else rc = 0" not in host
    assert "dpu_sync(" not in host


def test_v4_session_reuses_the_abi_for_an_explicit_sdk_simulator_target() -> None:
    host = (PLAN / "host.c").read_text(encoding="ascii")
    provider = (PLAN / "simplepim_provider.c").read_text(encoding="ascii")
    header = (PLAN / "protocol.h").read_text(encoding="ascii")

    assert "--target hardware|simulator" in host
    assert 'strcmp(target, "simulator")' in host
    assert 'strcmp(getenv("DPU_BACKEND"), "simulator")' in host
    assert "upmem_v4_provider_init_simulator" in host
    assert "dpu_alloc(requested_dpus, NULL, &provider->set)" in provider
    assert "EXECUTION_PLAN_V4_NATIVE_SIMULATOR_BACKEND_ID" in header
    assert "EXECUTION_PLAN_V4_NATIVE_SIMULATOR_EXECUTION_CLASS" in header
    assert "timing_claim_applicable" in host
    assert "scaling_claim_applicable" in host
    assert "speedup_claim_applicable" in host
    assert "energy_claim_applicable" in host


def test_v4_kernel_is_integer_or_float_tile_only_with_aligned_dma() -> None:
    dpu = (PLAN / "dpu.c").read_text(encoding="ascii")
    assert "__mram_noinit uint8_t V4_MRAM" in dpu
    assert (
        "V4_CONTROL.numeric_mode == EXECUTION_PLAN_V4_NUMERIC_HOST_PACKED_INT8" in dpu
    )
    assert "int8_t" in dpu
    assert "int_value +=" in dpu
    assert "float_value +=" in dpu
    assert "mram_read" in dpu and "mram_write" in dpu
    assert "sizeof(v4_input_window[me()])" in dpu
    assert "sizeof(v4_output_window[me()])" in dpu
    assert "BARRIER_INIT(v4_barrier, NR_TASKLETS)" in dpu
    assert "k_elements * 128u * 128u" in dpu


def test_v4_output_writer_closes_each_file_once() -> None:
    request = (PLAN / "request.c").read_text(encoding="ascii")
    writer = request.split("int execution_plan_v4_request_write_output", 1)[1].split(
        "void execution_plan_v4_request_free", 1
    )[0]
    assert "close_status = fclose(file);" in writer
    assert "if (file != NULL) fclose(file);" not in writer


def test_v4_makefile_has_tasklet_keyed_binaries_without_changing_v3_rules() -> None:
    makefile = (PLAN / "Makefile").read_text(encoding="ascii")
    assert "MAX_TASKLETS := 24" in makefile
    assert "bin/host_upmem_execution_plan_v4_t%" in makefile
    assert "bin/dpu_gemm_tile_v4_t%" in makefile
    assert "NR_TASKLETS=$*" in makefile
    assert "v4 requires NR_TASKLETS in [1,24]" in makefile
    assert "host_upmem_execution_plan_v3" not in makefile
    assert "dpu_resident_v3" not in makefile
    assert "execution_plan_v4" in makefile


def test_v4_runtime_has_no_historical_source_dependency() -> None:
    files = tuple(PLAN.glob("*"))
    text = "\n".join(
        path.read_text(encoding="ascii", errors="ignore")
        for path in files
        if path.is_file() and path.name != "simplepim_management_profile.patch"
    )
    assert "upmem_sdk_execution_plan" not in text
    assert "execution_plan_v2_common" not in text
    assert "execution_plan_v3_common" not in text
    assert "table_management_init_with_profile" not in text
    assert (PLAN / "simplepim_management_profile.patch").is_file()
