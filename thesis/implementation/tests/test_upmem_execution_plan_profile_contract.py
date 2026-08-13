from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/simplepim"
PROTOCOL = NATIVE / "upmem_sdk_generic_loop_resident/session_protocol.h"
MAKEFILE = NATIVE / "upmem_sdk_execution_plan/Makefile"
REQUEST = NATIVE / "upmem_sdk_execution_plan/plan_request.c"
SESSION_PROTOCOL = NATIVE / "upmem_sdk_generic_loop_resident/session_protocol.c"


def test_execution_plan_protocol_keeps_legacy_defaults_overridable() -> None:
    source = PROTOCOL.read_text(encoding="ascii")

    assert '#ifndef RESIDENT_ROUTE_ID' in source
    assert '#define RESIDENT_ROUTE_ID "upmem_tn_hardware_taskgraph_resident"' in source
    assert '#ifndef RESIDENT_BACKEND_ID' in source
    assert '#define RESIDENT_BACKEND_ID "upmem_sdk_hardware_taskgraph_resident"' in source
    assert '#ifndef RESIDENT_PROFILE_VERSION' in source
    assert '#define RESIDENT_PROFILE_VERSION "hardware_taskgraph_single_dpu_mram_resident_v1"' in source


def test_execution_plan_v1_and_v2_hosts_define_execution_plan_identity() -> None:
    source = MAKEFILE.read_text(encoding="ascii")

    assert "RESIDENT_ROUTE_ID ?= upmem_tn_hardware_execution_plan_resident" in source
    assert "RESIDENT_BACKEND_ID ?= upmem_sdk_hardware_execution_plan_resident" in source
    assert "RESIDENT_PROFILE_VERSION ?= hardware_taskgraph_execution_plan_resident_v1" in source
    assert '-DRESIDENT_ROUTE_ID=\\"$(RESIDENT_ROUTE_ID)\\"' in source
    assert '-DRESIDENT_BACKEND_ID=\\"$(RESIDENT_BACKEND_ID)\\"' in source
    assert '-DRESIDENT_PROFILE_VERSION=\\"$(RESIDENT_PROFILE_VERSION)\\"' in source
    assert source.count("$(EXECUTION_PLAN_RESIDENT_DEFINES)") == 2


def test_execution_plan_loaders_require_exact_manifest_dpu_count() -> None:
    source = REQUEST.read_text(encoding="ascii")
    protocol = SESSION_PROTOCOL.read_text(encoding="ascii")

    assert 'resident_uint_field((char *)manifest_bytes, "requested_dpu_count"' in protocol
    assert "manifest_requested_dpus != manifest_requested_dpu_count" in protocol
    assert source.count("request->resident.requested_dpus != request->schedule.header.dpu_count") == 1
    assert source.count("request->resident.requested_dpus != request->distributed_v2.header.dpu_count") == 1
    assert "request->resident.requested_dpus != 1u" not in source
