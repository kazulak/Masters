from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/simplepim/upmem_sdk_generic_loop_frontier_two_dpu"


def test_raw_frontier_target_remains_direct_sdk_and_distinct() -> None:
    makefile = (NATIVE / "Makefile").read_text(encoding="ascii")
    host = (NATIVE / "host.c").read_text(encoding="ascii")

    raw_all = next(line for line in makefile.splitlines() if line.startswith("all:"))
    assert raw_all == "all: bin/host_frontier_two_dpu bin/dpu_frontier_two_dpu"
    assert "bin/host_frontier_two_dpu_simplepim_management" not in raw_all
    assert "simplepim-all:" in makefile
    assert "bin/host_frontier_two_dpu_simplepim_management" in makefile
    assert "FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT=1" not in raw_all
    assert "FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT=1" in makefile
    assert "dpu_alloc(FRONTIER_TWO_DPU_COUNT, FRONTIER_ALLOCATION_PROFILE, &set)" in host
    assert "bin/host_frontier_two_dpu_simplepim_management" in makefile
    assert "NR_TASKLETS ?= 1" in makefile
    assert "DPU_BACKEND" in host
    assert "frontier_resolve_sibling_binary" in host
    assert '"/proc/self/exe"' in host
    assert '"dpu_simplepim_management_init"' in host
    assert '"bin/dpu_simplepim_management_init"' not in host


def test_simplepim_provider_uses_upstream_management_extension_not_dpu_alloc() -> None:
    provider = (NATIVE / "simplepim_provider.c").read_text(encoding="ascii")
    header = (NATIVE / "simplepim_provider.h").read_text(encoding="ascii")

    assert "simplepim_management_t" in header
    assert "table_management_init_with_profile" in provider
    assert "dpu_alloc(" not in provider
    assert "table_map" not in provider
    assert "table_zip" not in provider
    assert "simplepim_frontier_provider_release" in provider
    assert provider.count("dpu_free(") == 1


def test_staged_management_patch_is_physical_and_fail_closed() -> None:
    patch = (NATIVE / "simplepim_management_profile.patch").read_text(encoding="ascii")
    makefile = (NATIVE / "Makefile").read_text(encoding="ascii")

    assert "table_management_init_with_profile" in patch
    assert "int* allocation_used" in patch
    assert "int* release_attempted" in patch
    assert "int* release_succeeded" in patch
    assert "dpu_error_t* release_error_out" in patch
    assert 'strcmp(allocation_profile, "backend=hw") != 0' in patch
    assert 'dpu_alloc(num_dpus, allocation_profile, &set)' in patch
    assert "dpu_free(set)" in patch
    assert "SIMPLEPIM_STAGE" in makefile
    assert "git apply --recount --no-index" in makefile
    assert "management_profile_manifest.json" in makefile
    assert "rev-parse --verify HEAD" in makefile
    assert "status --porcelain --untracked-files=all" in makefile
    assert "staged_source_tree_sha256" in makefile
    assert "sha256sum simplepim_management_profile.patch" in makefile
    assert "DPU_BACKEND" not in patch
    assert "simulator" not in patch.lower()


def test_simplepim_management_host_link_places_dl_after_upmem_sdk_flags() -> None:
    makefile = (NATIVE / "Makefile").read_text(encoding="ascii")
    link_line = next(
        line
        for line in makefile.splitlines()
        if "$(HOST_CC)" in line and "simplepim_provider.c" in line
    )
    sdk_flags = "$$(dpu-pkg-config --cflags --libs dpu)"

    assert sdk_flags in link_line
    assert "-ldl" in link_line
    assert link_line.index(sdk_flags) < link_line.index("-ldl")


def test_response_contract_separates_management_provider_from_kernel_provider() -> None:
    common = (NATIVE / "common.h").read_text(encoding="ascii")
    host = (NATIVE / "host.c").read_text(encoding="ascii")

    assert 'FRONTIER_CONTROL_PROVIDER "simplepim_management"' in common
    assert 'FRONTIER_PROVIDER_ID "simplepim_management"' in common
    assert 'FRONTIER_KERNEL_PROVIDER "thesis_resident_generic_contract"' in common
    assert '\\"simplepim_operator_names\\":[]' in host
    assert '\\"simplepim_management_allocation_used\\":%s' in host
    assert '\\"simplepim_management_object_created\\":%s' in host
    assert '\\"raw_sdk_direct_allocation_used\\":%s' in host
    assert '\\"raw_sdk_load_used\\":%s' in host
    assert '\\"raw_sdk_transfer_used\\":%s' in host
    assert '\\"raw_sdk_launch_used\\":%s' in host
    assert '\\"raw_sdk_sync_used\\":%s' in host
    assert '\\"raw_sdk_control_calls_used\\":%s' in host
    assert '\\"any_task_completed\\":%s' in host
    assert '\\"all_tasks_completed\\":%s' in host
    assert '\\"complete_taskgraph_executed\\":%s' in host
    assert '\\"thesis_owned_kernel_executed\\":%s' in host
    assert '\\"thesis_resident_kernel_executed\\":%s' in host
    assert '\\"provider_release_attempted\\":%s' in host
    assert '\\"provider_release_succeeded\\":%s' in host
    assert '\\"provider_release_error\\":%d' in host
    assert "simplepim_management_init_called = FRONTIER_PROVIDER_SIMPLEPIM_MANAGEMENT && provider_init_called" in host
    assert "const int any_task_completed =" in host
    assert "const int all_tasks_completed =" in host
    assert "const int complete_taskgraph_executed = all_tasks_completed" in host
    assert '\\"complete_taskgraph_executed\\":%s' in host
    assert '\\"simplepim_kernel_executed\\":false' in host
    assert '\\"simplepim_heap_used\\":false' in host
    assert '\\"simplepim_table_transport_used\\":false' in host
    assert '\\"timing_is_bringup_only\\":true' in host
    assert '\\"hardware_speedup_applicable\\":false' in host


def test_provider_cleanup_preserves_ownership_after_failed_release() -> None:
    provider = (NATIVE / "simplepim_provider.c").read_text(encoding="ascii")
    provider_header = (NATIVE / "simplepim_provider.h").read_text(encoding="ascii")
    host = (NATIVE / "host.c").read_text(encoding="ascii")

    assert "int allocation_used;" in provider_header
    assert "provider->allocation_used = allocation_used;" in provider
    assert "provider->release_succeeded = release_succeeded;" in provider
    assert "provider->allocation_active = 0;" in provider
    assert "if (error != DPU_OK)" in provider
    assert "return error;" in provider
    assert "if (simplepim_provider.release_attempted)" in host
    assert "provider_release_attempted = simplepim_provider.release_attempted;" in host
    assert '\\"allocation_still_owned\\":%s' in host


def test_simplepim_stage_manifest_and_malformed_request_work_from_other_cwd(tmp_path: Path) -> None:
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK compiler is unavailable")

    subprocess.run(["make", "-C", str(NATIVE), "simplepim-stage"], cwd=tmp_path, check=True)
    manifest_path = NATIVE / "build/simplepim_frontier_two_dpu/staged/SimplePIM/management_profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest["source_commit"] == manifest["expected_source_commit"]
    assert manifest["source_worktree_dirty"] is False
    assert len(manifest["staged_source_tree_sha256"]) == 64
    assert manifest["patch_applied"] is True
    expected_sha = hashlib.sha256((NATIVE / "simplepim_management_profile.patch").read_bytes()).hexdigest()
    assert manifest["patch_sha256"] == expected_sha

    subprocess.run(["make", "-C", str(NATIVE), "simplepim-all"], cwd=tmp_path, check=True)
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text("{}", encoding="ascii")
    completed = subprocess.run(
        [
            str(NATIVE / "bin/host_frontier_two_dpu_simplepim_management"),
            "--frontier-package",
            str(request),
            "--frontier-response",
            str(response),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["failure_stage"] == "manifest_parse_failed"
    assert payload["provider_init_called"] is False
    assert payload["provider_init_succeeded"] is False
    assert payload["simplepim_management_init_called"] is False
    assert payload["simplepim_management_allocation_used"] is False
    assert payload["simplepim_management_object_created"] is False
    assert payload["raw_sdk_control_calls_used"] is False
    assert payload["raw_sdk_direct_allocation_used"] is False
    assert payload["raw_sdk_load_used"] is False
    assert payload["raw_sdk_transfer_used"] is False
    assert payload["raw_sdk_launch_used"] is False
    assert payload["raw_sdk_sync_used"] is False
    assert payload["any_task_completed"] is False
    assert payload["all_tasks_completed"] is False
    assert payload["complete_taskgraph_executed"] is False
    assert payload["thesis_resident_kernel_executed"] is False
    assert payload["provider_release_attempted"] is False
    assert payload["provider_release_succeeded"] is False
    assert payload["provider_release_error"] == 0
    assert payload["release"]["allocation_still_owned"] is False


@pytest.mark.skipif(
    shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None,
    reason="UPMEM SDK compiler is unavailable",
)
def test_simplepim_malformed_request_reports_no_runtime_activity(tmp_path: Path) -> None:
    subprocess.run(["make", "simplepim-all"], cwd=NATIVE, check=True)

    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text("{}", encoding="ascii")

    completed = subprocess.run(
        [
            str(NATIVE / "bin/host_frontier_two_dpu_simplepim_management"),
            "--frontier-package",
            str(request),
            "--frontier-response",
            str(response),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "manifest_parse_failed"
    assert payload["simplepim_management_init_called"] is False
    assert payload["raw_sdk_control_calls_used"] is False
    assert payload["thesis_owned_kernel_executed"] is False
    assert payload["thesis_resident_kernel_executed"] is False
    assert payload["simplepim_kernel_executed"] is False
    assert payload["allocation"]["attempted"] is False
    assert payload["load"]["attempted"] is False
