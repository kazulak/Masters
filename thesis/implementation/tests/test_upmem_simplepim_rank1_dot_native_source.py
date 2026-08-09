import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


IMPLEMENTATION = Path(__file__).resolve().parents[1]
ROUTE = IMPLEMENTATION / "native" / "upmem" / "simplepim" / "upmem_sdk_rank1_dot_m4_2"
SUBMODULE = IMPLEMENTATION / "external" / "SimplePIM"
PINNED_COMMIT = "1d639c53532555f01e9f71d872e7712b166d6cba"
SIMULATOR_ENV_KEYS = (
    "DPU_BACKEND",
    "DPU_PROFILE",
    "SIMPLEPIM_BACKEND",
    "UPMEM_BACKEND",
    "UPMEM_MODE",
    "UPMEM_TARGET",
    "UPMEM_PROFILE",
    "UPMEM_PROFILE_BASE",
)


def test_route_is_isolated_and_uses_the_pinned_clean_source():
    assert ROUTE.is_dir()
    assert subprocess.run(
        ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == PINNED_COMMIT
    assert subprocess.run(
        ["git", "-C", str(SUBMODULE), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""


def test_route_patch_changes_exactly_both_uninitialized_expressions():
    patch = (ROUTE / "simplepim_rank1_hardening.patch").read_text()
    assert patch.count("-    uint32_t unroll_block_rest = copy_block_size-unroll_block_rest;") == 2
    assert patch.count("+    uint32_t unroll_block_rest = copy_block_size-unroll_block_size;") == 2
    assert "MapProcessing.h" in patch
    assert "Map.c" in patch and "free_space_start_pos" in patch
    assert "ProcessingHelperHost.c" in patch and "if(succ != 0)" in patch
    assert "GenRed.c" in patch and "dlerror()" in patch


def test_route_owned_overlay_uses_int64_dot_product_callbacks():
    map_source = (ROUTE / "benchmark" / "dot_funcs" / "map.h").read_text()
    map_reduce = (ROUTE / "benchmark" / "dot_reduce_funcs" / "map_to_val_func.h").read_text()
    combine = (ROUTE / "benchmark" / "dot_reduce_funcs" / "init_combine_func.h").read_text()
    assert "simplepim" not in map_source.lower()
    assert "int64_t" in map_source
    assert "pair[0]" in map_source and "pair[1]" in map_source
    assert "int64_t" in map_reduce
    assert "*key = 0" in map_reduce
    assert "int64_t" in combine and "+=" in combine


def test_native_contract_is_physical_two_dpu_persistent_simplepim_only():
    source = (ROUTE / "host.c").read_text()
    makefile = (ROUTE / "Makefile").read_text()
    for required in (
        "UPMEM_ALLOW_PHYSICAL_HARDWARE",
        "M42_DPU_COUNT 2u",
        "M42_INITIALIZATION_TASKLETS 1u",
        "M42_OPERATOR_TASKLETS 12u",
        "M42_VECTOR_LENGTH 256u",
        "M42_WARMUPS 1u",
        "M42_REPEATS 5u",
        "table_management_init(M42_DPU_COUNT)",
        "simplepim_scatter",
        "table_zip",
        "table_map",
        "table_gen_red",
        "dpu_free",
        "table_reuse\\\":false",
        "bounded_table_growth\\\":",
        "persistent_allocation_requested\\\":true",
        "persistent_allocation_observed\\\":",
        "logical_payload_transfer_bytes_per_iteration",
        "logical_payload_transfer_bytes_total_session",
        "observed_table_count",
        "mram_high_water_bytes_per_dpu",
        "hardware_speedup_applicable\\\":false",
        "host_mediated_reduction\\\":",
    ):
        assert required in source, required
    assert "simplepim_allreduce(" not in source
    assert "simplepim_gather(" not in source
    assert "INITIALIZATION_TASKLETS ?= 1" in makefile
    assert "OPERATOR_TASKLETS ?= 12" in makefile
    assert "-ldl" in makefile and "-lm" in makefile and "-fopenmp" in makefile
    assert "simplepim_rank1_hardening.patch" in makefile
    assert '--stage-manifest "../../simplepim_stage_manifest.json"' in makefile
    assert 'create_handle("", ZIP)' not in source
    assert "zip_binary_hash" not in source
    assert "strdup(" not in source
    assert "state->hardware_execution = state->map_completed && state->reduction_completed" in source
    assert "strcmp(provenance->source_commit, M42_SOURCE_COMMIT) == 0" in source
    assert "thesis_direct_raw_sdk_allocation_used\\\":false" in source
    assert "simplepim_managed_allocation\\\":" in source
    assert "host_hash == NULL || init_hash == NULL || map_hash == NULL || genred_hash == NULL || reduce_so_hash == NULL" in source
    for key in SIMULATOR_ENV_KEYS:
        assert f'getenv("{key}") != NULL' in source


def test_make_dry_run_does_not_allocate_or_launch():
    result = subprocess.run(
        ["make", "-C", str(ROUTE), "--dry-run", "parser"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--mode parser" in result.stdout
    assert "dpu_alloc" not in result.stdout
    assert "--mode execute" not in result.stdout


def test_stage_applies_patch_without_mutating_submodule():
    subprocess.run(["make", "-C", str(ROUTE), "stage"], check=True)
    staged = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "staged"
    patched = (staged / "lib" / "processing" / "map" / "MapProcessing.h").read_text()
    patched_map = (staged / "lib" / "processing" / "map" / "Map.c").read_text()
    patched_helper = (staged / "lib" / "processing" / "ProcessingHelperHost.c").read_text()
    patched_genred = (staged / "lib" / "processing" / "gen_red" / "GenRed.c").read_text()
    manifest = json.loads((staged / "simplepim_stage_manifest.json").read_text())
    assert (staged / "benchmarks" / "rank1_dot" / "dot_funcs" / "map.h").is_file()
    assert (staged / "benchmarks" / "rank1_dot" / "dot_reduce_funcs" / "map_to_val_func.h").is_file()
    assert patched.count("unroll_block_rest = copy_block_size-unroll_block_size") == 2
    assert patched.count("unroll_block_rest = copy_block_size-unroll_block_rest") == 0
    assert patched_map.count("table_management->free_space_start_pos =") == 2
    assert "char func_bodyname[2048] = {0}" in patched_helper
    assert patched_helper.count("if(succ != 0)") >= 7
    assert "failure:" in patched_helper
    assert "init_error != NULL || combine_error != NULL" in patched_genred
    assert manifest["source_commit"] == PINNED_COMMIT
    assert manifest["expected_source_commit"] == PINNED_COMMIT
    assert manifest["source_worktree_dirty"] is False
    assert manifest["patch_applied"] is True
    assert subprocess.run(
        ["git", "-C", str(SUBMODULE), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""


def test_parser_mode_runs_without_hardware_when_sdk_is_available():
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK is not installed")
    subprocess.run(["make", "-C", str(ROUTE), "parser"], check=True)
    response_path = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "parser_response.json"
    response = json.loads(response_path.read_text())
    assert response["status"] == "prepared"
    assert response["parser_mode"] is True
    assert response["target_observed"] == "not_executed"
    assert response["allocated_dpu_count"] is None
    assert response["allocation_attempted"] is False
    assert response["initialization_tasklets_per_dpu"] == 1
    assert response["operator_tasklets_per_dpu"] == 12
    assert response["persistent_allocation_requested"] is True
    assert response["persistent_allocation_observed"] is False
    assert response["bounded_table_growth"] is False
    assert response["expected_table_count_session"] == 30
    assert response["observed_table_count"] is None
    assert response["mram_conservative_bound_bytes_per_dpu"] == 18624
    assert response["logical_payload_transfer_bytes_per_iteration"] == 2064
    assert response["logical_payload_transfer_bytes_total_session"] == 12384
    assert response["simplepim_operator_api_used"] is False
    assert response["thesis_direct_raw_sdk_allocation_used"] is False
    assert response["simplepim_managed_allocation"] is False
    for field in (
        "host_binary_hash",
        "initialization_binary_hash",
        "map_binary_hash",
        "genred_binary_hash",
        "genred_reduce_shared_object_hash",
    ):
        assert response[field] is None
    assert "tasklets_per_dpu" not in response
    assert "zip_binary_hash" not in response
    assert response["failure_stage"] is None


def test_execute_mode_requires_opt_in_and_writes_structured_failure_without_hardware():
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK is not installed")
    response_path = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "execute_response.json"
    env = os.environ.copy()
    for name in ("UPMEM_ALLOW_PHYSICAL_HARDWARE", *SIMULATOR_ENV_KEYS):
        env.pop(name, None)
    result = subprocess.run(
        ["make", "-C", str(ROUTE), "execute"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    response = json.loads(response_path.read_text())
    assert result.returncode != 0
    assert response["status"] == "failed"
    assert response["failure_stage"] == "opt_in"
    assert response["target_observed"] == "not_executed"
    assert response["allocation_attempted"] is False
    assert response["cpu_fallback_used"] is False
    assert response["simulator_kernel_executed"] is False


@pytest.mark.parametrize("selector", SIMULATOR_ENV_KEYS)
def test_execute_mode_rejects_simulator_selectors_before_allocation(selector):
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK is not installed")
    subprocess.run(["make", "-C", str(ROUTE), "build"], check=True)
    staged = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "staged"
    host = staged / "benchmarks" / "rank1_dot" / "bin" / "rank1_dot_host"
    response_path = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "simulator_selector_response.json"
    env = os.environ.copy()
    for name in SIMULATOR_ENV_KEYS:
        env.pop(name, None)
    env["UPMEM_ALLOW_PHYSICAL_HARDWARE"] = "1"
    env[selector] = "simulator"
    result = subprocess.run(
        [str(host), "--mode", "execute", "--response", str(response_path), "--stage-manifest", str(staged / "simplepim_stage_manifest.json")],
        env=env,
        check=False,
    )
    response = json.loads(response_path.read_text())
    assert result.returncode != 0
    assert response["failure_stage"] == "hardware_profile"
    assert response["reason"] == f"{selector}_must_be_unset"
    assert response["target_observed"] == "not_executed"
    assert response["allocation_attempted"] is False
