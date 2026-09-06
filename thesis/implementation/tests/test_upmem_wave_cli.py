"""Experimental wave policies must reach the canonical experiment runner."""

import copy
import json

import pytest
import yaml

from quantum_bench import cli
from quantum_bench.evidence import executable_id, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts
from tests.test_cli_report import _simulator_config
from tests.test_upmem_wave_runtime import binaries as _sdk_binaries


binaries = _sdk_binaries


def _configuration(tmp_path, native_paths, *, sliced=False):
    config = yaml.safe_load(_simulator_config())
    config["defaults"]["timeout_s"] = 30.0
    config["cases"]["bell"]["circuit"].update(
        name="quantization_stress", parameters={"n_qubits": 4, "repeat_layers": 2}
    )
    if sliced:
        config["plans"]["greedy"]["slicing"] = {
            "node_id": "contract_24",
            "minimum_slice_count": 4,
        }
    original = config["routes"].pop("simulator")
    for policy in ("split_complex_float32_v1", "complex_int8_shared_scale_v1"):
        for schedule in ("serial_nodes_v1", "static_dag_waves_v1"):
            for fuse in (False, True):
                name = f"{policy}-{schedule}-{int(fuse)}"
                route = copy.deepcopy(original)
                route["numeric_policy"] = policy
                route["options"].update(
                    dpu_count=3,
                    tasklets_per_dpu=8,
                    session_root=str(tmp_path / "sessions" / name),
                    host_binary=native_paths[0],
                    dpu_binary=native_paths[1],
                    initialization_binary=native_paths[2],
                    request_transport="packed_wave_v1",
                    schedule_policy=schedule,
                    fuse_complex=fuse,
                    geometry_policy="outer_k1_v1" if fuse else "panel_only_v1",
                )
                config["routes"][name] = route
    config["matrix"][0]["route_ids"] = list(config["routes"])
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_legacy_executable_identity_has_unchanged_payload(monkeypatch):
    monkeypatch.setattr(cli, "_sha256_file", lambda path: "b" * 64)
    route = dict(
        executor="upmem_physical",
        options=dict(
            host_binary="host", dpu_binary="dpu", initialization_binary="init"
        ),
    )
    expected = executable_id(
        dict(
            executor="upmem_physical",
            abi_version=4,
            static_file_sha256=dict(
                host_binary="b" * 64,
                dpu_binary="b" * 64,
                initialization_binary="b" * 64,
            ),
            request_transport="packed_operation_v1",
            source_commit=None,
            dependency_versions={},
        )
    )
    assert cli._executable_identity(route) == expected
    route["options"].update(
        request_transport="packed_operation_v1",
        schedule_policy="serial_nodes_v1",
        fuse_complex=False,
        geometry_policy="panel_only_v1",
    )
    assert cli._executable_identity(route) == expected


def test_prepared_executable_identity_binds_dispatch_but_not_schedule_or_precision(
    monkeypatch,
):
    monkeypatch.setattr(cli, "_sha256_file", lambda path: "b" * 64)
    route = dict(
        executor="upmem_sdk_simulator",
        numeric_policy="split_complex_float32_v1",
        options=dict(
            host_binary="host", dpu_binary="dpu", initialization_binary="init"
        ),
    )
    legacy = cli._executable_identity(route)
    route["options"]["request_transport"] = "packed_wave_v1"
    prepared = cli._executable_identity(route)
    assert prepared != legacy
    route["options"]["schedule_policy"] = "static_dag_waves_v1"
    route["numeric_policy"] = "complex_int8_shared_scale_v1"
    assert cli._executable_identity(route) == prepared
    route["options"]["fuse_complex"] = True
    fused = cli._executable_identity(route)
    assert fused != prepared
    route["options"]["geometry_policy"] = "outer_k1_v1"
    assert cli._executable_identity(route) not in (legacy, prepared, fused)


@pytest.mark.parametrize("sliced", [False, True])
def test_plan_run_and_canonical_verifier_preserve_wave_policy(
    tmp_path, binaries, monkeypatch, sliced
):
    path = _configuration(tmp_path, binaries[8], sliced=sliced)
    config = load_experiment_config(path)
    document = cli._plan_document(config)
    entries = {entry["route_id"]: entry for entry in document["entries"]}
    monkeypatch.setattr(
        cli, "open_upmem", lambda *a, **k: pytest.fail("physical route opened")
    )
    output = tmp_path / "evidence"
    result = cli.run_command(str(path), str(output), allow_physical=False)
    assert result["status"] == "completed"
    verified = verify_artifacts(output)
    assert verified["success_count"] == 8
    manifest, samples, sessions = load_artifacts(output)
    assert len(samples) == len(sessions) == 8
    assert manifest["status"] == "completed"
    assert all(
        row["session_protocol_id"] == "upmem_prepared_wave_abi_v5" for row in sessions
    )
    saw_parallel = False
    for sample in samples:
        route = config["routes"][sample["route_id"]]
        options = route["options"]
        plan_facts = entries[sample["route_id"]]["upmem"]
        assert (
            plan_facts["execution_policy"]["schedule_policy"]
            == options["schedule_policy"]
        )
        assert (
            sample["identities"]["physical_plan_id"] == plan_facts["physical_plan_id"]
        )
        assert sample["identities"]["executable_id"] == cli._executable_identity(route)
        facts = sample["backend_facts"]
        assert facts["request_transport"] == "packed_wave_v1"
        assert facts["schedule_policy"] == options["schedule_policy"]
        assert facts["geometry_kernel_policy"] == options["geometry_policy"]
        assert facts["simulator_kernel_executed"] and not facts["cpu_fallback_used"]
        assert facts["target_observed"] == "sdk_simulator"
        assert sample["validation"]["policy_reference_passed"]
        assert sample["validation"]["full_precision_passed"] is not False
        ops = facts["operation_facts"]
        if options["fuse_complex"]:
            assert sum(op["fused_tile_count"] for op in ops) > 0
            _, _, dag, _ = cli._plan_dag(
                cli._job(config["cases"]["bell"]), config["plans"]["greedy"]
            )
            compiled = cli.plan_upmem(
                dag,
                numeric_policy=route["numeric_policy"],
                topology=cli._topology(route),
                schedule_policy=options["schedule_policy"],
            )
            expected_outer = sum(
                unit.k_size == 1
                for stage in compiled.stages
                for unit in stage.work_units
            )
            assert sum(op["outer_product_tile_count"] for op in ops) == expected_outer
            # Internal slicing exposes K=1 work absent from the unsliced fixture.
            assert (expected_outer > 0) == sliced
        else:
            assert sum(op["fused_tile_count"] for op in ops) == 0
            assert sum(op["outer_product_tile_count"] for op in ops) == 0
        parallel = any(len(op["cohort_node_ids"]) > 1 for op in ops)
        if options["schedule_policy"] == "serial_nodes_v1":
            assert not parallel
        saw_parallel |= parallel
    assert saw_parallel
    # Plan output remains JSON-serializable alongside canonical evidence records.
    json.dumps(document, sort_keys=True)
