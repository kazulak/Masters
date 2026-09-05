from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quantum_bench.experiment import load_experiment_config
import quantum_bench.quantized_contraction as qc
from quantum_bench.model import ContractNode, TensorSpec, TensorView


IMPLEMENTATION = Path(__file__).resolve().parents[1]
SCRIPT_PATH = IMPLEMENTATION / "scripts" / "qualify_upmem_execution_integration.py"
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_upmem_execution_integration", SCRIPT_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
QUALIFIER = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(QUALIFIER)


def _config(name: str) -> dict[str, Any]:
    return json.loads(
        QUALIFIER.canonical_json(
            load_experiment_config(IMPLEMENTATION / "configs" / name)
        )
    )


def test_integration_matrices_are_bounded_and_cross_policy_complete() -> None:
    sdk = _config("tn_benchmark_upmem_execution_integration_sdk_v1.yml")
    physical = _config("tn_benchmark_upmem_execution_integration_physical_v1.yml")

    assert len(QUALIFIER._expected_cells(sdk)) == 14
    assert len(QUALIFIER._expected_cells(physical)) == 7
    assert set(sdk["cases"]) == {"bell2", "stress14"}
    assert set(physical["cases"]) == {"bell2", "stress14"}
    assert "stress18" not in sdk["cases"]
    assert "ghz14" not in sdk["cases"]
    assert "xor18" not in sdk["cases"]
    assert set(QUALIFIER.ROUTE_SPECS) == set(sdk["routes"]) == set(physical["routes"])
    assert {route["numeric_policy"] for route in sdk["routes"].values()} == {
        QUALIFIER.FLOAT32,
        QUALIFIER.INT8,
    }
    assert {route["numeric_policy"] for route in physical["routes"].values()} == {
        QUALIFIER.FLOAT32,
        QUALIFIER.INT8,
    }
    assert {
        (route["options"]["dpu_count"], route["options"]["tasklets_per_dpu"])
        for route in sdk["routes"].values()
    } == {(1, 1), (1, 8), (3, 8), (4, 8)}
    assert physical["routes"]["int8_3dpu_t8"]["options"]["dpu_count"] == 3
    assert physical["routes"]["int8_4dpu_t8"]["numeric_policy"] == QUALIFIER.INT8
    assert physical["routes"]["float32_4dpu_t8"]["numeric_policy"] == QUALIFIER.FLOAT32
    assert all(
        route["options"]["rank_paths"] == ["/dev/dpu_rank1"]
        for route in physical["routes"].values()
    )
    assert physical["matrix"] == [
        {"case_id": "bell2", "plan_id": "greedy", "route_ids": ["float32_1dpu_t1", "int8_1dpu_t1"]},
        {
            "case_id": "stress14",
            "plan_id": "greedy",
            "route_ids": [
                "float32_1dpu_t8",
                "int8_1dpu_t8",
                "float32_4dpu_t8",
                "int8_3dpu_t8",
                "int8_4dpu_t8",
            ],
        },
    ]


def test_prepare_uses_explicit_remote_paths_and_rehashes_identity(tmp_path: Path) -> None:
    template = IMPLEMENTATION / "configs" / "tn_benchmark_upmem_execution_integration_physical_v1.yml"
    output = tmp_path / "prepared.yml"
    QUALIFIER.prepare(
        template,
        output,
        rank_path="/dev/dpu_rank1",
        session_root=tmp_path / "sessions",
        expected_cpu=0,
        binary_root=tmp_path / "upmem-bin",
    )
    prepared = load_experiment_config(output)
    assert prepared["experiment_identity_payload"]["label"] == (
        "upmem-execution-integration-physical-v2"
    )
    assert QUALIFIER.identity_hash(
        "quantum_bench.experiment_id.v3", prepared["experiment_identity_payload"]
    ) == prepared["experiment_id"]
    assert prepared["collection"]["machine_policy"]["affinity"]["expected_cpus"] == (0,)
    for route_id, route in prepared["routes"].items():
        options = route["options"]
        assert options["rank_paths"] == ("/dev/dpu_rank1",)
        assert options["session_root"] == str((tmp_path / "sessions" / route_id).resolve())
        suffix = options["tasklets_per_dpu"]
        assert options["host_binary"].endswith(f"host_upmem_execution_plan_v4_t{suffix}")
    QUALIFIER._validate_frozen_config(prepared, kind="physical")


def _skinny_contract_node() -> ContractNode:
    left = TensorSpec("left", (0, 1), (2, 3), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (3, 1), "dense", dtype="complex128")
    output = TensorSpec(
        "output", (0, 2), (2, 1), "dense", dtype="complex128", produced_by="contract"
    )
    return ContractNode(
        node_id="contract",
        left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id=right.id, labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


@pytest.mark.parametrize("cpu,rank", [(1, "/dev/dpu_rank1"), (0, "/dev/dpu_rank2")])
def test_prepare_rejects_machine_drift_without_writing(
    tmp_path: Path, cpu: int, rank: str
) -> None:
    output = tmp_path / "prepared.yml"
    with pytest.raises(ValueError, match="CPU 0 and dpu_rank1"):
        QUALIFIER.prepare(
            IMPLEMENTATION / "configs/tn_benchmark_upmem_execution_integration_physical_v1.yml",
            output,
            rank_path=rank,
            session_root=tmp_path / "sessions",
            expected_cpu=cpu,
            binary_root=tmp_path / "bin",
        )
    assert not output.exists()


def test_cpu_zero_precision_boundary_and_dense_skinny_fixture_semantics() -> None:
    zero = qc.quantize_complex_shared_scale(np.zeros((2, 3), dtype=np.complex128))
    assert zero.scale == 1.0
    np.testing.assert_array_equal(zero.q_real, np.zeros((2, 3), dtype=np.int8))
    np.testing.assert_array_equal(zero.q_imag, np.zeros((2, 3), dtype=np.int8))

    precise = np.array([1.00000006 + 0.0j, 0.25 + 0.5j], dtype=np.complex128)
    encoded = qc.quantize_complex_shared_scale(precise)
    assert encoded.scale == float(np.float64(1.00000006) / 127.0)
    assert encoded.scale != float(np.float32(1.00000006) / 127.0)

    boundary = qc.quantize_complex_shared_scale(
        np.array([127.0 - 127.0j, 0.0 + 0.0j], dtype=np.complex128)
    )
    np.testing.assert_array_equal(boundary.q_real, np.array([127, 0], dtype=np.int8))
    np.testing.assert_array_equal(boundary.q_imag, np.array([-127, 0], dtype=np.int8))
    assert boundary.diagnostics.boundary_saturation_count == 2
    assert qc.theoretical_accumulator_bound(3) == 2 * 3 * 127**2

    # This is a dense complex fixture with a skinny (M=2, N=1) contraction.
    node = _skinny_contract_node()
    left = np.array(
        [[127.0 + 0.0j, 0.0 + 1.0j, -2.0 + 3.0j], [1.0 - 4.0j, 0.0j, 5.0 + 0.0j]],
        dtype=np.complex128,
    )
    right = np.array([[0.5 + 1.0j], [127.0 - 127.0j], [0.0 + 0.0j]], dtype=np.complex128)
    result, facts = qc.contract_complex_int8_reference(left, right, node)
    assert result.shape == (2, 1)
    assert result.dtype == np.dtype(np.complex64)
    assert (facts.M, facts.N, facts.K) == (2, 1, 3)
    assert facts.int32_theoretical_accumulator_bound == 2 * facts.K * 127**2


def _fixture_evidence(kind: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = _config(
        "tn_benchmark_upmem_execution_integration_sdk_v1.yml"
        if kind == "sdk"
        else "tn_benchmark_upmem_execution_integration_physical_v1.yml"
    )
    physical = kind == "physical"
    target = "physical_hardware" if physical else "sdk_simulator"
    cells = sorted(QUALIFIER.KINDS[kind]["cells"])
    contracts = QUALIFIER._expected_execution_contract(config)
    samples: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for index, (case_id, plan_id, route_id) in enumerate(cells):
        policy, dpus, ranks, tasklets = QUALIFIER.ROUTE_SPECS[route_id]
        session_id = f"{kind}-session-{index}"
        contract = contracts[(case_id, plan_id, route_id)]
        identities = contract["identities"]
        physical_plan_id = identities["physical_plan_id"]
        binary_hashes = {f"{field}_sha256": f"{tasklets * 10 + i:064x}" for i, field in enumerate(QUALIFIER.PATH_FIELDS)}
        executable_id = QUALIFIER._expected_executable_id(
            binary_hashes, executor=QUALIFIER.KINDS[kind]["executor"]
        )
        common = {
            "target_observed": target,
            "cpu_fallback_used": False,
            "requested_dpus": dpus,
            "allocated_dpus": dpus,
            "active_dpus": len(contract["active_dpu_ids"]),
            "active_ranks": contract["active_rank_indices"],
            "rank_count": ranks,
            "tasklets_per_dpu": tasklets,
            "observed_rank_count": ranks,
            "observed_tasklets_per_dpu": tasklets,
            "requested_dpu_count": dpus,
            "allocated_dpu_count": dpus,
            "observed_dpu_count": dpus,
            "startup_resource_admission_passed": True,
            **contract["admission"],
            "request_transport": QUALIFIER.PACKED,
            "kernel_policy": QUALIFIER.KERNEL_POLICY,
            "kernel_implementation_id": QUALIFIER.KERNEL_IMPLEMENTATION,
            "physical_plan_id": physical_plan_id,
            "logical_plan_id": identities["logical_plan_id"],
            "physical_target_verified": physical,
            "hardware_kernel_executed": physical,
            "simulator_kernel_executed": not physical,
        }
        terminal = {
            **common,
            **binary_hashes,
            "active_dpu_ids": contract["active_dpu_ids"],
            "active_rank_indices": contract["active_rank_indices"],
            "binary_identity_verified": True,
            "native_identity_verified": True,
            "simulator_target_verified": not physical,
            "hardware_release_verified": True,
        }
        sessions.append(
            {
                "case_id": case_id,
                "plan_id": plan_id,
                "route_id": route_id,
                "session_instance_id": session_id,
                "status": "success",
                "release_verified": True,
                "terminal_backend_facts": terminal,
            }
        )
        validation = {
            "policy_reference_passed": True,
            "full_precision_threshold_applicable": policy == QUALIFIER.FLOAT32,
            "full_precision_passed": True if policy == QUALIFIER.FLOAT32 else None,
            "accuracy_qualified": policy == QUALIFIER.FLOAT32,
        }
        samples.append(
            {
                "case_id": case_id,
                "plan_id": plan_id,
                "route_id": route_id,
                "attempt_kind": "measurement",
                "sample_index": 0,
                "block_id": 0,
                "observed_affinity": [0],
                "session_instance_id": session_id,
                "status": "success",
                "measurement": {"scope_id": "steady_execution_v1"},
                "output_sha256": "e" * 64,
                "identities": {
                    **identities,
                    "executable_id": executable_id,
                },
                "numeric_facts": {
                    "numeric_policy": policy,
                    "operations": [{"numeric_policy": policy}],
                },
                "validation": validation,
                "backend_facts": common,
            }
        )
    manifest = {
        "source_commit": "a" * 40,
        "source_worktree_dirty": False,
        "experiment_id": config["experiment_id"],
        "configuration": {"experiment": config},
    }
    summary = {
        "status": "completed",
        "sample_count": len(samples),
        "session_count": len(sessions),
        "success_count": len(samples),
        "failed_count": 0,
        "unsupported_count": 0,
    }
    return manifest, samples, sessions, summary


def _patch_inspector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest, samples, sessions, summary = fixture
    monkeypatch.setattr(
        QUALIFIER,
        "load_artifacts",
        lambda _root: (manifest, tuple(samples), tuple(sessions)),
    )
    monkeypatch.setattr(QUALIFIER, "verify_artifacts", lambda _root: summary)
    return manifest, samples, sessions


@pytest.mark.parametrize("kind", ["sdk", "physical"])
def test_inspector_accepts_exact_plan_coverage(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_evidence(kind)
    _patch_inspector(monkeypatch, tmp_path, fixture)
    result = QUALIFIER.inspect(tmp_path, kind=kind, expected_source="a" * 40)
    assert result["status"] == "passed"
    assert result["matrix_cell_count"] == (14 if kind == "sdk" else 7)
    assert result["correctness_only"] is (kind == "physical")


@pytest.mark.parametrize("kind", ["sdk", "physical"])
def test_v1_probe_identity_cannot_qualify_as_v2(kind: str) -> None:
    config = _config(f"tn_benchmark_upmem_execution_integration_{kind}_v1.yml")
    payload = config["experiment_identity_payload"]
    payload["label"] = f"upmem-execution-integration-{kind}-v1"
    config["experiment_id"] = QUALIFIER.identity_hash("quantum_bench.experiment_id.v3", payload)
    with pytest.raises(ValueError, match="identity label is not frozen"):
        QUALIFIER._validate_frozen_config(config, kind=kind)


def test_real_plans_cover_every_stress14_dpu_and_bell2_partial_waves() -> None:
    config = _config("tn_benchmark_upmem_execution_integration_sdk_v1.yml")
    QUALIFIER._validate_frozen_config(config, kind="sdk")
    assert config["experiment_identity_payload"]["label"] == "upmem-execution-integration-sdk-v2"
    assert config["cases"]["stress14"]["circuit"]["parameters"] == {
        "n_qubits": 14, "repeat_layers": 2,
    }
    contracts = QUALIFIER._expected_execution_contract(config)
    for (case_id, _, route_id), contract in contracts.items():
        dpus = QUALIFIER.ROUTE_SPECS[route_id][1]
        expected_active = dpus if case_id == "stress14" else 1
        assert contract["active_dpu_ids"] == [[0, dpu] for dpu in range(expected_active)]
        assert all(work > 0 for work in contract["work_by_dpu"].values())
        admission = contract["admission"]
        assert admission["execution_active_dpu_count"] == expected_active
        assert admission["execution_resource_admission_passed"] is (expected_active == dpus)
        assert admission["execution_resource_admission_reasons"] == (
            [] if expected_active == dpus else ["active_dpu_count_mismatch"]
        )

    # A smaller stress circuit cannot silently replace the multi-DPU coverage case.
    config["cases"]["stress14"]["circuit"]["parameters"]["n_qubits"] = 4
    with pytest.raises(ValueError, match="does not cover every requested DPU"):
        QUALIFIER._expected_execution_contract(config)


@pytest.mark.parametrize("kind", ["sdk", "physical"])
@pytest.mark.parametrize("mutation", [
    "missing_dpu", "wrong_active_count", "wrong_terminal_count", "wrong_requested_count",
    "physical_plan", "problem", "network", "logical_plan", "admission", "reasons",
])
def test_inspector_rejects_plan_coverage_and_identity_drift(
    kind: str, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_evidence(kind)
    _, samples, sessions, _ = fixture
    sample = next(row for row in samples if row["case_id"] == "stress14"
                  and row["route_id"] == "int8_4dpu_t8")
    terminal = next(row["terminal_backend_facts"] for row in sessions
                    if row["session_instance_id"] == sample["session_instance_id"])
    if mutation == "missing_dpu":
        terminal["active_dpu_ids"] = [[0, 0], [0, 1], [0, 2]]
    elif mutation == "wrong_active_count":
        sample["backend_facts"]["active_dpus"] = 3
    elif mutation == "wrong_terminal_count":
        terminal["allocated_dpu_count"] = 3
    elif mutation == "wrong_requested_count":
        sample["backend_facts"]["requested_dpus"] = 3
    elif mutation == "admission":
        sample["backend_facts"]["execution_resource_admission_passed"] = False
    elif mutation == "reasons":
        sample["backend_facts"]["execution_resource_admission_reasons"] = ["active_dpu_count_mismatch"]
    else:
        field = {"physical_plan": "physical_plan_id", "problem": "problem_id",
                 "network": "tensor_network_structure_id", "logical_plan": "logical_plan_id"}[mutation]
        sample["identities"][field] = "f" * 64
        # Internal agreement is insufficient; IDs must match the rebuilt plan.
        sample["backend_facts"][field] = "f" * 64
    _patch_inspector(monkeypatch, tmp_path, fixture)
    with pytest.raises(ValueError):
        QUALIFIER.inspect(tmp_path, kind=kind, expected_source="a" * 40)


@pytest.mark.parametrize("mutation", ["passed", "missing_reason", "extra_dpu"])
def test_bell2_partial_wave_requires_exact_admission_facts(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_evidence("sdk")
    _, samples, sessions, _ = fixture
    sample = next(row for row in samples if row["case_id"] == "bell2"
                  and row["route_id"] == "int8_3dpu_t8")
    if mutation == "passed":
        sample["backend_facts"]["execution_resource_admission_passed"] = True
    elif mutation == "missing_reason":
        sample["backend_facts"]["execution_resource_admission_reasons"] = []
    else:
        terminal = next(row["terminal_backend_facts"] for row in sessions
                        if row["session_instance_id"] == sample["session_instance_id"])
        terminal["active_dpu_ids"] = [[0, 0], [0, 1]]
    _patch_inspector(monkeypatch, tmp_path, fixture)
    with pytest.raises(ValueError):
        QUALIFIER.inspect(tmp_path, kind="sdk", expected_source="a" * 40)


@pytest.mark.parametrize("mutation", ["dirty", "transport", "int8_threshold", "executable", "binary"])
def test_inspector_fails_closed_on_provenance_transport_or_threshold_mutation(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_evidence("physical")
    manifest, samples, sessions, summary = fixture
    if mutation == "dirty":
        manifest["source_worktree_dirty"] = True
    elif mutation == "transport":
        samples[0]["backend_facts"]["request_transport"] = "directory_v1"
    elif mutation == "executable":
        samples[0]["identities"]["executable_id"] = "f" * 64
    elif mutation == "binary":
        sessions[0]["terminal_backend_facts"]["dpu_binary_sha256"] = "f" * 64
    else:
        int8_sample = next(
            sample
            for sample in samples
            if QUALIFIER.ROUTE_SPECS[sample["route_id"]][0] == QUALIFIER.INT8
        )
        int8_sample["validation"]["full_precision_threshold_applicable"] = True
    _patch_inspector(monkeypatch, tmp_path, (manifest, samples, sessions, summary))
    with pytest.raises(ValueError):
        QUALIFIER.inspect(tmp_path, kind="physical", expected_source="a" * 40)


def test_preserved_historical_192_file_checksums_match_current_bytes() -> None:
    manifest_path = (
        IMPLEMENTATION
        / "thesis_results"
        / "upmem_execution_integration_v1"
        / "prior_calibration_supersession.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_protocol"]["total_attempts"] == 192
    assert manifest["historical_files_modified"] is False
    for relative, expected in manifest["preserved_file_sha256"].items():
        path = IMPLEMENTATION / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, relative
