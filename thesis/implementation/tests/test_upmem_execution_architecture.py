from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from quantum_bench.providers.registry import UPMEM_PROVIDER_REGISTRY, route_registry
from quantum_bench.targets.upmem.execution_plan import (
    DpuResourceContext,
    UpmemExecutionPlan,
    UpmemSchedulePlan,
    UpmemValidationStatuses,
    execution_plan_hash,
    validate_execution_plan_graph_identity,
)
from quantum_bench.tn.execution_bundle import execution_hashes

from .support import minimal_real_graph, split_complex_graph


def test_execution_plan_hash_is_deterministic_and_separate_from_task_graph_hashes() -> None:
    graph = minimal_real_graph().graph
    plan = UpmemExecutionPlan.for_task_graph(graph)
    before = execution_hashes(graph)

    assert plan.execution_plan_hash == execution_plan_hash(plan)
    assert plan.execution_plan_hash == UpmemExecutionPlan.for_task_graph(graph).execution_plan_hash
    assert execution_hashes(graph) == before
    assert {
        graph.circuit_semantics_hash,
        graph.tensor_network_hash,
        graph.contraction_plan_hash,
    } == set(before.values())


def test_changed_execution_choice_changes_only_execution_plan_hash() -> None:
    graph = minimal_real_graph().graph
    original = UpmemExecutionPlan.for_task_graph(graph)
    changed = replace(original, schedule=replace(original.schedule, parallelism="tasklet_parallel"))

    assert changed.execution_plan_hash != original.execution_plan_hash
    assert execution_hashes(graph) == {
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "contraction_plan_hash": graph.contraction_plan_hash,
    }

    with pytest.raises(FrozenInstanceError):
        original.schedule = changed.schedule  # type: ignore[misc]
    with pytest.raises(TypeError, match="task_graph"):
        execution_plan_hash(original, task_graph=graph)  # type: ignore[call-arg]


def test_dpu_resource_context_represents_prepare_without_allocation() -> None:
    context = DpuResourceContext(
        requested_dpu_count=2,
        requested_tasklets_per_dpu=4,
    )

    assert context.allocation_status == "not_run"
    assert context.allocated_dpu_count is None
    assert context.allocated_tasklets_per_dpu is None

    with pytest.raises(ValueError, match="allocated counts must be absent"):
        DpuResourceContext(allocated_dpu_count=1)


def test_dpu_resource_context_rejects_verified_allocation_mismatch() -> None:
    context = DpuResourceContext(
        requested_dpu_count=2,
        requested_tasklets_per_dpu=4,
        allocated_dpu_count=2,
        allocated_tasklets_per_dpu=4,
        allocation_status="verified",
    )
    assert (context.allocated_dpu_count, context.allocated_tasklets_per_dpu) == (2, 4)

    with pytest.raises(ValueError, match="positive integer"):
        DpuResourceContext(allocation_status="verified")
    with pytest.raises(ValueError, match="allocated_dpu_count"):
        DpuResourceContext(
            requested_dpu_count=1,
            allocated_dpu_count=2,
            allocated_tasklets_per_dpu=1,
            allocation_status="verified",
        )
    with pytest.raises(ValueError, match="allocated_tasklets_per_dpu"):
        DpuResourceContext(
            requested_tasklets_per_dpu=2,
            allocated_dpu_count=1,
            allocated_tasklets_per_dpu=1,
            allocation_status="verified",
        )


def test_validation_statuses_derive_policy_pass_full_precision_fail() -> None:
    statuses = UpmemValidationStatuses.from_checks(
        execution_contract=True,
        policy_reference=True,
        full_precision_accuracy=False,
    )

    assert statuses.execution_contract_status == "passed"
    assert statuses.policy_reference_status == "passed"
    assert statuses.full_precision_accuracy_status == "failed"
    assert statuses.scientific_validation_status == "failed"

    with pytest.raises(ValueError, match="requires every validation status"):
        UpmemValidationStatuses.derive(
            execution_contract_status="passed",
            policy_reference_status="passed",
            full_precision_accuracy_status="failed",
            scientific_validation_status="passed",
        )


def test_graph_identity_validation_rejects_stale_and_mismatched_graphs() -> None:
    graph = minimal_real_graph().graph
    plan = UpmemExecutionPlan.for_task_graph(graph)

    stale = replace(graph, circuit_semantics_hash="stale")
    with pytest.raises(ValueError, match="circuit_semantics_hash"):
        UpmemExecutionPlan.for_task_graph(stale)

    with pytest.raises(ValueError, match="circuit_semantics_hash"):
        validate_execution_plan_graph_identity(plan, split_complex_graph().graph)


def test_nested_plan_types_and_default_factories_are_enforced() -> None:
    graph = minimal_real_graph().graph
    with pytest.raises(TypeError, match="schedule"):
        UpmemExecutionPlan.for_task_graph(graph, schedule=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="kernel"):
        UpmemExecutionPlan(kernel=object())  # type: ignore[arg-type]

    first = UpmemExecutionPlan()
    second = UpmemExecutionPlan()
    assert first.placement is not second.placement
    assert first.placement.resources is not second.placement.resources
    assert isinstance(first.schedule, UpmemSchedulePlan)


def test_upmem_provider_registry_is_fixed_and_truthful() -> None:
    assert tuple(UPMEM_PROVIDER_REGISTRY) == (
        "upmem_sdk_simulator",
        "upmem_resident_hardware",
        "simplepim",
        "pid_comm",
        "atim",
        "sparsep",
    )
    assert UPMEM_PROVIDER_REGISTRY["upmem_sdk_simulator"].qualification_scope == "sdk_simulator_execution_contract"
    assert UPMEM_PROVIDER_REGISTRY["upmem_sdk_simulator"].qualification_status == "validated"
    resident = UPMEM_PROVIDER_REGISTRY["upmem_resident_hardware"]
    assert resident.route_id is None
    assert resident.benchmark_surface_id == "upmem_tn_hardware_taskgraph_resident"
    assert resident.qualification_status == "guarded"
    simplepim = UPMEM_PROVIDER_REGISTRY["simplepim"]
    assert simplepim.qualification_scope == "bounded_physical_management_operator_qualification"
    assert simplepim.qualification_status == "guarded"
    assert simplepim.availability_status == "environment_dependent"
    assert simplepim.benchmark_surface_id == "upmem_tn_hardware_simplepim_bounded"
    assert "general TaskGraph executor" in " ".join(simplepim.notes)
    assert all(UPMEM_PROVIDER_REGISTRY[name].qualification_status == "planned" for name in ("pid_comm", "atim", "sparsep"))
    with pytest.raises(TypeError):
        UPMEM_PROVIDER_REGISTRY["new"] = UPMEM_PROVIDER_REGISTRY["simplepim"]  # type: ignore[index]


def test_provider_route_ids_are_referentially_integral(tmp_path) -> None:
    routes = route_registry(tmp_path)
    for descriptor in UPMEM_PROVIDER_REGISTRY.values():
        if descriptor.route_id is not None:
            assert descriptor.route_id in routes


def test_public_status_docs_do_not_keep_superseded_milestone_claims() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    architecture = (root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    roadmap = (root / "docs" / "slr_architecture_implementation_roadmap.md").read_text(encoding="utf-8")

    assert "balanced useful-slice acceptance is still open" not in readme
    assert "physical qualification\nis pending" not in readme
    assert "SimplePIM | External pinned repository | Task-specific target" not in architecture
    assert "Bounded physical management/operator qualification" in architecture
    assert "the current second partial is zero" not in architecture
    assert "does not prove that either planner optimized physical" in architecture
    assert "host-mediated transfer is the initial communication provider" in architecture
    assert "M4.5: descriptor-driven shared runtime" in roadmap
    assert "Complete M2.1 before interpreting" not in roadmap
    assert "bounded_taskgraph_executed" in architecture
