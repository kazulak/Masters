from __future__ import annotations

import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import TensorNetworkSpec, TensorSpec
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.planning import (
    PlannerEngine,
    PlannerRequest,
    plan_contractions,
    plan_opt_einsum,
    plan_upmem_greedy,
)
from quantum_bench.tn.upmem_path_cost_v2 import (
    DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2,
    DEFAULT_UPMEM_PATH_COST_POLICY_V2,
    UPMEM_PATH_OBJECTIVE_V2,
)
from quantum_bench.tn.upmem_planner import UPMEM_PATH_SELECTION_SCOPE_V2


def _chain_network() -> TensorNetworkSpec:
    return TensorNetworkSpec(
        circuit=builtin_circuit("bell_2q"),
        tensors=(
            TensorSpec("left", (0, 1), (2, 3), "dense"),
            TensorSpec("middle", (1, 2), (3, 4), "dense"),
            TensorSpec("right", (2, 3), (4, 5), "dense"),
        ),
        output_labels=(0, 3),
        einsum_expression="ab,bc,cd->ad",
    )


def _assert_complete_pairwise_path(path: tuple[tuple[int, ...], ...], tensor_count: int) -> None:
    active_count = tensor_count
    for step in path:
        assert len(step) == 2
        assert 0 <= step[0] < active_count
        assert 0 <= step[1] < active_count
        assert step[0] != step[1]
        active_count -= 1
    assert len(path) == tensor_count - 1
    assert active_count == 1


def test_opt_einsum_is_deterministic_from_spec_only() -> None:
    network = _chain_network()
    request = PlannerRequest(engine=PlannerEngine.OPT_EINSUM, optimize="greedy")

    first = plan_opt_einsum(network, request)
    second = plan_opt_einsum(network, request)

    assert first.path == second.path
    assert first.identity.planner_config_hash == second.identity.planner_config_hash
    assert first.metadata["input_source"] == "tensor_network_spec_labels_and_shapes"
    _assert_complete_pairwise_path(first.path, len(network.tensors))


def test_external_planner_identity_excludes_unused_upmem_policy() -> None:
    network = _chain_network()
    opt_default = plan_contractions(
        network, {"engine": "opt_einsum", "optimize": "greedy"}
    )
    opt_changed = plan_contractions(
        network,
        {
            "engine": "opt_einsum",
            "optimize": "greedy",
            "weight_profile": "compute_oriented",
            "normalization": "different_but_unused",
            "execution_policy": "different_but_unused",
            "input_representation": "real_float32",
        },
    )
    cot_default = plan_contractions(
        network, {"engine": "cotengra", "methods": "greedy", "objective": "flops"}
    )
    cot_changed = plan_contractions(
        network,
        {
            "engine": "cotengra",
            "methods": "greedy",
            "objective": "flops",
            "weight_profile": "compute_oriented",
            "normalization": "different_but_unused",
            "execution_policy": "different_but_unused",
            "input_representation": "real_float32",
        },
    )

    assert opt_default.identity.planner_config == {
        "engine": "opt_einsum",
        "optimize": "greedy",
        "opt_einsum_version": opt_default.identity.planner_config["opt_einsum_version"],
    }
    assert opt_default.identity.planner_config_hash == opt_changed.identity.planner_config_hash
    assert cot_default.identity.planner_config_hash == cot_changed.identity.planner_config_hash
    assert set(cot_default.identity.planner_config) == {
        "engine",
        "methods",
        "objective",
        "max_repeats",
        "seed",
        "cotengra_version",
        "optlib",
        "parallel",
        "progbar",
        "on_trial_error",
    }


def test_dispatch_selects_each_public_structural_adapter() -> None:
    network = _chain_network()

    opt_result = plan_contractions(network, {"engine": "opt_einsum", "optimize": "greedy"})
    cotengra_result = plan_contractions(
        network,
        {"engine": "cotengra", "objective": "flops", "max_repeats": 1, "seed": 0},
    )

    assert opt_result.identity.planner_engine == PlannerEngine.OPT_EINSUM.value
    assert cotengra_result.identity.planner_engine == PlannerEngine.COTENGRA.value
    _assert_complete_pairwise_path(opt_result.path, len(network.tensors))
    _assert_complete_pairwise_path(cotengra_result.path, len(network.tensors))


def test_invalid_engine_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported planner engine"):
        PlannerRequest(engine="does_not_exist")  # type: ignore[arg-type]


def test_adapters_do_not_require_tensor_arrays() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q")).spec
    result = plan_opt_einsum(network)
    assert result.path


def test_custom_upmem_plans_from_metadata_without_fabricated_arrays() -> None:
    result = plan_upmem_greedy(
        _chain_network(),
        {"engine": "custom_upmem", "input_representation": "real_float32"},
    )

    assert result.path
    assert result.metadata["selection_scope"] == UPMEM_PATH_SELECTION_SCOPE_V2
    assert result.metadata["numeric_flags"] == {
        "left": False,
        "middle": False,
        "right": False,
    }
    assert all(
        item["numeric_execution"]["representation"] == "real_float32"
        for item in result.metadata["step_trace"]
        if item["selected"]
    )


def test_custom_upmem_representation_assumption_controls_numeric_model() -> None:
    storage_complex_result = plan_upmem_greedy(
        _chain_network(),
        {"engine": "custom_upmem", "input_representation": "real_float32"},
    )
    real_network = TensorNetworkSpec(
        circuit=builtin_circuit("bell_2q"),
        tensors=tuple(
            TensorSpec(tensor.id, tensor.labels, tensor.shape, tensor.structure, dtype="float32")
            for tensor in _chain_network().tensors
        ),
        output_labels=_chain_network().output_labels,
        einsum_expression=_chain_network().einsum_expression,
    )
    real_result = plan_upmem_greedy(
        real_network,
        {"engine": "custom_upmem", "input_representation": "real_float32"},
    )

    assert storage_complex_result.metadata["numeric_flags"] == real_result.metadata["numeric_flags"]
    assert storage_complex_result.metadata["components"] == real_result.metadata["components"]

    split_result = plan_upmem_greedy(
        _chain_network(),
        {"engine": "custom_upmem", "input_representation": "split_real_imag"},
    )
    assert split_result.metadata["numeric_flags"] == {
        "left": True,
        "middle": True,
        "right": True,
    }
    assert all(
        item["numeric_execution"]["representation"] == "split_real_imag"
        for item in split_result.metadata["step_trace"]
        if item["selected"]
    )

    assert real_result.metadata["numeric_flags"] == {
        "left": False,
        "middle": False,
        "right": False,
    }
    assert split_result.metadata["components"]["numeric_component_invocations"] > real_result.metadata[
        "components"
    ]["numeric_component_invocations"]
    assert real_result.metadata["components"]["numeric_representation_penalty"] == 0.0
    assert split_result.metadata["components"]["numeric_representation_penalty"] > 0.0


def test_custom_upmem_defaults_are_explicit_v2_policy_fields() -> None:
    request = PlannerRequest(engine=PlannerEngine.CUSTOM_UPMEM)
    assert request.objective_version == UPMEM_PATH_OBJECTIVE_V2
    assert request.selection_scope == UPMEM_PATH_SELECTION_SCOPE_V2
    assert request.normalization == DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2
    assert request.execution_policy == DEFAULT_UPMEM_PATH_COST_POLICY_V2
    assert request.weight_profile == "balanced_literature_informed"
    assert request.input_representation == "split_real_imag"


def test_custom_upmem_identity_records_all_explicit_policy_fields() -> None:
    network = _chain_network()
    first = plan_upmem_greedy(network, {"engine": "custom_upmem", "algorithm": "greedy"})
    second = plan_upmem_greedy(
        network,
        {
            "engine": "custom_upmem",
            "algorithm": "greedy",
            "weight_profile": "compute_oriented",
        },
    )

    config = first.identity.planner_config
    assert config["objective_version"] == UPMEM_PATH_OBJECTIVE_V2
    assert config["selection_scope"] == UPMEM_PATH_SELECTION_SCOPE_V2
    assert config["weight_profile"] == "balanced_literature_informed"
    assert config["normalization"] == DEFAULT_UPMEM_PATH_COST_NORMALIZATION_V2
    assert config["execution_policy"] == DEFAULT_UPMEM_PATH_COST_POLICY_V2
    assert config["input_representation"] == "split_real_imag"
    real = plan_upmem_greedy(
        network,
        {
            "engine": "custom_upmem",
            "algorithm": "greedy",
            "input_representation": "real_float32",
        },
    )
    assert real.identity.planner_config["input_representation"] == "real_float32"
    assert first.identity.planner_config_hash != second.identity.planner_config_hash
    assert first.identity.planner_config_hash != real.identity.planner_config_hash


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("objective_version", "upmem_path_cost_v1", "objective version"),
        ("selection_scope", "local_step", "selection scope"),
        ("normalization", "unknown", "normalization"),
        ("execution_policy", "unknown", "policy"),
        ("weight_profile", "unknown", "profile"),
        ("input_representation", "complex128", "input representation"),
    ],
)
def test_custom_upmem_rejects_invalid_v2_policy_fields(
    field: str, value: str, message: str
) -> None:
    config = {"engine": "custom_upmem", field: value}
    with pytest.raises(ValueError, match=message):
        plan_upmem_greedy(_chain_network(), config)


def test_planner_request_rejects_mutable_and_unused_options() -> None:
    with pytest.raises(ValueError, match="JSON scalars"):
        PlannerRequest(options=(("nested", {"value": 1}),))

    with pytest.raises(ValueError, match="Unsupported opt_einsum"):
        plan_contractions(
            _chain_network(),
            PlannerRequest(options=(("unused", 1),)),
        )
