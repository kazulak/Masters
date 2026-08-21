from __future__ import annotations

import pytest

from quantum_bench.bench.route_specs import (
    ComparisonSpec,
    ModuleSpec,
    PipelineParameters,
    PipelineRoute,
)


def _route(
    *,
    route_id: str,
    label: str,
    numeric: str = "float32_real",
    optional_modules: tuple[ModuleSpec, ...] = (),
) -> PipelineRoute:
    return PipelineRoute(
        route_id=route_id,
        label=label,
        modules=(
            ModuleSpec("tensor_network", "quantum_gate_tn_v1", PipelineParameters({})),
            ModuleSpec(
                "planner",
                "opt_einsum",
                PipelineParameters({"engine": "opt_einsum", "optimize": "greedy"}),
            ),
            ModuleSpec("numeric", numeric, PipelineParameters({})),
            ModuleSpec("executor", "numpy_cpu", PipelineParameters({})),
            ModuleSpec("topology", "cpu", PipelineParameters({})),
            *optional_modules,
        ),
    )


def test_pipeline_parameters_are_canonical_json_safe_and_return_fresh_data() -> None:
    first = PipelineParameters({"b": 1, "a": {"nested": [3, 2, 1], "value": 1.0}})
    second = PipelineParameters({"a": {"value": 1.0, "nested": [3, 2, 1]}, "b": 1})

    assert first.canonical_json == second.canonical_json
    assert first.hash == second.hash
    copied = first.to_dict()
    copied["a"]["nested"].append(0)
    assert first.to_dict()["a"]["nested"] == [3, 2, 1]

    with pytest.raises(ValueError, match="top-level mapping"):
        PipelineParameters([("not", "a mapping")])
    with pytest.raises(ValueError, match="Non-finite"):
        PipelineParameters({"bad": float("nan")})
    with pytest.raises(ValueError, match="Unsupported JSON type"):
        PipelineParameters({"bad": object()})


def test_module_and_route_hashes_are_deterministic_and_role_checked() -> None:
    route_a = _route(route_id="a", label="first")
    route_b = _route(route_id="b", label="second")

    assert route_a.route_config_hash == route_b.route_config_hash
    assert route_a.module("planner").implementation == "opt_einsum"

    with pytest.raises(KeyError, match="missing role"):
        route_a.module("kernel")
    with pytest.raises(ValueError, match="required roles"):
        PipelineRoute(
            route_id="missing",
            label="bad",
            modules=route_a.modules[:-1],
        )
    with pytest.raises(ValueError, match="unique roles"):
        PipelineRoute(
            route_id="duplicate",
            label="bad",
            modules=route_a.modules + (route_a.module("planner"),),
        )
    with pytest.raises(ValueError, match="extra"):
        PipelineRoute(
            route_id="unknown",
            label="bad",
            modules=route_a.modules
            + (ModuleSpec("unknown", "bad", PipelineParameters({})),),
        )


def test_comparison_requires_exact_changed_roles_including_profile_declarations() -> (
    None
):
    baseline = _route(route_id="base", label="baseline")
    candidate = _route(
        route_id="kernel",
        label="kernel profile",
        optional_modules=(
            ModuleSpec("kernel", "reference_kernel", PipelineParameters({})),
        ),
    )

    comparison = ComparisonSpec(
        baseline_route=baseline,
        candidate_route=candidate,
        changed_roles=("kernel",),
        label="kernel declaration",
    )

    assert comparison.to_dict()["candidate_route_id"] == "kernel"
    assert comparison.to_dict()["changed_roles"] == ("kernel",)

    with pytest.raises(ValueError, match="changed_roles mismatch"):
        ComparisonSpec(
            baseline_route=baseline,
            candidate_route=candidate,
            changed_roles=(),
            label="wrong",
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        ComparisonSpec(
            baseline_route=baseline,
            candidate_route=candidate,
            changed_roles=("kernel", "kernel"),
            label="duplicate",
        )
