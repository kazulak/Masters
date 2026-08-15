from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from quantum_bench.core.jsonio import write_jsonl
from quantum_bench.core.target_estimates import TargetEstimateSet
from quantum_bench.bench import runner
from quantum_bench.targets.upmem.schedule import (
    DENSE_INT8_FORMAT,
    UPMEM_PROFILE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UpmemDataFormat,
    annotate_task_graph_with_upmem_estimates,
    estimate_dense_task_graph_sidecar,
    upmem_task_estimate_rows,
    upmem_target_path_summary,
)
from quantum_bench.tn.execution_bundle import with_execution_identity


def test_sidecar_is_complete_provenanced_and_does_not_mutate_graph(
    minimal_graph, tmp_path
) -> None:
    graph = minimal_graph.graph
    before_hash = graph.contraction_plan_hash
    assert all(task.target_estimates == {} for task in graph.tasks)
    assert graph.path_summary.missing_target_estimate_count == 0
    assert graph.path_summary.total_host_to_dpu_bytes == 0

    sidecar, schedule = estimate_dense_task_graph_sidecar(graph)

    assert sidecar.scientific_plan_hash == before_hash
    assert len(sidecar.rows) == len(graph.tasks)
    assert all(row.values for row in sidecar.rows)
    assert {spec.origin for spec in sidecar.metric_specs} == {
        "analytic_model",
        "scientific_plan",
    }
    assert all(spec.unit and spec.scope for spec in sidecar.metric_specs)
    assert all(task.target_estimates == {} for task in graph.tasks)
    assert graph.contraction_plan_hash == before_hash
    assert upmem_task_estimate_rows(graph, sidecar) == sidecar.jsonl_rows()

    summary = upmem_target_path_summary(sidecar, schedule)
    assert summary["scientific_plan_hash"] == before_hash
    assert summary["metric_provenance"]

    path = tmp_path / "estimates.jsonl"
    write_jsonl(path, upmem_task_estimate_rows(graph, sidecar))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [json.dumps(row, sort_keys=True) for row in sidecar.jsonl_rows()]


def test_legacy_adapter_mirrors_inline_estimates_without_changing_plan(
    minimal_graph,
) -> None:
    graph = minimal_graph.graph
    annotated, schedule = annotate_task_graph_with_upmem_estimates(graph)

    assert annotated.contraction_plan_hash == graph.contraction_plan_hash
    assert all(
        UPMEM_DENSE_ESTIMATE_KEY in task.target_estimates for task in annotated.tasks
    )
    assert all(task.target_estimates == {} for task in graph.tasks)
    assert schedule.tasks


def test_sidecar_does_not_change_execution_identity(minimal_graph) -> None:
    graph = minimal_graph.graph
    sidecar, _ = estimate_dense_task_graph_sidecar(graph)
    assert (
        with_execution_identity(graph).contraction_plan_hash
        == sidecar.scientific_plan_hash
    )


def test_runner_writes_sidecar_artifacts_without_persisting_target_fields(
    tmp_path: Path,
) -> None:
    case = {
        "case_id": "sidecar_runner",
        "circuit": {"kind": "builtin", "name": "bell_2q"},
    }
    suite = {
        "suite_id": "sidecar_runner_suite",
        "planner": {"engine": "opt_einsum", "optimize": "greedy"},
        "route_policy": {"routes": []},
    }
    case_dir = tmp_path / "run" / "cases" / "sidecar_runner"

    generated = runner._generate_case(case, suite, tmp_path, case_dir)

    assert all(task.target_estimates == {} for task in generated["graph"].tasks)
    estimate_path = (
        case_dir.parents[1] / generated["target_estimate_artifacts"]["upmem_dense_int8"]
    )
    rows = [
        json.loads(line)
        for line in estimate_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(generated["graph"].tasks)
    assert all(
        row["scientific_plan_hash"] == generated["graph"].contraction_plan_hash
        for row in rows
    )
    assert all(row["metric_provenance"] for row in rows)

    recreated = TargetEstimateSet.from_jsonl_rows(
        list(reversed(rows)),
        expected_target_id=generated["target_estimates"].target_id,
        expected_model_id=generated["target_estimates"].model_id,
    )
    assert recreated.to_json_dict() == generated["target_estimates"].to_json_dict()
    assert json.dumps(recreated.to_json_dict(), sort_keys=True) == json.dumps(
        generated["target_estimates"].to_json_dict(), sort_keys=True
    )
    assert (case_dir / "path_summary.json").exists()
    assert (case_dir / "target_estimates" / "upmem_path_summary.json").exists()


def test_upmem_adapter_rejects_wrong_sidecar_identity_and_descriptors(
    minimal_graph,
) -> None:
    graph = minimal_graph.graph
    sidecar, schedule = estimate_dense_task_graph_sidecar(graph)

    with pytest.raises(ValueError, match="target ID"):
        annotate_task_graph_with_upmem_estimates(
            graph,
            estimates=replace(sidecar, target_id="other_target"),
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="model ID"):
        annotate_task_graph_with_upmem_estimates(
            graph,
            estimates=replace(sidecar, model_id="other_model"),
            schedule=schedule,
        )

    with pytest.raises(ValueError, match="scientific plan hash"):
        annotate_task_graph_with_upmem_estimates(
            graph,
            estimates=replace(sidecar, scientific_plan_hash="0" * 64),
            schedule=schedule,
        )

    bad_row = replace(sidecar.rows[0], input_tensor_ids=("wrong", "tensor"))
    with pytest.raises(ValueError, match="input tensors"):
        annotate_task_graph_with_upmem_estimates(
            graph,
            estimates=replace(sidecar, rows=(bad_row, *sidecar.rows[1:])),
            schedule=schedule,
        )

    bad_output = replace(sidecar.rows[0], output_tensor_id="wrong_output")
    with pytest.raises(ValueError, match="output tensor"):
        annotate_task_graph_with_upmem_estimates(
            graph,
            estimates=replace(sidecar, rows=(bad_output, *sidecar.rows[1:])),
            schedule=schedule,
        )


def test_precomputed_sidecar_requires_matching_schedule_profile(minimal_graph) -> None:
    graph = minimal_graph.graph
    sidecar, schedule = estimate_dense_task_graph_sidecar(graph)

    with pytest.raises(ValueError, match="requires its matching schedule"):
        annotate_task_graph_with_upmem_estimates(graph, estimates=sidecar)
    with pytest.raises(ValueError, match="requires.*sidecar"):
        annotate_task_graph_with_upmem_estimates(graph, schedule=schedule)

    changed_profile = replace(
        schedule,
        hardware=UPMEM_PROFILE.__class__(
            name="other-profile",
            wram_bytes=UPMEM_PROFILE.wram_bytes,
            dpu_count=UPMEM_PROFILE.dpu_count,
        ),
    )
    with pytest.raises(ValueError, match="profile/task metadata"):
        annotate_task_graph_with_upmem_estimates(
            graph, estimates=sidecar, schedule=changed_profile
        )

    changed_format = replace(
        schedule,
        data_format=UpmemDataFormat(
            "other-format",
            DENSE_INT8_FORMAT.input_element_bytes,
            DENSE_INT8_FORMAT.output_element_bytes,
            DENSE_INT8_FORMAT.accumulator_element_bytes,
        ),
    )
    with pytest.raises(ValueError, match="profile/task metadata"):
        annotate_task_graph_with_upmem_estimates(
            graph, estimates=sidecar, schedule=changed_format
        )


def test_sidecar_rejects_duplicate_and_incomplete_metric_specs(minimal_graph) -> None:
    graph = minimal_graph.graph
    sidecar, _ = estimate_dense_task_graph_sidecar(graph)
    rows = [row.to_json_dict() for row in sidecar.rows]

    with pytest.raises(ValueError, match="duplicate metric specs"):
        TargetEstimateSet.from_rows(
            graph.contraction_plan_hash,
            sidecar.target_id,
            sidecar.model_id,
            rows,
            [*sidecar.metric_specs, sidecar.metric_specs[0]],
        )
    with pytest.raises(ValueError, match="do not cover every row"):
        TargetEstimateSet.from_rows(
            graph.contraction_plan_hash,
            sidecar.target_id,
            sidecar.model_id,
            rows,
            list(sidecar.metric_specs[1:]),
        )

    incomplete_rows = [rows[0], {**rows[1], sidecar.metric_specs[0].name: None}]
    incomplete_rows[1].pop(sidecar.metric_specs[1].name)
    with pytest.raises(ValueError, match="do not cover every row"):
        TargetEstimateSet.from_rows(
            graph.contraction_plan_hash,
            sidecar.target_id,
            sidecar.model_id,
            incomplete_rows,
            list(sidecar.metric_specs),
        )
