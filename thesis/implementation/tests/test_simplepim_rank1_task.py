from dataclasses import replace

import numpy as np
import pytest

from quantum_bench.targets.upmem.simplepim_rank1_task import (
    RANK1_DOT_LENGTH,
    RANK1_OUTPUT_TENSOR_ID,
    build_rank1_taskgraph_workload,
    validate_rank1_task,
)


def test_rank1_fixture_has_stable_execution_identity() -> None:
    first = build_rank1_taskgraph_workload()
    second = build_rank1_taskgraph_workload()

    assert first.graph.circuit_semantics_hash == second.graph.circuit_semantics_hash
    assert first.graph.tensor_network_hash == second.graph.tensor_network_hash
    assert first.graph.contraction_plan_hash == second.graph.contraction_plan_hash
    assert len(first.graph.contraction_plan_hash) == 64
    assert first.reference_int64 == second.reference_int64
    validate_rank1_task(first)


def test_rank1_fixture_is_a_valid_single_task_graph() -> None:
    workload = build_rank1_taskgraph_workload()

    validate_rank1_task(workload)
    task = workload.graph.tasks[0]
    assert len(workload.graph.tasks) == 1
    assert task.dependencies == ()
    assert task.input_shapes == ((RANK1_DOT_LENGTH,), (RANK1_DOT_LENGTH,))
    assert task.output_shape == ()
    assert (task.gemm_m, task.gemm_k, task.gemm_n) == (1, RANK1_DOT_LENGTH, 1)
    assert task.output_tensor_id == RANK1_OUTPUT_TENSOR_ID
    assert workload.left.dtype == np.int8
    assert workload.right.dtype == np.int8


@pytest.mark.parametrize(
    "mutate",
    [
        lambda workload: replace(workload, graph=replace(workload.graph, tasks=())),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], dependencies=("other",)),)),
        ),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], input_shapes=((255,), (256,))),)),
        ),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], contracted_labels=(1,)),)),
        ),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], gemm_k=255),)),
        ),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], structure="generic"),)),
        ),
        lambda workload: replace(
            workload,
            graph=replace(
                workload.graph,
                tasks=(replace(workload.graph.tasks[0], input_tensor_ids=("wrong", "rank1_right")),),
            ),
        ),
        lambda workload: replace(
            workload,
            graph=replace(workload.graph, tasks=(replace(workload.graph.tasks[0], output_tensor_id="rank1_right"),)),
        ),
        lambda workload: replace(workload, left=workload.left.astype(np.float32)),
    ],
)
def test_rank1_fixture_rejects_unsupported_variants(
    mutate: object,
) -> None:
    workload = mutate(build_rank1_taskgraph_workload())  # type: ignore[operator]

    with pytest.raises(ValueError):
        validate_rank1_task(workload)


def test_rank1_fixture_rejects_complex_or_non_int8_network_tensor() -> None:
    workload = build_rank1_taskgraph_workload()
    spec = replace(workload.network.tensors[0].spec, dtype="complex64")
    network = replace(
        workload.network,
        tensors=[replace(workload.network.tensors[0], spec=spec, array=workload.left.astype(np.complex64)), workload.network.tensors[1]],
    )

    with pytest.raises(ValueError, match="dtype|array"):
        validate_rank1_task(replace(workload, network=network))
