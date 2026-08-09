import numpy as np
import pytest

from quantum_bench.targets.upmem.simplepim_chain_task import (
    CHAIN_INTERMEDIATE_ID,
    build_simplepim_chain_workload,
    validate_simplepim_chain_workload,
)


def test_chain_fixture_is_deterministic_and_exact() -> None:
    first = build_simplepim_chain_workload()
    second = build_simplepim_chain_workload()
    assert first.reference_int64 == -11654
    assert first.operand_sha256 == second.operand_sha256
    assert all(np.array_equal(left, right) for left, right in zip(first.operands, second.operands, strict=True))


def test_chain_has_two_tasks_and_resident_intermediate_dependency() -> None:
    workload = build_simplepim_chain_workload()
    first, second = workload.graph.tasks
    assert first.output_tensor_id == CHAIN_INTERMEDIATE_ID
    assert second.dependencies == (first.id,)
    assert CHAIN_INTERMEDIATE_ID in second.input_tensor_ids


def test_chain_validates_and_has_four_tiles() -> None:
    workload = build_simplepim_chain_workload()
    validate_simplepim_chain_workload(workload)
    assert [(tile.start, tile.stop) for tile in workload.tiles] == [(0, 64), (64, 128), (128, 192), (192, 256)]


def test_chain_rejects_wrong_dependency() -> None:
    workload = build_simplepim_chain_workload()
    tasks = list(workload.graph.tasks)
    tasks[1] = tasks[1].__class__(**{**tasks[1].__dict__, "dependencies": ()})
    broken = workload.__class__(**{**workload.__dict__, "graph": workload.graph.__class__(**{**workload.graph.__dict__, "tasks": tuple(tasks)})})
    with pytest.raises(ValueError, match="dependency"):
        validate_simplepim_chain_workload(broken)
