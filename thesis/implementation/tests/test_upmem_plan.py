from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quantum_bench.execution.contracts import UpmemPlan as LegacyUpmemPlan
from quantum_bench.lowering import contraction_dag_hash
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorSpec, TensorView
from quantum_bench.results import UnsupportedExecution
from quantum_bench.upmem.plan import (
    PLAN_SCHEMA_VERSION,
    UpmemPlan,
    UpmemResources,
    UpmemStage,
    UpmemTopology,
    UpmemWorkUnit,
    _INT8_PRODUCT,
    _INT64_MAX,
    _validate_final_int8_bounds,
    physical_plan_id,
    plan_upmem,
    validate_upmem_plan,
)
from quantum_bench.upmem import plan as upmem_plan_module
from quantum_bench.upmem.tiling import TileLoweringError
from quantum_bench.upmem.protocol import INT32_MAX, MAX_CONTRACTED


def _view(tensor_id: str, labels: tuple[int, ...], shape: tuple[int, ...]) -> TensorView:
    return TensorView(tensor_id=tensor_id, labels=labels, shape=shape)


def _contract(
    node_id: str,
    left_id: str,
    right_id: str,
    output_id: str,
    left_labels: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_labels: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
    dependencies: tuple[str, ...] = (),
) -> ContractNode:
    contracted = tuple(
        label
        for label in left_labels
        if label in right_labels and label not in output_labels
    )
    return ContractNode(
        node_id=node_id,
        left=_view(left_id, left_labels, left_shape),
        right=_view(right_id, right_labels, right_shape),
        output=TensorSpec(output_id, output_labels, output_shape, "dense"),
        contracted_labels=contracted,
        output_labels=output_labels,
        dependencies=dependencies,
    )


def _dag() -> ContractionDAG:
    node = _contract(
        "contract_0",
        "a",
        "b",
        "out",
        (0, 1),
        (2, 3),
        (1, 2),
        (3, 4),
        (0, 2),
        (2, 4),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("a", (0, 1), (2, 3), "dense"),
            TensorSpec("b", (1, 2), (3, 4), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (2, 4)),
    )


def _diamond_with_unrelated_node() -> ContractionDAG:
    first = _contract(
        "a_node", "a0", "a1", "p", (0, 1), (2, 3), (1, 4), (3, 2), (0, 4), (2, 2)
    )
    second = _contract(
        "b_node", "b0", "b1", "q", (0, 3), (2, 3), (3, 4), (3, 2), (0, 4), (2, 2)
    )
    middle = _contract(
        "middle", "u0", "u1", "u", (4, 5), (2, 3), (5, 3), (3, 2), (4, 3), (2, 2)
    )
    reduced = ReduceNode(
        node_id="reduce_node",
        inputs=(_view("p", (0, 4), (2, 2)), _view("q", (0, 4), (2, 2))),
        output=TensorSpec("result", (0, 4), (2, 2), "dense"),
        dependencies=("a_node", "b_node"),
    )
    final = _contract(
        "final_node",
        "result",
        "u",
        "final_result",
        (0, 4),
        (2, 2),
        (4, 3),
        (2, 2),
        (0, 3),
        (2, 2),
        dependencies=("reduce_node", "middle"),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("a0", (0, 1), (2, 3), "dense"),
            TensorSpec("a1", (1, 4), (3, 2), "dense"),
            TensorSpec("b0", (0, 3), (2, 3), "dense"),
            TensorSpec("b1", (3, 4), (3, 2), "dense"),
            TensorSpec("u0", (4, 5), (2, 3), "dense"),
            TensorSpec("u1", (5, 3), (3, 2), "dense"),
        ),
        nodes=(first, second, middle, reduced, final),
        output=_view("final_result", (0, 3), (2, 2)),
    )


def _topology(**kwargs: int) -> UpmemTopology:
    return UpmemTopology(
        dpu_count=kwargs.get("dpu_count", 2),
        tasklets_per_dpu=kwargs.get("tasklets_per_dpu", 1),
        rank_count=kwargs.get("rank_count", 1),
    )


def test_mapping_is_deterministic_and_uses_dag_hash() -> None:
    dag = _dag()
    first = plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology())
    second = plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology())

    assert PLAN_SCHEMA_VERSION == 1
    assert first == second
    assert first.logical_plan_id == contraction_dag_hash(dag)
    assert physical_plan_id(first) == physical_plan_id(second)


def test_real_tile_byte_and_mac_semantics_are_explicit() -> None:
    dag = _dag()
    float_plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology())
    int_plan = plan_upmem(dag, numeric_policy="split_complex_int8_shared_scale_v1", topology=_topology())

    float_unit = float_plan.stages[0].work_units[0]
    int_unit = int_plan.stages[0].work_units[0]
    assert (float_unit.estimated_input_bytes, float_unit.estimated_output_bytes) == (6 * 4 + 12 * 4, 8 * 4)
    assert (int_unit.estimated_input_bytes, int_unit.estimated_output_bytes) == (6 + 12, 8 * 4)
    assert float_unit.aligned_mram_bytes == 6 * 4 + 12 * 4 + 8 * 4
    assert int_unit.aligned_mram_bytes == 8 + 16 + 32
    assert float_unit.estimated_arithmetic_work == 2 * 4 * 3
    assert int_unit.estimated_arithmetic_work == 2 * 4 * 3


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "split_complex_float32_v1",
            (64, 144, 88, 296),
        ),
        (
            "split_complex_int8_shared_scale_v1",
            (16, 40, 88, 144),
        ),
    ],
)
def test_remainder_tile_alignment_and_byte_sum(
    policy: str, expected: tuple[int, int, int, int]
) -> None:
    unit = plan_upmem(
        _remainder_dag(), numeric_policy=policy, topology=_topology()
    ).stages[0].work_units[0]
    left, right, output, total = expected
    assert unit.aligned_mram_bytes == total
    assert unit.aligned_mram_bytes == left + right + output
    assert (unit.estimated_input_bytes, unit.estimated_output_bytes) == (
        (15 + 35) * (1 if policy.endswith("int8_shared_scale_v1") else 4),
        21 * 4,
    )


def test_reduce_stage_allows_unrelated_ready_node_between_producers() -> None:
    plan = plan_upmem(
        _diamond_with_unrelated_node(),
        numeric_policy="split_complex_float32_v1",
        topology=_topology(),
    )
    assert [stage.stage_id for stage in plan.stages] == [
        "contract_batch:a_node",
        "contract_batch:b_node",
        "contract_batch:middle",
        "host_reduce:reduce_node",
        "contract_batch:final_node",
    ]
    assert plan.stages[-2].work_units == ()


def test_work_units_are_sorted_by_required_final_key() -> None:
    dag = _contract_dag_with_large_tiles()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=2, rank_count=2),
    )
    units = plan.stages[0].work_units
    keys = [
        (u.logical_rank, u.logical_dpu, u.wave, u.batch_start, u.m_start, u.n_start, u.k_start, u.stable_tile_id)
        for u in units
    ]
    assert keys == sorted(keys)
    assert [
        (unit.wave, unit.logical_rank, unit.logical_dpu, unit.stable_tile_id)
        for unit in units
    ] == [
        (0, 0, 0, "b_0:out_0_0:k_0"),
        (1, 0, 0, "b_0:out_0_0:k_1"),
        (2, 0, 0, "b_0:out_0_0:k_2"),
        (0, 0, 1, "b_0:out_0_1:k_0"),
        (1, 0, 1, "b_0:out_0_1:k_1"),
        (2, 0, 1, "b_0:out_0_1:k_2"),
        (0, 1, 0, "b_0:out_1_0:k_0"),
        (1, 1, 0, "b_0:out_1_0:k_1"),
        (2, 1, 0, "b_0:out_1_0:k_2"),
        (0, 1, 1, "b_0:out_1_1:k_0"),
        (1, 1, 1, "b_0:out_1_1:k_1"),
        (2, 1, 1, "b_0:out_1_1:k_2"),
    ]
    geometry = [
        (u.batch_start, u.m_start, u.m_size, u.n_start, u.n_size, u.k_start, u.k_size)
        for u in units
    ]
    assert len(geometry) == len(set(geometry)) == 12


def _contract_dag_with_large_tiles() -> ContractionDAG:
    node = _contract(
        "large",
        "left",
        "right",
        "large_out",
        (0, 1),
        (300, 300),
        (1, 2),
        (300, 300),
        (0, 2),
        (300, 300),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (300, 300), "dense"),
            TensorSpec("right", (1, 2), (300, 300), "dense"),
        ),
        nodes=(node,),
        output=_view("large_out", (0, 2), (300, 300)),
    )


def _remainder_dag() -> ContractionDAG:
    node = _contract(
        "remainder",
        "left",
        "right",
        "out",
        (0, 1),
        (3, 5),
        (1, 2),
        (5, 7),
        (0, 2),
        (3, 7),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (3, 5), "dense"),
            TensorSpec("right", (1, 2), (5, 7), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (3, 7)),
    )


def _oversized_k_dag() -> ContractionDAG:
    node = _contract(
        "oversized_k",
        "left",
        "right",
        "out",
        (0, 1),
        (1, MAX_CONTRACTED + 1),
        (1, 2),
        (MAX_CONTRACTED + 1, 1),
        (0, 2),
        (1, 1),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (1, MAX_CONTRACTED + 1), "dense"),
            TensorSpec("right", (1, 2), (MAX_CONTRACTED + 1, 1), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (1, 1)),
    )


def test_identity_changes_for_policy_topology_stage_and_work_order() -> None:
    dag = _dag()
    base = plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology())
    assert physical_plan_id(base) != physical_plan_id(
        plan_upmem(dag, numeric_policy="split_complex_int8_shared_scale_v1", topology=_topology())
    )
    assert physical_plan_id(base) != physical_plan_id(
        plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology(dpu_count=4))
    )
    changed_stage_order = replace(base.stages[0], stage_id="contract_batch:changed")
    changed_stage_plan = replace(base, stages=(changed_stage_order,))
    assert physical_plan_id(base) != physical_plan_id(changed_stage_plan)
    large = plan_upmem(
        _contract_dag_with_large_tiles(),
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=4, rank_count=2),
    )
    reversed_units = replace(
        large.stages[0], work_units=tuple(reversed(large.stages[0].work_units))
    )
    reordered_plan = replace(large, stages=(reversed_units,))
    assert physical_plan_id(large) != physical_plan_id(reordered_plan)
    unit = base.stages[0].work_units[0]
    changed_unit = replace(unit, stable_tile_id=unit.stable_tile_id + ":changed")
    changed_stage = replace(base.stages[0], work_units=(changed_unit,))
    changed_plan = replace(base, stages=(changed_stage,))
    assert physical_plan_id(base) != physical_plan_id(changed_plan)


def test_validator_rejects_tampering() -> None:
    dag = _dag()
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1", topology=_topology())
    validate_upmem_plan(dag, plan)
    tampered = replace(plan, kernel_policy="different")
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        validate_upmem_plan(dag, tampered)


@pytest.mark.parametrize("tamper", ["stage_order", "node_id", "tile", "bytes", "topology"])
def test_validator_rejects_structurally_valid_tampering(tamper: str) -> None:
    dag = _contract_dag_with_large_tiles()
    if tamper == "stage_order":
        dag = _diamond_with_unrelated_node()
    plan = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    if tamper == "stage_order":
        tampered = replace(plan, stages=tuple(reversed(plan.stages)))
    elif tamper == "node_id":
        stage = plan.stages[0]
        units = tuple(replace(unit, node_id="wrong_node") for unit in stage.work_units)
        changed = replace(stage, node_ids=("wrong_node",), work_units=units)
        tampered = replace(plan, stages=(changed, *plan.stages[1:]))
    elif tamper == "tile":
        stage = plan.stages[0]
        changed = replace(stage.work_units[0], m_start=stage.work_units[0].m_start + 1)
        tampered = replace(
            plan, stages=(replace(stage, work_units=(changed, *stage.work_units[1:])), *plan.stages[1:])
        )
    elif tamper == "bytes":
        stage = plan.stages[0]
        changed = replace(
            stage.work_units[0],
            estimated_input_bytes=stage.work_units[0].estimated_input_bytes + 1,
        )
        tampered = replace(
            plan, stages=(replace(stage, work_units=(changed, *stage.work_units[1:])), *plan.stages[1:])
        )
    else:
        tampered = replace(
            plan,
            topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=1, rank_count=2),
        )
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        validate_upmem_plan(dag, tampered)


def test_structurally_invalid_stage_records_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="host_reduce"):
        UpmemStage(stage_id="reduce", kind="host_reduce", node_ids=("a", "b"), work_units=())
    with pytest.raises(ValueError, match="host_reduce"):
        UpmemStage(
            stage_id="reduce",
            kind="host_reduce",
            node_ids=("a",),
            work_units=(
                UpmemWorkUnit(
                    node_id="a",
                    stable_tile_id="tile",
                    wave=0,
                    logical_rank=0,
                    logical_dpu=0,
                    batch_start=0,
                    batch_size=1,
                    m_start=0,
                    m_size=1,
                    n_start=0,
                    n_size=1,
                    k_start=0,
                    k_size=1,
                    estimated_input_bytes=0,
                    estimated_output_bytes=0,
                    aligned_mram_bytes=0,
                    estimated_arithmetic_work=1,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("wave", True), ("m_size", 0), ("node_id", ""), ("k_start", -1)],
)
def test_work_unit_intrinsic_validation(field: str, value: object) -> None:
    kwargs = dict(
        node_id="node",
        stable_tile_id="tile",
        wave=0,
        logical_rank=0,
        logical_dpu=0,
        batch_start=0,
        batch_size=1,
        m_start=0,
        m_size=1,
        n_start=0,
        n_size=1,
        k_start=0,
        k_size=1,
        estimated_input_bytes=0,
        estimated_output_bytes=0,
        aligned_mram_bytes=0,
        estimated_arithmetic_work=1,
    )
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        UpmemWorkUnit(**kwargs)


@pytest.mark.parametrize(
    ("dpu_count", "rank_count", "tasklets", "message"),
    [
        (0, 1, 1, "positive"),
        (1, 0, 1, "positive"),
        (3, 2, 1, "divisible"),
        (65, 1, 1, "at most 64"),
        (1, 1, 0, r"\[1, 24\]"),
        (1, 1, 25, r"\[1, 24\]"),
    ],
)
def test_mapping_rejects_topology_boundaries(
    dpu_count: int, rank_count: int, tasklets: int, message: str
) -> None:
    with pytest.raises(UnsupportedExecution, match=message):
        plan_upmem(
            _dag(),
            numeric_policy="split_complex_float32_v1",
            topology=UpmemTopology(
                dpu_count=dpu_count,
                tasklets_per_dpu=tasklets,
                rank_count=rank_count,
            ),
        )


def test_mapping_rejects_abi_max_contracted_shape_only() -> None:
    with pytest.raises(UnsupportedExecution, match="max_contracted"):
        plan_upmem(
            _oversized_k_dag(),
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )


@pytest.mark.parametrize(
    ("contracted_size", "tile_k", "should_fail"),
    [
        (INT32_MAX // _INT8_PRODUCT, INT32_MAX // _INT8_PRODUCT, False),
        (INT32_MAX // _INT8_PRODUCT, INT32_MAX // _INT8_PRODUCT + 1, True),
    ],
)
def test_final_int8_int32_boundary(
    contracted_size: int, tile_k: int, should_fail: bool
) -> None:
    tile = SimpleNamespace(id="tile", k_size=tile_k)
    if should_fail:
        with pytest.raises(UnsupportedExecution, match="int32"):
            _validate_final_int8_bounds("node", contracted_size, (tile,))
    else:
        _validate_final_int8_bounds("node", contracted_size, (tile,))


def test_final_int8_int64_boundary() -> None:
    boundary = _INT64_MAX // (2 * _INT8_PRODUCT)
    _validate_final_int8_bounds("node", boundary, ())
    with pytest.raises(UnsupportedExecution, match="int64"):
        _validate_final_int8_bounds("node", boundary + 1, ())


def test_mapping_rejects_unknown_numeric_policy() -> None:
    with pytest.raises(UnsupportedExecution, match="numeric policy"):
        plan_upmem(_dag(), numeric_policy="unknown", topology=_topology())  # type: ignore[arg-type]


def test_mapping_rejects_valid_dag_without_contract_work() -> None:
    input_view = _view("input", (0,), (2,))
    dag = ContractionDAG(
        tensors=(TensorSpec("input", (0,), (2,), "dense"),),
        nodes=(),
        output=input_view,
    )

    with pytest.raises(UnsupportedExecution) as exc_info:
        plan_upmem(
            dag,
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )

    assert exc_info.value.stage == "mapping"
    assert exc_info.value.capability == "upmem_no_contract_work"


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (ValueError("injected mapper defect"), "injected mapper defect"),
        (TileLoweringError("injected unsupported tile"), "not representable"),
    ],
)
def test_final_mapper_preserves_expected_exception_boundary(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_message: str,
) -> None:
    def raise_error(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(upmem_plan_module, "plan_tile_shapes", raise_error)
    expected_exception = (
        ValueError if isinstance(error, ValueError) and not isinstance(error, TileLoweringError)
        else UnsupportedExecution
    )

    with pytest.raises(expected_exception, match=expected_message) as exc_info:
        plan_upmem(
            _dag(),
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )

    if isinstance(error, TileLoweringError):
        assert exc_info.value.stage == "mapping"
        assert exc_info.value.capability == "upmem_v4_geometry"


def test_resources_are_immutable_and_callback_is_not_identity() -> None:
    def callback() -> None:
        return None
    first = UpmemResources(
        session_root="session",
        host_binary="host",
        dpu_binary="dpu",
        initialization_binary="init",
        rank_paths=("rank0",),
        session_opener=callback,
    )
    second = replace(first, session_opener=lambda: None)
    assert first == second
    assert "session_opener" not in repr(first)
    with pytest.raises((AttributeError, TypeError)):
        first.rank_paths = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="tuple"):
        UpmemResources(
            session_root="session",
            host_binary="host",
            dpu_binary="dpu",
            initialization_binary="init",
            rank_paths=["rank0"],  # type: ignore[arg-type]
        )


def test_final_records_are_distinct_from_legacy_records() -> None:
    assert UpmemPlan is not LegacyUpmemPlan
