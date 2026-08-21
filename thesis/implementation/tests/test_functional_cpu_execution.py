from __future__ import annotations

import hashlib

import numpy as np
import pytest

from quantum_bench.core.records import ContractionTask, TensorNetworkSpec, TensorSpec
from quantum_bench.execution import (
    CpuCompileRequest,
    ExecutionPlan,
    NumericMode,
    RunContext,
    Target,
    UpmemCompileRequest,
    UpmemPlan,
    UpmemTopology,
    compile_execution,
    compile_cpu,
    execute,
)
from quantum_bench.execution.contracts import ExecutionFailure, UnsupportedExecution
from quantum_bench.tn.graph import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    SliceSpec,
    TensorView,
    build_contraction_dag,
    contraction_dag_hash,
    apply_slicing,
)
from quantum_bench.tn.network import TensorInput, TensorInputs
from quantum_bench.whole_circuit.policies import Float32RealPolicy, HostPackedInt8Policy


def _hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _dag(spec: TensorNetworkSpec, path: tuple[tuple[int, int], ...]) -> ContractionDAG:
    return build_contraction_dag(spec, path)


def _inputs(*arrays: tuple[str, np.ndarray]) -> TensorInputs:
    return TensorInputs(
        values=tuple(TensorInput(tensor_id=tensor_id, array=array) for tensor_id, array in arrays)
    )


def _compile(
    dag: ContractionDAG, mode: NumericMode = NumericMode.FLOAT32_REAL
) -> ExecutionPlan:
    return compile_cpu(
        dag,
        CpuCompileRequest(
            contraction_dag_hash=contraction_dag_hash(dag),
            numeric_mode=mode,
        ),
    )


def _matrix_dag(dtype: str = "float64") -> ContractionDAG:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense", dtype=dtype),
        TensorSpec("b", (1, 2), (3, 2), "dense", dtype=dtype),
    )
    return _dag(
        TensorNetworkSpec(None, tensors, (0, 2), "ab,bc->ac"),  # type: ignore[arg-type]
        ((0, 1),),
    )


def _matrix_task() -> ContractionTask:
    return ContractionTask(
        id="contract",
        input_tensor_ids=("a", "b"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=((2, 3), (3, 2)),
        output_shape=(2, 2),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=3,
        gemm_n=2,
        structure="dense",
        estimated_flops=24,
        estimated_bytes=56,
    )


def _run_matrix(mode: NumericMode, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dag = _matrix_dag(str(np.asarray(left).dtype))
    return execute(
        _compile(dag, mode),
        dag,
        _inputs(("a", left), ("b", right)),
        RunContext(run_id=mode.value, target=Target.CPU),
    ).output


def test_numeric_modes_match_existing_real_policy_implementations() -> None:
    left = np.array([[0.25, -1.5, 2.0], [3.25, 0.5, -0.75]], dtype=np.float64)
    right = np.array([[1.0, -2.0], [0.25, 0.5], [-1.25, 0.75]], dtype=np.float64)
    task = _matrix_task()

    expected_float, _ = Float32RealPolicy().contract(task, left, right)
    expected_int8, _ = HostPackedInt8Policy().contract(task, left, right)

    actual_float = _run_matrix(NumericMode.FLOAT32_REAL, left, right)
    actual_int8 = _run_matrix(NumericMode.HOST_PACKED_INT8_PER_TASK_V1, left, right)

    np.testing.assert_array_equal(actual_float, expected_float)
    np.testing.assert_array_equal(actual_int8, expected_int8)
    assert actual_float.dtype == np.float32
    assert actual_int8.dtype == np.float32


def test_complex128_mode_preserves_complex_contraction() -> None:
    left = np.array(
        [
            [1.0 + 2.0j, -0.5j, 0.25],
            [2.0, 0.25 - 1.0j, -0.75j],
        ]
    )
    right = np.array(
        [
            [0.5 - 1.0j, 2.0],
            [1.5j, -0.25 + 0.5j],
            [0.75 + 0.25j, -1.0j],
        ]
    )

    actual = _run_matrix(NumericMode.COMPLEX128, left, right)
    expected = np.einsum("ab,bc->ac", left, right)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.complex128


def test_float32_real_rejects_nonzero_imaginary_values() -> None:
    left = np.ones((2, 3), dtype=np.complex128)
    left[0, 0] += 1.0j
    right = np.ones((3, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="real-valued"):
        _run_matrix(NumericMode.FLOAT32_REAL, left, right)


def test_numeric_execution_does_not_mutate_inputs() -> None:
    left = np.array([[0.25, -1.5, 2.0], [3.25, 0.5, -0.75]], dtype=np.float64)
    right = np.array([[1.0, -2.0], [0.25, 0.5], [-1.25, 0.75]], dtype=np.float64)
    left_before = left.copy()
    right_before = right.copy()

    _run_matrix(NumericMode.HOST_PACKED_INT8_PER_TASK_V1, left, right)

    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)


def test_execution_plan_hash_changes_for_each_numeric_mode() -> None:
    dag = _matrix_dag()
    plans = {
        mode: _compile(dag, mode)
        for mode in (
            NumericMode.COMPLEX128,
            NumericMode.FLOAT32_REAL,
            NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
        )
    }
    from quantum_bench.execution import execution_plan_hash

    assert len({execution_plan_hash(plan) for plan in plans.values()}) == 3


def test_chain_matches_einsum_and_does_not_mutate_inputs() -> None:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense", dtype="float64"),
        TensorSpec("b", (1, 2), (3, 4), "dense", dtype="float64"),
        TensorSpec("c", (2, 3), (4, 5), "dense", dtype="float64"),
    )
    spec = TensorNetworkSpec(None, tensors, (0, 3), "ab,bc,cd->ad")  # type: ignore[arg-type]
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    c = np.arange(20.0).reshape(4, 5)
    originals = (a.copy(), b.copy(), c.copy())
    dag = _dag(spec, ((1, 2), (0, 1)))

    result = execute(
        _compile(dag),
        dag,
        _inputs(("a", a), ("b", b), ("c", c)),
        RunContext(run_id="chain", target=Target.CPU),
    )

    expected = np.einsum("ab,bc,cd->ad", a, b, c).astype(np.float32)
    assert np.array_equal(result.output, expected)
    assert result.output_hash == _hash(expected)
    assert result.executed_node_ids == ("contract_0", "contract_1")
    assert all(np.array_equal(current, original) for current, original in zip((a, b, c), originals))


def test_topological_order_comes_from_dependencies() -> None:
    a = TensorSpec("a", (0, 1), (2, 2), "dense", dtype="float64")
    b = TensorSpec("b", (1, 2), (2, 2), "dense", dtype="float64")
    c = TensorSpec("c", (2, 3), (2, 2), "dense", dtype="float64")
    first = ContractNode(
        node_id="first",
        left=TensorView(tensor_id="a", labels=a.labels, shape=a.shape),
        right=TensorView(tensor_id="b", labels=b.labels, shape=b.shape),
        output=TensorSpec("mid", (0, 2), (2, 2), "dense", dtype="float64"),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    second = ContractNode(
        node_id="second",
        left=TensorView(tensor_id="mid", labels=(0, 2), shape=(2, 2)),
        right=TensorView(tensor_id="c", labels=c.labels, shape=c.shape),
        output=TensorSpec("out", (0, 3), (2, 2), "dense", dtype="float64"),
        contracted_labels=(2,),
        output_labels=(0, 3),
        dependencies=("first",),
    )
    dag = ContractionDAG(
        tensors=(a, b, c),
        nodes=(second, first),
        output=TensorView(tensor_id="out", labels=(0, 3), shape=(2, 2)),
    )

    plan = _compile(dag)
    assert plan.payload.node_order == ("first", "second")
    result = execute(
        plan,
        dag,
        _inputs(("a", np.eye(2)), ("b", np.ones((2, 2))), ("c", np.eye(2))),
        RunContext(run_id="branch-order", target=Target.CPU),
    )
    assert np.array_equal(result.output, np.ones((2, 2)))
    assert result.executed_node_ids == ("first", "second")


def test_reduce_node_sums_inputs() -> None:
    x = TensorSpec("x", (0,), (2,), "dense", dtype="float64")
    y = TensorSpec("y", (0,), (2,), "dense", dtype="float64")
    output = TensorSpec("out", (0,), (2,), "dense", dtype="float64")
    node = ReduceNode(
        node_id="reduce",
        inputs=(
            TensorView(tensor_id="x", labels=(0,), shape=(2,)),
            TensorView(tensor_id="y", labels=(0,), shape=(2,)),
        ),
        output=output,
    )
    dag = ContractionDAG(
        tensors=(x, y),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=(0,), shape=(2,)),
    )
    result = execute(
        _compile(dag),
        dag,
        _inputs(("x", np.array([1.0, 2.0])), ("y", np.array([3.0, 4.0]))),
        RunContext(run_id="reduce", target=Target.CPU),
    )
    assert np.array_equal(result.output, np.array([4.0, 6.0]))
    assert result.executed_node_ids == ("reduce",)


def test_sliced_dag_executes_fixed_indices_on_original_axes() -> None:
    dag = _matrix_dag()
    sliced = apply_slicing(dag, SliceSpec(node_id="contract_0", label=1))
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(6.0).reshape(3, 2)

    result = execute(
        _compile(sliced),
        sliced,
        _inputs(("a", left), ("b", right)),
        RunContext(run_id="sliced", target=Target.CPU),
    )

    np.testing.assert_array_equal(result.output, left @ right)


def test_sliced_view_with_multiple_original_axes_uses_correct_indices() -> None:
    left_spec = TensorSpec("left", (0, 1, 2), (2, 3, 4), "dense", dtype="float64")
    right_spec = TensorSpec("right", (1, 3), (3, 2), "dense", dtype="float64")
    node = ContractNode(
        node_id="contract",
        left=TensorView(
            tensor_id="left",
            labels=(1,),
            shape=(3,),
            slice_spec=((0, 1), (2, 2)),
        ),
        right=TensorView(tensor_id="right", labels=(1, 3), shape=(3, 2)),
        output=TensorSpec("out", (3,), (2,), "dense", dtype="float64"),
        contracted_labels=(1,),
        output_labels=(3,),
    )
    dag = ContractionDAG(
        tensors=(left_spec, right_spec),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=(3,), shape=(2,)),
    )
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(6.0).reshape(3, 2)

    result = execute(
        _compile(dag),
        dag,
        _inputs(("left", left), ("right", right)),
        RunContext(run_id="multi-axis-sliced", target=Target.CPU),
    )

    expected = left[1, :, 2] @ right
    np.testing.assert_array_equal(result.output, expected.astype(np.float32))


def test_compile_rejects_dag_hash_mismatch() -> None:
    dag = _dag(
        TensorNetworkSpec(
            None,
            (TensorSpec("a", (0,), (2,), "dense"), TensorSpec("b", (0,), (2,), "dense")),
            (),
            "a,b->",
        ),
        ((0, 1),),
    )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hash"):
        compile_cpu(dag, CpuCompileRequest(contraction_dag_hash="wrong"))


def test_gpu_is_explicitly_unsupported() -> None:
    dag = _dag(
        TensorNetworkSpec(
            None,
            (TensorSpec("a", (0,), (2,), "dense"), TensorSpec("b", (0,), (2,), "dense")),
            (),
            "a,b->",
        ),
        ((0, 1),),
    )  # type: ignore[arg-type]
    result = compile_execution(dag, Target.GPU)
    assert isinstance(result, UnsupportedExecution)
    assert result.target is Target.GPU


def test_upmem_execution_is_explicitly_unsupported_without_fallback() -> None:
    dag = _dag(
        TensorNetworkSpec(
            None,
            (TensorSpec("a", (0,), (2,), "dense"), TensorSpec("b", (0,), (2,), "dense")),
            (),
            "a,b->",
        ),
        ((0, 1),),
    )  # type: ignore[arg-type]
    compiled = compile_execution(
        dag,
        UpmemCompileRequest(
            contraction_dag_hash=contraction_dag_hash(dag),
            numeric_mode=NumericMode.FLOAT32,
            topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
        ),
    )
    assert isinstance(compiled, ExecutionPlan)
    plan = ExecutionPlan(
        contraction_dag_hash=contraction_dag_hash(dag),
        target=Target.UPMEM,
        payload=UpmemPlan(
            topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
            numeric_mode=NumericMode.FLOAT32,
            kernel_id="test",
            decomposition_id="test",
            placement_id="test",
            reduction_id="test",
        ),
    )
    failure = execute(
        plan,
        dag,
        _inputs(("a", np.ones(2)), ("b", np.ones(2))),
        RunContext(run_id="upmem", target=Target.UPMEM),
    )
    assert isinstance(failure, ExecutionFailure)
    assert "not implemented" in failure.reason
