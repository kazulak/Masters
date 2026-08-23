from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import quantum_bench.cpu as cpu
from quantum_bench.cpu import run_complex128_reference, run_cpu_once
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode, TensorSpec, TensorView
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    Measurement,
    UnsupportedExecution,
)


FLOAT = "split_complex_float32_v1"
INT8 = "split_complex_int8_shared_scale_v1"


def _contract_dag(*, scrambled: bool = False) -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    x = TensorSpec("x", (0, 1), (2, 2), "dense", dtype="complex128")
    y = TensorSpec("y", (1, 2), (2, 2), "dense", dtype="complex128")
    z = TensorSpec("z", (2, 3), (2, 2), "dense", dtype="complex128")
    first_output = TensorSpec("p", (0, 2), (2, 2), "dense", dtype="complex128", produced_by="first")
    second_output = TensorSpec("q", (0, 3), (2, 2), "dense", dtype="complex128", produced_by="second")
    first = ContractNode(
        node_id="first",
        left=TensorView(tensor_id="x", labels=x.labels, shape=x.shape),
        right=TensorView(tensor_id="y", labels=y.labels, shape=y.shape),
        output=first_output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    second = ContractNode(
        node_id="second",
        left=TensorView(tensor_id="p", labels=first_output.labels, shape=first_output.shape),
        right=TensorView(tensor_id="z", labels=z.labels, shape=z.shape),
        output=second_output,
        contracted_labels=(2,),
        output_labels=(0, 3),
        dependencies=("first",),
    )
    nodes = (second, first) if scrambled else (first, second)
    dag = ContractionDAG(
        tensors=(x, y, z),
        nodes=nodes,
        output=TensorView(tensor_id="q", labels=(0, 3), shape=(2, 2)),
    )
    inputs = {
        "x": np.array([[1 + 1j, 2], [3, 4 - 1j]], dtype=np.complex128),
        "y": np.array([[2, 1j], [1 - 1j, 3]], dtype=np.complex128),
        "z": np.array([[1, 2], [3j, 4]], dtype=np.complex128),
    }
    return dag, inputs


def _slice_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    source = TensorSpec("source", (0, 1, 2), (2, 3, 4), "dense", dtype="complex128")
    dag = ContractionDAG(
        tensors=(source,),
        nodes=(),
        output=TensorView(
            tensor_id="source",
            labels=(2,),
            shape=(4,),
            slice_spec=((0, 1), (1, 2)),
        ),
    )
    return dag, {"source": np.arange(24, dtype=np.float64).reshape(2, 3, 4).astype(np.complex128)}


def _reduce_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    a = TensorSpec("a", (0,), (1,), "dense", dtype="complex128")
    b = TensorSpec("b", (0,), (1,), "dense", dtype="complex128")
    out = TensorSpec("sum", (0,), (1,), "dense", dtype="complex128", produced_by="reduce")
    dag = ContractionDAG(
        tensors=(a, b),
        nodes=(
            ReduceNode(
                node_id="reduce",
                inputs=(
                    TensorView(tensor_id="a", labels=(0,), shape=(1,)),
                    TensorView(tensor_id="b", labels=(0,), shape=(1,)),
                ),
                output=out,
            ),
        ),
        output=TensorView(tensor_id="sum", labels=(0,), shape=(1,)),
    )
    return dag, {
        "a": np.array([1 + 0j], dtype=np.complex128),
        "b": np.array([2 + 0j], dtype=np.complex128),
    }


def _int8_reduce_dag(*, mixed_raw_input: bool) -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    a = TensorSpec("a", (0, 1), (1, 1), "dense", dtype="complex128")
    b = TensorSpec("b", (1, 2), (1, 1), "dense", dtype="complex128")

    def contract(node_id: str, output_id: str) -> ContractNode:
        output = TensorSpec(
            output_id,
            (0, 2),
            (1, 1),
            "dense",
            dtype="complex128",
            produced_by=node_id,
        )
        return ContractNode(
            node_id=node_id,
            left=TensorView(tensor_id="a", labels=a.labels, shape=a.shape),
            right=TensorView(tensor_id="b", labels=b.labels, shape=b.shape),
            output=output,
            contracted_labels=(1,),
            output_labels=(0, 2),
        )

    first = contract("first", "p")
    tensors = [a, b]
    inputs: dict[str, np.ndarray] = {
        "a": np.array([[2 + 1j]], dtype=np.complex128),
        "b": np.array([[3 - 1j]], dtype=np.complex128),
    }
    if mixed_raw_input:
        raw = TensorSpec("raw", (0, 2), (1, 1), "dense", dtype="complex128")
        tensors.append(raw)
        inputs["raw"] = np.array([[1 + 0j]], dtype=np.complex128)
        second_view = TensorView(tensor_id="raw", labels=(0, 2), shape=(1, 1))
        nodes: tuple[ContractNode | ReduceNode, ...] = (first,)
        dependencies = ("first",)
    else:
        second = contract("second", "q")
        second_view = TensorView(tensor_id="q", labels=(0, 2), shape=(1, 1))
        nodes = (first, second)
        dependencies = ("first", "second")
    reduced = TensorSpec(
        "sum", (0, 2), (1, 1), "dense", dtype="complex128", produced_by="reduce"
    )
    reduce = ReduceNode(
        node_id="reduce",
        inputs=(
            TensorView(tensor_id="p", labels=(0, 2), shape=(1, 1)),
            second_view,
        ),
        output=reduced,
        dependencies=dependencies,
    )
    return (
        ContractionDAG(
            tensors=tuple(tensors),
            nodes=(*nodes, reduce),
            output=TensorView(tensor_id="sum", labels=(0, 2), shape=(1, 1)),
        ),
        inputs,
    )


def test_measurement_and_facts_are_validated_and_frozen() -> None:
    with pytest.raises(ValueError):
        Measurement("", 0.0)
    with pytest.raises(ValueError):
        Measurement("scope", -1.0)
    with pytest.raises(ValueError):
        Measurement("scope", float("nan"))
    with pytest.raises(ValueError):
        Measurement("scope", 0.0, h2d_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Measurement("scope", 0.0, d2h_bytes=-1)

    output = np.array([1 + 2j], dtype=np.complex64)
    backend = {"nested": {"values": [1, 2]}}
    sample = ExecutionSample(
        output=output,
        measurement=Measurement("scope", 0.0),
        backend_facts=backend,
        numeric_facts={"ok": True},
    )
    output[0] = 9
    backend["nested"]["values"].append(3)  # type: ignore[index]
    assert sample.output[0] == 1 + 2j
    with pytest.raises((TypeError, ValueError)):
        sample.output[0] = 2
    with pytest.raises(TypeError):
        sample.backend_facts["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        sample.backend_facts["nested"]["values"] += (3,)  # type: ignore[index]
    with pytest.raises(TypeError):
        ExecutionSample(
            output=np.zeros(1),
            measurement=Measurement("scope", 0.0),
            backend_facts={"bad": np.int64(1)},
            numeric_facts={},
        )


def test_failures_preserve_contract_fields() -> None:
    unsupported = UnsupportedExecution("preflight", "not available", "timing_scope")
    assert (unsupported.stage, unsupported.reason, unsupported.capability) == (
        "preflight",
        "not available",
        "timing_scope",
    )
    failed = ExecutionFailed("kernel", "crashed", {"rank": 0})
    assert failed.stage == "kernel"
    assert failed.reason == "crashed"
    assert failed.backend_facts["rank"] == 0


def test_execution_sample_requires_measurement() -> None:
    with pytest.raises(TypeError, match="Measurement"):
        ExecutionSample(  # type: ignore[arg-type]
            output=np.zeros(1),
            measurement=object(),
            backend_facts={},
            numeric_facts={},
        )


def test_policy_preflight_rejects_unknown_and_inapplicable_int8() -> None:
    empty_dag, empty_inputs = _slice_dag()
    with pytest.raises(UnsupportedExecution) as unknown:
        run_cpu_once(empty_dag, empty_inputs, "unknown")  # type: ignore[arg-type]
    assert unknown.value.capability == "numeric_policy"
    with pytest.raises(UnsupportedExecution) as empty_int8:
        run_cpu_once(empty_dag, empty_inputs, INT8)
    assert empty_int8.value.capability == "numeric_policy_applicability"

    reduce_dag, reduce_inputs = _reduce_dag()
    with pytest.raises(UnsupportedExecution) as reduce_int8:
        run_cpu_once(reduce_dag, reduce_inputs, INT8)
    assert reduce_int8.value.capability == "numeric_policy_applicability"


def test_int8_admission_follows_requested_output_dataflow() -> None:
    disconnected, disconnected_inputs = _contract_dag()
    disconnected = replace(
        disconnected,
        output=TensorView(tensor_id="x", labels=(0, 1), shape=(2, 2)),
    )
    with pytest.raises(UnsupportedExecution) as raw_output:
        run_cpu_once(disconnected, disconnected_inputs, INT8)
    assert raw_output.value.capability == "numeric_policy_applicability"

    mixed, mixed_inputs = _int8_reduce_dag(mixed_raw_input=True)
    with pytest.raises(UnsupportedExecution) as mixed_output:
        run_cpu_once(mixed, mixed_inputs, INT8)
    assert mixed_output.value.capability == "numeric_policy_applicability"

    derived, derived_inputs = _int8_reduce_dag(mixed_raw_input=False)
    sample = run_cpu_once(derived, derived_inputs, INT8)
    assert sample.numeric_facts["numeric_policy"] == INT8
    assert sample.measurement.host_reduce_s is not None


def test_int8_reduction_uses_producer_id_order(monkeypatch: pytest.MonkeyPatch) -> None:
    left = TensorSpec("left", (), (), "dense", dtype="complex128")
    right = TensorSpec("right", (), (), "dense", dtype="complex128")

    def contract(node_id: str, output_id: str) -> ContractNode:
        output = TensorSpec(
            output_id, (), (), "dense", dtype="complex128", produced_by=node_id
        )
        return ContractNode(
            node_id=node_id,
            left=TensorView(tensor_id="left", labels=(), shape=()),
            right=TensorView(tensor_id="right", labels=(), shape=()),
            output=output,
            contracted_labels=(),
            output_labels=(),
        )

    first = contract("producer_a", "a")
    second = contract("producer_b", "b")
    third = contract("producer_c", "c")
    reduced = TensorSpec("sum", (), (), "dense", dtype="complex128", produced_by="reduce")
    reduce = ReduceNode(
        node_id="reduce",
        inputs=(
            TensorView(tensor_id="c", labels=(), shape=()),
            TensorView(tensor_id="b", labels=(), shape=()),
            TensorView(tensor_id="a", labels=(), shape=()),
        ),
        output=reduced,
        dependencies=("producer_c", "producer_b", "producer_a"),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(third, reduce, second, first),
        output=TensorView(tensor_id="sum", labels=(), shape=()),
    )
    decoded = iter((1e20, -1e20, 1.0))

    def cancellation_values(*args: object, **kwargs: object) -> np.ndarray:
        return np.asarray(next(decoded), dtype=np.complex64)

    monkeypatch.setattr(cpu, "decode_complex_products", cancellation_values)
    sample = run_cpu_once(
        dag,
        {
            "left": np.asarray(1 + 0j, dtype=np.complex128),
            "right": np.asarray(1 + 0j, dtype=np.complex128),
        },
        INT8,
    )
    assert sample.output == np.asarray(1 + 0j, dtype=np.complex64)


def test_cpu_obeys_dependencies_and_matches_complex128_reference() -> None:
    dag, inputs = _contract_dag(scrambled=True)
    sample = run_cpu_once(dag, inputs, FLOAT)
    expected = run_complex128_reference(dag, inputs)
    np.testing.assert_allclose(sample.output, expected, rtol=1e-6, atol=1e-6)
    assert sample.output.dtype == np.complex64
    assert sample.measurement.kernel_s is not None
    assert sample.measurement.encode_s is not None
    assert sample.measurement.decode_s is not None
    assert sample.backend_facts == {"backend_id": "numpy_cpu_v1", "execution_class": "cpu_host"}
    assert sample.numeric_facts["numeric_policy"] == FLOAT


def test_cpu_rejects_cycles_and_nonsteady_scope() -> None:
    dag, inputs = _contract_dag()
    with pytest.raises(UnsupportedExecution, match="timing scope"):
        run_cpu_once(dag, inputs, FLOAT, scope_id="simulation_end_to_end_v1")
    first, second = dag.nodes
    cyclic = replace(
        dag,
        nodes=(
            replace(first, dependencies=("second",)),
            replace(second, dependencies=("first",)),
        ),
    )
    with pytest.raises(ValueError):
        run_cpu_once(cyclic, inputs, FLOAT)


@pytest.mark.parametrize(
    ("stage", "operation"),
    [
        ("encode", "encode_complex_tensor"),
        ("kernel", "contract_complex_products"),
        ("decode", "decode_complex_products"),
    ],
)
def test_runtime_errors_are_wrapped_by_stage(monkeypatch: pytest.MonkeyPatch, stage: str, operation: str) -> None:
    dag, inputs = _contract_dag()

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("controlled failure")

    monkeypatch.setattr(cpu, operation, fail)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == stage
    assert error.value.backend_facts == {"backend_id": "numpy_cpu_v1", "execution_class": "cpu_host"}


def test_wrong_decode_shape_is_wrapped_as_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    dag, inputs = _contract_dag()

    def wrong_shape(*args: object, **kwargs: object) -> np.ndarray:
        return np.zeros((1,), dtype=np.complex64)

    monkeypatch.setattr(cpu, "decode_complex_products", wrong_shape)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == "decode"


def test_empty_exception_message_uses_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    dag, inputs = _contract_dag()

    def fail(*args: object, **kwargs: object) -> object:
        raise MemoryError()

    monkeypatch.setattr(cpu, "encode_complex_tensor", fail)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == "encode"
    assert error.value.reason == "MemoryError"


def test_reduce_and_finalize_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    reduce_dag, reduce_inputs = _reduce_dag()

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("controlled failure")

    monkeypatch.setattr(cpu, "_to_complex64", fail)
    with pytest.raises(ExecutionFailed) as reduced:
        run_cpu_once(reduce_dag, reduce_inputs, FLOAT)
    assert reduced.value.stage == "host_reduce"

    empty_dag, empty_inputs = _slice_dag()
    with pytest.raises(ExecutionFailed) as finalized:
        run_cpu_once(empty_dag, empty_inputs, FLOAT)
    assert finalized.value.stage == "finalize"


def test_zero_duration_phases_are_recorded_and_topology_precedes_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, inputs = _contract_dag()
    original_order = cpu._topological_order
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 10.0

    def checked_order(value: ContractionDAG):
        assert clock_calls == 0
        return original_order(value)

    monkeypatch.setattr(cpu.time, "perf_counter", clock)
    monkeypatch.setattr(cpu, "_topological_order", checked_order)
    measurement = run_cpu_once(dag, inputs, FLOAT).measurement
    assert measurement.total_wall_s == 0.0
    assert measurement.encode_s == 0.0
    assert measurement.kernel_s == 0.0
    assert measurement.decode_s == 0.0


def test_reduce_node_is_deterministic_and_complex64() -> None:
    a = TensorSpec("a", (0,), (2,), "dense", dtype="complex128")
    b = TensorSpec("b", (0,), (2,), "dense", dtype="complex128")
    out = TensorSpec("sum", (0,), (2,), "dense", dtype="complex128", produced_by="reduce")
    dag = ContractionDAG(
        tensors=(a, b),
        nodes=(
            ReduceNode(
                node_id="reduce",
                inputs=(
                    TensorView(tensor_id="a", labels=(0,), shape=(2,)),
                    TensorView(tensor_id="b", labels=(0,), shape=(2,)),
                ),
                output=out,
            ),
        ),
        output=TensorView(tensor_id="sum", labels=(0,), shape=(2,)),
    )
    inputs = {"a": np.array([1 + 2j, 3], dtype=np.complex128), "b": np.array([4, -1j], dtype=np.complex128)}
    first = run_cpu_once(dag, inputs, FLOAT)
    second = run_cpu_once(dag, inputs, FLOAT)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.output, np.array([5 + 2j, 3 - 1j], dtype=np.complex64))
    assert first.measurement.host_reduce_s is not None
    assert first.measurement.kernel_s is None


def test_views_apply_multiple_fixed_axes_against_original_tensor() -> None:
    dag, inputs = _slice_dag()
    expected = inputs["source"][1, 2, :]
    sample = run_cpu_once(dag, inputs, FLOAT)
    reference = run_complex128_reference(dag, inputs)
    np.testing.assert_array_equal(sample.output, expected.astype(np.complex64))
    np.testing.assert_array_equal(reference, expected)


def test_reduction_and_finalization_reject_float32_overflow() -> None:
    reduce_dag, reduce_inputs = _reduce_dag()
    reduce_inputs["a"][0] = float(np.finfo(np.float32).max) * 2
    with pytest.raises(ExecutionFailed, match="host_reduce"):
        run_cpu_once(reduce_dag, reduce_inputs, FLOAT)

    empty_dag, empty_inputs = _slice_dag()
    empty_inputs["source"][1, 2, 0] = float(np.finfo(np.float32).max) * 2
    with pytest.raises(ExecutionFailed, match="finalize"):
        run_cpu_once(empty_dag, empty_inputs, FLOAT)


def test_empty_dag_and_scalar_output() -> None:
    scalar = TensorSpec("scalar", (), (), "dense", dtype="complex128")
    dag = ContractionDAG(
        tensors=(scalar,),
        nodes=(),
        output=TensorView(tensor_id="scalar", labels=(), shape=()),
    )
    inputs = {"scalar": np.asarray(2 + 3j, dtype=np.complex128)}
    sample = run_cpu_once(dag, inputs, FLOAT)
    reference = run_complex128_reference(dag, inputs)
    assert sample.output.shape == ()
    assert sample.output.dtype == np.complex64
    assert reference.shape == ()
    assert reference.dtype == np.complex128


@pytest.mark.parametrize("policy", [FLOAT, INT8])
def test_input_arrays_are_unchanged_and_runs_are_deterministic(policy: str) -> None:
    dag, inputs = _contract_dag()
    before = {key: value.copy() for key, value in inputs.items()}
    first = run_cpu_once(dag, inputs, policy)
    second = run_cpu_once(dag, inputs, policy)
    np.testing.assert_array_equal(first.output, second.output)
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], before[key])


def test_complex128_reference_preserves_output_order() -> None:
    dag, inputs = _contract_dag()
    node = dag.nodes[-1]
    assert isinstance(node, ContractNode)
    reordered = replace(
        node,
        output_labels=(3, 0),
        output=replace(node.output, labels=(3, 0), shape=(2, 2)),
    )
    reordered_dag = replace(
        dag,
        nodes=(dag.nodes[0], reordered),
        output=TensorView(tensor_id="q", labels=(3, 0), shape=(2, 2)),
    )
    # The descriptor is intentionally changed with the declared output order.
    result = run_complex128_reference(reordered_dag, inputs)
    expected = run_complex128_reference(dag, inputs).T
    np.testing.assert_array_equal(result, expected)


def test_complex128_reference_rejects_nonfinite_output() -> None:
    dag, inputs = _slice_dag()
    inputs["source"][1, 2, 0] = np.nan + 0j
    with pytest.raises(ValueError, match="nonfinite"):
        run_complex128_reference(dag, inputs)
