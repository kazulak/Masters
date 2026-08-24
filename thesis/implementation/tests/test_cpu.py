from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

import quantum_bench.cpu as cpu
from quantum_bench.cpu import (
    replay_upmem_plan_once,
    run_complex128_reference,
    run_cpu_once,
)
from quantum_bench.lowering import slice_contraction
from quantum_bench.model import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorSpec,
    TensorView,
)
from quantum_bench.numerics import decode_complex_products, encode_complex_tensor
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    Measurement,
    UnsupportedExecution,
)
from quantum_bench.upmem import plan as upmem_plan_module
from quantum_bench.upmem.tiling import M5TileLimits, lower_binary_contraction


FLOAT = "split_complex_float32_v1"
INT8 = "split_complex_int8_shared_scale_v1"


def _contract_dag(
    *, scrambled: bool = False
) -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    x = TensorSpec("x", (0, 1), (2, 2), "dense", dtype="complex128")
    y = TensorSpec("y", (1, 2), (2, 2), "dense", dtype="complex128")
    z = TensorSpec("z", (2, 3), (2, 2), "dense", dtype="complex128")
    first_output = TensorSpec(
        "p", (0, 2), (2, 2), "dense", dtype="complex128", produced_by="first"
    )
    second_output = TensorSpec(
        "q", (0, 3), (2, 2), "dense", dtype="complex128", produced_by="second"
    )
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
        left=TensorView(
            tensor_id="p", labels=first_output.labels, shape=first_output.shape
        ),
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
    return dag, {
        "source": np.arange(24, dtype=np.float64).reshape(2, 3, 4).astype(np.complex128)
    }


def _four_way_sliced_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    left = TensorSpec("left", (0, 1), (2, 4), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (4, 2), "dense", dtype="complex128")
    node = ContractNode(
        node_id="sliced",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=TensorSpec(
            "out", (0, 2), (2, 2), "dense", dtype="complex128", produced_by="sliced"
        ),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = slice_contraction(
        ContractionDAG(
            tensors=(left, right),
            nodes=(node,),
            output=TensorView(tensor_id="out", labels=(0, 2), shape=(2, 2)),
        ),
        node_id="sliced",
        labels=(1,),
    )
    inputs = {
        "left": np.array(
            [[1 + 2j, 20 - 1j, 3 + 4j, 400 - 2j], [2 - 1j, 40 + 3j, 5, 800 + 1j]],
            dtype=np.complex128,
        ),
        "right": np.array(
            [[3 - 1j, 4], [5 + 2j, 6], [70 - 3j, 80], [9 + 1j, 10]],
            dtype=np.complex128,
        ),
    }
    return dag, inputs


def _reduce_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    a = TensorSpec("a", (0,), (1,), "dense", dtype="complex128")
    b = TensorSpec("b", (0,), (1,), "dense", dtype="complex128")
    out = TensorSpec(
        "sum", (0,), (1,), "dense", dtype="complex128", produced_by="reduce"
    )
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


def _int8_reduce_dag(
    *, mixed_raw_input: bool
) -> tuple[ContractionDAG, dict[str, np.ndarray]]:
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
    reduced = TensorSpec(
        "sum", (), (), "dense", dtype="complex128", produced_by="reduce"
    )
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
    assert sample.backend_facts == {
        "backend_id": "numpy_cpu_v1",
        "execution_class": "cpu_host",
    }
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
def test_runtime_errors_are_wrapped_by_stage(
    monkeypatch: pytest.MonkeyPatch, stage: str, operation: str
) -> None:
    dag, inputs = _contract_dag()

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("controlled failure")

    monkeypatch.setattr(cpu, operation, fail)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == stage
    assert error.value.backend_facts == {
        "backend_id": "numpy_cpu_v1",
        "execution_class": "cpu_host",
    }


def test_wrong_decode_shape_is_wrapped_as_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, inputs = _contract_dag()

    def wrong_shape(*args: object, **kwargs: object) -> np.ndarray:
        return np.zeros((1,), dtype=np.complex64)

    monkeypatch.setattr(cpu, "decode_complex_products", wrong_shape)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == "decode"


def test_empty_exception_message_uses_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, inputs = _contract_dag()

    def fail(*args: object, **kwargs: object) -> object:
        raise MemoryError()

    monkeypatch.setattr(cpu, "encode_complex_tensor", fail)
    with pytest.raises(ExecutionFailed) as error:
        run_cpu_once(dag, inputs, FLOAT)
    assert error.value.stage == "encode"
    assert error.value.reason == "MemoryError"


def test_reduce_and_finalize_errors_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    out = TensorSpec(
        "sum", (0,), (2,), "dense", dtype="complex128", produced_by="reduce"
    )
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
    inputs = {
        "a": np.array([1 + 2j, 3], dtype=np.complex128),
        "b": np.array([4, -1j], dtype=np.complex128),
    }
    first = run_cpu_once(dag, inputs, FLOAT)
    second = run_cpu_once(dag, inputs, FLOAT)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(
        first.output, np.array([5 + 2j, 3 - 1j], dtype=np.complex64)
    )
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


@pytest.mark.parametrize("policy", [FLOAT, INT8])
def test_replay_matches_physical_plan_reference_and_records_facts(policy: str) -> None:
    dag, inputs = _contract_dag()
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=policy,
        topology=upmem_plan_module.UpmemTopology(dpu_count=2, tasklets_per_dpu=1),
    )
    before = {key: value.copy() for key, value in inputs.items()}
    sample = replay_upmem_plan_once(dag, plan, inputs)
    reference = run_complex128_reference(dag, inputs)
    if policy == FLOAT:
        np.testing.assert_allclose(sample.output, reference, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(
            sample.output,
            run_cpu_once(dag, inputs, INT8).output,
            rtol=1e-5,
            atol=1e-5,
        )
    assert sample.backend_facts["backend_id"] == "cpu_upmem_plan_replay_v1"
    assert sample.backend_facts["execution_class"] == "cpu_physical_plan_reference"
    assert sample.backend_facts["physical_plan_consumed"] is True
    assert sample.backend_facts["hardware_execution"] is False
    assert sample.measurement.total_wall_s >= 0.0
    assert sample.measurement.preparation_s is not None
    assert sample.measurement.encode_s is not None
    assert sample.measurement.kernel_s is not None
    assert sample.measurement.decode_s is not None
    assert sample.measurement.h2d_s is None
    assert sample.measurement.d2h_s is None
    assert sample.measurement.energy_j is None
    assert sample.numeric_facts["numeric_policy"] == policy
    assert len(sample.numeric_facts["operand_records"]) == 4
    assert len(sample.numeric_facts["raw_lane_records"]) == 8
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], before[key])
    with pytest.raises((TypeError, ValueError)):
        sample.output[0, 0] = 0


def test_replay_int8_raw_lane_hashes_are_exact_little_endian_int32() -> None:
    dag, inputs = _int8_reduce_dag(mixed_raw_input=False)
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=INT8,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    sample = replay_upmem_plan_once(dag, plan, inputs)
    record = next(
        item
        for item in sample.numeric_facts["raw_lane_records"]
        if item["node_id"] == "first" and item["lane"] == "rr"
    )
    expected = np.asarray([[127 * 127]], dtype="<i4")
    assert record["dtype"] == "<i4"
    assert record["exact"] is True
    assert record["shape"] == (1, 1)
    assert record["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()


def test_raw_lane_evidence_uses_runtime_canonical_order() -> None:
    value = np.zeros((1, 1), dtype=np.int32)
    unordered = (
        ("contract_2", "tile", "ir", value),
        ("contract_10", "tile", "ri", value),
        ("contract_10", "tile", "rr", value),
        ("contract_2", "tile", "ii", value),
    )

    ordered = sorted(unordered, key=cpu._raw_lane_sort_key)

    assert [(item[0], item[2]) for item in ordered] == [
        ("contract_10", "rr"),
        ("contract_10", "ri"),
        ("contract_2", "ii"),
        ("contract_2", "ir"),
    ]


@pytest.mark.parametrize("policy", [FLOAT, INT8])
def test_replay_unilateral_contraction_reduces_before_encoding(policy: str) -> None:
    left = TensorSpec("left", (0, 1), (2, 3), "dense", dtype="complex128")
    right = TensorSpec("right", (2,), (4,), "dense", dtype="complex128")
    output = TensorSpec(
        "out", (0, 2), (2, 4), "dense", dtype="complex128", produced_by="contract"
    )
    node = ContractNode(
        node_id="contract",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
    )
    inputs = {
        "left": np.array(
            [
                [1.1 + 0.2j, -0.7 + 1.4j, 0.3 - 0.9j],
                [2.0 - 1.1j, -1.2 + 0.4j, 0.6 + 0.8j],
            ],
            dtype=np.complex128,
        ),
        "right": np.array(
            [1.0 + 0.5j, -0.4 + 1.2j, 0.7 - 0.2j, 1.3 + 0.1j],
            dtype=np.complex128,
        ),
    }
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=policy,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )

    # This is the independent reference for the physical policy: lower the
    # unilateral label into a reduced operand before applying numeric policy.
    reduced_left = TensorSpec("left_reduced", (0,), (2,), "dense", dtype="complex128")
    reduced_node = ContractNode(
        node_id="contract",
        left=TensorView(
            tensor_id="left_reduced",
            labels=reduced_left.labels,
            shape=reduced_left.shape,
        ),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(),
        output_labels=(0, 2),
    )
    reduced_dag = ContractionDAG(
        tensors=(reduced_left, right),
        nodes=(reduced_node,),
        output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
    )
    left_float32 = np.asarray(inputs["left"].real, dtype=np.float32) + 1j * np.asarray(
        inputs["left"].imag, dtype=np.float32
    )
    reduced_inputs = {
        "left_reduced": np.asarray(
            left_float32.real.sum(axis=1, dtype=np.float32)
            + 1j * left_float32.imag.sum(axis=1, dtype=np.float32),
            dtype=np.complex128,
        ),
        "right": inputs["right"],
    }
    expected = run_cpu_once(reduced_dag, reduced_inputs, policy).output

    sample = replay_upmem_plan_once(dag, plan, inputs)
    np.testing.assert_allclose(sample.output, expected, rtol=1e-5, atol=1e-5)
    if policy == INT8:
        naive = run_cpu_once(dag, inputs, policy).output
        assert not np.allclose(sample.output, naive, rtol=1e-6, atol=1e-6)


def test_replay_int8_multiple_k_chunks_records_each_exact_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = TensorSpec("left", (0, 1), (1, 5), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (5, 1), "dense", dtype="complex128")
    output = TensorSpec(
        "out", (0, 2), (1, 1), "dense", dtype="complex128", produced_by="contract"
    )
    node = ContractNode(
        node_id="contract",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
    )
    inputs = {
        "left": np.arange(1, 6, dtype=np.float64).reshape(1, 5).astype(np.complex128),
        "right": np.arange(2, 7, dtype=np.float64).reshape(5, 1).astype(np.complex128),
    }
    limits = M5TileLimits.host_packed_int8(
        max_tile_dim=2, max_elements=64, max_packed_k=2
    )
    monkeypatch.setattr(
        upmem_plan_module, "tile_limits_for_numeric_mode", lambda mode: limits
    )
    monkeypatch.setattr(cpu, "_replay_limits", lambda policy: limits)
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=INT8,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    assert len({unit.k_start for unit in plan.stages[0].work_units}) == 3

    sample = replay_upmem_plan_once(dag, plan, inputs)
    lowering = lower_binary_contraction(
        node,
        np.asarray(inputs["left"].real, dtype=np.float32),
        np.asarray(inputs["right"].real, dtype=np.float32),
        limits=limits,
    )
    left_encoded = encode_complex_tensor(
        np.asarray(lowering.canonical.left, dtype=np.complex64), INT8
    )
    right_encoded = encode_complex_tensor(
        np.asarray(lowering.canonical.right, dtype=np.complex64), INT8
    )
    lane_operands = (
        (left_encoded.real, right_encoded.real),
        (left_encoded.imag, right_encoded.imag),
        (left_encoded.real, right_encoded.imag),
        (left_encoded.imag, right_encoded.real),
    )
    records = {
        (item["stable_tile_id"], item["lane"]): item
        for item in sample.numeric_facts["raw_lane_records"]
    }
    expected_totals = [np.zeros((1, 1), dtype=np.int64) for _ in range(4)]
    for unit in plan.stages[0].work_units:
        for lane_index, (left_plane, right_plane) in enumerate(lane_operands):
            accumulator = np.zeros((unit.m_size, unit.n_size), dtype=np.int32)
            for k_index in range(unit.k_size):
                product = np.multiply(
                    left_plane[
                        unit.batch_start,
                        unit.m_start : unit.m_start + unit.m_size,
                        unit.k_start + k_index,
                    ][:, None],
                    right_plane[
                        unit.batch_start,
                        unit.k_start + k_index,
                        unit.n_start : unit.n_start + unit.n_size,
                    ][None, :],
                    dtype=np.int32,
                )
                accumulator = np.add(accumulator, product, dtype=np.int32)
            expected = np.asarray(accumulator, dtype="<i4")
            record = records[
                (unit.stable_tile_id, ("rr", "ii", "ri", "ir")[lane_index])
            ]
            assert record["dtype"] == "<i4"
            assert record["exact"] is True
            assert record["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
            expected_totals[lane_index] = np.add(
                expected_totals[lane_index], expected, dtype=np.int64
            )

    expected_output = decode_complex_products(
        tuple(expected_totals), left_encoded.scale, right_encoded.scale, INT8
    )
    np.testing.assert_array_equal(sample.output, expected_output)


def test_replay_canonicalizes_operands_before_encoding() -> None:
    left = TensorSpec("left", (0, 1), (2, 3), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (3, 2), "dense", dtype="complex128")
    output = TensorSpec(
        "out", (0, 2), (2, 2), "dense", dtype="complex128", produced_by="contract"
    )
    node = ContractNode(
        node_id="contract",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
    )
    inputs = {
        "left": np.arange(6, dtype=np.float64).reshape(2, 3).astype(np.complex128),
        "right": np.array([[1, 2], [3, 4j], [5, 6]], dtype=np.complex128),
    }
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=FLOAT,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    replay = replay_upmem_plan_once(dag, plan, inputs)
    np.testing.assert_allclose(
        replay.output, run_complex128_reference(dag, inputs), rtol=1e-5, atol=1e-5
    )
    assert replay.numeric_facts["operand_records"][0]["shape"] == (1, 2, 3)


def test_replay_handles_remainder_tiles_and_multiple_k_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = TensorSpec("left", (0, 1), (3, 5), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (5, 7), "dense", dtype="complex128")
    output = TensorSpec(
        "out", (0, 2), (3, 7), "dense", dtype="complex128", produced_by="contract"
    )
    node = ContractNode(
        node_id="contract",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    dag = ContractionDAG(
        tensors=(left, right),
        nodes=(node,),
        output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
    )
    inputs = {
        "left": np.arange(15, dtype=np.float64).reshape(3, 5).astype(np.complex128),
        "right": np.arange(35, dtype=np.float64).reshape(5, 7).astype(np.complex128),
    }
    limits = M5TileLimits.float32(max_tile_dim=2, max_elements=64, max_packed_k=64)
    monkeypatch.setattr(
        upmem_plan_module, "tile_limits_for_numeric_mode", lambda mode: limits
    )
    monkeypatch.setattr(cpu, "_replay_limits", lambda policy: limits)
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=FLOAT,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    assert len(plan.stages[0].work_units) == 24
    sample = replay_upmem_plan_once(dag, plan, inputs)
    np.testing.assert_allclose(
        sample.output, run_complex128_reference(dag, inputs), rtol=1e-5, atol=1e-5
    )


def test_replay_reduces_decoded_branches_in_producer_order() -> None:
    a = TensorSpec("a", (0, 1), (1, 1), "dense", dtype="complex128")
    b = TensorSpec("b", (1, 2), (1, 1), "dense", dtype="complex128")
    c = TensorSpec("c", (0, 1), (1, 1), "dense", dtype="complex128")
    d = TensorSpec("d", (1, 2), (1, 1), "dense", dtype="complex128")
    p = TensorSpec(
        "p", (0, 2), (1, 1), "dense", dtype="complex128", produced_by="first"
    )
    q = TensorSpec(
        "q", (0, 2), (1, 1), "dense", dtype="complex128", produced_by="second"
    )
    summed = TensorSpec(
        "sum", (0, 2), (1, 1), "dense", dtype="complex128", produced_by="reduce"
    )

    def contract(
        node_id: str, left: TensorSpec, right: TensorSpec, output: TensorSpec
    ) -> ContractNode:
        return ContractNode(
            node_id=node_id,
            left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
            right=TensorView(
                tensor_id=right.id, labels=right.labels, shape=right.shape
            ),
            output=output,
            contracted_labels=(1,),
            output_labels=(0, 2),
        )

    first = contract("first", a, b, p)
    second = contract("second", c, d, q)
    reduce = ReduceNode(
        node_id="reduce",
        inputs=(
            TensorView(tensor_id="q", labels=q.labels, shape=q.shape),
            TensorView(tensor_id="p", labels=p.labels, shape=p.shape),
        ),
        output=summed,
        dependencies=("first", "second"),
    )
    dag = ContractionDAG(
        tensors=(a, b, c, d),
        nodes=(second, reduce, first),
        output=TensorView(tensor_id="sum", labels=summed.labels, shape=summed.shape),
    )
    inputs = {
        "a": np.array([[2 + 1j]], dtype=np.complex128),
        "b": np.array([[3 - 1j]], dtype=np.complex128),
        "c": np.array([[20 + 10j]], dtype=np.complex128),
        "d": np.array([[30 - 10j]], dtype=np.complex128),
    }
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=INT8,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    sample = replay_upmem_plan_once(dag, plan, inputs)
    expected = run_cpu_once(dag, inputs, INT8).output
    np.testing.assert_allclose(sample.output, expected, rtol=1e-5, atol=1e-5)
    assert sample.measurement.host_reduce_s is not None
    assert {item["node_id"] for item in sample.numeric_facts["operand_records"]} == {
        "first",
        "second",
    }
    scales = {
        item["node_id"]: item["scale"]
        for item in sample.numeric_facts["operand_records"]
        if item["side"] == "left"
    }
    assert scales["first"] != scales["second"]


def test_replay_grouped_slices_decodes_each_int8_branch_before_reduction() -> None:
    dag, inputs = _four_way_sliced_dag()
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=INT8,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    replay = replay_upmem_plan_once(dag, plan, inputs)
    expected = run_cpu_once(dag, inputs, INT8)
    np.testing.assert_array_equal(replay.output, expected.output)
    branch_ids = tuple(plan.stages[0].node_ids)
    assert (
        tuple(
            item["node_id"]
            for item in replay.numeric_facts["operand_records"]
            if item["side"] == "left"
        )
        == branch_ids
    )
    scales = {
        item["node_id"]: item["scale"]
        for item in replay.numeric_facts["operand_records"]
        if item["side"] == "left"
    }
    assert len(set(scales.values())) > 1


def test_replay_supports_zero_and_pure_imaginary_inputs() -> None:
    dag, _ = _contract_dag()
    inputs = {
        "x": np.zeros((2, 2), dtype=np.complex128),
        "y": 1j * np.ones((2, 2), dtype=np.complex128),
        "z": np.zeros((2, 2), dtype=np.complex128),
    }
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=INT8,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    sample = replay_upmem_plan_once(dag, plan, inputs)
    np.testing.assert_array_equal(sample.output, np.zeros((2, 2), dtype=np.complex64))
    assert sample.numeric_facts["saturation_real"] == 0
    assert sample.numeric_facts["saturation_imag"] == 0


def test_replay_rejects_tampered_plan_before_timing() -> None:
    dag, inputs = _contract_dag()
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=FLOAT,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    unit = plan.stages[0].work_units[0]
    tampered = replace(
        plan,
        stages=(
            replace(
                plan.stages[0],
                work_units=(
                    replace(unit, estimated_input_bytes=unit.estimated_input_bytes + 1),
                ),
            ),
            *plan.stages[1:],
        ),
    )
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        replay_upmem_plan_once(dag, tampered, inputs)


def test_replay_rejects_tampered_grouped_stage_and_nonsteady_scope() -> None:
    dag, inputs = _contract_dag()
    plan = upmem_plan_module.plan_upmem(
        dag,
        numeric_policy=FLOAT,
        topology=upmem_plan_module.UpmemTopology(dpu_count=1, tasklets_per_dpu=1),
    )
    grouped = replace(
        plan,
        stages=(replace(plan.stages[0], node_ids=("first", "second")),),
    )
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        replay_upmem_plan_once(dag, grouped, inputs)
    with pytest.raises(UnsupportedExecution, match="timing scope"):
        replay_upmem_plan_once(dag, plan, inputs, scope_id="simulation_end_to_end_v1")
