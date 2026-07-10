from __future__ import annotations

import json

import numpy as np

from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    generic_loop_reference_float32,
    generic_loop_reference_int32,
    prepare_generic_task,
)


def _task(
    task_id: str = "generic",
    *,
    left_shape: tuple[int, ...] = (2, 3),
    right_shape: tuple[int, ...] = (3, 4),
    output_shape: tuple[int, ...] = (2, 4),
    index_expression: str = "ab,bc->ac",
    left_labels: tuple[int, ...] = (0, 1),
    right_labels: tuple[int, ...] = (1, 2),
    contracted_labels: tuple[int, ...] = (1,),
    output_labels: tuple[int, ...] = (0, 2),
) -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=(f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=f"{task_id}_out",
        dependencies=(),
        index_expression=index_expression,
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        gemm_m=0,
        gemm_k=0,
        gemm_n=0,
        structure="generic",
        estimated_flops=0,
        estimated_bytes=0,
    )


def _tensors(task: ContractionTask, *, complex_values: bool = False) -> tuple[TensorValue, TensorValue]:
    left_size = int(np.prod(task.input_shapes[0]))
    right_size = int(np.prod(task.input_shapes[1]))
    left = ((np.arange(left_size, dtype=np.float64).reshape(task.input_shapes[0]) % 13.0) - 6.0) / 13.0
    right = ((np.arange(right_size, dtype=np.float64).reshape(task.input_shapes[1]) % 11.0) - 5.0) / 11.0
    if complex_values:
        left = left + 0.25j
    return (
        TensorValue(TensorSpec(task.input_tensor_ids[0], task.left_labels, task.input_shapes[0], "dense", dtype=str(left.dtype)), left),
        TensorValue(TensorSpec(task.input_tensor_ids[1], task.right_labels, task.input_shapes[1], "dense", dtype=str(right.dtype)), right),
    )


def test_generic_preparation_is_quantization_aware_and_json_safe() -> None:
    task = _task()
    left, right = _tensors(task)
    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert result.status == "prepared"
    assert result.reason is None
    assert result.prepared_operands is not None
    assert result.validation_metrics["reference_kind"] == "expected_quantized_reference_vs_python_generic_loop_int32"
    assert result.validation_metrics["passed"] is True
    assert result.full_precision_error_metrics["reference_kind"] == "full_precision_vs_expected_quantized_reference"
    assert result.metadata["validation_target"] == "expected_quantized_reference_output"
    assert result.metadata["full_precision_reference_is_validation_target"] is False

    payload = result.to_json_dict()
    encoded = json.dumps(payload)
    assert "prepared_operands" not in payload
    assert "left_quantized" not in encoded
    assert "right_quantized" not in encoded
    assert payload["native_index_metadata"]["output_to_left_axes"] == [0, -1]
    assert payload["native_index_metadata"]["output_to_right_axes"] == [-1, 1]


def test_generic_loop_reference_matches_einsum_on_quantized_inputs() -> None:
    task = _task()
    left, right = _tensors(task)
    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))
    assert result.prepared_operands is not None

    reference_i32 = generic_loop_reference_int32(
        result.prepared_operands.left_quantized,
        result.prepared_operands.right_quantized,
        output_shape=result.output_shape,
        left_strides=result.left_strides,
        right_strides=result.right_strides,
        output_strides=result.output_strides,
        output_to_left_axes=result.output_to_left_axes,
        output_to_right_axes=result.output_to_right_axes,
        contracted_to_left_axes=result.contracted_to_left_axes,
        contracted_to_right_axes=result.contracted_to_right_axes,
        contracted_dims=result.contracted_dims,
    )
    expected_i32 = np.einsum(
        task.index_expression,
        result.prepared_operands.left_quantized.astype(np.int32),
        result.prepared_operands.right_quantized.astype(np.int32),
        optimize=False,
    )
    np.testing.assert_array_equal(reference_i32, expected_i32)


def test_generic_preparation_supports_float32_no_quant_mode() -> None:
    task = _task()
    left, right = _tensors(task)
    result = prepare_generic_task(
        GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right, quantization_mode="none")
    )

    assert result.status == "prepared"
    assert result.reason is None
    assert result.left_conversion is None
    assert result.right_conversion is None
    assert result.prepared_operands is not None
    assert result.prepared_operands.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT
    assert result.metadata["quantization_mode"] == "none"
    assert result.metadata["input_dtype_on_dpu"] == "float32"
    assert result.metadata["accumulator_dtype_on_dpu"] == "float32"
    assert result.metadata["scaling_applied"] is False

    reference_f32 = generic_loop_reference_float32(
        result.prepared_operands.left_operand,
        result.prepared_operands.right_operand,
        output_shape=result.output_shape,
        left_strides=result.left_strides,
        right_strides=result.right_strides,
        output_strides=result.output_strides,
        output_to_left_axes=result.output_to_left_axes,
        output_to_right_axes=result.output_to_right_axes,
        contracted_to_left_axes=result.contracted_to_left_axes,
        contracted_to_right_axes=result.contracted_to_right_axes,
        contracted_dims=result.contracted_dims,
    )
    expected = np.einsum(
        task.index_expression,
        np.asarray(left.array, dtype=np.float32),
        np.asarray(right.array, dtype=np.float32),
        optimize=False,
    )
    np.testing.assert_allclose(reference_f32, expected, rtol=1e-6, atol=1e-6)


def test_generic_preparation_rejects_complex_values_explicitly() -> None:
    task = _task()
    left, right = _tensors(task, complex_values=True)
    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert result.status == "unsupported_shape"
    assert result.reason == "complex_generic_loop_not_implemented"
    assert result.prepared_operands is None


def test_generic_preparation_rejects_int32_accumulation_overflow_risk() -> None:
    task = _task(
        "overflow",
        left_shape=(1, 200000),
        right_shape=(200000, 1),
        output_shape=(1, 1),
    )
    left, right = _tensors(task)
    result = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            caps=GenericTaskPreparationCaps(max_tensor_elements=300000, max_contracted_combinations=300000),
        )
    )

    assert result.status == "unsupported_shape"
    assert result.reason == "int32_accumulation_overflow_risk"


def test_generic_preparation_rejects_element_cap_before_bridge() -> None:
    task = _task("cap", left_shape=(65537, 1), right_shape=(1, 1), output_shape=(65537, 1))
    left, right = _tensors(task)
    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert result.status == "unsupported_shape"
    assert result.reason == "element_count_cap_exceeded"


def test_generic_preparation_default_element_cap_accepts_65536() -> None:
    task = _task("element_boundary", left_shape=(256, 256), right_shape=(256, 1), output_shape=(256, 1))
    left, right = _tensors(task)
    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert GenericTaskPreparationCaps().max_tensor_elements == 65536
    assert result.status == "prepared"
    assert result.reason is None


def test_generic_preparation_default_rank_cap_accepts_rank_sixteen() -> None:
    rank = 16
    task = _task(
        "rank_sixteen",
        left_shape=(1,) * rank,
        right_shape=(1,),
        output_shape=(1,) * (rank - 1),
        index_expression="abcdefghijklmnop,p->abcdefghijklmno",
        left_labels=tuple(range(rank)),
        right_labels=(rank - 1,),
        contracted_labels=(rank - 1,),
        output_labels=tuple(range(rank - 1)),
    )
    left, right = _tensors(task)

    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert GenericTaskPreparationCaps().max_rank == rank
    assert result.status == "prepared"
    assert result.reason is None


def test_generic_preparation_default_rank_cap_rejects_rank_seventeen() -> None:
    rank = 17
    task = _task(
        "rank_seventeen",
        left_shape=(1,) * rank,
        right_shape=(1,),
        output_shape=(1,) * (rank - 1),
        index_expression="abcdefghijklmnopq,q->abcdefghijklmnop",
        left_labels=tuple(range(rank)),
        right_labels=(rank - 1,),
        contracted_labels=(rank - 1,),
        output_labels=tuple(range(rank - 1)),
    )
    left, right = _tensors(task)

    result = prepare_generic_task(GenericTaskPreparationInput(task=task, left_tensor=left, right_tensor=right))

    assert result.status == "unsupported_shape"
    assert result.reason == "rank_cap_exceeded"


def test_generic_preparation_contract_cap_remains_4096() -> None:
    task = _task(
        "contracted_cap",
        left_shape=(4097, 1),
        right_shape=(4097, 1),
        output_shape=(1, 1),
        index_expression="ab,ac->bc",
        left_labels=(0, 1),
        right_labels=(0, 2),
        contracted_labels=(0,),
        output_labels=(1, 2),
    )
    left, right = _tensors(task)
    result = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            caps=GenericTaskPreparationCaps(max_tensor_elements=65536),
        )
    )

    assert result.status == "unsupported_shape"
    assert result.reason == "contracted_combination_cap_exceeded"
