from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.targets.upmem import (
    GENERIC_BRIDGE_ID,
    GENERIC_LOOP_BACKEND_ID,
    execute_generic_bridge,
    generic_bridge_backend_registry,
    read_generic_bridge_input_manifest,
    read_generic_bridge_output_manifest,
    write_generic_bridge_input_manifest,
)


def _prepared_task():
    task = ContractionTask(
        id="generic_bridge_task",
        input_tensor_ids=("left", "right"),
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
        estimated_flops=0,
        estimated_bytes=0,
    )
    left = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
    right = np.array([[0.2, 0.3], [-0.4, 0.5], [0.6, -0.7]], dtype=np.float64)
    return prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=TensorValue(TensorSpec("left", task.left_labels, task.input_shapes[0], "dense", dtype="float64"), left),
            right_tensor=TensorValue(TensorSpec("right", task.right_labels, task.input_shapes[1], "dense", dtype="float64"), right),
        )
    )


def test_generic_bridge_manifest_has_label_free_native_contract(tmp_path: Path) -> None:
    preparation = _prepared_task()
    manifest = write_generic_bridge_input_manifest(preparation, tmp_path)
    payload = json.loads((tmp_path / "input_manifest.json").read_text(encoding="utf-8"))

    assert manifest.bridge_id == GENERIC_BRIDGE_ID
    assert payload["kernel_family"] == "generic_loop_fallback"
    assert payload["metadata"]["native_contract_uses_string_labels"] is False
    assert payload["metadata"]["simplepim_api_used"] is False
    assert payload["metadata"]["native_sdk_control_path"] is True
    assert payload["expected_quantized_reference_output"]["relative_path"] == "references/expected_quantized_reference_output.npy"
    assert payload["full_precision_reference_output"]["relative_path"] == "references/full_precision_reference_output.npy"
    assert "left_labels" not in payload["native_index_metadata"]
    assert "right_labels" not in payload["native_index_metadata"]
    json.dumps(payload)

    read_back = read_generic_bridge_input_manifest(tmp_path / "input_manifest.json")
    assert read_back.native_index_metadata["contracted_combination_count"] == 3
    assert not Path(read_back.operands["left"]["relative_path"]).is_absolute()
    assert (tmp_path / read_back.operands["left"]["relative_path"]).exists()


def test_generic_bridge_rejects_manifest_path_escape(tmp_path: Path) -> None:
    preparation = _prepared_task()
    write_generic_bridge_input_manifest(preparation, tmp_path)
    payload = json.loads((tmp_path / "input_manifest.json").read_text(encoding="utf-8"))
    payload["operands"]["left"]["relative_path"] = "../escape.npy"
    (tmp_path / "input_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    result = execute_generic_bridge(tmp_path / "input_manifest.json", execute_external=False)

    assert result.execution_status == "failed"
    assert result.reason == "input_manifest_invalid"
    assert result.external_command_executed is False
    output = read_generic_bridge_output_manifest(tmp_path / "output_manifest.json")
    assert output.status == "failed"
    assert output.validation_metrics["reason"] == "input_manifest_invalid"


def test_generic_bridge_external_disabled_writes_nonexecuting_output_manifest(tmp_path: Path) -> None:
    preparation = _prepared_task()
    write_generic_bridge_input_manifest(preparation, tmp_path)

    result = execute_generic_bridge(tmp_path / "input_manifest.json", execute_external=False)
    output = read_generic_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "not_implemented"
    assert result.reason == "generic_external_execution_disabled"
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert output.status == "not_implemented"
    assert output.output_blob is None
    assert output.metadata["kernel_family"] == "generic_loop_fallback"
    assert output.metadata["simplepim_api_used"] is False


def test_generic_backend_registry_is_explicit() -> None:
    registry = generic_bridge_backend_registry()

    assert tuple(registry) == (
        GENERIC_LOOP_BACKEND_ID,
        "upmem_sdk_hardware_generic_loop",
    )
    identity = registry[GENERIC_LOOP_BACKEND_ID]
    assert identity.kernel_family == "generic_loop_fallback"
    assert identity.external_command_capable is True
    assert identity.implemented is True
    assert registry["upmem_sdk_hardware_generic_loop"].implemented is True
