from __future__ import annotations

import time

import numpy as np

from tnsim.core.model import ExecutionRun, TensorNetwork
from .energy import estimate_energy


def execute_cpu_task_graph(graph: dict, network: TensorNetwork, config: dict) -> ExecutionRun:
    arrays = {tensor.id: tensor.array for tensor in network.tensors}
    labels = {tensor.id: tensor.labels for tensor in network.tensors}
    profiles = []
    start_total = time.perf_counter()

    for task in graph["tasks"]:
        if task["selected_route"] != "cpu_reference":
            raise ValueError(f"CPU executor received non-CPU task route: {task['selected_route']}")
        left_id, right_id = task["input_tensor_ids"]
        output_id = task["output_tensor_id"]

        prepare_start = time.perf_counter()
        prepare_end = time.perf_counter()
        execute_start = time.perf_counter()
        result = np.einsum(task["index_expression"], arrays[left_id], arrays[right_id], optimize=False)
        execute_end = time.perf_counter()

        arrays[output_id] = result
        labels[output_id] = tuple(task["labels"]["output"])
        profiles.append(
            {
                "task_id": task["id"],
                "route": task["selected_route"],
                "data_format": task["selected_data_format"]["name"],
                "status": "ok",
                "prepare_seconds": prepare_end - prepare_start,
                "execute_seconds": execute_end - execute_start,
                "total_seconds": execute_end - prepare_start,
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
                "host_tensor_read_bytes": int(arrays[left_id].nbytes + arrays[right_id].nbytes),
                "host_tensor_write_bytes": int(result.nbytes),
                "output_tensor_id": output_id,
            }
        )

    final_id = graph["meta"]["final_tensor_id"]
    if final_id is None:
        raise ValueError("Task graph did not produce a final tensor")
    result = arrays[final_id]
    final_labels = labels[final_id]
    wanted_labels = tuple(graph["meta"]["output_labels"])
    if final_labels != wanted_labels:
        axes = [final_labels.index(label) for label in wanted_labels]
        result = np.transpose(result, axes)

    execution_seconds = time.perf_counter() - start_total
    energy_joules, energy_source, watts = estimate_energy(execution_seconds, config)
    for profile in profiles:
        profile["energy_joules"] = profile["total_seconds"] * watts
        profile["energy_source"] = energy_source
        profile["estimated_power_watts"] = watts

    return ExecutionRun(
        output=result,
        profiles=profiles,
        execution_seconds=execution_seconds,
        energy_joules=energy_joules,
        energy_source=energy_source,
        estimated_power_watts=watts,
    )

