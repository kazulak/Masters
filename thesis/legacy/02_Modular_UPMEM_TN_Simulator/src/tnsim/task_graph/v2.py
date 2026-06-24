from __future__ import annotations

from pathlib import Path
import os
import platform
import sys
import time

import numpy as np
import opt_einsum as oe

from tnsim import __version__
from tnsim.core.model import TensorNetwork, TensorValue
from tnsim.core.utils import density, index_symbols, label_dim, shape_product
from tnsim.dispatch import KNOWN_ROUTES, dispatch_task


TASK_GRAPH_SCHEMA = "task_graph_v2_stage1a-0.1"


def plan_task_graph(network: TensorNetwork, config: dict, root_dir: Path) -> tuple[dict, list[TensorValue], float]:
    start = time.perf_counter()
    planner = config["planner"]
    if planner["engine"] != "opt_einsum":
        raise ValueError("Stage 1A supports only planner.engine=opt_einsum")

    path, path_info = oe.contract_path(
        network.einsum_expression,
        *(tensor.array for tensor in network.tensors),
        optimize=planner["optimize"],
    )
    active = [
        TensorValue(tensor.id, tensor.labels, np.empty(tensor.array.shape, dtype=np.complex128), tensor.structure)
        for tensor in network.tensors
    ]
    tensor_records = [tensor_metadata(tensor, tensor.produced_by) for tensor in network.tensors]
    produced_by: dict[str, str | None] = {tensor.id: tensor.produced_by for tensor in network.tensors}
    symbols = index_symbols([tensor.labels for tensor in network.tensors], network.output_labels)
    tasks = []
    route_decisions = []

    for step_index, contraction in enumerate(path):
        if len(contraction) != 2:
            raise ValueError(
                "Stage 1A requires pairwise contraction paths; "
                f"opt_einsum returned step {contraction}"
            )

        i, j = sorted(contraction)
        left = active[i]
        right = active[j]
        contracted = tuple(
            label for label in left.labels if label in set(right.labels) and label not in network.output_labels
        )
        free_left = tuple(label for label in left.labels if label not in contracted)
        free_right = tuple(label for label in right.labels if label not in contracted)
        output_labels = free_left + free_right
        output_shape = tuple(label_dim(label, left, right) for label in output_labels)
        output_id = f"result_{step_index}"
        task_id = f"task_{step_index}"
        expression = (
            "".join(symbols[label] for label in left.labels)
            + ","
            + "".join(symbols[label] for label in right.labels)
            + "->"
            + "".join(symbols[label] for label in output_labels)
        )
        dependencies = [
            dep for dep in (produced_by.get(left.id), produced_by.get(right.id)) if dep and dep.startswith("task_")
        ]
        m = shape_product([label_dim(label, left, right) for label in free_left])
        k = shape_product([label_dim(label, left, right) for label in contracted])
        n = shape_product([label_dim(label, left, right) for label in free_right])
        task = {
            "id": task_id,
            "op_kind": "contraction",
            "input_tensor_ids": [left.id, right.id],
            "output_tensor_id": output_id,
            "dependencies": dependencies,
            "index_expression": expression,
            "input_shapes": [list(left.array.shape), list(right.array.shape)],
            "output_shape": list(output_shape),
            "labels": {
                "left": list(left.labels),
                "right": list(right.labels),
                "free_left": list(free_left),
                "contracted": list(contracted),
                "free_right": list(free_right),
                "output": list(output_labels),
            },
            "gemm_shape": {"m": m, "k": k, "n": n},
            "structure": "dense",
            "density_estimate": 1.0,
            "nnz_estimate": int(shape_product(output_shape)),
            "candidate_routes": [],
            "selected_route": None,
            "selected_data_format": config["execution"]["data_format"],
            "rejected_routes_with_reasons": [],
            "estimated_cost": {
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
                "host_tensor_read_bytes": int(left.array.nbytes + right.array.nbytes),
                "host_tensor_write_bytes": int(np.prod(output_shape, dtype=np.int64) * np.dtype(np.complex128).itemsize),
                "floating_ops_estimate": int(8 * m * k * n),
                "preparation_cost_seconds": None,
                "conversion_cost_seconds": 0.0,
                "reduction_cost_seconds": 0.0,
                "quantization_error": 0.0,
            },
            "validation_policy": {
                "reference_route": "cpu_reference",
                "metrics": ["max_abs_error", "max_rel_error", "norm_drift", "fidelity"],
            },
        }
        decision = dispatch_task(task, config)
        task.update(
            {
                "candidate_routes": decision["candidate_routes"],
                "selected_route": decision["selected_route"],
                "rejected_routes_with_reasons": decision["rejected_routes_with_reasons"],
                "route_reason": decision["reason"],
            }
        )
        route_decisions.append(decision)
        tasks.append(task)

        output_tensor = TensorValue(output_id, output_labels, np.empty(output_shape, dtype=np.complex128), "dense", task_id)
        tensor_records.append(
            tensor_metadata(
                output_tensor,
                task_id,
                density_estimate=1.0,
                nnz_estimate=int(shape_product(output_shape)),
            )
        )
        produced_by[output_id] = task_id
        active.pop(j)
        active.pop(i)
        active.insert(i, output_tensor)

    final_tensor_id = active[0].id if active else None
    graph = {
        "schema_version": TASK_GRAPH_SCHEMA,
        "meta": {
            "experiment_id": config["experiment"]["id"],
            "tnsim_version": __version__,
            "source": network.circuit.source,
            "circuit": {
                "name": network.circuit.name,
                "n_qubits": network.circuit.n_qubits,
                "n_operations": len(network.circuit.operations),
            },
            "planner": {
                "name": "opt_einsum",
                "version": getattr(oe, "__version__", "unknown"),
                "optimize": planner["optimize"],
                "path_length": len(path),
                "path": [list(step) for step in path],
                "path_info": str(path_info),
            },
            "root_dir": str(root_dir),
            "full_einsum_expression": network.einsum_expression,
            "output_labels": list(network.output_labels),
            "final_tensor_id": final_tensor_id,
        },
        "hardware": host_hardware_record(),
        "ablation": ablation_record(config),
        "tensors": tensor_records,
        "tasks": tasks,
        "route_decisions": route_decisions,
        "profiles": [],
        "validation": [],
    }
    return graph, active, time.perf_counter() - start


def tensor_metadata(
    tensor: TensorValue,
    produced_by: str | None,
    density_estimate: float | None = None,
    nnz_estimate: int | None = None,
) -> dict:
    record_density = density(tensor.array) if density_estimate is None else density_estimate
    nnz = int(np.count_nonzero(tensor.array)) if nnz_estimate is None else nnz_estimate
    return {
        "id": tensor.id,
        "shape": list(tensor.array.shape),
        "labels": list(tensor.labels),
        "logical_dtype": "complex_f64",
        "structure": tensor.structure,
        "density_estimate": record_density,
        "nnz_estimate": nnz,
        "format": {
            "name": "complex_f64_host",
            "accumulator": "complex_f64",
            "scale_scope": "none",
            "metadata": {},
        },
        "storage": {
            "location": "host",
            "kind": "in_memory_numpy",
            "path": None,
            "byte_length": int(tensor.array.nbytes),
        },
        "lifetime": {
            "produced_by": produced_by,
            "consumed_by": [],
        },
    }


def ablation_record(config: dict) -> dict:
    routes = config["execution"]["routes"]
    enabled = list(routes.get("enabled", []))
    disabled = list(routes.get("disabled", []))
    for route in KNOWN_ROUTES:
        if route not in enabled and route not in disabled:
            disabled.append(route)
    return {
        "enabled_routes": enabled,
        "disabled_routes": disabled,
        "forced_route": routes.get("forced"),
        "format_policy": config["execution"]["data_format"]["name"],
        "cost_model": "rules_v0_static_cpu",
    }


def host_hardware_record() -> dict:
    return {
        "profile_id": "local_host_cpu",
        "host_cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "upmem_sdk": os.environ.get("UPMEM_HOME", "not_used_by_cpu_stage_1a"),
        "rank_count": 0,
        "dpu_count": 0,
        "notes": ["CPU shared-path benchmark; no DPU hardware used."],
    }

