#!/usr/bin/env python3
import json
import math
import os
import sys

import numpy as np
import opt_einsum as oe

TILE_ROWS = 16
TILE_K = 256
TILE_N = 64
NR_TASKLETS = 8

MVP_SCHEMA_VERSION = "mvp-raw-dense-0.2"
RAW_ROUTE = "raw_upmem_dense"
REFERENCE_ROUTE = "cpu_reference"
DATA_FORMAT = {
    "name": "complex_i8_tile_scaled",
    "logical_dtype": "complex_f64",
    "accumulator": "int32",
    "scale_scope": "tile",
    "complex_layout": "split_real_imag",
}


def c_gemm_tile_input_bytes():
    return 3 * 4 + TILE_ROWS * TILE_K + TILE_K * TILE_N


def c_gemm_tile_output_bytes():
    return TILE_ROWS * TILE_N * 4


def density_estimate(tensor):
    if tensor.size == 0:
        return 0.0
    return float(np.count_nonzero(tensor) / tensor.size)


def fresh(counter):
    value = counter[0]
    counter[0] += 1
    return value


def build_tensor_network(circuit_fn, n_qubits):
    tensors = []
    index_labels = []
    counter = [0]

    wire_idx = {}
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    for wire in range(n_qubits):
        idx = fresh(counter)
        wire_idx[wire] = idx
        tensors.append(zero.copy())
        index_labels.append((idx,))

    for matrix, wires in circuit_fn():
        in_idxs = [wire_idx[w] for w in wires]
        out_idxs = [fresh(counter) for _ in wires]
        for wire, out_idx in zip(wires, out_idxs):
            wire_idx[wire] = out_idx

        k = len(wires)
        tensor = np.asarray(matrix, dtype=np.complex128).reshape([2] * (2 * k))
        tensors.append(tensor)
        index_labels.append(tuple(in_idxs + out_idxs))

    output_indices = tuple(wire_idx[w] for w in range(n_qubits))
    return tensors, index_labels, output_indices


def index_chars(index_labels, output_indices):
    all_idxs = sorted(set(i for lab in index_labels for i in lab) | set(output_indices))
    symbols = list(oe.get_symbol(i) for i in range(len(all_idxs)))
    return {idx: sym for idx, sym in zip(all_idxs, symbols)}


def prod2(labels):
    return 1 << len(labels)


def find_contraction_path(tensors, index_labels, output_indices):
    idx_to_char = index_chars(index_labels, output_indices)
    operand_subs = ["".join(idx_to_char[i] for i in lab) for lab in index_labels]
    out_sub = "".join(idx_to_char[i] for i in output_indices)
    einsum_str = ",".join(operand_subs) + "->" + out_sub

    path, _ = oe.contract_path(einsum_str, *tensors, optimize="optimal")
    steps = []
    current_labels = list(index_labels)
    current_tensors = list(tensors)

    for i, j in path:
        i, j = sorted((i, j))
        t_a = current_tensors[i]
        t_b = current_tensors[j]
        lab_a = tuple(current_labels[i])
        lab_b = tuple(current_labels[j])

        contracted = [x for x in lab_a if x in set(lab_b) and x not in output_indices]
        free_a = [x for x in lab_a if x not in contracted]
        free_b = [x for x in lab_b if x not in contracted]
        out_lab = tuple(free_a + free_b)

        m = prod2(free_a)
        k = prod2(contracted)
        n = prod2(free_b)
        needs_k_tiling = k > TILE_K

        sub_a = "".join(idx_to_char[x] for x in free_a + contracted)
        sub_b = "".join(idx_to_char[x] for x in contracted + free_b)
        sub_c = "".join(idx_to_char[x] for x in out_lab)

        steps.append({
            "einsum": f"{sub_a},{sub_b}->{sub_c}",
            "shape_A": list(t_a.shape),
            "shape_B": list(t_b.shape),
            "shape_out": [2] * len(out_lab),
            "labels_A": list(lab_a),
            "labels_B": list(lab_b),
            "labels_out": list(out_lab),
            "free_A": list(free_a),
            "contracted": list(contracted),
            "free_B": list(free_b),
            "m": m,
            "k": k,
            "n": n,
            "n_row_blocks": math.ceil(m / TILE_ROWS),
            "n_col_blocks": math.ceil(n / TILE_N),
            "operand_i": i,
            "operand_j": j,
            "needs_k_tiling": needs_k_tiling,
        })

        result_tensor = np.zeros([2] * len(out_lab), dtype=np.complex128)
        current_tensors.pop(j)
        current_labels.pop(j)
        current_tensors.pop(i)
        current_labels.pop(i)
        current_tensors.insert(i, result_tensor)
        current_labels.insert(i, out_lab)

    return steps


def write_task_graph(initial_tensors, index_labels, steps, output_path):
    initial_list = []
    offset = 0
    for idx, tensor in enumerate(initial_tensors):
        n_elem = int(tensor.size)
        density = density_estimate(tensor)
        initial_list.append({
            "id": idx,
            "key": f"tensor_{idx}",
            "shape": list(tensor.shape),
            "labels": list(index_labels[idx]),
            "logical_dtype": "complex_f64",
            "structure": "dense",
            "density_estimate": density,
            "nnz_estimate": int(np.count_nonzero(tensor)),
            "format": {
                "name": "complex_f64_host",
                "accumulator": "complex_f64",
                "scale_scope": "none",
            },
            "n_elements": n_elem,
            "storage": {
                "location": "host",
                "kind": "host_binary_split_real_imag_f64",
                "path": "data_exchange/tensor_data.bin",
                "offset_real_bytes": offset,
                "offset_imag_bytes": offset + n_elem * 8,
                "byte_length": n_elem * 8 * 2,
            },
            # Compatibility fields consumed by the C MVP host.
            "offset_real_bytes": offset,
            "offset_imag_bytes": offset + n_elem * 8,
        })
        offset += n_elem * 8 * 2

    tasks = []
    keys = [f"tensor_{i}" for i in range(len(initial_tensors))]
    for s, step in enumerate(steps):
        i = step["operand_i"]
        j = step["operand_j"]
        out_key = f"result_{s}"
        n_tiles = step["n_row_blocks"] * step["n_col_blocks"]
        gemm_invocations = n_tiles * 4
        h2d_bytes = gemm_invocations * c_gemm_tile_input_bytes()
        d2h_bytes = gemm_invocations * c_gemm_tile_output_bytes()
        task = {
            "task_id": s,
            "id": f"task_{s}",
            "op_kind": "contraction",
            "input_A_key": keys[i],
            "input_B_key": keys[j],
            "input_tensor_ids": [keys[i], keys[j]],
            "output_key": out_key,
            "output_tensor_id": out_key,
            "einsum": step["einsum"],
            "index_expression": step["einsum"],
            "shape_A": step["shape_A"],
            "shape_B": step["shape_B"],
            "shape_out": step["shape_out"],
            "input_shapes": [step["shape_A"], step["shape_B"]],
            "output_shape": step["shape_out"],
            "labels_A": step["labels_A"],
            "labels_B": step["labels_B"],
            "labels_out": step["labels_out"],
            "free_A": step["free_A"],
            "contracted": step["contracted"],
            "free_B": step["free_B"],
            "m": step["m"],
            "k": step["k"],
            "n": step["n"],
            "n_row_blocks": step["n_row_blocks"],
            "n_col_blocks": step["n_col_blocks"],
            "needs_k_tiling": step["needs_k_tiling"],
            "structure": "dense",
            "density_estimate": 1.0,
            "nnz_estimate": step["m"] * step["n"],
            "candidate_routes": [REFERENCE_ROUTE, RAW_ROUTE],
            "selected_route": RAW_ROUTE,
            "selected_data_format": DATA_FORMAT,
            "route_reason": "conference_mvp_validation_path_raw_upmem_dense",
            "rejected_routes_with_reasons": [
                {
                    "route": "simplepim_default",
                    "reason": "not_part_of_conference_mvp_path",
                },
                {
                    "route": "sparsep",
                    "reason": "task_classified_dense",
                },
                {
                    "route": "pidcomm_collective",
                    "reason": "single_dpu_no_collective_required",
                },
            ],
            "estimated_cost": {
                "host_to_dpu_bytes": h2d_bytes,
                "dpu_to_host_bytes": d2h_bytes,
                "mram_wram_bytes": h2d_bytes + d2h_bytes,
                "integer_ops": 4 * step["m"] * step["k"] * step["n"],
                "gemm_tile_invocations": gemm_invocations,
                "conversion_cost": "measured_by_host_execution_log",
                "reduction_cost": 0,
                "quantization_error": "measured_by_validation_record",
            },
            "tiling": {
                "tile_rows": TILE_ROWS,
                "tile_k": TILE_K,
                "tile_cols": TILE_N,
                "n_row_blocks": step["n_row_blocks"],
                "n_col_blocks": step["n_col_blocks"],
                "k_tiling": False,
                "estimated_wram_bytes": (
                    TILE_ROWS * TILE_K
                    + TILE_K * TILE_N
                    + TILE_ROWS * TILE_N * 4
                ),
            },
            "is_final": s == len(steps) - 1,
        }
        tasks.append(task)
        keys.pop(j)
        keys.pop(i)
        keys.insert(i, out_key)

    doc = {
        "schema_version": MVP_SCHEMA_VERSION,
        "meta": {
            "mvp_role": "raw_upmem_dense_baseline",
            "architecture_note": (
                "Fast conference MVP retained as the raw UPMEM dense validation "
                "path for TaskGraphV2 replay."
            ),
            "n_initial_tensors": len(initial_tensors),
            "n_tasks": len(steps),
            "tile_rows": TILE_ROWS,
            "tile_k": TILE_K,
            "tile_n": TILE_N,
            "nr_tasklets": NR_TASKLETS,
            "planner": {
                "name": "opt_einsum",
                "optimize": "optimal",
                "seed": 0,
            },
            "default_route": RAW_ROUTE,
            "reference_route": REFERENCE_ROUTE,
            "data_format": DATA_FORMAT,
            "route_contract": {
                "can_execute": "implicit_dense_compatibility_check",
                "estimate": "static_bytes_and_ops_in_task_graph",
                "prepare": "host_pack_quantize_and_dpu_symbol_buffers",
                "execute": "single_dpu_int8_gemm_tile_kernel",
            },
            "hardware_constraints": {
                "dpu_count": 1,
                "no_dynamic_inter_dpu_communication": True,
                "wram_safe_bytes": 48 * 1024,
            },
        },
        "initial_tensors": initial_list,
        "tasks": tasks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[generate_plan] Wrote {output_path}")


def write_tensor_binary(initial_tensors, output_path):
    with open(output_path, "wb") as f:
        for tensor in initial_tensors:
            flat = tensor.flatten()
            f.write(flat.real.astype("<f8").tobytes())
            f.write(flat.imag.astype("<f8").tobytes())
    print(f"[generate_plan] Wrote {output_path} ({os.path.getsize(output_path)} bytes)")


def validate_against_numpy(initial_tensors, index_labels, output_indices):
    idx_to_char = index_chars(index_labels, output_indices)
    operand_subs = ["".join(idx_to_char[i] for i in lab) for lab in index_labels]
    out_sub = "".join(idx_to_char[i] for i in output_indices)
    einsum_str = ",".join(operand_subs) + "->" + out_sub
    return np.einsum(einsum_str, *initial_tensors, optimize="optimal")


def circuit_bell_2q():
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex128)
    yield h, [0]
    yield cnot, [0, 1]


def circuit_ghz_4q():
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=np.complex128)
    yield h, [0]
    for ctrl in range(3):
        yield cnot, [ctrl, ctrl + 1]


def main():
    circuit_name = sys.argv[1] if len(sys.argv) > 1 else "bell_2q"
    if circuit_name == "bell_2q":
        circuit_fn = circuit_bell_2q
        n_qubits = 2
    elif circuit_name == "ghz_4q":
        circuit_fn = circuit_ghz_4q
        n_qubits = 4
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "data_exchange")
    os.makedirs(out_dir, exist_ok=True)

    tensors, labels, output_indices = build_tensor_network(circuit_fn, n_qubits)
    steps = find_contraction_path(tensors, labels, output_indices)

    write_task_graph(tensors, labels, steps, os.path.join(out_dir, "task_graph.json"))
    write_tensor_binary(tensors, os.path.join(out_dir, "tensor_data.bin"))

    ref = validate_against_numpy(tensors, labels, output_indices)
    np.save(os.path.join(out_dir, "reference_output.npy"), ref)
    print(f"[generate_plan] Reference amplitude shape: {ref.shape}")
    print(f"[generate_plan] Reference (flat): {ref.flatten()}")
    print(f"[generate_plan] Done. {len(steps)} contraction steps.")


if __name__ == "__main__":
    main()
