from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from quantum_bench.core.records import CircuitOperation, CircuitSpec


def load_circuit(case: dict, root_dir: Path) -> CircuitSpec:
    circuit = case.get("circuit", {})
    kind = circuit.get("kind", "builtin")
    if kind == "synthetic_pressure":
        raise ValueError(
            "synthetic_pressure workloads are analysis-only; use "
            "benchmark-matrix-report or upmem-multi-dpu-assignment"
        )
    if kind == "planner_motif":
        raise ValueError(
            "planner_motif workloads are modeled-planning-only; use "
            "compare-planners"
        )
    if kind == "builtin":
        return builtin_circuit(str(circuit["name"]), circuit)
    if kind == "quest_compatible":
        return quest_compatible_circuit(str(circuit["name"]), circuit)
    if kind == "qasm_file":
        path = Path(circuit["path"])
        if not path.is_absolute():
            path = root_dir / path
        return parse_openqasm2(path)
    raise ValueError(f"Unsupported circuit kind: {kind}")


def quest_compatible_circuit(name: str, params: dict | None = None) -> CircuitSpec:
    params = params or {}
    lowered = name.lower()
    n_qubits = int(params.get("n_qubits", params.get("qubits", 0)) or 0)
    depth = int(params.get("depth", 1))
    repeat_layers = int(params.get("repeat_layers", 1) or 1)
    if repeat_layers < 1:
        raise ValueError("quest-compatible repeat_layers must be >= 1")

    if lowered == "qrng":
        n = n_qubits or 4
        ops = tuple(CircuitOperation("h", (wire,)) for wire in range(n))
        return CircuitSpec(f"quest_qrng_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if lowered in {"bb84", "bb_n"}:
        n = n_qubits or 4
        ops: list[CircuitOperation] = []
        for wire in range(n):
            ops.append(CircuitOperation("h", (wire,)))
            ops.append(CircuitOperation("x", (wire,)))
        return CircuitSpec(f"quest_bb84_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if lowered in {"bv", "bernstein_vazirani"}:
        n = n_qubits or 4
        target = n - 1
        ops = [CircuitOperation("h", (wire,)) for wire in range(n)]
        ops.extend(CircuitOperation("cx", (control, target)) for control in range(target))
        ops.append(CircuitOperation("x", (target,)))
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(target))
        return CircuitSpec(f"quest_bv_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if lowered in {"edc", "dense_coding"}:
        n = max(n_qubits or 2, 2)
        ops = [CircuitOperation("h", (wire,)) for wire in range(n)]
        ops.extend(CircuitOperation("cx", (wire, wire + 1)) for wire in range(n - 1))
        ops.extend(CircuitOperation("cx", (wire, wire - 1)) for wire in range(n - 1, 0, -1))
        ops.extend(CircuitOperation("x", (wire,)) for wire in range(n))
        return CircuitSpec(f"quest_edc_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if lowered in {"xor", "parity"}:
        n = n_qubits or 4
        ops = tuple(CircuitOperation("cx", (wire, wire + 1)) for wire in range(n - 1))
        return CircuitSpec(f"quest_xor_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if lowered in {"hs", "hidden_shift"}:
        allocated = int(params.get("allocated_qubits", n_qubits or 4))
        if allocated % 2 != 0:
            raise ValueError("quest-compatible HS requires an even allocated qubit count")
        logical = allocated // 2
        ops = []
        for wire in range(logical):
            ops.extend(
                (
                    CircuitOperation("h", (wire,)),
                    CircuitOperation("x", (wire,)),
                    CircuitOperation("h", (wire,)),
                    CircuitOperation("cx", (wire, wire + logical)),
                    CircuitOperation("cx", (wire, wire + logical)),
                    CircuitOperation("h", (wire,)),
                    CircuitOperation("x", (wire,)),
                    CircuitOperation("h", (wire,)),
                )
            )
        return CircuitSpec(
            f"quest_hs_{allocated}q",
            allocated,
            _repeat_ops(ops, repeat_layers),
            _source_quest(name, logical_qubits=logical, allocated_qubits=allocated, depth=depth, repeat_layers=repeat_layers),
        )

    raise ValueError(f"Unknown quest-compatible circuit: {name}")


def builtin_circuit(name: str, params: dict | None = None) -> CircuitSpec:
    params = params or {}
    lowered = name.lower()
    n_qubits = int(params.get("n_qubits", params.get("qubits", 0)) or 0)
    depth = int(params.get("depth", 1))

    if lowered == "bell_2q":
        return CircuitSpec(name, 2, (CircuitOperation("h", (0,)), CircuitOperation("cx", (0, 1))), _source(name))

    if lowered in {"ghz_4q", "ghz_chain"}:
        return _ghz_chain(4 if lowered == "ghz_4q" else max(n_qubits, 2), name)

    if lowered == "qrng":
        n = n_qubits or 4
        ops = tuple(CircuitOperation("h", (wire,)) for wire in range(n))
        return CircuitSpec(f"qrng_{n}q", n, ops, _source(name, n_qubits=n))

    if lowered in {"quantization_stress", "quantization-stress", "quant_stress"}:
        n = n_qubits or 4
        if n < 2:
            raise ValueError("quantization_stress requires at least two qubits")
        repeat_layers = int(params.get("repeat_layers", max(depth, 2)))
        if repeat_layers < 1:
            raise ValueError("quantization_stress repeat_layers must be >= 1")
        # Keep the angles fixed across sizes so changes in the records are
        # attributable to the circuit size and not generated parameters.
        fixed_angles = (math.pi / 7, -math.pi / 5, math.pi / 3, -math.pi / 4, math.pi / 6, -math.pi / 8)
        ops: list[CircuitOperation] = []
        for layer in range(repeat_layers):
            ops.extend(CircuitOperation("h", (wire,)) for wire in range(n))
            ops.extend(CircuitOperation("rz", (wire,), (fixed_angles[wire % len(fixed_angles)],)) for wire in range(n))
            ops.extend(CircuitOperation("cx", (wire, wire + 1)) for wire in range(n - 1))
            if layer % 2 == 1:
                ops.extend(CircuitOperation("rz", (wire,), (fixed_angles[(wire + 2) % len(fixed_angles)],)) for wire in range(n))
        return CircuitSpec(
            f"quantization_stress_{n}q",
            n,
            tuple(ops),
            _source(
                name,
                n_qubits=n,
                repeat_layers=repeat_layers,
                deterministic_unitary=True,
                measurement_mode="pre_measurement_statevector",
            ),
        )

    if lowered in {"bv", "bernstein_vazirani"}:
        n = n_qubits or 4
        ops: list[CircuitOperation] = [CircuitOperation("x", (n - 1,))]
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(n))
        ops.extend(CircuitOperation("cx", (wire, n - 1)) for wire in range(n - 1) if wire % 2 == 0)
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(n - 1))
        return CircuitSpec(f"bv_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if lowered in {"xor", "parity"}:
        n = n_qubits or 4
        ops = [CircuitOperation("h", (0,))]
        ops.extend(CircuitOperation("cx", (wire, n - 1)) for wire in range(n - 1))
        return CircuitSpec(f"xor_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if lowered in {"bb84", "bb_n"}:
        n = n_qubits or 4
        ops = []
        for wire in range(n):
            ops.append(CircuitOperation("h" if wire % 2 else "x", (wire,)))
            if wire % 3 == 0:
                ops.append(CircuitOperation("h", (wire,)))
        return CircuitSpec(f"bb84_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if lowered in {"edc", "dense_coding"}:
        n = max(n_qubits or 2, 2)
        ops = [CircuitOperation("h", (0,)), CircuitOperation("cx", (0, 1)), CircuitOperation("z", (0,))]
        if n > 2:
            ops.extend(CircuitOperation("cx", (wire - 1, wire)) for wire in range(2, n))
        return CircuitSpec(f"edc_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if lowered in {"hs", "hidden_shift"}:
        logical = int(params.get("logical_qubits", max(1, n_qubits // 2 if n_qubits else 2)))
        allocated = int(params.get("allocated_qubits", n_qubits or 2 * logical))
        ops = []
        for layer in range(max(depth, 1)):
            ops.extend(CircuitOperation("h", (wire,)) for wire in range(allocated))
            ops.extend(CircuitOperation("cz", (wire, wire + 1)) for wire in range(0, allocated - 1, 2))
            if layer % 2 == 0:
                ops.extend(CircuitOperation("x", (wire,)) for wire in range(logical))
        return CircuitSpec(f"hs_{allocated}q", allocated, tuple(ops), _source(name, logical_qubits=logical, allocated_qubits=allocated))

    raise ValueError(f"Unknown builtin circuit: {name}")


def gate_matrix(gate: str, params: Iterable[float] = ()) -> np.ndarray:
    gate = gate.lower()
    params = tuple(params)
    one = np.complex128(1.0)
    zero = np.complex128(0.0)
    if gate == "i":
        return np.eye(2, dtype=np.complex128)
    if gate == "h":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    if gate == "x":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if gate == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if gate == "z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if gate == "s":
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if gate == "t":
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    if gate == "rz":
        (theta,) = params
        return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=np.complex128)
    if gate == "ry":
        (theta,) = params
        half_theta = 0.5 * theta
        return np.array(
            [
                [math.cos(half_theta), -math.sin(half_theta)],
                [math.sin(half_theta), math.cos(half_theta)],
            ],
            dtype=np.complex128,
        )
    if gate in {"cx", "cnot"}:
        return np.array([[one, zero, zero, zero], [zero, one, zero, zero], [zero, zero, zero, one], [zero, zero, one, zero]], dtype=np.complex128)
    if gate == "cz":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if gate == "swap":
        return np.array([[one, zero, zero, zero], [zero, zero, one, zero], [zero, one, zero, zero], [zero, zero, zero, one]], dtype=np.complex128)
    raise ValueError(f"Unsupported gate: {gate}")


def gate_tensor(op: CircuitOperation) -> np.ndarray:
    matrix = gate_matrix(op.gate, op.params)
    n_wires = len(op.wires)
    tensor = matrix.reshape([2] * n_wires + [2] * n_wires)
    axes = list(range(n_wires, 2 * n_wires)) + list(range(n_wires))
    return np.transpose(tensor, axes).astype(np.complex128, copy=False)


def parse_openqasm2(path: Path) -> CircuitSpec:
    n_qubits = None
    operations: list[CircuitOperation] = []
    qreg_re = re.compile(r"qreg\s+q\[(\d+)\]\s*;")
    gate_re = re.compile(r"([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s+(.+);")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith("OPENQASM") or line.startswith("include"):
            continue
        qreg_match = qreg_re.fullmatch(line)
        if qreg_match:
            n_qubits = int(qreg_match.group(1))
            continue
        match = gate_re.fullmatch(line)
        if not match:
            raise ValueError(f"Unsupported QASM line in {path}: {raw_line}")
        gate = match.group(1).lower()
        params = _parse_params(match.group(2))
        wires = tuple(int(value) for value in re.findall(r"q\[(\d+)\]", match.group(3)))
        operations.append(CircuitOperation(gate, wires, params))
    if n_qubits is None:
        raise ValueError(f"No qreg declaration found in {path}")
    return CircuitSpec(path.stem, n_qubits, tuple(operations), {"kind": "qasm_file", "path": str(path)})


def gate_structure(gate: str) -> str:
    if gate.lower() in {"z", "s", "t", "rz", "cz"}:
        return "diagonal"
    if gate.lower() in {"x", "cx", "cnot", "swap"}:
        return "permutation"
    return "dense"


def manifest(circuit: CircuitSpec) -> dict:
    one_q = sum(1 for op in circuit.operations if len(op.wires) == 1)
    two_q = sum(1 for op in circuit.operations if len(op.wires) == 2)
    return {
        "name": circuit.name,
        "n_qubits": circuit.n_qubits,
        "depth_proxy": len(circuit.operations),
        "gate_counts": {"1q": one_q, "2q": two_q, "total": len(circuit.operations)},
        "gate_set": sorted({op.gate for op in circuit.operations}),
        "source": circuit.source,
        "workload_kind": "workload-shape reproduction" if circuit.source.get("name", "").lower() in {"bb84", "bb_n", "hs", "edc", "xor", "bv", "qrng"} else "textbook/smoke circuit",
    }


def _ghz_chain(n_qubits: int, name: str) -> CircuitSpec:
    operations = [CircuitOperation("h", (0,))]
    operations.extend(CircuitOperation("cx", (i, i + 1)) for i in range(n_qubits - 1))
    return CircuitSpec(f"ghz_{n_qubits}q", n_qubits, tuple(operations), _source(name, n_qubits=n_qubits))


def _source(name: str, **extra: object) -> dict:
    return {"kind": "builtin", "name": name, **extra}


def _source_quest(name: str, **extra: object) -> dict:
    return {
        "kind": "quest_compatible",
        "name": name,
        "deterministic_unitary": True,
        "measurement_mode": "pre_measurement_statevector",
        **extra,
    }


def _repeat_ops(ops: Iterable[CircuitOperation], repeat_layers: int) -> tuple[CircuitOperation, ...]:
    base = tuple(ops)
    if repeat_layers == 1:
        return base
    return tuple(op for _ in range(repeat_layers) for op in base)


def _parse_params(raw: str | None) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(_eval_float_expr(item.strip()) for item in raw.split(","))


def _eval_float_expr(raw: str) -> float:
    tree = ast.parse(raw, mode="eval")
    return float(_eval_ast_node(tree.body, raw))


def _eval_ast_node(node: ast.AST, raw: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast_node(node.operand, raw)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_ast_node(node.left, raw)
        right = _eval_ast_node(node.right, raw)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right
    raise ValueError(f"Unsupported numeric expression in QASM parameter: {raw}")
