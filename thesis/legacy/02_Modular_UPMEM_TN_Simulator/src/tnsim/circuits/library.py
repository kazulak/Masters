from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CircuitOperation:
    gate: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Circuit:
    name: str
    n_qubits: int
    operations: tuple[CircuitOperation, ...]
    source: dict


def builtin_circuit(name: str, workload: dict) -> Circuit:
    if name == "bell_2q":
        return Circuit(
            name="bell_2q",
            n_qubits=2,
            operations=(
                CircuitOperation("h", (0,)),
                CircuitOperation("cx", (0, 1)),
            ),
            source={"kind": "builtin_circuit", "name": name},
        )

    if name == "ghz_4q":
        return _ghz_chain(4, name)

    if name == "ghz_chain":
        n_qubits = int(workload.get("n_qubits", 4))
        if n_qubits < 2:
            raise ValueError("ghz_chain requires n_qubits >= 2")
        return _ghz_chain(n_qubits, name)

    raise ValueError(f"Unknown builtin circuit: {name}")


def load_circuit(workload: dict, root_dir: Path) -> Circuit:
    kind = workload.get("kind", "builtin_circuit")
    if kind == "builtin_circuit":
        name = str(workload["name"])
        return builtin_circuit(name, workload)

    if kind == "qasm_file":
        path = Path(workload["path"])
        if not path.is_absolute():
            path = root_dir / path
        return parse_openqasm2(path)

    raise ValueError(f"Unsupported workload kind: {kind}")


def gate_tensor(op: CircuitOperation) -> np.ndarray:
    matrix = gate_matrix(op.gate, op.params)
    n_wires = len(op.wires)
    tensor = matrix.reshape([2] * n_wires + [2] * n_wires)
    axes = list(range(n_wires, 2 * n_wires)) + list(range(n_wires))
    return np.transpose(tensor, axes).astype(np.complex128, copy=False)


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
        return np.array(
            [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]],
            dtype=np.complex128,
        )
    if gate in {"cx", "cnot"}:
        return np.array(
            [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
                [zero, zero, one, zero],
            ],
            dtype=np.complex128,
        )
    if gate == "cz":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if gate == "swap":
        return np.array(
            [
                [one, zero, zero, zero],
                [zero, zero, one, zero],
                [zero, one, zero, zero],
                [zero, zero, zero, one],
            ],
            dtype=np.complex128,
        )

    raise ValueError(f"Unsupported gate: {gate}")


def parse_openqasm2(path: Path) -> Circuit:
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

    return Circuit(
        name=path.stem,
        n_qubits=n_qubits,
        operations=tuple(operations),
        source={"kind": "qasm_file", "path": str(path)},
    )


def _ghz_chain(n_qubits: int, name: str) -> Circuit:
    operations = [CircuitOperation("h", (0,))]
    operations.extend(CircuitOperation("cx", (i, i + 1)) for i in range(n_qubits - 1))
    return Circuit(
        name=name if name != "ghz_chain" else f"ghz_{n_qubits}q",
        n_qubits=n_qubits,
        operations=tuple(operations),
        source={"kind": "builtin_circuit", "name": name, "n_qubits": n_qubits},
    )


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

