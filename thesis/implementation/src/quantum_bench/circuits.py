from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from quantum_bench.model import CircuitOperation, CircuitSpec


_BUILTIN_FAMILIES = {
    "bell_2q": "bell_2q",
    "ghz_4q": "ghz_4q",
    "ghz_chain": "ghz_chain",
    "qrng": "qrng",
    "quantization_stress": "quantization_stress",
    "quantization-stress": "quantization_stress",
    "quant_stress": "quantization_stress",
    "bv": "bv",
    "bernstein_vazirani": "bv",
    "xor": "xor",
    "parity": "xor",
    "bb84": "bb84",
    "bb_n": "bb84",
    "edc": "edc",
    "dense_coding": "edc",
    "hs": "hs",
    "hidden_shift": "hs",
}
_BUILTIN_PARAMETER_SCHEMAS = {
    "bell_2q": frozenset(),
    "ghz_4q": frozenset({"n_qubits", "qubits"}),
    "ghz_chain": frozenset({"n_qubits", "qubits"}),
    "qrng": frozenset({"n_qubits", "qubits"}),
    "quantization_stress": frozenset(
        {"n_qubits", "qubits", "depth", "repeat_layers"}
    ),
    "bv": frozenset({"n_qubits", "qubits"}),
    "xor": frozenset({"n_qubits", "qubits"}),
    "bb84": frozenset({"n_qubits", "qubits"}),
    "edc": frozenset({"n_qubits", "qubits"}),
    "hs": frozenset(
        {"n_qubits", "qubits", "depth", "logical_qubits", "allocated_qubits"}
    ),
}
_QUEST_FAMILIES = {
    "qrng": "qrng",
    "bb84": "bb84",
    "bb_n": "bb84",
    "bv": "bv",
    "bernstein_vazirani": "bv",
    "edc": "edc",
    "dense_coding": "edc",
    "xor": "xor",
    "parity": "xor",
    "hs": "hs",
    "hidden_shift": "hs",
}
_QUEST_PARAMETER_SCHEMAS = {
    family: frozenset({"n_qubits", "qubits", "repeat_layers"})
    for family in {"qrng", "bb84", "bv", "edc", "xor"}
}
_QUEST_PARAMETER_SCHEMAS["hs"] = frozenset(
    {
        "n_qubits",
        "qubits",
        "depth",
        "repeat_layers",
        "logical_qubits",
        "allocated_qubits",
    }
)


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
        return builtin_circuit(circuit["name"], circuit)
    if kind == "quest_compatible":
        return quest_compatible_circuit(circuit["name"], circuit)
    if kind == "qasm_file":
        path = Path(circuit["path"])
        if not path.is_absolute():
            path = root_dir / path
        return parse_openqasm2(path)
    raise ValueError(f"Unsupported circuit kind: {kind}")


def _resolve_family(name: object, aliases: dict[str, str], kind: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} circuit name must be a nonempty string")
    family = aliases.get(name.lower())
    if family is None:
        raise ValueError(f"Unknown {kind} circuit: {name}")
    return family


def _validate_generator_params(
    name: str,
    params: dict | None,
    *,
    kind: str,
    scientific_keys: frozenset[str],
    source_metadata_keys: frozenset[str],
) -> dict:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(f"{kind} circuit parameters must be a dictionary")
    if any(not isinstance(key, str) for key in params):
        raise ValueError(f"{kind} circuit parameter keys must be strings")

    config_metadata_keys = frozenset({"kind", "name", "path"})
    allowed_keys = scientific_keys | source_metadata_keys | config_metadata_keys
    unknown_keys = sorted(set(params) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown {kind} circuit parameter(s) for {name!r}: "
            f"{', '.join(unknown_keys)}"
        )

    if "kind" in params and params["kind"] != kind:
        raise ValueError(f"circuit source kind must be {kind!r}")
    if "name" in params and params["name"] != name:
        raise ValueError("circuit source name must match the requested circuit name")
    if "path" in params and params["path"] is not None:
        raise ValueError(f"circuit source path must be None for {kind}")
    if (
        "deterministic_unitary" in params
        and params["deterministic_unitary"] is not True
    ):
        raise ValueError("circuit source deterministic_unitary must be true")
    if (
        "measurement_mode" in params
        and params["measurement_mode"] != "pre_measurement_statevector"
    ):
        raise ValueError(
            "circuit source measurement_mode must be "
            "'pre_measurement_statevector'"
        )
    return params


def _optional_count_parameter(
    params: dict, key: str, *, minimum: int = 1
) -> int | None:
    if key not in params:
        return None
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"circuit parameter {key!r} must be a non-bool integer")
    if value < minimum:
        raise ValueError(f"circuit parameter {key!r} must be >= {minimum}")
    return value


def _count_parameter(
    params: dict, key: str, *, default: int, minimum: int = 1
) -> int:
    value = _optional_count_parameter(params, key, minimum=minimum)
    return default if value is None else value


def _optional_qubit_count(params: dict, *, minimum: int = 1) -> int | None:
    present = [key for key in ("n_qubits", "qubits") if key in params]
    if len(present) > 1:
        raise ValueError("circuit parameters n_qubits and qubits are aliases; use one")
    if not present:
        return None
    return _optional_count_parameter(params, present[0], minimum=minimum)


def _qubit_count(params: dict, *, default: int, minimum: int = 1) -> int:
    value = _optional_qubit_count(params, minimum=minimum)
    return default if value is None else value


def quest_compatible_circuit(name: str, params: dict | None = None) -> CircuitSpec:
    family = _resolve_family(name, _QUEST_FAMILIES, "quest-compatible")
    params = _validate_generator_params(
        name,
        params,
        kind="quest_compatible",
        scientific_keys=_QUEST_PARAMETER_SCHEMAS[family],
        source_metadata_keys=frozenset(
            {"deterministic_unitary", "measurement_mode"}
        ),
    )
    repeat_layers = _count_parameter(params, "repeat_layers", default=1)

    if family == "qrng":
        n = _qubit_count(params, default=4)
        ops = tuple(CircuitOperation("h", (wire,)) for wire in range(n))
        return CircuitSpec(f"quest_qrng_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if family == "bb84":
        n = _qubit_count(params, default=4)
        ops: list[CircuitOperation] = []
        for wire in range(n):
            ops.append(CircuitOperation("h", (wire,)))
            ops.append(CircuitOperation("x", (wire,)))
        return CircuitSpec(f"quest_bb84_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if family == "bv":
        n = _qubit_count(params, default=4)
        target = n - 1
        ops = [CircuitOperation("h", (wire,)) for wire in range(n)]
        ops.extend(CircuitOperation("cx", (control, target)) for control in range(target))
        ops.append(CircuitOperation("x", (target,)))
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(target))
        return CircuitSpec(f"quest_bv_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if family == "edc":
        n = _qubit_count(params, default=2, minimum=2)
        ops = [CircuitOperation("h", (wire,)) for wire in range(n)]
        ops.extend(CircuitOperation("cx", (wire, wire + 1)) for wire in range(n - 1))
        ops.extend(CircuitOperation("cx", (wire, wire - 1)) for wire in range(n - 1, 0, -1))
        ops.extend(CircuitOperation("x", (wire,)) for wire in range(n))
        return CircuitSpec(f"quest_edc_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if family == "xor":
        n = _qubit_count(params, default=4)
        ops = tuple(CircuitOperation("cx", (wire, wire + 1)) for wire in range(n - 1))
        return CircuitSpec(f"quest_xor_{n}q", n, _repeat_ops(ops, repeat_layers), _source_quest(name, n_qubits=n, repeat_layers=repeat_layers))

    if family == "hs":
        n_qubits = _optional_qubit_count(params)
        allocated = _optional_count_parameter(params, "allocated_qubits")
        logical = _optional_count_parameter(params, "logical_qubits")
        if n_qubits is not None and allocated is not None and n_qubits != allocated:
            raise ValueError(
                "quest-compatible HS n_qubits and allocated_qubits must match"
            )
        if logical is not None and allocated is None and n_qubits is None:
            raise ValueError(
                "quest-compatible HS logical_qubits requires allocated_qubits"
            )
        allocated = allocated if allocated is not None else n_qubits or 4
        if allocated % 2 != 0:
            raise ValueError("quest-compatible HS requires an even allocated qubit count")
        expected_logical = allocated // 2
        if logical is not None and logical != expected_logical:
            raise ValueError(
                "quest-compatible HS logical_qubits must be half allocated_qubits"
            )
        logical = expected_logical
        depth = _count_parameter(params, "depth", default=1)
        if depth != 1:
            raise ValueError("quest-compatible HS depth must be 1")
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

    raise AssertionError(f"Unhandled quest-compatible circuit family: {family}")


def builtin_circuit(name: str, params: dict | None = None) -> CircuitSpec:
    family = _resolve_family(name, _BUILTIN_FAMILIES, "builtin")
    source_metadata_keys = (
        frozenset({"deterministic_unitary", "measurement_mode"})
        if family == "quantization_stress"
        else frozenset()
    )
    params = _validate_generator_params(
        name,
        params,
        kind="builtin",
        scientific_keys=_BUILTIN_PARAMETER_SCHEMAS[family],
        source_metadata_keys=source_metadata_keys,
    )

    if family == "bell_2q":
        return CircuitSpec(name, 2, (CircuitOperation("h", (0,)), CircuitOperation("cx", (0, 1))), _source(name))

    if family == "ghz_4q":
        if _qubit_count(params, default=4) != 4:
            raise ValueError("ghz_4q requires exactly four qubits")
        return _ghz_chain(4, name)

    if family == "ghz_chain":
        return _ghz_chain(_qubit_count(params, default=2, minimum=2), name)

    if family == "qrng":
        n = _qubit_count(params, default=4)
        ops = tuple(CircuitOperation("h", (wire,)) for wire in range(n))
        return CircuitSpec(f"qrng_{n}q", n, ops, _source(name, n_qubits=n))

    if family == "quantization_stress":
        n = _qubit_count(params, default=4, minimum=2)
        depth = _count_parameter(params, "depth", default=1)
        repeat_layers = _count_parameter(
            params, "repeat_layers", default=max(depth, 2)
        )
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

    if family == "bv":
        n = _qubit_count(params, default=4)
        ops: list[CircuitOperation] = [CircuitOperation("x", (n - 1,))]
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(n))
        ops.extend(CircuitOperation("cx", (wire, n - 1)) for wire in range(n - 1) if wire % 2 == 0)
        ops.extend(CircuitOperation("h", (wire,)) for wire in range(n - 1))
        return CircuitSpec(f"bv_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if family == "xor":
        n = _qubit_count(params, default=4)
        ops = [CircuitOperation("h", (0,))]
        ops.extend(CircuitOperation("cx", (wire, n - 1)) for wire in range(n - 1))
        return CircuitSpec(f"xor_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if family == "bb84":
        n = _qubit_count(params, default=4)
        ops = []
        for wire in range(n):
            ops.append(CircuitOperation("h" if wire % 2 else "x", (wire,)))
            if wire % 3 == 0:
                ops.append(CircuitOperation("h", (wire,)))
        return CircuitSpec(f"bb84_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if family == "edc":
        n = _qubit_count(params, default=2, minimum=2)
        ops = [CircuitOperation("h", (0,)), CircuitOperation("cx", (0, 1)), CircuitOperation("z", (0,))]
        if n > 2:
            ops.extend(CircuitOperation("cx", (wire - 1, wire)) for wire in range(2, n))
        return CircuitSpec(f"edc_{n}q", n, tuple(ops), _source(name, n_qubits=n))

    if family == "hs":
        n_qubits = _optional_qubit_count(params)
        logical = _optional_count_parameter(params, "logical_qubits")
        allocated = _optional_count_parameter(params, "allocated_qubits")
        if n_qubits is not None and allocated is not None and n_qubits != allocated:
            raise ValueError("builtin HS n_qubits and allocated_qubits must match")
        if allocated is None:
            allocated = n_qubits if n_qubits is not None else 2 * (logical or 2)
        if allocated % 2 != 0:
            raise ValueError("builtin HS requires an even allocated qubit count")
        expected_logical = allocated // 2
        if logical is not None and logical != expected_logical:
            raise ValueError("builtin HS logical_qubits must be half allocated_qubits")
        logical = expected_logical
        depth = _count_parameter(params, "depth", default=1)
        ops = []
        for layer in range(depth):
            ops.extend(CircuitOperation("h", (wire,)) for wire in range(allocated))
            ops.extend(CircuitOperation("cz", (wire, wire + 1)) for wire in range(0, allocated - 1, 2))
            if layer % 2 == 0:
                ops.extend(CircuitOperation("x", (wire,)) for wire in range(logical))
        return CircuitSpec(f"hs_{allocated}q", allocated, tuple(ops), _source(name, logical_qubits=logical, allocated_qubits=allocated))

    raise AssertionError(f"Unhandled builtin circuit family: {family}")


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
