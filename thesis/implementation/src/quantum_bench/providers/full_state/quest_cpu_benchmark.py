from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from quantum_bench.core.records import (
    BenchmarkContext,
    ExecutionProfile,
    RouteCapabilities,
    RouteEstimate,
    RouteIdentity,
    RouteOutput,
    RouteProbe,
    RouteResult,
    TaskGraph,
)
from quantum_bench.tn.network import TensorNetworkValue


class QuestCpuFullStateBenchmarkRoute:
    name = "quest_cpu_full_state_benchmark"
    backend_family = "cpu_full_state"
    identity = RouteIdentity(
        route_id=name,
        display_name="QuEST CPU full-state benchmark",
        role="baseline",
        simulation_method="full_state_vector",
        kernel_family="external_full_state",
        hardware_target="cpu",
        execution_mode="external_process",
        output_contract="metrics_only",
        validation_mode="benchmark_only",
    )

    def __init__(self, root_dir: Path):
        self.root = root_dir / "native" / "quest_cpu"
        self.runner = self.root / "bin" / "quest_runner"

    def probe(self) -> RouteProbe:
        if not self.root.exists():
            return RouteProbe(self.name, False, f"QuEST implementation not found at {self.root}")
        if not self.runner.exists():
            return RouteProbe(self.name, False, f"QuEST runner not built at {self.runner}; run make in {self.root}")
        return RouteProbe(self.name, True, metadata={"runner": str(self.runner), "quest_root": str(self.root)})

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("BB84", "BV", "EDC", "HS", "QRNG", "XOR", "RANDOM"),
            can_return_output=False,
            can_measure_energy=True,
            metadata={"available": probe.available, "reason": probe.reason, **probe.metadata},
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        algo = self._algo(context)
        if algo is None:
            return False, "QuEST adapter supports only BB84, BV, EDC, HS, QRNG, XOR, RANDOM benchmark families"
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        allocated = graph.network.circuit.n_qubits
        bytes_estimate = (2**allocated) * 16
        return RouteEstimate(self.name, sum(task.estimated_flops for task in graph.tasks), bytes_estimate, bytes_estimate)

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": graph, "network": network}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        graph: TaskGraph = dict(prepared)["graph"]  # type: ignore[arg-type]
        algo = self._algo(context)
        if algo is None:
            return self._failed("unsupported QuEST algorithm")
        cmd = [str(self.runner), "--algo", algo, "--json"]
        if algo == "HS" and "logical_qubits" in context.case.get("circuit", {}):
            cmd.extend(["--logical-qubits", str(context.case["circuit"]["logical_qubits"])])
        else:
            cmd.extend(["--qubits", str(graph.network.circuit.n_qubits)])
        depth = context.case.get("circuit", {}).get("depth")
        if depth is not None:
            cmd.extend(["--depth", str(depth)])
        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, check=False)
        total_s = time.perf_counter() - start
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._failed((result.stderr or result.stdout or "QuEST did not emit JSON").strip(), total_s)
        quest_status = str(payload.get("status") or "")
        status = "passed" if result.returncode == 0 and quest_status in {"ok", "passed"} else "failed"
        return RouteResult(
            self.name,
            self.backend_family,
            status,
            RouteOutput(contract=self.identity.output_contract, metadata={"quest": payload}),
            ExecutionProfile(kernel_s=float(payload.get("time_s") or 0.0), total_s=total_s),
            payload.get("energy_joules"),
            str(payload.get("energy_source") or "unavailable"),
            None if status == "passed" else str(payload.get("error") or result.stderr or "QuEST failed"),
            {"quest": payload, "command": cmd},
        )

    def _algo(self, context: BenchmarkContext) -> str | None:
        name = str(context.case.get("circuit", {}).get("name", "")).upper()
        return name if name in {"BB84", "BV", "EDC", "HS", "QRNG", "XOR", "RANDOM"} else None

    def _failed(self, error: str, total_s: float = 0.0) -> RouteResult:
        return RouteResult(
            self.name,
            self.backend_family,
            "failed",
            RouteOutput(contract=self.identity.output_contract),
            ExecutionProfile(total_s=total_s),
            None,
            "unavailable",
            error,
        )
