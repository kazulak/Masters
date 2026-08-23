from __future__ import annotations

import inspect
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from quantum_bench.bench import m5_circuit_study


ROOT = Path(__file__).parents[1]


def test_active_imports_do_not_load_historical_taskgraph() -> None:
    script = """
import sys
import quantum_bench.tn.graph
import quantum_bench.execution.contracts
import quantum_bench.execution.compiler
assert 'quantum_bench.upmem.tiling' not in sys.modules
assert 'quantum_bench.upmem.protocol' not in sys.modules
import quantum_bench.targets.upmem.v4_executor
import quantum_bench.bench.m5_circuit_study
assert 'quantum_bench.tn.task_graph' not in sys.modules
assert 'quantum_bench.whole_circuit' not in sys.modules
assert 'quantum_bench.whole_circuit.core' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_active_m5_study_plans_once_then_uses_functional_dag(
    monkeypatch,
) -> None:
    config = m5_circuit_study.load_study_config(
        ROOT / "configs/suites/m5_circuit_smoke.yml"
    )
    calls = 0
    original = m5_circuit_study.plan_contractions

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(m5_circuit_study, "plan_contractions", counted)
    plans = m5_circuit_study._build_plans(ROOT, config)

    assert calls == len(plans)
    assert all(plan.dag.nodes for plan in plans)
    reversed_dag = replace(plans[0].dag, nodes=tuple(reversed(plans[0].dag.nodes)))
    resources = m5_circuit_study._estimate_resources(
        reversed_dag, config["resource_limits"]
    )
    assert resources["dag_node_count"] == len(reversed_dag.nodes)
    source = inspect.getsource(m5_circuit_study)
    assert "WholeGraphExecutor" not in source
    assert "plan_task_graph_with_config" not in source
    assert "TaskGraph" not in source
    assert "ContractionTask" not in source
    assert "TensorNetworkValue" not in source
    assert "TensorValue" not in source
    assert "materialize_task_graph_from_planner_result" not in source
    assert all(
        not hasattr(plan, "graph") and not hasattr(plan, "network") for plan in plans
    )
