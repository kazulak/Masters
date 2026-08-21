from __future__ import annotations

import inspect
from pathlib import Path

from quantum_bench.bench import m5_circuit_study


ROOT = Path(__file__).parents[1]


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
    source = inspect.getsource(m5_circuit_study)
    assert "WholeGraphExecutor" not in source
    assert "plan_task_graph_with_config" not in source
