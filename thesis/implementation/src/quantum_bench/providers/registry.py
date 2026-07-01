from __future__ import annotations

from pathlib import Path

from quantum_bench.providers.base import ExecutionRoute
from quantum_bench.providers.exact_tn import (
    CpuTnEinsumExactRoute,
    QuimbTnExactRoute,
    UpmemDenseInt8PlaceholderRoute,
    UpmemTnSdkSimulatorQuantizedRoute,
)
from quantum_bench.providers.full_state import QuestCpuFullStateBenchmarkRoute, QuestCpuFullStateExactRoute, QuestGpuFullStateExactRoute


def route_registry(root_dir: Path) -> dict[str, ExecutionRoute]:
    routes: list[ExecutionRoute] = [
        CpuTnEinsumExactRoute(),
        QuimbTnExactRoute(),
        QuestCpuFullStateBenchmarkRoute(root_dir),
        QuestCpuFullStateExactRoute(root_dir),
        QuestGpuFullStateExactRoute(root_dir),
        UpmemDenseInt8PlaceholderRoute(),
        UpmemTnSdkSimulatorQuantizedRoute(),
    ]
    return {route.name: route for route in routes}
