from __future__ import annotations

from pathlib import Path

from tnsim.core.model import ExecutionRun, TensorNetwork
from .cpu_einsum import execute_cpu_task_graph
from .mvp_upmem import execute_mvp_upmem
from .quest_exact import execute_quest_exact


def execute_backend(graph: dict, network: TensorNetwork, config: dict, output_dir: Path) -> ExecutionRun:
    forced = config["execution"]["routes"].get("forced")
    if forced == "cpu_reference":
        return execute_cpu_task_graph(graph, network, config)
    if forced == "quest_exact_statevector":
        return execute_quest_exact(network, config, output_dir)
    if forced == "raw_upmem_dense":
        return execute_mvp_upmem(network, config, output_dir)
    raise ValueError(f"No executor implemented for forced route: {forced}")
