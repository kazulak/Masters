from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.core.records import TensorNetworkSpec, TensorSpec
from quantum_bench.execution import (
    ExecutionPlan,
    NumericMode,
    RunContext,
    Target,
    UpmemCompileRequest,
    UpmemRuntimeResources,
    UpmemTopology,
    compile_execution,
    execution_plan_hash,
    run_upmem,
    UnsupportedExecution,
)
from quantum_bench.tn.graph import (
    ContractNode,
    SliceSpec,
    build_contraction_dag,
    contraction_dag_hash,
    apply_slicing,
)
from quantum_bench.targets.upmem.execution_plan_v4 import MAX_CONTRACTED


def _dag() -> object:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense", dtype="float64"),
        TensorSpec("b", (1, 2), (3, 2), "dense", dtype="float64"),
    )
    return build_contraction_dag(
        TensorNetworkSpec(None, tensors, (0, 2), "ab,bc->ac"),  # type: ignore[arg-type]
        ((0, 1),),
    )


def _chain_dag() -> object:
    tensors = (
        TensorSpec("a", (0, 1), (2, 3), "dense", dtype="float64"),
        TensorSpec("b", (1, 2), (3, 4), "dense", dtype="float64"),
        TensorSpec("c", (2, 3), (4, 2), "dense", dtype="float64"),
    )
    return build_contraction_dag(
        TensorNetworkSpec(None, tensors, (0, 3), "ab,bc,cd->ad"),  # type: ignore[arg-type]
        ((1, 2), (0, 1)),
    )


def _request(
    dag: object, mode: NumericMode = NumericMode.FLOAT32_REAL
) -> UpmemCompileRequest:
    return UpmemCompileRequest(
        contraction_dag_hash=contraction_dag_hash(dag),  # type: ignore[arg-type]
        numeric_mode=mode,
        topology=UpmemTopology(
            dpu_count=2,
            tasklets_per_dpu=1,
            rank_count=1,
        ),
    )


def _resources(tmp_path: Path) -> UpmemRuntimeResources:
    host = tmp_path / "host"
    dpu = tmp_path / "dpu"
    init = tmp_path / "init"
    host.write_bytes(b"host")
    dpu.write_bytes(b"dpu")
    init.write_bytes(b"init")
    host.chmod(0o755)
    return UpmemRuntimeResources(
        session_root=str(tmp_path / "session"),
        host_binary=str(host),
        dpu_binary=str(dpu),
        initialization_binary=str(init),
        rank_paths=("/dev/dpu_rank0",),
    )


class _FakeSession:
    def __init__(self, terminal_metadata: dict[str, object] | None = None) -> None:
        self.calls: list[str] = []
        self.closed = False
        self.terminal_metadata = terminal_metadata or {
            "backend_id": "fake_m5",
            "target_observed": "physical_hardware",
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": True,
            "hardware_release_verified": True,
            "hardware_release_confirmed": True,
            "requested_dpu_count": 2,
            "allocated_dpu_count": 2,
            "observed_rank_count": 1,
            "observed_tasklets_per_dpu": 1,
            "tasklets_per_dpu": 1,
            "profile": "m5_whole_circuit_v4_v1",
            "abi": "execution_plan_v4",
            "session_protocol": "persistent_rank_session_v1",
            "dispatch_mode": "bulk_set_synchronous_v1",
            "kernel_identity": "dpu_gemm_tile_v4",
            "execution_class": "physical_v4_output_tile",
            "graph_intermediate_placement": "host_managed",
            "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
            "native_identity_verified": True,
            "ready_verified": True,
            "physical_target_verified": True,
            "binary_identity_verified": True,
            "failure_stage": None,
        }

    def execute(
        self,
        node: ContractNode,
        left: np.ndarray,
        right: np.ndarray,
        *,
        node_plan: object | None = None,
    ) -> object:
        from quantum_bench.whole_circuit.core import EngineTaskResult

        self.calls.append(node.node_id)
        output = np.einsum(
            left,
            list(node.left.labels),
            right,
            list(node.right.labels),
            list(node.output_labels),
        )
        return EngineTaskResult(
            output=np.asarray(output),
            metadata={
                "backend_id": "fake_m5",
                "target_observed": "physical_hardware",
                "native_kernel_executed": True,
                "hardware_kernel_executed": True,
                "simulator_kernel_executed": False,
                "cpu_fallback_used": False,
                "physical_plan_consumed": node_plan is not None,
                "application_visible_h2d_bytes": 8,
                "application_visible_d2h_bytes": 4,
                "timing": {
                    "h2d_time_s": 1.0,
                    "kernel_time_s": 2.0,
                    "d2h_time_s": 3.0,
                    "total_route_time_s": 6.0,
                },
            },
        )

    def close(self) -> dict[str, object]:
        self.closed = True
        return self.terminal_metadata


def _inputs(*items: tuple[str, np.ndarray]) -> dict[str, np.ndarray]:
    return dict(items)


def test_compile_upmem_is_deterministic_and_portable() -> None:
    dag = _dag()
    first = compile_execution(dag, _request(dag))
    second = compile_execution(dag, _request(dag))

    assert isinstance(first, ExecutionPlan)
    assert first == second
    assert tuple(node.node_id for node in first.payload.node_plans) == ("contract_0",)
    node_plan = first.payload.node_plans[0]
    assert node_plan.canonical_shape == (1, 2, 3, 2)
    assert len(node_plan.work_units) == 1
    assert node_plan.work_units[0].stable_tile_id == "b_0:out_0_0:k_0"
    assert execution_plan_hash(first) == execution_plan_hash(second)


def test_compile_upmem_assigns_static_units_in_k_wave_rank_dpu_order() -> None:
    dag = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0, 1), (300, 300), "dense", dtype="float64"),
                TensorSpec("b", (1, 2), (300, 300), "dense", dtype="float64"),
            ),
            (0, 2),
            "ab,bc->ac",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )
    compiled = compile_execution(
        dag,
        replace(
            _request(dag),
            topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=1, rank_count=2),
        ),
    )
    assert isinstance(compiled, ExecutionPlan)
    units = compiled.payload.node_plans[0].work_units
    assert len(units) > 4
    assert [unit.k_start for unit in units] == sorted(unit.k_start for unit in units)
    for wave in {unit.wave for unit in units}:
        assigned = [unit for unit in units if unit.wave == wave]
        assert len(assigned) <= 4
        assert [(unit.logical_rank, unit.logical_dpu) for unit in assigned] == [
            (slot // 2, slot % 2) for slot in range(len(assigned))
        ]


def test_engine_plan_assignment_drives_rank_local_dpu_request() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
        M5WholeCircuitSession,
    )
    from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
        M5TileLimits,
        lower_binary_contraction,
    )

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    node = dag.nodes[0]
    assert isinstance(node, ContractNode)
    lowering = lower_binary_contraction(
        node,
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
        limits=M5TileLimits.float32(),
    )
    node_plan = compiled.payload.node_plans[0]
    unit = node_plan.work_units[0]
    moved = replace(unit, logical_dpu=1)
    moved_plan = replace(node_plan, work_units=(moved,))
    session = object.__new__(M5WholeCircuitSession)
    session.ranks = (SimpleNamespace(index=0, local_dpus=2),)

    waves, requests = session._requests_from_plan(node, lowering, moved_plan)

    assert waves[0][0].id == unit.stable_tile_id
    assert requests[0][0][0].index == 0
    assert requests[0][0][1] == [(waves[0][0], 1)]


def test_engine_rejects_plan_tile_extent_tampering_before_requests() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
        M5WholeCircuitSession,
    )
    from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
        M5TileLimits,
        lower_binary_contraction,
    )

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    node = dag.nodes[0]
    assert isinstance(node, ContractNode)
    lowering = lower_binary_contraction(
        node,
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
        limits=M5TileLimits.float32(),
    )
    node_plan = compiled.payload.node_plans[0]
    unit = node_plan.work_units[0]
    tampered = replace(unit, m_size=unit.m_size + 1)
    tampered_plan = replace(node_plan, work_units=(tampered,))
    session = object.__new__(M5WholeCircuitSession)
    session.ranks = (SimpleNamespace(index=0, local_dpus=2),)

    with pytest.raises(ValueError, match="extents"):
        session._requests_from_plan(node, lowering, tampered_plan)


def test_compile_rejects_unsupported_topology_and_int8_k() -> None:
    dag = _dag()
    with pytest.raises(ValueError, match="64 DPUs per rank"):
        compile_execution(
            dag,
            replace(
                _request(dag),
                topology=UpmemTopology(
                    dpu_count=65,
                    tasklets_per_dpu=1,
                    rank_count=1,
                ),
            ),
        )

    large = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0,), (200_000,), "dense", dtype="float64"),
                TensorSpec("b", (0,), (200_000,), "dense", dtype="float64"),
            ),
            (),
            "a,b->",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )
    unsupported = compile_execution(
        large, _request(large, NumericMode.HOST_PACKED_INT8_PER_TASK_V1)
    )
    assert isinstance(unsupported, UnsupportedExecution)
    assert unsupported.capability == "upmem_max_contracted_elements"


def test_compile_accepts_exact_v4_contracted_boundary() -> None:
    boundary = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0,), (MAX_CONTRACTED,), "dense", dtype="float64"),
                TensorSpec("b", (0,), (MAX_CONTRACTED,), "dense", dtype="float64"),
            ),
            (),
            "a,a->",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )

    compiled = compile_execution(
        boundary, _request(boundary, NumericMode.HOST_PACKED_INT8_PER_TASK_V1)
    )
    assert isinstance(compiled, ExecutionPlan)


def test_compile_does_not_count_unilateral_reduction_in_v4_k() -> None:
    """The native canonicalizer pre-sums labels that occur on only one input."""

    unilateral = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec(
                    "a", (0, 1), (MAX_CONTRACTED + 1, 2), "dense", dtype="float64"
                ),
                TensorSpec("b", (2,), (3,), "dense", dtype="float64"),
            ),
            (1, 2),
            "ab,c->bc",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )

    compiled = compile_execution(
        unilateral,
        _request(unilateral, NumericMode.HOST_PACKED_INT8_PER_TASK_V1),
    )
    assert isinstance(compiled, ExecutionPlan)


def test_compile_rejects_zero_sized_contracted_dimension_before_runtime() -> None:
    empty_k = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0, 1), (2, 0), "dense", dtype="float64"),
                TensorSpec("b", (1, 2), (0, 3), "dense", dtype="float64"),
            ),
            (0, 2),
            "ab,bc->ac",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )

    compiled = compile_execution(empty_k, _request(empty_k))
    assert isinstance(compiled, UnsupportedExecution)
    assert compiled.capability == "upmem_v4_positive_canonical_geometry"
    assert "label_dimension_is_not_positive" in compiled.reason


def test_compile_rejects_only_v4_unencodable_geometry() -> None:
    batch_overflow = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0, 1), (2**32, 1), "dense", dtype="float64"),
                TensorSpec("b", (0, 2), (2**32, 1), "dense", dtype="float64"),
            ),
            (0, 1, 2),
            "ab,ac->abc",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )
    unsupported_batch = compile_execution(batch_overflow, _request(batch_overflow))
    assert isinstance(unsupported_batch, UnsupportedExecution)
    assert unsupported_batch.capability == "upmem_v4_batch_count"

    output_overflow = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec("a", (0, 1), (2**63, 1), "dense", dtype="float64"),
                TensorSpec("b", (2,), (2,), "dense", dtype="float64"),
            ),
            (0, 1, 2),
            "ab,c->abc",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )
    unsupported_output = compile_execution(output_overflow, _request(output_overflow))
    assert isinstance(unsupported_output, UnsupportedExecution)
    assert unsupported_output.capability == "upmem_v4_uint64_element_count"


def test_compile_accepts_large_rank_and_logical_tensors_when_tiling_can_lower_them() -> (
    None
):
    high_rank_shape = (2,) * 17
    large_logical = build_contraction_dag(
        TensorNetworkSpec(
            None,
            (
                TensorSpec(
                    "a", tuple(range(17)), high_rank_shape, "dense", dtype="float64"
                ),
                TensorSpec("b", (16, 17), (2, 2), "dense", dtype="float64"),
            ),
            tuple(range(16)) + (17,),
            "abcdefghijklmnopq,qr->abcdefghijklmnopr",
        ),  # type: ignore[arg-type]
        ((0, 1),),
    )

    compiled = compile_execution(large_logical, _request(large_logical))
    assert isinstance(compiled, ExecutionPlan)


def test_run_upmem_uses_one_session_in_plan_order_and_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _chain_dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    result = run_upmem(
        compiled,
        dag,  # type: ignore[arg-type]
        _inputs(
            ("a", np.arange(6.0).reshape(2, 3)),
            ("b", np.arange(12.0).reshape(3, 4)),
            ("c", np.arange(8.0).reshape(4, 2)),
        ),
        RunContext(
            run_id="fake",
            target=Target.UPMEM,
            target_resources=_resources(tmp_path),
        ),
    )

    assert session.calls == ["contract_0", "contract_1"]
    assert session.closed
    assert result.executed_node_ids == ("contract_0", "contract_1")
    assert result.h2d_bytes == 16
    assert result.d2h_bytes == 8
    assert result.transfer_bytes == 24
    assert result.timing.kernel_s == 4.0
    assert result.timing.route_total_s is not None
    assert result.timing.route_total_s > 0.0
    assert result.backend_facts is not None
    assert result.backend_facts.hardware_release_confirmed
    assert result.backend_facts.native_identity_verified is True
    assert (
        result.backend_facts.intermediate_placement_origin == "m5_host_coordinator_v1"
    )
    assert result.backend_facts.observed_rank_count == 1
    assert result.backend_facts.rank_binding_sha256
    assert result.backend_facts.host_binary_sha256
    expected = np.einsum(
        "ab,bc,cd->ad",
        np.arange(6.0).reshape(2, 3),
        np.arange(12.0).reshape(3, 4),
        np.arange(8.0).reshape(4, 2),
    )
    np.testing.assert_array_equal(result.output, expected)


def test_run_upmem_reports_completed_host_nodes_not_planned_node_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)

    def completed_once(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), ("host-completed-node",)

    monkeypatch.setattr(module, "_execute_once", completed_once)
    result = run_upmem(
        compiled,
        dag,
        _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
        RunContext(
            run_id="observed-completion",
            target=Target.UPMEM,
            target_resources=_resources(tmp_path),
        ),
    )

    assert result.executed_node_ids == ("host-completed-node",)


def test_run_upmem_rejects_static_plan_tampering_before_session_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    node_plan = compiled.payload.node_plans[0]
    unit = node_plan.work_units[0]
    tampered = replace(
        compiled,
        payload=replace(
            compiled.payload,
            node_plans=(
                replace(
                    node_plan,
                    work_units=(replace(unit, stable_tile_id="tampered"),),
                ),
            ),
        ),
    )
    opened = False

    def opener(plan: object, context: object) -> _FakeSession:
        nonlocal opened
        opened = True
        return _FakeSession()

    monkeypatch.setattr(module, "_open_session", opener)
    with pytest.raises(ValueError, match="pure v4 recomputation"):
        run_upmem(
            tampered,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="tampered",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )
    assert not opened


def test_execute_once_evicts_only_produced_intermediates_after_last_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _chain_dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    observed_final_tensor_maps: list[set[str]] = []
    original_resolve = module._resolve_view

    def capture(view, tensors):
        if view.tensor_id == dag.output.tensor_id:
            observed_final_tensor_maps.append(set(tensors))
        return original_resolve(view, tensors)

    monkeypatch.setattr(module, "_resolve_view", capture)
    module._execute_once(
        _FakeSession(),
        dag,
        {
            "a": np.ones((2, 3)),
            "b": np.ones((3, 4)),
            "c": np.ones((4, 2)),
        },
        compiled.payload,
        resources=None,
        aggregate=None,
    )

    assert observed_final_tensor_maps
    assert dag.nodes[0].output.id not in observed_final_tensor_maps[-1]
    assert {"a", "b", "c"} <= observed_final_tensor_maps[-1]


def test_run_upmem_hashes_every_measured_output_and_rejects_nondeterminism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    outputs = iter(
        (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    )

    def nondeterministic_once(*args, **kwargs):
        return next(outputs), ("contract_0",)

    hashes: list[str] = []
    original_hash = module._array_hash
    monkeypatch.setattr(module, "_execute_once", nondeterministic_once)
    monkeypatch.setattr(
        module,
        "_array_hash",
        lambda value: hashes.append(original_hash(value)) or hashes[-1],
    )
    with pytest.raises(RuntimeError, match="non-deterministic"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="nondeterministic",
                target=Target.UPMEM,
                repetitions=2,
                target_resources=_resources(tmp_path),
            ),
        )
    assert len(hashes) == 2
    assert session.closed


def test_packed_tile_assembly_and_dequantization_are_separate() -> None:
    from types import SimpleNamespace

    import quantum_bench.execution.upmem as module

    aggregate = module._Aggregate()
    aggregate.add(
        SimpleNamespace(
            metadata={
                "physical_plan_consumed": True,
                "application_visible_h2d_bytes": 8,
                "application_visible_d2h_bytes": 4,
                "timing": {
                    "host_quantization_time_s": 0.25,
                    "preparation_time_s": 0.5,
                    "host_tile_assembly_time_s": 0.25,
                    "host_dequantization_time_s": 0.25,
                },
                "target_observed": "physical_hardware",
                "simulator_kernel_executed": False,
                "cpu_fallback_used": False,
            }
        )
    )

    assert aggregate.reduction_s == pytest.approx(0.25)
    assert aggregate.host_quantization_s == pytest.approx(0.25)
    assert aggregate.preparation_s == pytest.approx(0.5)
    assert aggregate.host_dequantization_s == pytest.approx(0.25)


def test_run_upmem_reduces_sliced_contracts_on_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = apply_slicing(_dag(), SliceSpec(node_id="contract_0", label=1))
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(6.0).reshape(3, 2)
    result = run_upmem(
        compiled,
        dag,
        _inputs(("a", left), ("b", right)),
        RunContext(
            run_id="slice",
            target=Target.UPMEM,
            target_resources=_resources(tmp_path),
        ),
    )

    assert session.calls == [
        "contract_0__slice_1_0",
        "contract_0__slice_1_1",
        "contract_0__slice_1_2",
    ]
    np.testing.assert_array_equal(result.output, left @ right)
    assert result.timing.reduction_s is not None
    assert result.timing.reduction_s >= 0.0


def test_run_upmem_closes_session_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    class FailingSession(_FakeSession):
        def execute(
            self,
            node: ContractNode,
            left: np.ndarray,
            right: np.ndarray,
            *,
            node_plan: object | None = None,
        ) -> object:
            self.calls.append(node.node_id)
            raise RuntimeError("native failure")

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = FailingSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)

    with pytest.raises(RuntimeError, match="native failure"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="failure",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )
    assert session.closed


def test_terminal_physical_facts_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession({"target_observed": "physical_hardware"})
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    with pytest.raises(RuntimeError, match="terminal metadata"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="missing-terminal-facts",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("profile", "wrong-profile"),
        ("abi", "wrong-abi"),
        ("session_protocol", "wrong-session"),
        ("dispatch_mode", "wrong-dispatch"),
        ("kernel_identity", "wrong-kernel"),
        ("execution_class", "wrong-execution-class"),
        ("graph_intermediate_placement", "wrong-placement"),
        ("graph_intermediate_placement_origin", "wrong-origin"),
        ("native_identity_verified", False),
    ),
)
def test_terminal_observed_execution_identity_must_match_compiled_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    metadata = dict(_FakeSession().terminal_metadata)
    metadata[field] = value
    monkeypatch.setattr(
        module, "_open_session", lambda plan, context: _FakeSession(metadata)
    )

    with pytest.raises(RuntimeError, match=field):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id=f"observed-{field}",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


@pytest.mark.parametrize(
    ("canonical", "alias", "expected"),
    (
        ("profile", "physical_profile", "m5_whole_circuit_v4_v1"),
        ("abi", "abi_version", "execution_plan_v4"),
    ),
)
def test_terminal_accepts_documented_identity_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical: str,
    alias: str,
    expected: str,
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    metadata = dict(_FakeSession().terminal_metadata)
    metadata.pop(canonical)
    metadata[alias] = expected
    monkeypatch.setattr(
        module, "_open_session", lambda plan, context: _FakeSession(metadata)
    )

    result = run_upmem(
        compiled,
        dag,
        _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
        RunContext(
            run_id=f"alias-{canonical}",
            target=Target.UPMEM,
            target_resources=_resources(tmp_path),
        ),
    )
    assert result.backend_facts is not None
    assert getattr(result.backend_facts, f"{canonical}_id") == expected


def test_terminal_requires_observed_backend_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    metadata = dict(_FakeSession().terminal_metadata)
    metadata.pop("backend_id")
    monkeypatch.setattr(
        module, "_open_session", lambda plan, context: _FakeSession(metadata)
    )
    with pytest.raises(RuntimeError, match="missing backend_id"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="missing-backend-id",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


@pytest.mark.parametrize(
    "mode",
    (NumericMode.FLOAT32_REAL, NumericMode.HOST_PACKED_INT8_PER_TASK_V1),
)
def test_nonzero_imaginary_inputs_are_rejected_before_session_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: NumericMode
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag, mode))
    assert isinstance(compiled, ExecutionPlan)
    opened = False

    def opener(plan: object, context: object) -> _FakeSession:
        nonlocal opened
        opened = True
        return _FakeSession()

    monkeypatch.setattr(module, "_open_session", opener)
    with pytest.raises(ValueError, match="nonzero imaginary"):
        run_upmem(
            compiled,
            dag,
            _inputs(
                ("a", np.ones((2, 3), dtype=np.complex64) * (1 + 1j)),
                ("b", np.ones((3, 2), dtype=np.complex64)),
            ),
            RunContext(
                run_id="complex-input",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )
    assert not opened


def test_missing_task_transfer_bytes_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    class MissingBytesSession(_FakeSession):
        def execute(
            self,
            task: object,
            left: np.ndarray,
            right: np.ndarray,
            *,
            node_plan: object | None = None,
        ) -> object:
            result = super().execute(task, left, right, node_plan=node_plan)
            result.metadata.pop("application_visible_h2d_bytes")
            result.metadata.pop("application_visible_d2h_bytes")
            return result

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    monkeypatch.setattr(
        module, "_open_session", lambda plan, context: MissingBytesSession()
    )
    with pytest.raises(RuntimeError, match="missing application_visible_h2d_bytes"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="missing-bytes",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


def test_terminal_release_failure_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession(
        {
            "target_observed": "physical_hardware",
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_allocation_verified": True,
            "hardware_release_verified": False,
            "hardware_release_confirmed": False,
        }
    )
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    with pytest.raises(RuntimeError, match="hardware_release_verified"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="release-failure",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


def test_simulator_terminal_fact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    metadata = dict(_FakeSession().terminal_metadata)
    metadata["simulator_kernel_executed"] = True
    session = _FakeSession(metadata)
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    with pytest.raises(RuntimeError, match="simulator_kernel_executed"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="simulator-terminal",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


def test_session_close_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    class CloseFailingSession(_FakeSession):
        def close(self) -> dict[str, object]:
            self.closed = True
            raise RuntimeError("release failed")

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = CloseFailingSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)
    with pytest.raises(RuntimeError, match="session close failed"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="close-failure",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )
    assert session.closed


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allocated_dpu_count", 1),
        ("observed_rank_count", 2),
        ("observed_tasklets_per_dpu", 2),
    ),
)
def test_terminal_allocation_must_match_compiled_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    metadata = dict(_FakeSession().terminal_metadata)
    metadata[field] = value
    session = _FakeSession(metadata)
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)

    with pytest.raises(RuntimeError, match=field):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id=f"mismatch-{field}",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


def test_invalid_explicit_timeout_is_rejected_before_session_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    opened = False

    def opener(plan: object, context: object) -> _FakeSession:
        nonlocal opened
        opened = True
        return _FakeSession()

    resources = _resources(tmp_path)
    monkeypatch.setattr(
        module,
        "_open_session",
        lambda plan, context: opener(plan, context),
    )
    for timeout_s in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout_s"):
            run_upmem(
                compiled,
                dag,
                _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
                RunContext(
                    run_id=f"timeout-{timeout_s}",
                    target=Target.UPMEM,
                    timeout_s=timeout_s,
                    target_resources=resources,
                ),
            )
    assert not opened


def test_execution_and_close_failures_are_both_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    class DualFailingSession(_FakeSession):
        def execute(
            self,
            task: object,
            left: np.ndarray,
            right: np.ndarray,
            *,
            node_plan: object | None = None,
        ) -> object:
            raise RuntimeError("native failure")

        def close(self) -> dict[str, object]:
            self.closed = True
            raise RuntimeError("release failed")

    dag = _dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = DualFailingSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)

    with pytest.raises(RuntimeError, match="native failure.*release failed"):
        run_upmem(
            compiled,
            dag,
            _inputs(("a", np.ones((2, 3))), ("b", np.ones((3, 2)))),
            RunContext(
                run_id="dual-failure",
                target=Target.UPMEM,
                target_resources=_resources(tmp_path),
            ),
        )


def test_binary_hashes_are_computed_once_outside_task_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.execution.upmem as module

    calls: list[Path] = []

    def fake_hash(path: Path) -> str:
        calls.append(path)
        return f"hash-{path.name}"

    monkeypatch.setattr(module, "_file_sha256", fake_hash)
    dag = _chain_dag()
    compiled = compile_execution(dag, _request(dag))
    assert isinstance(compiled, ExecutionPlan)
    session = _FakeSession()
    monkeypatch.setattr(module, "_open_session", lambda plan, context: session)

    run_upmem(
        compiled,
        dag,
        _inputs(
            ("a", np.arange(6.0).reshape(2, 3)),
            ("b", np.arange(12.0).reshape(3, 4)),
            ("c", np.arange(8.0).reshape(4, 2)),
        ),
        RunContext(
            run_id="hash-count",
            target=Target.UPMEM,
            target_resources=_resources(tmp_path),
        ),
    )

    assert len(calls) == 3
