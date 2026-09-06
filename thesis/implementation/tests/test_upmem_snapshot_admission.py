"""Snapshot limits must precede payload conversion, files and native allocation."""

from dataclasses import replace
import sys

import pytest

from quantum_bench.results import UnsupportedExecution
from quantum_bench.upmem import packed_wave, runtime
from quantum_bench.upmem.plan import UpmemResources, UpmemTopology, plan_upmem
from quantum_bench.upmem.wave_protocol import COMPLETION_BYTES
from tests.test_upmem_packed_wave import control, envelope, operation, tile
from tests.test_upmem_wave_runtime import fork_join


def test_sizes_match_input_wire_bytes_and_unpadded_result_geometry():
    item = control(dpu=0, m=3, n=5, k=1)
    encoded = envelope(operations=(operation(m=3, n=5, k=1),), waves=((tile(item),),))
    request, response = packed_wave.wave_snapshot_sizes(1, ((item,),))
    assert request == len(encoded)
    assert response == COMPLETION_BYTES + 4 * 3 * 5 * 4
    assert response < COMPLETION_BYTES + sum(size for _, size in item.planes[4:])


@pytest.mark.parametrize("kind", ["input", "result", "descriptors"])
def test_oversized_snapshots_are_rejected_before_payload_validation(monkeypatch, kind):
    # Lower the same production cap so no large allocation is needed in a test.
    item = control(dpu=0, m=16, n=16, k=1) if kind == "result" else control(dpu=0)
    limit = {"input": 450, "result": 1000, "descriptors": 400}[kind]
    monkeypatch.setattr(packed_wave, "MAX_ENVELOPE_BYTES", limit)

    def forbidden(*args, **kwargs):
        pytest.fail("payload validation/copy happened before size rejection")

    monkeypatch.setattr(packed_wave, "_validate_cohort", forbidden)
    with pytest.raises(ValueError, match="snapshot admission limit"):
        envelope(operations=(operation(m=item.m, n=item.n, k=item.k),), waves=((tile(item),),))


def test_mutable_oversized_plane_is_rejected_before_bytes_conversion():
    class CopyTrap(bytearray):
        def __bytes__(self):
            pytest.fail("oversized mutable plane was copied")

    item = tile(control(dpu=0))
    bad = replace(item, inputs=(CopyTrap(1024), *item.inputs[1:]))
    with pytest.raises(ValueError, match="length differs from control"):
        envelope(waves=((bad,),))


def test_digest_size_is_checked_before_mutable_copy():
    class CopyTrap(bytearray):
        def __bytes__(self):
            pytest.fail("oversized digest was copied")

    with pytest.raises(ValueError, match="32 digest bytes"):
        packed_wave._digest_bytes(CopyTrap(1024), "plan")


def test_multibyte_memoryview_uses_nbytes_not_element_count():
    import array

    item = tile(control(dpu=0))
    inputs = tuple(memoryview(array.array("f", [1.0] * (len(payload) // 4))) for payload in item.inputs)
    assert envelope(waves=((replace(item, inputs=inputs),),)) == envelope(waves=((item,),))


@pytest.mark.parametrize("target", ["simulator", "hardware"])
@pytest.mark.parametrize("schedule", ["serial_nodes_v1", "static_dag_waves_v1"])
def test_plan_size_rejection_precedes_session_or_engine_creation(tmp_path, monkeypatch, target, schedule):
    dag, _ = fork_join(k=1)
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1", schedule_policy=schedule,
                      topology=UpmemTopology(dpu_count=3, tasklets_per_dpu=8, rank_count=1))
    root = tmp_path / "must-not-exist"
    resources = UpmemResources(session_root=str(root), host_binary=sys.executable, dpu_binary=sys.executable,
                               initialization_binary=sys.executable, request_transport="packed_wave_v1",
                               rank_paths=("/dev/dpu_rank1",) if target == "hardware" else ())
    monkeypatch.setattr(packed_wave, "MAX_ENVELOPE_BYTES", 1)

    def forbidden(*args, **kwargs):
        pytest.fail("native executor constructed before snapshot admission")

    monkeypatch.setattr(runtime, "UpmemV4Executor", forbidden)
    opener = runtime.open_upmem if target == "hardware" else runtime.open_upmem_simulator
    with pytest.raises(UnsupportedExecution) as failure:
        opener(dag, plan, resources, fuse_complex=True)
    assert failure.value.stage == "preflight"
    assert failure.value.capability == "prepared_wave_snapshot_limit"
    assert not root.exists()
