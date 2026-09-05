from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_complex_batching.py"
SPEC = importlib.util.spec_from_file_location("batching_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def operation():
    # Odd output widths, two K chunks, and three idle DPU slots per wave.
    return {"node_id": "contract_0", "b": 1, "m": 3, "n": 7, "k": 5,
            "work_units": [{"wave": i, "logical_rank": 0, "logical_dpu": 0,
                            "batch_start": 0, "batch_size": 1, "m_start": 0, "m_size": 3,
                            "n_start": 0, "n_size": 7, "k_start": k, "k_size": size}
                           for i, (k, size) in enumerate(((0, 3), (3, 2)))]}


@pytest.mark.parametrize("policy", ["split_complex_float32_v1", "complex_int8_shared_scale_v1"])
@pytest.mark.parametrize("dpus", [1, 4])
def test_exact_bytes_order_hashes_and_output_paths(tmp_path, policy, dpus):
    results, decoded = [], []
    for arm in probe.ARMS:
        root = tmp_path / arm
        root.mkdir()
        row, sequence = probe.prepare_operation(operation(), policy, dpus, root, arm, 19)
        assert sequence == 27
        assert row["request_count"] == 8
        assert row["native_processes_started"] == row["dpu_launches_executed"] == 0
        assert all(row[key] >= 0 for key in ("preparation_ns", "cpu_ns", "setup_ns", "validation_ns"))
        results.append(row)
        decoded.append([record for path in sorted(root.glob("*.bin"))
                        for record in probe.unpack_envelope(path.read_bytes())])
    assert decoded[0] == decoded[1]
    assert [r[0] for r in decoded[0]] == list(range(19, 27))
    assert all(r[1] == dpus for r in decoded[0])
    assert results[0]["bytes_written"] - results[1]["bytes_written"] == 288
    assert (results[0]["files_created"], results[1]["files_created"]) == (4, 1)
    assert results[0]["request_equivalence_sha256"] == results[1]["request_equivalence_sha256"]
    assert results[0]["output_paths_sha256"] == results[1]["output_paths_sha256"]


def resign(data):
    data[64:96] = hashlib.sha256(data[:64] + bytes(32) + data[96:]).digest()
    return bytes(data)


@pytest.mark.parametrize("mutation", ["truncate", "digest", "count", "offset", "order", "trailing", "version"])
def test_independent_decoder_rejects_corruption(tmp_path, mutation):
    probe.prepare_operation(operation(), "split_complex_float32_v1", 4, tmp_path, probe.ARMS[1], 0)
    data = bytearray(next(tmp_path.glob("*.bin")).read_bytes())
    if mutation == "truncate":
        data = data[:80]
    elif mutation == "digest":
        data[-1] ^= 1
    elif mutation == "count":
        struct.pack_into("<I", data, 16, 0xFFFFFFFF)
        data = resign(data)
    elif mutation == "offset":
        struct.pack_into("<Q", data, 104, 2**64 - 1)
        data[264:296] = hashlib.sha256(data[96:264] + bytes(32)).digest()
        data = resign(data)
    elif mutation == "order":
        data[96:296], data[296:496] = data[296:496], data[96:296]
        data = resign(data)
    elif mutation == "trailing":
        data += b"x"
    else:
        struct.pack_into("<I", data, 8, 9)
    with pytest.raises(ValueError):
        probe.unpack_envelope(bytes(data))


@pytest.mark.parametrize("policy", ["split_complex_float32_v1", "complex_int8_shared_scale_v1"])
def test_synthetic_complex_encoding_and_split_k_replay(policy):
    a = probe.encode_complex_tensor(probe.synthetic((3, 5), 1), policy)
    b = probe.encode_complex_tensor(probe.synthetic((5, 7), 7), policy)
    dtype = np.int32 if policy == "complex_int8_shared_scale_v1" else np.float32
    products = []
    for left, right in ((a.real, b.real), (a.imag, b.imag), (a.real, b.imag), (a.imag, b.real)):
        left, right = left.astype(dtype), right.astype(dtype)
        chunked = left[:, :3] @ right[:3] + left[:, 3:] @ right[3:]
        np.testing.assert_allclose(chunked, left @ right, atol=2e-7, rtol=2e-6)
        products.append(chunked)
    actual = ((products[0] - products[1]) + 1j * (products[2] + products[3])) * a.scale * b.scale
    reference = ((a.real.astype(np.float64) + 1j * a.imag) @
                 (b.real.astype(np.float64) + 1j * b.imag)) * a.scale * b.scale
    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=3e-6)


def test_failure_is_returned_without_replacement(tmp_path):
    cell = {"numeric_policy": "invalid", "topology": {"dpu_count": 1}, "operations": [operation()]}
    result = probe.run_arm(cell, probe.ARMS[0], tmp_path)
    assert result["status"] == "failed"
    assert not result["operations"]
    assert list(tmp_path.iterdir())  # Failed operation retained for diagnosis.


def test_summary_uses_complete_paired_blocks_not_pooled_operation_ratios():
    rows = [{"cell_id": "a", "block": block, "arm": arm,
             "operations": [{"preparation_ns": value}]}
            for block, values in enumerate(((999, 1), (20, 10), (40, 20)))
            for arm, value in zip(probe.ARMS, values)]
    result = probe.summarize(rows)[0]
    assert result["paired_median_speedup"] == 2
    assert result["paired_median_saved_ns"] == 15
    assert result["arms"][probe.ARMS[0]]["measured_count"] == 2
