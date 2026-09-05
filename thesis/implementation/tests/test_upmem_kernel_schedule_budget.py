"""The approved hardware ceiling includes controls and qualification packets."""

import json
from pathlib import Path


def test_kernel_schedule_budget_is_closed_and_float32_primary():
    root = Path(__file__).resolve().parents[1]
    budget = json.loads(
        (root / "configs/upmem_kernel_schedule_budget_v1.json").read_text()
    )
    packets = budget["packets"]
    assert len({packet["id"] for packet in packets}) == len(packets)
    assert all(type(packet["attempt_ceiling"]) is int for packet in packets)
    assert all(packet["attempt_ceiling"] > 0 for packet in packets)
    assert sum(packet["attempt_ceiling"] for packet in packets) == 1051
    assert budget["total_attempt_ceiling"] == 1051
    assert budget["primary_numeric_policy"] == "split_complex_float32_v1"
    assert budget["physical_rank_count"] == 1
    counts = {packet["id"]: packet["attempt_ceiling"] for packet in packets}
    assert sum(count for name, count in counts.items() if name.startswith("path_")) == 540
    assert sum(counts[name] for name in (
        "changed_checkpoint_correctness", "principal_mechanism_confirmation",
        "final_hierarchical_scaling", "scalar_wram_ablation",
    )) == 96
    assert budget["rules"]["hardware_per_weight_vector"] is False
    assert budget["rules"]["replace_failed_samples"] is False
    assert budget["rules"]["physical_acceptance_requires_two_verified_copies"] is True
