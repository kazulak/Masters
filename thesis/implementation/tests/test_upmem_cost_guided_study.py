from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "upmem_cost_guided_study_test", ROOT / "scripts/upmem_cost_guided_path.py"
)
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_frozen_packet_and_effective_budget():
    config, workload, budget = study.load_study()
    assert len(workload["instances"]) == 12
    assert budget == {
        "development_cells": 12,
        "evaluation_cells": 12,
        "initial_paths_per_cell": 4,
        "stages": {"initial": 192, "feedback_1": 144, "feedback_2": 144, "evaluation": 288},
        "maximum_attempts": 768,
        "unallocated_attempts": 24,
    }
    assert config["campaign"]["evaluation_methods"] == ["G", "F", "R", "U"]
    assert config["score"]["historical_observations_allowed"] is False
    assert config["score"]["evaluation_observations_allowed_for_fit"] is False
    assert config["search"]["proposals"] == 128
    assert config["search"]["max_repeats"] == 1
    assert config["search"]["accel"] is config["search"]["parallel"] is False


@pytest.mark.parametrize("args", [(0, 12, 792), (12, 0, 792), (12, 12, True), (12, 12, 600)])
def test_budget_rejects_invalid_or_insufficient_counts(args):
    with pytest.raises(ValueError):
        study.attempt_budget(*args)


def test_unconstrained_budget_formula():
    result = study.attempt_budget(12, 12, 864)
    assert result["initial_paths_per_cell"] == 6
    assert result["maximum_attempts"] == 864


@pytest.mark.parametrize("field,value", [("effective_attempt_cap", 864), ("initial_paths_per_cell", 6)])
def test_declared_budget_drift_rejected(tmp_path, field, value):
    config = json.loads(study.DEFAULT_STUDY.read_text())
    config["campaign"][field] = value
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Declared"):
        study.load_study(path)


def test_manifest_hash_drift_rejected(tmp_path):
    config = json.loads(study.DEFAULT_STUDY.read_text())
    config["workload"]["sha256"] = "0" * 64
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        study.load_study(path)
