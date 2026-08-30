from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_circuit_resource_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("freeze_circuit_resource_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_bundle_contract_is_fixed_to_the_canonical_diagnostic() -> None:
    assert module.EXPECTED_SOURCE == "89ecc5527f182f42dc471101f96edf86b0dadefa"
    assert module.EXPECTED_TAG == "thesis-upmem-circuit-resource-sensitivity-diagnostic-v1"
    assert module.EXPECTED_CASES == (
        "quantization_stress_18q_l2",
        "hs_18q_d1",
        "ghz_chain_18q",
    )
    assert len(module.EXPECTED_ROUTES) == 7


def test_safe_extract_rejects_links_and_parent_paths(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 0
        stream.addfile(info)

    with pytest.raises(ValueError, match="unsafe archive member"):
        module._safe_extract(archive, tmp_path / "extract")
