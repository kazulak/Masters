from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import tarfile

import pytest

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_m7b.py"
PHYSICAL_SCRIPT = ROOT / "scripts" / "qualify_m7c_physical.py"
SELECTION_SCRIPT = ROOT / "scripts" / "select_m7c_workload.py"
SCALING_SCRIPT = ROOT / "scripts" / "run_m7c_scaling_campaign.py"


def _load_script(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualifier():
    return _load_script(SCRIPT, "qualify_m7b")


def _physical_qualifier():
    return _load_script(PHYSICAL_SCRIPT, "qualify_m7c_physical")


def _selector():
    return _load_script(SELECTION_SCRIPT, "select_m7c_workload")


def _scaling_campaign():
    return _load_script(SCALING_SCRIPT, "run_m7c_scaling_campaign")


def _archive(path: Path, member_name: str, *, kind: str = "file") -> None:
    source = path.parent / "payload.txt"
    source.write_text("payload\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        if kind == "file":
            bundle.add(source, arcname=member_name)
            return
        member = tarfile.TarInfo(member_name)
        member.type = tarfile.SYMTYPE
        member.linkname = "payload.txt"
        bundle.addfile(member)


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_extracts_regular_relative_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, "evidence/manifest.json")

    _qualifier()._safe_extract_tar(archive, tmp_path / "output")

    assert (tmp_path / "output" / "evidence" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "payload\n"


def test_qualifier_verifies_bundled_hashes_and_records_external_provenance(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "m7a-bundle"
    evidence = bundle / "cpu-run" / "manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (bundle / "SHA256SUMS").write_text(
        f"{digest}  runs/{bundle.name}/cpu-run/manifest.json\n"
        + "0" * 64
        + "  native/upmem/runtime/bin/dpu\n",
        encoding="utf-8",
    )

    external = _qualifier()._verify_internal_hashes(tmp_path)

    assert external == ("native/upmem/runtime/bin/dpu",)


def test_physical_preparation_preserves_template_resolved_paths(tmp_path: Path) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_smoke.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "smoke.yml"
    prepared = _physical_qualifier().prepare_config(
        template=template,
        output=output,
        mode="float32-smoke",
        rank_path="/dev/dpu_rank42",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[2, 4],
    )

    assert prepared == output
    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    source_options = source["routes"]["upmem_float32_1dpu"]["options"]
    copied_options = copied["routes"]["upmem_float32_1dpu"]["options"]
    for field in ("host_binary", "dpu_binary", "initialization_binary"):
        assert copied_options[field] == source_options[field]
    assert copied_options["session_root"] == str((tmp_path / "sessions").resolve())
    assert copied_options["rank_paths"] == ("/dev/dpu_rank42",)
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (2, 4),
    }


def test_physical_preparation_probe_has_one_measurement(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "probe.yml"
    _physical_qualifier().prepare_config(
        template=ROOT / "configs" / "tn_benchmark_physical_smoke.yml",
        output=output,
        mode="probe",
        rank_path="/dev/dpu_rank0",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[0],
    )

    config = load_experiment_config(output)
    assert config["collection"]["warmup_blocks"] == 0
    assert config["collection"]["measurement_blocks"] == 1
    assert config["routes"]["upmem_float32_1dpu"]["numeric_policy"] == (
        "split_complex_float32_v1"
    )


def test_m7c_workload_selection_is_deterministic_and_preregistered(
    tmp_path: Path,
) -> None:
    selector = _selector()
    first = selector.build_selection()
    second = selector.build_selection()

    assert first == second
    assert first["selected_primary"] == "quantization_stress_18q_l2"
    assert first["selected_secondary"] == "ghz_chain_18q"
    primary = next(
        candidate
        for candidate in first["candidates"]
        if candidate["candidate_id"] == first["selected_primary"]
    )
    assert primary["logical_plan_id"] == (
        "d504919e20d95bac608dd906d46abb122f9680873679710b0584e71981648fb5"
    )
    assert primary["topologies"]["dpu4_tasklet8"][
        "collection_resource_admission_passed"
    ] is True

    path = tmp_path / "selection.json"
    selector.write_selection(path)
    selector.check_selection(path)


def test_m7c_committed_selection_matches_scaling_config() -> None:
    selector = _selector()
    selection = ROOT / "configs" / "m7c_workload_selection.json"
    for config in (
        "tn_benchmark_physical_scaling_diagnostic.yml",
        "tn_benchmark_physical_scaling.yml",
        "tn_benchmark_physical_scaling_confirmation.yml",
    ):
        selector.check_selection(selection, ROOT / "configs" / config)


def test_m7c_scaling_preparation_preserves_all_resolved_route_paths(
    tmp_path: Path,
) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "diagnostic.yml"
    _scaling_campaign().prepare_config(
        template=template,
        output=output,
        rank_paths=["/dev/dpu_rank19"],
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[1, 3],
    )

    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    for route_id, route in source["routes"].items():
        if route["executor"] != "upmem_physical":
            continue
        source_options = route["options"]
        copied_options = copied["routes"][route_id]["options"]
        for field in ("host_binary", "dpu_binary", "initialization_binary"):
            assert copied_options[field] == source_options[field]
        assert copied_options["rank_paths"] == ("/dev/dpu_rank19",)
        assert copied_options["session_root"] == str(
            (tmp_path / "sessions" / route_id).resolve()
        )
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (1, 3),
    }
