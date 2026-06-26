from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.targets.upmem.simplepim import probe_simplepim, simplepim_probe_metadata


def test_missing_simplepim_configuration_is_unavailable() -> None:
    probe = probe_simplepim(env={}, path_lookup=lambda command: None)
    payload = probe.to_json_dict()

    assert probe.simplepim_available is False
    assert probe.simplepim_probe_status == "unavailable"
    assert probe.simplepim_command_path is None
    assert probe.skip_reason
    assert payload["metadata"]["external_command_executed"] is False
    assert simplepim_probe_metadata(payload)["backend"] == "simplepim_unavailable"
    json.dumps(payload)


def test_simplepim_bin_reports_available_without_executing_command(tmp_path: Path) -> None:
    binary = tmp_path / "simplepim"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    probe = probe_simplepim(env={"SIMPLEPIM_BIN": str(binary)}, path_lookup=lambda command: None)
    payload = probe.to_json_dict()
    metadata = simplepim_probe_metadata(payload)

    assert probe.simplepim_available is True
    assert probe.simplepim_probe_status == "available"
    assert probe.simplepim_command_path == str(binary)
    assert probe.simplepim_version is None
    assert payload["metadata"]["source"] == "SIMPLEPIM_BIN"
    assert payload["metadata"]["external_command_executed"] is False
    assert metadata["backend"] == "simplepim_future"
    assert metadata["simplepim_available"] is True


def test_command_discovery_reports_available_without_version_probe() -> None:
    probe = probe_simplepim(
        env={},
        path_lookup=lambda command: "/usr/local/bin/simplepim" if command == "simplepim" else None,
    )
    payload = probe.to_json_dict()

    assert probe.simplepim_available is True
    assert probe.simplepim_probe_status == "available"
    assert probe.simplepim_command_path == "/usr/local/bin/simplepim"
    assert probe.simplepim_version is None
    assert payload["metadata"]["source"] == "command_discovery"
    assert payload["metadata"]["command"] == "simplepim"
    assert payload["metadata"]["external_command_executed"] is False


def test_simplepim_home_alone_is_configured_but_unverified(tmp_path: Path) -> None:
    probe = probe_simplepim(env={"SIMPLEPIM_HOME": str(tmp_path)}, path_lookup=lambda command: None)
    payload = probe.to_json_dict()

    assert probe.simplepim_available is False
    assert probe.simplepim_probe_status == "configured_but_unverified"
    assert probe.simplepim_home == str(tmp_path)
    assert probe.simplepim_command_path is None
    assert probe.skip_reason
    assert payload["metadata"]["source"] == "SIMPLEPIM_HOME"
    assert payload["metadata"]["external_command_executed"] is False
    assert simplepim_probe_metadata(payload)["backend"] == "simplepim_unavailable"
