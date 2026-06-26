from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping

from quantum_bench.core.records import JsonDict


SIMPLEPIM_PROBE_KEY = "simplepim"
SIMPLEPIM_COMMAND_CANDIDATES = ("simplepim", "simplepim-run")
SimplePimProbeStatus = Literal["available", "unavailable", "configured_but_unverified"]


@dataclass(frozen=True)
class SimplePimProbeResult:
    simplepim_available: bool
    simplepim_probe_status: SimplePimProbeStatus
    simplepim_version: str | None = None
    simplepim_home: str | None = None
    simplepim_bin: str | None = None
    simplepim_library_path: str | None = None
    simplepim_command_path: str | None = None
    skip_reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return {
            "simplepim_available": self.simplepim_available,
            "simplepim_probe_status": self.simplepim_probe_status,
            "simplepim_version": self.simplepim_version,
            "simplepim_home": self.simplepim_home,
            "simplepim_bin": self.simplepim_bin,
            "simplepim_library_path": self.simplepim_library_path,
            "simplepim_command_path": self.simplepim_command_path,
            "skip_reason": self.skip_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SimplePimDenseMicrobenchSpec:
    case_id: str
    task_id: str
    route_id: str
    fixed_point_spec: JsonDict
    tile_plan: JsonDict
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    host_to_dpu_bytes: int
    dpu_to_host_bytes: int
    mram_to_wram_bytes: int


def probe_simplepim(
    env: Mapping[str, str] | None = None,
    path_lookup: Callable[[str], str | None] | None = None,
) -> SimplePimProbeResult:
    probe_env = env if env is not None else os.environ
    lookup = path_lookup or shutil.which
    simplepim_home = _clean_path(probe_env.get("SIMPLEPIM_HOME"))
    simplepim_bin = _clean_path(probe_env.get("SIMPLEPIM_BIN"))
    simplepim_library_path = _clean_path(probe_env.get("SIMPLEPIM_LIB"))

    if simplepim_bin:
        return SimplePimProbeResult(
            simplepim_available=True,
            simplepim_probe_status="available",
            simplepim_version=None,
            simplepim_home=simplepim_home,
            simplepim_bin=simplepim_bin,
            simplepim_library_path=simplepim_library_path,
            simplepim_command_path=simplepim_bin,
            skip_reason=None,
            metadata={"source": "SIMPLEPIM_BIN", "external_command_executed": False},
        )

    for command in SIMPLEPIM_COMMAND_CANDIDATES:
        command_path = lookup(command)
        if command_path:
            return SimplePimProbeResult(
                simplepim_available=True,
                simplepim_probe_status="available",
                simplepim_version=None,
                simplepim_home=simplepim_home,
                simplepim_bin=None,
                simplepim_library_path=simplepim_library_path,
                simplepim_command_path=str(command_path),
                skip_reason=None,
                metadata={
                    "source": "command_discovery",
                    "command": command,
                    "external_command_executed": False,
                },
            )

    if simplepim_home:
        return SimplePimProbeResult(
            simplepim_available=False,
            simplepim_probe_status="configured_but_unverified",
            simplepim_version=None,
            simplepim_home=simplepim_home,
            simplepim_bin=None,
            simplepim_library_path=simplepim_library_path,
            simplepim_command_path=None,
            skip_reason="SIMPLEPIM_HOME is set, but no SimplePIM executable was configured or found",
            metadata={"source": "SIMPLEPIM_HOME", "external_command_executed": False},
        )

    if simplepim_library_path:
        return SimplePimProbeResult(
            simplepim_available=False,
            simplepim_probe_status="configured_but_unverified",
            simplepim_version=None,
            simplepim_home=None,
            simplepim_bin=None,
            simplepim_library_path=simplepim_library_path,
            simplepim_command_path=None,
            skip_reason="SIMPLEPIM_LIB is set, but no SimplePIM executable was configured or found",
            metadata={"source": "SIMPLEPIM_LIB", "external_command_executed": False},
        )

    return SimplePimProbeResult(
        simplepim_available=False,
        simplepim_probe_status="unavailable",
        simplepim_version=None,
        simplepim_home=None,
        simplepim_bin=None,
        simplepim_library_path=None,
        simplepim_command_path=None,
        skip_reason="SimplePIM is not configured; set SIMPLEPIM_BIN or put a SimplePIM command on PATH",
        metadata={"source": "none", "external_command_executed": False},
    )


def simplepim_probe_metadata(probe: SimplePimProbeResult | JsonDict | None = None) -> JsonDict:
    if probe is None:
        payload = probe_simplepim().to_json_dict()
    elif isinstance(probe, SimplePimProbeResult):
        payload = probe.to_json_dict()
    else:
        payload = dict(probe)

    return {
        "backend": "simplepim_future" if bool(payload.get("simplepim_available")) else "simplepim_unavailable",
        "simplepim_available": bool(payload.get("simplepim_available", False)),
        "simplepim_probe_status": payload.get("simplepim_probe_status"),
        "simplepim_version": payload.get("simplepim_version"),
        "simplepim_command_path": payload.get("simplepim_command_path"),
        "simplepim_library_path": payload.get("simplepim_library_path"),
        "simplepim_skip_reason": payload.get("skip_reason"),
    }


def _clean_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    cleaned = str(Path(stripped).expanduser())
    return cleaned or None
