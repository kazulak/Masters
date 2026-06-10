from __future__ import annotations


def estimate_energy(seconds: float, config: dict) -> tuple[float, str, float]:
    energy_cfg = config.get("measurement", {}).get("energy", {})
    watts = float(energy_cfg.get("cpu_watts", 65.0))
    return seconds * watts, "estimated_static_power", watts

