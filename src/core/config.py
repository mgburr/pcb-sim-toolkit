"""Simulation configuration, result types, and enums.

Separated from simulator.py to avoid circular imports with analysis modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SimulationType(Enum):
    SPICE_DC = "dc"
    SPICE_AC = "ac"
    SPICE_TRANSIENT = "transient"
    SIGNAL_INTEGRITY = "signal_integrity"
    THERMAL = "thermal"
    FULL = "full"


@dataclass
class SimulationConfig:
    sim_type: SimulationType = SimulationType.FULL
    spice_options: dict[str, Any] = field(default_factory=dict)
    frequency_range: tuple[float, float] = (1e3, 1e9)  # Hz
    time_step: float = 1e-9  # seconds
    duration: float = 1e-6  # seconds
    ambient_temp: float = 25.0  # Celsius
    output_dir: Path = field(default_factory=lambda: Path("./sim_output"))

    @classmethod
    def from_yaml(cls, path: Path) -> SimulationConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            sim_type=SimulationType(data.get("sim_type", "full")),
            spice_options=data.get("spice_options", {}),
            frequency_range=tuple(data.get("frequency_range", [1e3, 1e9])),
            time_step=data.get("time_step", 1e-9),
            duration=data.get("duration", 1e-6),
            ambient_temp=data.get("ambient_temp", 25.0),
            output_dir=Path(data.get("output_dir", "./sim_output")),
        )


@dataclass
class SimulationResult:
    success: bool
    sim_type: SimulationType
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        out = {
            "success": self.success,
            "sim_type": self.sim_type.value,
            "data": serialize(self.data),
            "errors": self.errors,
            "warnings": self.warnings,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


def serialize(obj: Any) -> Any:
    """Make numpy arrays and other non-JSON types serializable."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj
