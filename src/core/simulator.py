"""Main simulation orchestrator that coordinates all analysis stages."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..analysis.signal_integrity import SignalIntegrityAnalyzer
from ..analysis.thermal import ThermalAnalyzer
from ..analysis.spice import SpiceSimulator
from ..exporters.report import ReportExporter
from .config import SimulationConfig, SimulationResult, SimulationType
from .models import PCBDesign


class PCBSimulator:
    """Top-level simulator that orchestrates all analysis stages."""

    def __init__(self, design: PCBDesign, config: SimulationConfig | None = None):
        self.design = design
        self.config = config or SimulationConfig()
        self.results: list[SimulationResult] = []
        self._spice = SpiceSimulator()
        self._si = SignalIntegrityAnalyzer()
        self._thermal = ThermalAnalyzer()

    def run(self) -> list[SimulationResult]:
        """Execute the configured simulation pipeline."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results.clear()

        stages = self._resolve_stages()
        for stage in stages:
            result = self._run_stage(stage)
            self.results.append(result)
            if not result.success:
                break

        self._export_report()
        return self.results

    def _resolve_stages(self) -> list[SimulationType]:
        if self.config.sim_type == SimulationType.FULL:
            return [
                SimulationType.SPICE_DC,
                SimulationType.SPICE_AC,
                SimulationType.SPICE_TRANSIENT,
                SimulationType.SIGNAL_INTEGRITY,
                SimulationType.THERMAL,
            ]
        return [self.config.sim_type]

    def _run_stage(self, stage: SimulationType) -> SimulationResult:
        print(f"[pcb-sim] Running {stage.value} analysis...")
        try:
            if stage in (
                SimulationType.SPICE_DC,
                SimulationType.SPICE_AC,
                SimulationType.SPICE_TRANSIENT,
            ):
                return self._spice.run(self.design, stage, self.config)
            elif stage == SimulationType.SIGNAL_INTEGRITY:
                return self._si.run(self.design, self.config)
            elif stage == SimulationType.THERMAL:
                return self._thermal.run(self.design, self.config)
            else:
                return SimulationResult(
                    success=False,
                    sim_type=stage,
                    errors=[f"Unknown simulation type: {stage}"],
                )
        except Exception as exc:
            return SimulationResult(
                success=False, sim_type=stage, errors=[str(exc)]
            )

    def _export_report(self) -> None:
        exporter = ReportExporter(self.config.output_dir)
        exporter.export(self.design, self.results)

    @staticmethod
    def check_dependencies() -> dict[str, bool]:
        """Check which external tools are available on the system."""
        tools = {
            "ngspice": ["ngspice", "--version"],
            "kicad-cli": ["kicad-cli", "--version"],
            "openems": ["python3", "-c", "import CSXCAD; import openEMS"],
        }
        available = {}
        for name, cmd in tools.items():
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                available[name] = True
            except FileNotFoundError:
                available[name] = False
        return available
