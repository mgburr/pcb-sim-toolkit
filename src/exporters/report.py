"""Export simulation results as HTML/JSON reports with plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.models import PCBDesign
from ..core.config import SimulationResult, SimulationType


class ReportExporter:
    """Generate simulation reports in multiple formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def export(
        self, design: PCBDesign, results: list[SimulationResult]
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._export_json(design, results)
        self._export_html(design, results)
        self._generate_plots(results)

    def _export_json(
        self, design: PCBDesign, results: list[SimulationResult]
    ) -> None:
        out = {
            "design": {
                "name": design.name,
                "width_mm": design.width,
                "height_mm": design.height,
                "n_components": len(design.components),
                "n_nets": len(design.nets),
                "n_traces": len(design.traces),
            },
            "results": [],
        }
        for r in results:
            out["results"].append(
                {
                    "sim_type": r.sim_type.value,
                    "success": r.success,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "data": _serialize(r.data),
                }
            )
        path = self.output_dir / "report.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def _export_html(
        self, design: PCBDesign, results: list[SimulationResult]
    ) -> None:
        html_parts: list[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append(f"<title>PCB Simulation Report - {design.name}</title>")
        html_parts.append(self._css())
        html_parts.append("</head><body>")
        html_parts.append(f"<h1>PCB Simulation Report: {design.name}</h1>")

        # Design summary
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Design Summary</h2>")
        html_parts.append(f"<p>Board size: {design.width} x {design.height} mm</p>")
        html_parts.append(f"<p>Components: {len(design.components)}</p>")
        html_parts.append(f"<p>Nets: {len(design.nets)}</p>")
        html_parts.append(f"<p>Traces: {len(design.traces)}</p>")
        html_parts.append("</div>")

        # Results per stage
        for r in results:
            status = "pass" if r.success else "fail"
            html_parts.append(f"<div class='section {status}'>")
            html_parts.append(
                f"<h2>{r.sim_type.value.replace('_', ' ').title()} "
                f"[{'PASS' if r.success else 'FAIL'}]</h2>"
            )

            if r.warnings:
                html_parts.append("<div class='warnings'>")
                for w in r.warnings:
                    html_parts.append(f"<p class='warning'>Warning: {w}</p>")
                html_parts.append("</div>")

            if r.errors:
                html_parts.append("<div class='errors'>")
                for e in r.errors:
                    html_parts.append(f"<p class='error'>Error: {e}</p>")
                html_parts.append("</div>")

            # Render key data points
            html_parts.append(self._render_data(r))
            html_parts.append("</div>")

        html_parts.append("</body></html>")

        path = self.output_dir / "report.html"
        path.write_text("\n".join(html_parts))

    def _render_data(self, result: SimulationResult) -> str:
        """Render simulation data as HTML tables."""
        parts: list[str] = []

        if result.sim_type == SimulationType.SIGNAL_INTEGRITY:
            traces = result.data.get("traces", [])
            if traces:
                parts.append("<h3>Trace Impedances</h3>")
                parts.append(
                    "<table><tr><th>Net</th><th>Z0 (&Omega;)</th>"
                    "<th>Length (mm)</th><th>Delay (ps)</th>"
                    "<th>Loss (dB/mm @ 1GHz)</th></tr>"
                )
                for t in traces:
                    parts.append(
                        f"<tr><td>{t['net']}</td><td>{t['z0_ohms']}</td>"
                        f"<td>{t['length_mm']}</td><td>{t['total_delay_ps']}</td>"
                        f"<td>{t['loss_db_per_mm_1ghz']}</td></tr>"
                    )
                parts.append("</table>")

            eye = result.data.get("eye_diagram", {})
            eye_traces = eye.get("traces", [])
            if eye_traces:
                parts.append("<h3>Eye Diagram Estimates</h3>")
                parts.append(
                    f"<p>Bit rate: {eye.get('bit_rate_gbps', 0)} Gbps</p>"
                )
                parts.append(
                    "<table><tr><th>Net</th><th>Eye Height</th>"
                    "<th>Eye Width (ps)</th><th>Margin %</th></tr>"
                )
                for et in eye_traces:
                    parts.append(
                        f"<tr><td>{et['net']}</td>"
                        f"<td>{et['eye_height_normalized']}</td>"
                        f"<td>{et['eye_width_ps']}</td>"
                        f"<td>{et['margin_pct']}</td></tr>"
                    )
                parts.append("</table>")

        elif result.sim_type == SimulationType.THERMAL:
            parts.append(
                f"<p>Total power: {result.data.get('total_power_w', 0)} W</p>"
            )
            parts.append(
                f"<p>Average board temp: "
                f"{result.data.get('avg_board_temp_c', 0)} &deg;C</p>"
            )
            grid_info = result.data.get("thermal_grid", {})
            if grid_info:
                parts.append(
                    f"<p>Thermal range: {grid_info.get('min_temp_c', 0)} - "
                    f"{grid_info.get('max_temp_c', 0)} &deg;C</p>"
                )
                parts.append(
                    "<p><em>See thermal_heatmap.png for visualization</em></p>"
                )

        elif result.sim_type in (
            SimulationType.SPICE_DC,
            SimulationType.SPICE_AC,
            SimulationType.SPICE_TRANSIENT,
        ):
            if "node_voltages" in result.data:
                parts.append("<h3>DC Operating Point</h3>")
                parts.append("<table><tr><th>Node</th><th>Voltage (V)</th></tr>")
                for node, v in result.data["node_voltages"].items():
                    parts.append(f"<tr><td>{node}</td><td>{v}</td></tr>")
                parts.append("</table>")
            if "analysis" in result.data:
                parts.append(
                    f"<p>Analysis type: {result.data['analysis']}</p>"
                )

        return "\n".join(parts)

    def _generate_plots(self, results: list[SimulationResult]) -> None:
        """Generate matplotlib plots for simulation results."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        for r in results:
            if r.sim_type == SimulationType.THERMAL:
                grid_data = r.data.get("thermal_grid", {}).get("grid")
                if grid_data:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(
                        np.array(grid_data),
                        cmap="hot",
                        interpolation="bilinear",
                        origin="lower",
                    )
                    fig.colorbar(im, label="Temperature (C)")
                    ax.set_title("Thermal Heatmap")
                    ax.set_xlabel("X (cells)")
                    ax.set_ylabel("Y (cells)")
                    fig.savefig(
                        self.output_dir / "thermal_heatmap.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            elif r.sim_type == SimulationType.SPICE_AC:
                freqs = r.data.get("frequencies")
                impedances = r.data.get("impedances", {})
                if freqs and impedances:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for ref, zdata in impedances.items():
                        ax.loglog(freqs, zdata["magnitude"], label=ref)
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Impedance (Ohms)")
                    ax.set_title("AC Impedance Sweep")
                    ax.legend()
                    ax.grid(True, which="both", alpha=0.3)
                    fig.savefig(
                        self.output_dir / "ac_impedance.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            elif r.sim_type == SimulationType.SPICE_TRANSIENT:
                time = r.data.get("time")
                waveforms = r.data.get("waveforms", {})
                if time and waveforms:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    t_us = [t * 1e6 for t in time]
                    for ref, wf in waveforms.items():
                        ax.plot(t_us, wf, label=ref)
                    ax.set_xlabel("Time (us)")
                    ax.set_ylabel("Voltage (V)")
                    ax.set_title("Transient Simulation")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.savefig(
                        self.output_dir / "transient.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

    @staticmethod
    def _css() -> str:
        return """
<style>
  body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #34495e; }
  .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .section.fail { border-left: 4px solid #e74c3c; }
  .section.pass { border-left: 4px solid #2ecc71; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background: #3498db; color: white; }
  tr:nth-child(even) { background: #f2f2f2; }
  .warning { color: #f39c12; }
  .error { color: #e74c3c; font-weight: bold; }
</style>"""


def _serialize(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj
