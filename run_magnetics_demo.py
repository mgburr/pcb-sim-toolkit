"""Run magnetic field analysis on the differential pair design and generate plots."""

from pathlib import Path

from src.parsers.yaml_loader import load_design
from src.core.config import SimulationConfig
from src.analysis.magnetics import MagneticsAnalyzer
from src.exporters.magnetics_plots import generate_magnetics_plots


def main():
    design_path = Path("examples/differential_pair/design.yaml")
    output_dir = Path("diff_pair_output/magnetics")

    print("[magnetics] Loading design...")
    design = load_design(design_path)
    config = SimulationConfig()

    print("[magnetics] Computing magnetic fields (Biot-Savart)...")
    analyzer = MagneticsAnalyzer()
    result = analyzer.run(design, config, resolution_mm=0.25)

    print("[magnetics] Trace currents:")
    for tc in result.data.get("trace_currents", []):
        print(f"  {tc['net']:20s}  {tc['current_ma']:8.3f} mA  @ {tc['frequency_hz']:.0f} Hz")

    bfield = result.data.get("bfield_grid", {})
    print(f"[magnetics] B-field grid: {bfield.get('width_cells')}x{bfield.get('height_cells')} cells")
    print(f"[magnetics] B-field range: {bfield.get('min_b_tesla', 0)*1e6:.4f} - {bfield.get('max_b_tesla', 0)*1e6:.4f} uT")

    print("[magnetics] Trace B-field profiles:")
    for prof in result.data.get("trace_profiles", []):
        print(f"  {prof['net']:20s}  peak = {prof['peak_b_microtesla']:.3f} uT")

    # Prepare design data dict for the plotter (traces for overlay)
    design_data = {
        "name": design.name,
        "width": design.width,
        "height": design.height,
        "traces": [
            {
                "net": t.net,
                "width": t.width,
                "layer": t.layer,
                "points": list(t.points),
            }
            for t in design.traces
        ],
    }

    print("[magnetics] Generating plots...")
    plots = generate_magnetics_plots(design_data, result.data, output_dir)
    print(f"[magnetics] Generated {len(plots)} images:")
    for p in plots:
        print(f"  {p}")

    print("[magnetics] Done.")


if __name__ == "__main__":
    main()
