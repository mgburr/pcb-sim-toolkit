# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCB design simulation toolkit — SPICE, signal integrity, thermal, and magnetics analysis. Python 3.10+, MIT license.

## Common Commands

```bash
# Install (development)
python -m pip install -e ".[dev]"

# Run tests
pytest tests/ -v
pytest tests/test_simulation.py::TestSpiceSimulator -v          # single class
pytest tests/test_simulation.py::TestSpiceSimulator::test_netlist_generation -v  # single test

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# CLI usage
pcb-sim simulate examples/simple_led/design.yaml --type full
pcb-sim check          # check external tool availability
pcb-sim netlist examples/simple_led/design.yaml -o output.cir

# GUI
pcb-sim-gui
```

## Architecture

The simulation pipeline follows this flow:

```
Design File (YAML / KiCad .kicad_pcb / IPC-2581 XML)
  → Parser → PCBDesign model
  → PCBSimulator orchestrator
    ├→ SpiceSimulator (DC, AC, Transient — uses ngspice or fallback analytical solver)
    ├→ SignalIntegrityAnalyzer (Z0 via Hammerstad-Jensen, delay, crosstalk, eye diagram)
    ├→ ThermalAnalyzer (power estimation, 2D finite-difference thermal grid)
    └→ MagneticsAnalyzer (Biot-Savart B-field computation)
  → SimulationResult
  → Exporters (JSON report, HTML report, matplotlib plots)
```

**Key modules:**

- `src/core/models.py` — Dataclasses for PCBDesign, Component, Trace, Net, Stackup, Layer, Pad
- `src/core/simulator.py` — `PCBSimulator` orchestrator; `_resolve_stages()` expands `FULL` into individual stages
- `src/core/config.py` — `SimulationType` enum, `SimulationConfig`, `SimulationResult`
- `src/core/cli.py` — argparse CLI with `simulate`, `check`, `netlist` subcommands
- `src/analysis/spice.py` — Netlist generation and ngspice subprocess execution; includes `_fallback_simulation()` for when ngspice is unavailable
- `src/analysis/signal_integrity.py` — Characteristic impedance, propagation delay, crosstalk coupling
- `src/analysis/thermal.py` — Component power dissipation, board thermal resistance, 2D FD solver
- `src/analysis/magnetics.py` — Biot-Savart B-field grid computation, trace current estimation
- `src/parsers/` — Three loaders: YAML, KiCad S-expression, IPC-2581 XML
- `src/exporters/report.py` — HTML (Jinja2) and JSON report generation
- `src/exporters/magnetics_plots.py` — Matplotlib heatmaps, vector fields, contour plots
- `src/gui/app.py` — Tkinter GUI with design loading, visualization tabs, and result export

## External Tool Dependencies

ngspice, kicad-cli, and OpenEMS are optional. The simulator detects their availability at runtime (`check_dependencies()`) and uses analytical fallback solvers when ngspice is missing.

## Test Conventions

Tests use pytest with two main fixtures: `led_design` (simple_led example) and `usb_design` (differential_pair example). Tests validate against both ngspice and fallback solver paths. Output isolation uses pytest's `tmp_path` fixture.

## Example Designs

- `examples/simple_led/` — 3-component LED circuit (basic validation)
- `examples/differential_pair/` — USB 2.0 DP/DN with matched-length traces (advanced validation)
