# pcb-sim-toolkit

Open-source PCB design simulation toolkit. Runs SPICE circuit analysis, signal integrity checks, thermal simulation, and magnetic field analysis on PCB designs defined in YAML or imported from KiCad.

Features both a command-line interface and a full graphical user interface (GUI).

## Architecture

```
pcb-sim-toolkit/
  src/
    core/          # Data models, orchestrator, CLI
    analysis/      # SPICE, signal integrity, thermal, magnetics engines
    parsers/       # YAML and KiCad .kicad_pcb loaders
    exporters/     # HTML/JSON report generation with plots
    gui/           # Tkinter-based graphical user interface
  examples/        # Sample PCB designs
  configs/         # Simulation configuration presets
  tests/           # pytest test suite
```

## Simulation Pipeline

1. **SPICE DC** - Operating point analysis (node voltages, branch currents)
2. **SPICE AC** - Frequency-domain impedance sweep
3. **SPICE Transient** - Time-domain waveform simulation
4. **Signal Integrity** - Characteristic impedance (Z0), propagation delay, crosstalk (NEXT), eye diagram estimation
5. **Thermal** - 2D finite-difference steady-state temperature distribution, per-component junction temperature
6. **Magnetics** - Magnetic field (B-field) distribution using Biot-Savart law for current-carrying traces

When `ngspice` is installed, SPICE stages run natively. Otherwise, built-in analytical solvers provide approximate results.

### Magnetics Analysis

The magnetics module computes magnetic field maps around PCB traces:

```bash
python run_magnetics_demo.py
```

Outputs:
- `mag_field_magnitude.png` - Full-board B-field heatmap
- `mag_field_vectors.png` - Vector field showing B direction
- `mag_field_contour.png` - Iso-B contour lines
- `mag_field_log_scale.png` - Logarithmic scale for weak-field regions
- `mag_field_profiles.png` - Cross-section profiles per trace
- `mag_field_overview.png` - Combined dashboard view

## Setup

```bash
cd pcb-sim-toolkit
python -m pip install -e ".[dev]"
```

### Optional external tools

| Tool | Purpose | Install |
|------|---------|---------|
| ngspice | Full SPICE simulation | `brew install ngspice` / `apt install ngspice` |
| KiCad | Import .kicad_pcb files | https://www.kicad.org |
| matplotlib | Plot generation | Included in dependencies |

## Graphical User Interface (GUI)

Launch the GUI application:

```bash
python run_gui.py
# or after installation:
pcb-sim-gui
```

### GUI Features

- **Design Panel** (left): Load YAML or KiCad designs, view component list
- **Visualization Panel** (center): Tabbed views for Board Overview, Thermal, Signal Integrity, Magnetics, and Transient plots
- **Results Panel** (right): Hierarchical results tree, simulation log, export options
- **Configuration**: Set frequency range, time step, duration, ambient temperature
- **Toolbar**: Run Full Simulation, Run Magnetics Analysis, Open Reports

### GUI Workflow

1. Click **Load Design** to open a YAML or .kicad_pcb file
2. Review components and board layout in the Overview tab
3. Adjust simulation parameters if needed
4. Click **Run Full Simulation** or choose a specific analysis from the Simulation menu
5. View results in the tabbed plot area and results tree
6. Click **Open Report** to view HTML report in browser

## Command-Line Usage

### Check dependencies

```bash
pcb-sim check
```

### Run full simulation

```bash
pcb-sim simulate examples/simple_led/design.yaml -o ./output
```

### Run specific analysis

```bash
pcb-sim simulate examples/differential_pair/design.yaml --type signal_integrity
pcb-sim simulate examples/differential_pair/design.yaml --type thermal
pcb-sim simulate examples/simple_led/design.yaml --type dc
```

### Export SPICE netlist

```bash
pcb-sim netlist examples/simple_led/design.yaml -o circuit.cir
```

### Use a config file

```bash
pcb-sim simulate design.yaml --config configs/default.yaml
```

## Design Format (YAML)

```yaml
name: my_board
width: 100       # mm
height: 80       # mm

stackup:
  - name: F.Cu
    type: signal
    thickness: 0.035
    material: copper
  - name: Dielectric
    type: dielectric
    thickness: 0.2
    material: FR4
    er: 4.4

components:
  - ref: R1
    type: R
    value: 10K
    pads:
      - { name: "1", x: 10, y: 20 }
      - { name: "2", x: 15, y: 20 }

nets:
  - name: NET1
    nodes: ["R1.1", "U1.3"]

traces:
  - net: NET1
    width: 0.2
    layer: F.Cu
    points: [[10, 20], [15, 20]]
```

## Outputs

Simulations produce:
- `report.json` - Machine-readable results
- `report.html` - Human-readable report with tables
- `thermal_heatmap.png` - Board temperature visualization
- `ac_impedance.png` - Frequency response plot
- `transient.png` - Time-domain waveforms
- `*_netlist.cir` - Generated SPICE netlists

## Tests

```bash
pytest tests/ -v
```

## License

MIT
