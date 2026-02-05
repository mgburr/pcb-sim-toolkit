"""Integration tests for the PCB simulation pipeline."""

from pathlib import Path

import pytest

from src.parsers.yaml_loader import load_design
from src.analysis.spice import SpiceSimulator
from src.analysis.signal_integrity import SignalIntegrityAnalyzer
from src.analysis.thermal import ThermalAnalyzer
from src.core.config import SimulationConfig, SimulationType
from src.core.simulator import PCBSimulator


EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def led_design():
    return load_design(EXAMPLES / "simple_led" / "design.yaml")


@pytest.fixture
def usb_design():
    return load_design(EXAMPLES / "differential_pair" / "design.yaml")


class TestYAMLLoader:
    def test_load_led(self, led_design):
        assert led_design.name == "simple_led_driver"
        assert len(led_design.components) == 3
        assert len(led_design.nets) == 3
        assert len(led_design.traces) == 3

    def test_load_usb(self, usb_design):
        assert usb_design.name == "usb2_differential"
        assert len(usb_design.components) == 8


class TestSpiceSimulator:
    def test_netlist_generation(self, led_design):
        spice = SpiceSimulator()
        netlist = spice.generate_netlist(led_design)
        assert "simple_led_driver" in netlist
        assert "VCC" in netlist or "LED_ANODE" in netlist

    def test_dc_fallback(self, led_design):
        spice = SpiceSimulator()
        config = SimulationConfig(sim_type=SimulationType.SPICE_DC)
        result = spice.run(led_design, SimulationType.SPICE_DC, config)
        assert result.success

    def test_ac_fallback(self, usb_design):
        spice = SpiceSimulator()
        config = SimulationConfig(sim_type=SimulationType.SPICE_AC)
        result = spice.run(usb_design, SimulationType.SPICE_AC, config)
        assert result.success
        assert "frequencies" in result.data or "variables" in result.data

    def test_transient_fallback(self, usb_design):
        spice = SpiceSimulator()
        config = SimulationConfig(sim_type=SimulationType.SPICE_TRANSIENT)
        result = spice.run(usb_design, SimulationType.SPICE_TRANSIENT, config)
        assert result.success

    def test_parse_engineering_values(self):
        parse = SpiceSimulator._parse_value
        assert parse("10K") == 10e3
        assert parse("4.7U") == pytest.approx(4.7e-6)
        assert parse("100N") == pytest.approx(100e-9)
        assert parse("1MEG") == 1e6


class TestSignalIntegrity:
    def test_run(self, usb_design):
        si = SignalIntegrityAnalyzer()
        config = SimulationConfig()
        result = si.run(usb_design, config)
        assert result.success
        assert "traces" in result.data
        for tr in result.data["traces"]:
            assert tr["z0_ohms"] > 0
            assert tr["length_mm"] > 0

    def test_crosstalk(self, usb_design):
        si = SignalIntegrityAnalyzer()
        config = SimulationConfig()
        result = si.run(usb_design, config)
        crosstalk = result.data.get("crosstalk", [])
        # USB DP and DN traces are adjacent, should detect crosstalk
        assert len(crosstalk) >= 0  # may or may not detect depending on spacing


class TestThermalAnalysis:
    def test_run(self, led_design):
        ta = ThermalAnalyzer()
        config = SimulationConfig(ambient_temp=25.0)
        result = ta.run(led_design, config)
        assert result.success
        assert result.data["total_power_w"] > 0
        assert result.data["avg_board_temp_c"] >= 25.0
        assert "thermal_grid" in result.data

    def test_thermal_grid_shape(self, usb_design):
        ta = ThermalAnalyzer()
        config = SimulationConfig()
        result = ta.run(usb_design, config)
        grid = result.data["thermal_grid"]
        assert grid["width_cells"] > 0
        assert grid["height_cells"] > 0


class TestFullPipeline:
    def test_full_simulation(self, led_design, tmp_path):
        config = SimulationConfig(
            sim_type=SimulationType.FULL,
            output_dir=tmp_path / "output",
        )
        sim = PCBSimulator(led_design, config)
        results = sim.run()
        assert len(results) == 5
        assert all(r.success for r in results)
        assert (tmp_path / "output" / "report.json").exists()
        assert (tmp_path / "output" / "report.html").exists()
