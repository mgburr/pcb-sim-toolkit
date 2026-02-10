"""Tests for PCB mechanical data extraction and visualization."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from src.core.models import (
    PCBDesign,
    Component,
    ComponentType,
    Pad,
)
from src.parsers.ipc2581_loader import (
    load_ipc2581,
    _parse_pad_definitions,
    _parse_packages,
    _compute_fallback_mechanical,
)
from src.gui.mechanical_view import compute_board_summary, plot_mechanical_summary

import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IPC2581_TEST_FILE = (
    Path(__file__).parent.parent / "examples" / "ipc2581_test" / "test_board.cvg"
)


@pytest.fixture
def fig():
    return Figure(figsize=(8, 6), dpi=72)


@pytest.fixture
def ipc_design() -> PCBDesign:
    assert IPC2581_TEST_FILE.exists(), f"Test file missing: {IPC2581_TEST_FILE}"
    return load_ipc2581(IPC2581_TEST_FILE)


@pytest.fixture
def ipc_root() -> ET.Element:
    tree = ET.parse(IPC2581_TEST_FILE)
    return tree.getroot()


# ---------------------------------------------------------------------------
# Pad definition parsing
# ---------------------------------------------------------------------------

class TestPadDefinitions:
    def test_pad_count(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        assert len(pads) == 2
        assert "PAD_0402" in pads
        assert "PAD_QFP48" in pads

    def test_pad_0402_shape(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        p = pads["PAD_0402"]
        assert p.shape_type == "rect"
        assert p.width == pytest.approx(0.6)
        assert p.height == pytest.approx(0.5)

    def test_pad_qfp48_shape(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        p = pads["PAD_QFP48"]
        assert p.shape_type == "rect"
        assert p.width == pytest.approx(0.3)
        assert p.height == pytest.approx(1.6)

    def test_smd_detection(self, ipc_root):
        """Pads without drill info should be SMD (drill_diameter == 0)."""
        pads = _parse_pad_definitions(ipc_root, "MM")
        for pad in pads.values():
            assert pad.drill_diameter == 0.0


# ---------------------------------------------------------------------------
# Package parsing
# ---------------------------------------------------------------------------

class TestPackageParsing:
    def test_package_count(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert len(pkgs) == 2
        assert "0402" in pkgs
        assert "QFP-48" in pkgs

    def test_0402_body_dimensions(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        pkg = pkgs["0402"]
        assert pkg.body_width == pytest.approx(1.0)
        assert pkg.body_height == pytest.approx(0.6)

    def test_0402_courtyard(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        pkg = pkgs["0402"]
        assert pkg.courtyard_width == pytest.approx(1.4)
        assert pkg.courtyard_height == pytest.approx(1.0)

    def test_0402_pin_count(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert pkgs["0402"].pin_count == 2

    def test_0402_pin_pitch(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert pkgs["0402"].pin_pitch == pytest.approx(0.7)

    def test_qfp48_body_dimensions(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        pkg = pkgs["QFP-48"]
        assert pkg.body_width == pytest.approx(7.0)
        assert pkg.body_height == pytest.approx(7.0)

    def test_qfp48_pin_count(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert pkgs["QFP-48"].pin_count == 12  # subset in test data

    def test_qfp48_pin_pitch(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert pkgs["QFP-48"].pin_pitch == pytest.approx(0.5)

    def test_pad_shape_linked(self, ipc_root):
        pads = _parse_pad_definitions(ipc_root, "MM")
        pkgs = _parse_packages(ipc_root, "MM", pads)
        assert pkgs["0402"].pad_shape is not None
        assert pkgs["0402"].pad_shape.shape_type == "rect"


# ---------------------------------------------------------------------------
# Component-to-package linking
# ---------------------------------------------------------------------------

class TestPackageLinking:
    def test_r1_linked(self, ipc_design):
        r1 = ipc_design.get_component("R1")
        assert r1 is not None
        assert r1.package_def is not None
        assert r1.package_def.name == "0402"

    def test_u1_linked(self, ipc_design):
        u1 = ipc_design.get_component("U1")
        assert u1 is not None
        assert u1.package_def is not None
        assert u1.package_def.name == "QFP-48"

    def test_c1_linked(self, ipc_design):
        c1 = ipc_design.get_component("C1")
        assert c1 is not None
        assert c1.package_def is not None
        assert c1.package_def.name == "0402"

    def test_packages_dict_populated(self, ipc_design):
        assert len(ipc_design.packages) == 2


# ---------------------------------------------------------------------------
# Height estimation
# ---------------------------------------------------------------------------

class TestHeightEstimation:
    def test_0402_height(self, ipc_design):
        r1 = ipc_design.get_component("R1")
        assert r1.package_def.height == pytest.approx(0.35)

    def test_qfp_height(self, ipc_design):
        u1 = ipc_design.get_component("U1")
        # QFP pattern should match
        assert u1.package_def.height == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fallback synthesis
# ---------------------------------------------------------------------------

class TestFallbackSynthesis:
    def test_fallback_for_missing_package(self):
        """Components with no Package element get synthesized package_def."""
        design = PCBDesign(
            name="fallback_test",
            width=20,
            height=15,
            components=[
                Component(
                    reference="R5",
                    component_type=ComponentType.RESISTOR,
                    value="4.7k",
                    footprint="unknown_pkg",
                    pads=[
                        Pad(name="1", x=5, y=5),
                        Pad(name="2", x=6, y=5),
                    ],
                ),
            ],
        )
        _compute_fallback_mechanical(design)
        r5 = design.components[0]
        assert r5.package_def is not None
        assert r5.package_def.source == "computed"
        assert r5.package_def.body_width > 0
        assert r5.package_def.height > 0

    def test_fallback_body_from_pads(self):
        """Fallback body size should be pad extent + margin."""
        design = PCBDesign(
            name="pad_extent",
            width=20,
            height=15,
            components=[
                Component(
                    reference="R6",
                    component_type=ComponentType.RESISTOR,
                    value="1k",
                    pads=[
                        Pad(name="1", x=0, y=0),
                        Pad(name="2", x=2, y=0),
                    ],
                ),
            ],
        )
        _compute_fallback_mechanical(design)
        pkg = design.components[0].package_def
        # Pad extent is 2mm, +0.6 margin = 2.6
        assert pkg.body_width == pytest.approx(2.6)
        assert pkg.body_height == pytest.approx(0.6)

    def test_no_overwrite_parsed_package(self, ipc_design):
        """Parsed packages should not be overwritten by fallback."""
        r1 = ipc_design.get_component("R1")
        assert r1.package_def.source == "parsed"


# ---------------------------------------------------------------------------
# Board summary computation
# ---------------------------------------------------------------------------

class TestBoardSummary:
    def test_summary_keys(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        assert "board_area_mm2" in summary
        assert "component_count" in summary
        assert "total_pin_count" in summary
        assert "components" in summary

    def test_board_area(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        assert summary["board_area_mm2"] == pytest.approx(50 * 40)

    def test_component_count(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        assert summary["component_count"] == 3

    def test_smd_count(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        # All test components are SMD (no drill)
        assert summary["smd_count"] == 3
        assert summary["through_hole_count"] == 0

    def test_board_weight_positive(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        assert summary["estimated_board_weight_grams"] > 0

    def test_component_details(self, ipc_design):
        summary = compute_board_summary(ipc_design)
        refdes_list = [c["refdes"] for c in summary["components"]]
        assert "R1" in refdes_list
        assert "U1" in refdes_list
        assert "C1" in refdes_list


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------

class TestMechanicalPlot:
    def test_renders_without_error(self, fig, ipc_design):
        plot_mechanical_summary(fig, ipc_design)
        assert len(fig.axes) == 2  # text + pie

    def test_renders_empty_design(self, fig):
        design = PCBDesign(name="empty", width=10, height=10)
        _compute_fallback_mechanical(design)
        plot_mechanical_summary(fig, design)
        assert len(fig.axes) == 2

    def test_figure_is_cleared(self, fig, ipc_design):
        plot_mechanical_summary(fig, ipc_design)
        plot_mechanical_summary(fig, ipc_design)
        assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_design_without_packages_field(self):
        """Old designs without packages dict should work gracefully."""
        design = PCBDesign(name="legacy", width=10, height=10)
        assert design.packages == {}

    def test_component_without_package_def(self):
        """Components default to package_def=None."""
        comp = Component(
            reference="R1",
            component_type=ComponentType.RESISTOR,
            value="10k",
        )
        assert comp.package_def is None

    def test_link_packages_no_crash(self):
        """link_packages on design with no packages should not crash."""
        design = PCBDesign(
            name="no_pkgs",
            width=10,
            height=10,
            components=[
                Component(
                    reference="R1",
                    component_type=ComponentType.RESISTOR,
                    value="10k",
                    footprint="0402",
                ),
            ],
        )
        design.link_packages()
        assert design.components[0].package_def is None

    def test_summary_without_packages(self):
        """Board summary should work for designs without package data."""
        design = PCBDesign(
            name="no_pkgs",
            width=20,
            height=15,
            components=[
                Component(
                    reference="R1",
                    component_type=ComponentType.RESISTOR,
                    value="10k",
                    pads=[Pad(name="1", x=5, y=5)],
                ),
            ],
        )
        summary = compute_board_summary(design)
        assert summary["component_count"] == 1
