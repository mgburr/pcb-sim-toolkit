"""Tests for board view renderers and IPC-2581 parser enhancements."""

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
    Net,
    Trace,
    Pad,
    Stackup,
    Layer,
    LayerType,
)
from src.gui.board_view import plot_board_2d, plot_board_3d
from src.parsers.ipc2581_loader import load_ipc2581


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IPC2581_TEST_FILE = Path(__file__).parent.parent / "examples" / "ipc2581_test" / "test_board.cvg"


@pytest.fixture
def fig():
    return Figure(figsize=(6, 4), dpi=72)


@pytest.fixture
def minimal_design() -> PCBDesign:
    """A minimal in-code PCBDesign with no outline and no stackup."""
    return PCBDesign(
        name="minimal",
        width=20,
        height=15,
        components=[
            Component(
                reference="R1",
                component_type=ComponentType.RESISTOR,
                value="10k",
                pads=[
                    Pad(name="1", x=5, y=5),
                    Pad(name="2", x=6, y=5),
                ],
            ),
        ],
        nets=[Net(name="VCC", nodes=["R1.1"])],
        traces=[
            Trace(net="VCC", width=0.25, layer="Top", points=[(5, 5), (10, 5)]),
        ],
    )


@pytest.fixture
def full_design() -> PCBDesign:
    """A richer design with outline, stackup, and multiple components."""
    return PCBDesign(
        name="full_test",
        width=50,
        height=40,
        outline=[(0, 0), (50, 0), (50, 40), (0, 40), (0, 0)],
        stackup=Stackup(layers=[
            Layer(name="Top", layer_type=LayerType.SIGNAL, thickness=0.035, material="copper"),
            Layer(name="Dielectric1", layer_type=LayerType.DIELECTRIC, thickness=0.2,
                  material="FR-4", dielectric_constant=4.5),
            Layer(name="Bottom", layer_type=LayerType.SIGNAL, thickness=0.035, material="copper"),
        ]),
        components=[
            Component(
                reference="R1",
                component_type=ComponentType.RESISTOR,
                value="10k",
                pads=[Pad(name="1", x=10, y=15), Pad(name="2", x=11, y=15)],
                rotation=0.0,
                layer="Top",
            ),
            Component(
                reference="U1",
                component_type=ComponentType.IC,
                value="MCU",
                pads=[
                    Pad(name="1", x=30, y=25),
                    Pad(name="2", x=31, y=25),
                    Pad(name="3", x=32, y=25),
                ],
                rotation=45.0,
                layer="Top",
            ),
            Component(
                reference="C1",
                component_type=ComponentType.CAPACITOR,
                value="100nF",
                pads=[Pad(name="1", x=25, y=15), Pad(name="2", x=26, y=15)],
                rotation=0.0,
                layer="Bottom",
            ),
        ],
        nets=[
            Net(name="VCC", nodes=["R1.1", "C1.1", "U1.1"]),
            Net(name="GND", nodes=["R1.2", "C1.2", "U1.2"]),
            Net(name="DATA", nodes=["U1.3"]),
        ],
        traces=[
            Trace(net="VCC", width=0.25, layer="Top", points=[(10, 15), (25, 15), (30, 25)]),
            Trace(net="GND", width=0.2, layer="Top", points=[(11, 15), (26, 15)]),
        ],
    )


# ---------------------------------------------------------------------------
# 2D rendering tests
# ---------------------------------------------------------------------------

class TestPlotBoard2D:
    def test_renders_without_error(self, fig, full_design):
        plot_board_2d(fig, full_design)
        assert len(fig.axes) == 1

    def test_renders_minimal_design(self, fig, minimal_design):
        """Minimal design (no outline, no stackup) should render fine."""
        plot_board_2d(fig, minimal_design)
        assert len(fig.axes) == 1

    def test_empty_design(self, fig):
        """Completely empty design (no components, no traces)."""
        design = PCBDesign(name="empty", width=10, height=10)
        plot_board_2d(fig, design)
        assert len(fig.axes) == 1

    def test_highlight_net(self, fig, full_design):
        """Net highlighting should not raise."""
        plot_board_2d(fig, full_design, highlight_net="VCC")
        assert len(fig.axes) == 1

    def test_highlight_nonexistent_net(self, fig, full_design):
        """Highlighting a net that doesn't exist should not raise."""
        plot_board_2d(fig, full_design, highlight_net="NONEXISTENT")
        assert len(fig.axes) == 1

    def test_figure_is_cleared(self, fig, full_design):
        """Calling plot_board_2d twice should clear previous content."""
        plot_board_2d(fig, full_design)
        plot_board_2d(fig, full_design)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# 3D rendering tests
# ---------------------------------------------------------------------------

class TestPlotBoard3D:
    def test_renders_without_error(self, fig, full_design):
        plot_board_3d(fig, full_design)
        assert len(fig.axes) == 1

    def test_renders_minimal_design(self, fig, minimal_design):
        """Minimal design (no stackup) should render fine."""
        plot_board_3d(fig, minimal_design)
        assert len(fig.axes) == 1

    def test_empty_stackup(self, fig):
        """Design with empty stackup should not raise."""
        design = PCBDesign(name="no_stackup", width=10, height=10)
        plot_board_3d(fig, design)
        assert len(fig.axes) == 1

    def test_bottom_layer_component(self, fig, full_design):
        """Components on the bottom layer should render."""
        plot_board_3d(fig, full_design)
        # C1 is on Bottom layer — just verify no errors
        assert len(fig.axes) == 1

    def test_figure_is_cleared(self, fig, full_design):
        """Calling plot_board_3d twice should clear previous content."""
        plot_board_3d(fig, full_design)
        plot_board_3d(fig, full_design)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# IPC-2581 parser enhancement tests
# ---------------------------------------------------------------------------

class TestIPC2581ParserEnhancements:
    @pytest.fixture
    def ipc_design(self) -> PCBDesign:
        assert IPC2581_TEST_FILE.exists(), f"Test file missing: {IPC2581_TEST_FILE}"
        return load_ipc2581(IPC2581_TEST_FILE)

    def test_outline_parsed(self, ipc_design):
        """Board outline polygon should be extracted."""
        assert len(ipc_design.outline) == 5  # closed rectangle: 5 points
        # First point should be origin
        assert ipc_design.outline[0] == (0.0, 0.0)
        # Last point should close the polygon
        assert ipc_design.outline[-1] == ipc_design.outline[0]

    def test_dimensions_from_outline(self, ipc_design):
        """Width and height should match the outline bounding box."""
        assert ipc_design.width == pytest.approx(50.0)
        assert ipc_design.height == pytest.approx(40.0)

    def test_components_parsed(self, ipc_design):
        """All three components should be parsed."""
        assert len(ipc_design.components) == 3
        refs = {c.reference for c in ipc_design.components}
        assert refs == {"R1", "C1", "U1"}

    def test_component_rotation_default(self, ipc_design):
        """Components without explicit rotation should default to 0."""
        for comp in ipc_design.components:
            assert comp.rotation == pytest.approx(0.0)

    def test_component_layer_default(self, ipc_design):
        """Components without explicit layer should default to 'Top'."""
        for comp in ipc_design.components:
            assert comp.layer == "Top"

    def test_2d_render_with_ipc_data(self, fig, ipc_design):
        """2D rendering should work with parsed IPC-2581 data."""
        plot_board_2d(fig, ipc_design)
        assert len(fig.axes) == 1

    def test_3d_render_with_ipc_data(self, fig, ipc_design):
        """3D rendering should work with parsed IPC-2581 data."""
        plot_board_3d(fig, ipc_design)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# Model backward compatibility
# ---------------------------------------------------------------------------

class TestModelBackwardCompat:
    def test_component_defaults(self):
        """Component should work without rotation/layer args."""
        comp = Component(
            reference="R1",
            component_type=ComponentType.RESISTOR,
            value="10k",
        )
        assert comp.rotation == 0.0
        assert comp.layer == "Top"

    def test_design_defaults(self):
        """PCBDesign should work without outline arg."""
        design = PCBDesign(name="test", width=10, height=10)
        assert design.outline == []
