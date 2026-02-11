"""Headless GUI tests for component highlighting and selection sync.

Requires Xvfb (virtual framebuffer) to run without a physical display.
These tests launch the real Tkinter GUI, load a design, simulate user
clicks, and verify the selection state and replot pass-through.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gui.app import PCBSimulatorGUI


# ---------------------------------------------------------------------------
# Xvfb fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def virtual_display():
    """Start Xvfb for the module if no display is set."""
    if os.environ.get("DISPLAY"):
        yield
        return
    proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    yield
    proc.terminate()
    proc.wait()


# ---------------------------------------------------------------------------
# GUI fixture
# ---------------------------------------------------------------------------

TEST_DESIGN = Path(__file__).parent.parent / "examples" / "ipc2581_test" / "test_board.cvg"


@pytest.fixture
def gui():
    """Create a GUI instance, load the test design, and tear down after."""
    root = tk.Tk()
    root.withdraw()  # hide window
    app = PCBSimulatorGUI(root)
    app._load_design(TEST_DESIGN)
    root.update_idletasks()
    yield app
    root.destroy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGUIComponentSelection:
    """Verify selection state and sync between treeviews."""

    def test_initial_state_no_selection(self, gui):
        """No component should be selected after loading a design."""
        assert gui._selected_component is None
        assert gui.comp_tree.selection() == ()
        assert gui.mech_tree.selection() == ()

    def test_select_in_comp_tree(self, gui):
        """Selecting a row in comp_tree should update _selected_component."""
        items = gui.comp_tree.get_children()
        assert len(items) > 0
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        refdes = gui.comp_tree.item(items[0], "values")[0]
        assert gui._selected_component == refdes

    def test_sync_comp_to_mech(self, gui):
        """Selecting in comp_tree should sync selection to mech_tree."""
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        refdes = gui.comp_tree.item(items[0], "values")[0]

        # mech_tree should have matching selection
        mech_sel = gui.mech_tree.selection()
        assert len(mech_sel) == 1
        mech_refdes = gui.mech_tree.item(mech_sel[0], "values")[0]
        assert mech_refdes == refdes

    def test_sync_mech_to_comp(self, gui):
        """Selecting in mech_tree should sync selection to comp_tree."""
        items = gui.mech_tree.get_children()
        assert len(items) > 0
        gui.mech_tree.selection_set(items[0])
        gui.mech_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        refdes = gui.mech_tree.item(items[0], "values")[0]

        comp_sel = gui.comp_tree.selection()
        assert len(comp_sel) == 1
        comp_refdes = gui.comp_tree.item(comp_sel[0], "values")[0]
        assert comp_refdes == refdes
        assert gui._selected_component == refdes

    def test_select_each_component(self, gui):
        """Selecting each component in turn should update state correctly."""
        for item in gui.comp_tree.get_children():
            gui.comp_tree.selection_set(item)
            gui.comp_tree.event_generate("<<TreeviewSelect>>")
            gui.root.update_idletasks()

            refdes = gui.comp_tree.item(item, "values")[0]
            assert gui._selected_component == refdes

    def test_clear_selection(self, gui):
        """Clearing selection should reset _selected_component to None."""
        # First select something
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()
        assert gui._selected_component is not None

        # Simulate clicking empty space by calling the handler directly
        gui.comp_tree.selection_remove(*gui.comp_tree.selection())
        gui.mech_tree.selection_remove(*gui.mech_tree.selection())
        gui._selected_component = None
        gui._plot_board_overview()
        gui._plot_board_3d()
        gui.root.update_idletasks()

        assert gui._selected_component is None
        assert gui.comp_tree.selection() == ()
        assert gui.mech_tree.selection() == ()


class TestGUIHighlightPassThrough:
    """Verify that selection triggers replot with highlight_component."""

    def test_2d_plot_receives_highlight(self, gui):
        """After selecting a component, the 2D figure should show green text."""
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        refdes = gui._selected_component
        ax = gui.overview_fig.axes[0]
        for txt in ax.texts:
            if txt.get_text() == refdes:
                assert txt.get_color() == "#00ff00"
                break
        else:
            pytest.fail(f"Text label for {refdes} not found in 2D plot")

    def test_3d_plot_receives_highlight(self, gui):
        """After selecting a component, the 3D figure should show green text."""
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        refdes = gui._selected_component
        ax = gui.board3d_fig.axes[0]
        for txt in ax.texts:
            if txt.get_text() == refdes:
                assert txt.get_color() == "#00ff00"
                break
        else:
            pytest.fail(f"Text label for {refdes} not found in 3D plot")

    def test_no_highlight_after_clear(self, gui):
        """After clearing selection, no text should be green in 2D plot."""
        # Select then clear
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()

        gui.comp_tree.selection_remove(*gui.comp_tree.selection())
        gui.mech_tree.selection_remove(*gui.mech_tree.selection())
        gui._selected_component = None
        gui._plot_board_overview()
        gui._plot_board_3d()
        gui.root.update_idletasks()

        ax = gui.overview_fig.axes[0]
        for txt in ax.texts:
            assert txt.get_color() != "#00ff00", (
                f"Text '{txt.get_text()}' should not be green after clearing"
            )

    def test_load_design_resets_selection(self, gui):
        """Loading a new design should clear the selection."""
        items = gui.comp_tree.get_children()
        gui.comp_tree.selection_set(items[0])
        gui.comp_tree.event_generate("<<TreeviewSelect>>")
        gui.root.update_idletasks()
        assert gui._selected_component is not None

        gui._load_design(TEST_DESIGN)
        gui.root.update_idletasks()
        assert gui._selected_component is None
