#!/usr/bin/env python3
"""Stage 2 diagnostic — replicate full app structure to find freeze."""

import sys
import queue
import tkinter as tk
from tkinter import ttk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.parsers.ipc2581_loader import load_ipc2581
from src.gui.board_view import plot_board_2d, plot_board_3d
from src.gui.mechanical_view import plot_mechanical_summary, compute_board_summary

TEST_FILE = Path(__file__).parent.parent / "examples" / "ipc2581_test" / "test_board.cvg"


class DiagApp:
    def __init__(self, root):
        self.root = root
        root.title("Diagnostic - Full Structure")
        root.geometry("1200x800")

        self.design = None
        self._selected_component = None
        self._syncing_selection = False
        self._loading = False
        self.message_queue = queue.Queue()

        # --- Layout ---
        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left = ttk.Frame(paned, width=250)
        paned.add(left, weight=0)

        ttk.Button(left, text="Load Design", command=self._load).pack(fill=tk.X, padx=5, pady=5)

        self.comp_tree = ttk.Treeview(left, columns=("ref", "type"), show="headings", height=8)
        self.comp_tree.heading("ref", text="Ref")
        self.comp_tree.heading("type", text="Type")
        self.comp_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.comp_tree.bind("<<TreeviewSelect>>", self._on_component_selected)
        self.comp_tree.bind("<Button-1>", self._on_tree_click, add=True)

        # Center panel with notebook (7 tabs like full app)
        center = ttk.Frame(paned)
        paned.add(center, weight=1)

        self.notebook = ttk.Notebook(center)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.canvases = {}
        self.figs = {}
        tab_names = ["overview", "board3d", "mechanical", "thermal", "si", "magnetics", "transient"]
        for name in tab_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name.replace("_", " ").title())
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title(f"{name} - no data")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray")
            canvas = FigureCanvasTkAgg(fig, master=frame)
            print(f"  Drawing initial canvas: {name}...")
            canvas.draw()
            print(f"    done")
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figs[name] = fig
            self.canvases[name] = canvas

        # Mechanical treeview
        self.mech_frame = ttk.Frame(center)
        self.mech_tree = ttk.Treeview(self.mech_frame, columns=("refdes", "package"), show="headings", height=5)
        self.mech_tree.heading("refdes", text="RefDes")
        self.mech_tree.heading("package", text="Package")
        self.mech_tree.pack(fill=tk.BOTH, expand=True)
        self.mech_tree.bind("<<TreeviewSelect>>", self._on_component_selected)
        self.mech_tree.bind("<Button-1>", self._on_tree_click, add=True)

        # Status
        self.status = ttk.Label(root, text="Ready")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Periodic queue check (like full app)
        self.root.after(100, self._process_queue)
        print("Init complete. App is responsive.")

    def _process_queue(self):
        try:
            while True:
                self.message_queue.get_nowait()
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def _load(self):
        print("Loading design...")
        self._loading = True
        self._selected_component = None
        self.status.config(text="Loading...")
        self.root.update_idletasks()

        try:
            self.design = load_ipc2581(TEST_FILE)
            print(f"  Parsed: {self.design.name}, {len(self.design.components)} components")

            # Populate comp_tree
            for item in self.comp_tree.get_children():
                self.comp_tree.delete(item)
            for comp in self.design.components:
                self.comp_tree.insert("", tk.END, values=(comp.reference, comp.component_type.value))
            self.root.update_idletasks()
            print("  comp_tree populated")

            # Populate mech_tree
            for item in self.mech_tree.get_children():
                self.mech_tree.delete(item)
            for comp in self.design.components:
                self.mech_tree.insert("", tk.END, values=(comp.reference, "PKG"))
            self.root.update_idletasks()
            print("  mech_tree populated")

            # Plot 2D
            print("  Plotting 2D...")
            plot_board_2d(self.figs["overview"], self.design)
            self.canvases["overview"].draw_idle()
            self.root.update_idletasks()
            print("    done")

            # Plot 3D
            print("  Plotting 3D...")
            plot_board_3d(self.figs["board3d"], self.design)
            self.canvases["board3d"].draw_idle()
            self.root.update_idletasks()
            print("    done")

            self.status.config(text=f"Loaded: {self.design.name}")
            print("  Load complete!")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._loading = False

    def _on_component_selected(self, event):
        if self._syncing_selection or self._loading:
            return
        tree = event.widget
        sel = tree.selection()
        if not sel:
            return
        values = tree.item(sel[0], "values")
        if not values:
            return
        refdes = values[0]

        if refdes == self._selected_component:
            return

        print(f"  Selected: {refdes}")
        self._selected_component = refdes

        self._syncing_selection = True
        try:
            other = self.mech_tree if tree is self.comp_tree else self.comp_tree
            other.selection_remove(*other.selection())
            for iid in other.get_children():
                if other.item(iid, "values")[0] == refdes:
                    other.selection_set(iid)
                    other.see(iid)
                    break
        finally:
            self._syncing_selection = False

        if self.design:
            print("  Replotting 2D...")
            plot_board_2d(self.figs["overview"], self.design, highlight_component=refdes)
            self.canvases["overview"].draw_idle()
            print("  Replotting 3D...")
            plot_board_3d(self.figs["board3d"], self.design, highlight_component=refdes)
            self.canvases["board3d"].draw_idle()
            print("  Replot done")

        self.status.config(text=f"Selected: {refdes}")

    def _on_tree_click(self, event):
        tree = event.widget
        if tree.identify_region(event.x, event.y) == "nothing":
            print("  Click on empty space - clearing")
            self._syncing_selection = True
            try:
                self.comp_tree.selection_remove(*self.comp_tree.selection())
                self.mech_tree.selection_remove(*self.mech_tree.selection())
            finally:
                self._syncing_selection = False
            self._selected_component = None
            if self.design:
                plot_board_2d(self.figs["overview"], self.design)
                self.canvases["overview"].draw_idle()
                plot_board_3d(self.figs["board3d"], self.design)
                self.canvases["board3d"].draw_idle()
            self.status.config(text="Ready")


if __name__ == "__main__":
    print("=== Full Structure Diagnostic ===")
    root = tk.Tk()
    app = DiagApp(root)
    root.mainloop()
    print("Exited cleanly.")
