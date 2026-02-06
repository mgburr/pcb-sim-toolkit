"""PCB Simulation Toolkit - Graphical User Interface.

A complete GUI for loading PCB designs, configuring simulations,
running analyses, and viewing results with interactive plots.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Callable
import webbrowser

# Matplotlib integration with Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

# Local imports
import sys
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parsers.yaml_loader import load_design
from src.parsers.kicad_loader import load_kicad_pcb
from src.parsers.ipc2581_loader import load_ipc2581
from src.core.config import SimulationConfig, SimulationType
from src.core.simulator import PCBSimulator
from src.core.models import PCBDesign
from src.analysis.magnetics import MagneticsAnalyzer
from src.resource_path import get_examples_dir


class PCBSimulatorGUI:
    """Main GUI application for PCB simulation toolkit."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PCB Simulation Toolkit")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)

        # State
        self.design: PCBDesign | None = None
        self.design_path: Path | None = None
        self.results: list = []
        self.magnetics_data: dict | None = None
        if getattr(sys, "frozen", False):
            self.output_dir = Path.home() / "Documents" / "PCBSimToolkit" / "sim_output"
        else:
            self.output_dir = Path("./sim_output")
        self.sim_thread: threading.Thread | None = None
        self.message_queue: queue.Queue = queue.Queue()

        # Style configuration
        self._configure_styles()

        # Build UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Periodic queue check for thread-safe UI updates
        self.root.after(100, self._process_queue)

    def _configure_styles(self):
        """Configure ttk styles for modern appearance."""
        style = ttk.Style()
        style.theme_use("clam")

        # Custom styles
        style.configure("Title.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 10))
        style.configure("Run.TButton", font=("Helvetica", 11, "bold"))
        style.configure(
            "Treeview",
            rowheight=25,
            font=("Helvetica", 10),
        )
        style.configure(
            "Treeview.Heading",
            font=("Helvetica", 10, "bold"),
        )

    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Design...", command=self._open_design, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Recent", state="disabled")
        file_menu.add_separator()
        file_menu.add_command(label="Set Output Directory...", command=self._set_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")

        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Run Full Simulation", command=lambda: self._run_simulation("full"))
        sim_menu.add_separator()
        sim_menu.add_command(label="Run SPICE DC", command=lambda: self._run_simulation("dc"))
        sim_menu.add_command(label="Run SPICE AC", command=lambda: self._run_simulation("ac"))
        sim_menu.add_command(label="Run SPICE Transient", command=lambda: self._run_simulation("transient"))
        sim_menu.add_command(label="Run Signal Integrity", command=lambda: self._run_simulation("signal_integrity"))
        sim_menu.add_command(label="Run Thermal", command=lambda: self._run_simulation("thermal"))
        sim_menu.add_separator()
        sim_menu.add_command(label="Run Magnetics Analysis", command=self._run_magnetics)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Open Output Folder", command=self._open_output_folder)
        view_menu.add_command(label="Open HTML Report", command=self._open_html_report)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._open_design())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    def _create_main_layout(self):
        """Create the main three-panel layout."""
        # Main container with PanedWindow for resizable panels
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Design & Config
        self.left_panel = ttk.Frame(self.main_paned, width=320)
        self.main_paned.add(self.left_panel, weight=0)
        self._create_left_panel()

        # Center panel - Visualization
        self.center_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.center_panel, weight=1)
        self._create_center_panel()

        # Right panel - Results
        self.right_panel = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(self.right_panel, weight=0)
        self._create_right_panel()

    def _create_left_panel(self):
        """Create the left panel with design info and configuration."""
        # Design Info Section
        design_frame = ttk.LabelFrame(self.left_panel, text="Design", padding=10)
        design_frame.pack(fill=tk.X, padx=5, pady=5)

        # Load button
        btn_frame = ttk.Frame(design_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_frame, text="Load Design...", command=self._open_design).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Reload", command=self._reload_design).pack(side=tk.LEFT, padx=5)

        # Design info display
        self.design_info = ttk.Label(design_frame, text="No design loaded", style="Status.TLabel")
        self.design_info.pack(fill=tk.X)

        # Component list
        comp_frame = ttk.LabelFrame(self.left_panel, text="Components", padding=5)
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.comp_tree = ttk.Treeview(
            comp_frame,
            columns=("ref", "type", "value"),
            show="headings",
            height=8,
        )
        self.comp_tree.heading("ref", text="Ref")
        self.comp_tree.heading("type", text="Type")
        self.comp_tree.heading("value", text="Value")
        self.comp_tree.column("ref", width=50)
        self.comp_tree.column("type", width=60)
        self.comp_tree.column("value", width=80)

        comp_scroll = ttk.Scrollbar(comp_frame, orient=tk.VERTICAL, command=self.comp_tree.yview)
        self.comp_tree.configure(yscrollcommand=comp_scroll.set)
        self.comp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        comp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Simulation Configuration
        config_frame = ttk.LabelFrame(self.left_panel, text="Simulation Config", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Frequency range
        ttk.Label(config_frame, text="AC Frequency Range (Hz):").pack(anchor=tk.W)
        freq_frame = ttk.Frame(config_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        self.freq_start = ttk.Entry(freq_frame, width=12)
        self.freq_start.insert(0, "1e3")
        self.freq_start.pack(side=tk.LEFT)
        ttk.Label(freq_frame, text=" to ").pack(side=tk.LEFT)
        self.freq_end = ttk.Entry(freq_frame, width=12)
        self.freq_end.insert(0, "1e9")
        self.freq_end.pack(side=tk.LEFT)

        # Time step
        ttk.Label(config_frame, text="Transient Time Step (s):").pack(anchor=tk.W, pady=(5, 0))
        self.time_step = ttk.Entry(config_frame, width=12)
        self.time_step.insert(0, "1e-9")
        self.time_step.pack(anchor=tk.W, pady=2)

        # Duration
        ttk.Label(config_frame, text="Transient Duration (s):").pack(anchor=tk.W)
        self.duration = ttk.Entry(config_frame, width=12)
        self.duration.insert(0, "1e-6")
        self.duration.pack(anchor=tk.W, pady=2)

        # Ambient temp
        ttk.Label(config_frame, text="Ambient Temperature (C):").pack(anchor=tk.W, pady=(5, 0))
        self.ambient_temp = ttk.Entry(config_frame, width=12)
        self.ambient_temp.insert(0, "25.0")
        self.ambient_temp.pack(anchor=tk.W, pady=2)

        # Run buttons
        run_frame = ttk.LabelFrame(self.left_panel, text="Run Simulation", padding=10)
        run_frame.pack(fill=tk.X, padx=5, pady=5)

        self.run_full_btn = ttk.Button(
            run_frame,
            text="Run Full Simulation",
            style="Run.TButton",
            command=lambda: self._run_simulation("full"),
        )
        self.run_full_btn.pack(fill=tk.X, pady=2)

        self.run_magnetics_btn = ttk.Button(
            run_frame,
            text="Run Magnetics Analysis",
            command=self._run_magnetics,
        )
        self.run_magnetics_btn.pack(fill=tk.X, pady=2)

        # Progress
        self.progress = ttk.Progressbar(run_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(10, 0))

    def _create_center_panel(self):
        """Create the center panel with matplotlib visualization."""
        # Notebook for multiple plot tabs
        self.plot_notebook = ttk.Notebook(self.center_panel)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Board Overview
        self.overview_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.overview_frame, text="Board Overview")
        self._create_plot_canvas(self.overview_frame, "overview")

        # Tab 2: Thermal
        self.thermal_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.thermal_frame, text="Thermal")
        self._create_plot_canvas(self.thermal_frame, "thermal")

        # Tab 3: Signal Integrity
        self.si_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.si_frame, text="Signal Integrity")
        self._create_plot_canvas(self.si_frame, "si")

        # Tab 4: Magnetics
        self.mag_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.mag_frame, text="Magnetics")
        self._create_plot_canvas(self.mag_frame, "magnetics")

        # Tab 5: Transient
        self.transient_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.transient_frame, text="Transient")
        self._create_plot_canvas(self.transient_frame, "transient")

    def _create_plot_canvas(self, parent: ttk.Frame, name: str):
        """Create a matplotlib canvas in the given frame."""
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(f"Load a design and run simulation")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store references
        setattr(self, f"{name}_fig", fig)
        setattr(self, f"{name}_canvas", canvas)

    def _create_right_panel(self):
        """Create the right panel with results and log."""
        # Results notebook
        results_notebook = ttk.Notebook(self.right_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results tree tab
        results_frame = ttk.Frame(results_notebook)
        results_notebook.add(results_frame, text="Results")

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("value",),
            show="tree headings",
        )
        self.results_tree.heading("#0", text="Parameter")
        self.results_tree.heading("value", text="Value")
        self.results_tree.column("#0", width=180)
        self.results_tree.column("value", width=120)

        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scroll.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Log tab
        log_frame = ttk.Frame(results_notebook)
        results_notebook.add(log_frame, text="Log")

        self.log_text = tk.Text(log_frame, height=20, width=40, font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Export buttons
        export_frame = ttk.Frame(self.right_panel)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(export_frame, text="Open Report", command=self._open_html_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="Export JSON", command=self._export_json).pack(side=tk.LEFT, padx=2)

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            style="Status.TLabel",
            padding=(10, 5),
        )
        self.status_label.pack(side=tk.LEFT)

        self.output_label = ttk.Label(
            self.status_frame,
            text=f"Output: {self.output_dir}",
            style="Status.TLabel",
            padding=(10, 5),
        )
        self.output_label.pack(side=tk.RIGHT)

    # ---- Actions ----

    def _open_design(self):
        """Open a design file dialog."""
        filetypes = [
            ("All supported", "*.yaml *.yml *.kicad_pcb *.cvg"),
            ("YAML files", "*.yaml *.yml"),
            ("KiCad PCB", "*.kicad_pcb"),
            ("IPC-2581", "*.cvg"),
        ]
        path = filedialog.askopenfilename(
            title="Open PCB Design",
            filetypes=filetypes,
            initialdir=get_examples_dir(),
        )
        if path:
            self._load_design(Path(path))

    def _load_design(self, path: Path):
        """Load a design from the given path."""
        try:
            self._log(f"Loading design: {path}")
            if path.suffix == ".kicad_pcb":
                self.design = load_kicad_pcb(path)
            elif path.suffix == ".cvg":
                self.design = load_ipc2581(path)
            else:
                self.design = load_design(path)
            self.design_path = path
            self._update_design_display()
            self._plot_board_overview()
            self._set_status(f"Loaded: {path.name}")
            self._log(f"Design loaded: {self.design.name}")
            self._log(f"  Size: {self.design.width} x {self.design.height} mm")
            self._log(f"  Components: {len(self.design.components)}")
            self._log(f"  Nets: {len(self.design.nets)}")
            self._log(f"  Traces: {len(self.design.traces)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load design:\n{e}")
            self._log(f"ERROR: {e}")

    def _reload_design(self):
        """Reload the current design."""
        if self.design_path:
            self._load_design(self.design_path)

    def _update_design_display(self):
        """Update the design info display."""
        if not self.design:
            self.design_info.config(text="No design loaded")
            return

        info = (
            f"Name: {self.design.name}\n"
            f"Size: {self.design.width} x {self.design.height} mm\n"
            f"Components: {len(self.design.components)}\n"
            f"Nets: {len(self.design.nets)}\n"
            f"Traces: {len(self.design.traces)}"
        )
        self.design_info.config(text=info)

        # Update component tree
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)
        for comp in self.design.components:
            self.comp_tree.insert(
                "",
                tk.END,
                values=(comp.reference, comp.component_type.value, comp.value),
            )

    def _get_config(self) -> SimulationConfig:
        """Build SimulationConfig from UI inputs."""
        try:
            freq_start = float(self.freq_start.get())
            freq_end = float(self.freq_end.get())
            time_step = float(self.time_step.get())
            duration = float(self.duration.get())
            ambient = float(self.ambient_temp.get())
        except ValueError:
            raise ValueError("Invalid numeric input in configuration")

        return SimulationConfig(
            frequency_range=(freq_start, freq_end),
            time_step=time_step,
            duration=duration,
            ambient_temp=ambient,
            output_dir=self.output_dir,
        )

    def _run_simulation(self, sim_type: str):
        """Run a simulation in a background thread."""
        if not self.design:
            messagebox.showwarning("Warning", "Please load a design first.")
            return

        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showwarning("Warning", "Simulation already running.")
            return

        try:
            config = self._get_config()
            config.sim_type = SimulationType(sim_type)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self._set_status(f"Running {sim_type} simulation...")
        self.progress.start()
        self._disable_buttons()

        def run():
            try:
                sim = PCBSimulator(self.design, config)
                results = sim.run()
                self.message_queue.put(("results", results))
            except Exception as e:
                self.message_queue.put(("error", str(e)))

        self.sim_thread = threading.Thread(target=run, daemon=True)
        self.sim_thread.start()

    def _run_magnetics(self):
        """Run magnetics analysis in a background thread."""
        if not self.design:
            messagebox.showwarning("Warning", "Please load a design first.")
            return

        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showwarning("Warning", "Simulation already running.")
            return

        self._set_status("Running magnetics analysis...")
        self.progress.start()
        self._disable_buttons()

        def run():
            try:
                config = self._get_config()
                analyzer = MagneticsAnalyzer()
                result = analyzer.run(self.design, config, resolution_mm=0.5)
                self.message_queue.put(("magnetics", result.data))
            except Exception as e:
                self.message_queue.put(("error", str(e)))

        self.sim_thread = threading.Thread(target=run, daemon=True)
        self.sim_thread.start()

    def _process_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "results":
                    self.results = data
                    self._on_simulation_complete()
                elif msg_type == "magnetics":
                    self.magnetics_data = data
                    self._on_magnetics_complete()
                elif msg_type == "error":
                    self._on_simulation_error(data)
        except queue.Empty:
            pass

        self.root.after(100, self._process_queue)

    def _on_simulation_complete(self):
        """Handle simulation completion."""
        self.progress.stop()
        self._enable_buttons()

        failed = [r for r in self.results if not r.success]
        if failed:
            self._set_status(f"Simulation completed with {len(failed)} errors")
            for r in failed:
                for err in r.errors:
                    self._log(f"ERROR [{r.sim_type.value}]: {err}")
        else:
            self._set_status("Simulation completed successfully")

        self._log("Simulation complete.")
        for r in self.results:
            status = "OK" if r.success else "FAILED"
            self._log(f"  [{r.sim_type.value}] {status}")
            for w in r.warnings:
                self._log(f"    Warning: {w}")

        self._update_results_tree()
        self._update_plots()

    def _on_magnetics_complete(self):
        """Handle magnetics analysis completion."""
        self.progress.stop()
        self._enable_buttons()
        self._set_status("Magnetics analysis complete")
        self._log("Magnetics analysis complete.")

        bfield = self.magnetics_data.get("bfield_grid", {})
        self._log(f"  B-field range: {bfield.get('min_b_tesla', 0)*1e6:.2f} - {bfield.get('max_b_tesla', 0)*1e6:.2f} uT")

        for prof in self.magnetics_data.get("trace_profiles", []):
            self._log(f"  {prof['net']}: peak = {prof['peak_b_microtesla']:.2f} uT")

        self._plot_magnetics()
        self.plot_notebook.select(self.mag_frame)

    def _on_simulation_error(self, error: str):
        """Handle simulation error."""
        self.progress.stop()
        self._enable_buttons()
        self._set_status("Simulation failed")
        self._log(f"ERROR: {error}")
        messagebox.showerror("Simulation Error", error)

    def _disable_buttons(self):
        """Disable run buttons during simulation."""
        self.run_full_btn.state(["disabled"])
        self.run_magnetics_btn.state(["disabled"])

    def _enable_buttons(self):
        """Enable run buttons after simulation."""
        self.run_full_btn.state(["!disabled"])
        self.run_magnetics_btn.state(["!disabled"])

    def _update_results_tree(self):
        """Populate the results tree with simulation data."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        for r in self.results:
            parent = self.results_tree.insert(
                "",
                tk.END,
                text=r.sim_type.value.upper(),
                values=("OK" if r.success else "FAILED",),
                open=True,
            )

            data = r.data
            if r.sim_type == SimulationType.THERMAL:
                self.results_tree.insert(parent, tk.END, text="Total Power", values=(f"{data.get('total_power_w', 0):.4f} W",))
                self.results_tree.insert(parent, tk.END, text="Board Temp", values=(f"{data.get('avg_board_temp_c', 0):.1f} C",))
            elif r.sim_type == SimulationType.SIGNAL_INTEGRITY:
                traces = data.get("traces", [])
                si_parent = self.results_tree.insert(parent, tk.END, text="Traces", values=(f"{len(traces)}",))
                for t in traces[:5]:  # limit
                    self.results_tree.insert(si_parent, tk.END, text=t["net"], values=(f"Z0={t['z0_ohms']} ohm",))
            elif r.sim_type == SimulationType.SPICE_DC:
                for node, v in data.get("node_voltages", {}).items():
                    self.results_tree.insert(parent, tk.END, text=node, values=(f"{v} V",))

    def _update_plots(self):
        """Update all plot tabs with simulation results."""
        self._plot_board_overview()

        for r in self.results:
            if r.sim_type == SimulationType.THERMAL:
                self._plot_thermal(r.data)
            elif r.sim_type == SimulationType.SIGNAL_INTEGRITY:
                self._plot_signal_integrity(r.data)
            elif r.sim_type == SimulationType.SPICE_TRANSIENT:
                self._plot_transient(r.data)

    def _plot_board_overview(self):
        """Plot board overview with traces and components."""
        if not self.design:
            return

        fig = self.overview_fig
        fig.clear()
        ax = fig.add_subplot(111)

        # Draw traces
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(self.design.traces), 1)))
        for i, trace in enumerate(self.design.traces):
            if len(trace.points) >= 2:
                xs = [p[0] for p in trace.points]
                ys = [p[1] for p in trace.points]
                ax.plot(xs, ys, color=colors[i], linewidth=2, label=trace.net)

        # Draw component locations
        for comp in self.design.components:
            if comp.pads:
                cx = sum(p.x for p in comp.pads) / len(comp.pads)
                cy = sum(p.y for p in comp.pads) / len(comp.pads)
                ax.plot(cx, cy, "s", markersize=8, color="red")
                ax.annotate(comp.reference, (cx, cy), fontsize=7, ha="center", va="bottom")

        ax.set_xlim(0, self.design.width)
        ax.set_ylim(0, self.design.height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Board Overview: {self.design.name}")
        ax.grid(True, alpha=0.3)

        if len(self.design.traces) <= 8:
            ax.legend(loc="upper right", fontsize=7)

        fig.tight_layout()
        self.overview_canvas.draw()

    def _plot_thermal(self, data: dict):
        """Plot thermal heatmap."""
        grid = data.get("thermal_grid", {}).get("grid")
        if not grid:
            return

        fig = self.thermal_fig
        fig.clear()
        ax = fig.add_subplot(111)

        grid_arr = np.array(grid)
        im = ax.imshow(grid_arr, cmap="hot", origin="lower", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="Temperature (C)")

        ax.set_title(f"Thermal Map (max: {data.get('thermal_grid', {}).get('max_temp_c', 0):.1f} C)")
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")

        fig.tight_layout()
        self.thermal_canvas.draw()
        self.plot_notebook.select(self.thermal_frame)

    def _plot_signal_integrity(self, data: dict):
        """Plot signal integrity results."""
        traces = data.get("traces", [])
        if not traces:
            return

        fig = self.si_fig
        fig.clear()
        ax = fig.add_subplot(111)

        nets = [t["net"] for t in traces]
        z0s = [t["z0_ohms"] for t in traces]
        colors = ["green" if 45 <= z <= 55 else "orange" if 40 <= z <= 60 else "red" for z in z0s]

        bars = ax.barh(nets, z0s, color=colors)
        ax.axvline(50, color="blue", linestyle="--", alpha=0.7, label="50 ohm target")
        ax.set_xlabel("Characteristic Impedance Z0 (ohms)")
        ax.set_title("Trace Impedances")
        ax.legend()

        for bar, z in zip(bars, z0s):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{z:.1f}", va="center", fontsize=8)

        fig.tight_layout()
        self.si_canvas.draw()

    def _plot_transient(self, data: dict):
        """Plot transient waveforms."""
        time = data.get("time")
        waveforms = data.get("waveforms", {})
        if not time or not waveforms:
            return

        fig = self.transient_fig
        fig.clear()
        ax = fig.add_subplot(111)

        t_us = [t * 1e6 for t in time]
        for name, wf in waveforms.items():
            ax.plot(t_us, wf, label=name)

        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Transient Simulation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.transient_canvas.draw()

    def _plot_magnetics(self):
        """Plot magnetics results."""
        if not self.magnetics_data:
            return

        bfield = self.magnetics_data.get("bfield_grid", {})
        bmag = bfield.get("bmag_tesla")
        if not bmag:
            return

        fig = self.magnetics_fig
        fig.clear()
        ax = fig.add_subplot(111)

        bmag_arr = np.array(bmag) * 1e6  # to uT
        grid_x = np.array(bfield.get("grid_x_mm", []))
        grid_y = np.array(bfield.get("grid_y_mm", []))

        im = ax.pcolormesh(grid_x, grid_y, bmag_arr, cmap="inferno", shading="gouraud")
        fig.colorbar(im, ax=ax, label="B-field (uT)")

        # Overlay traces
        if self.design:
            for trace in self.design.traces:
                if len(trace.points) >= 2:
                    xs = [p[0] for p in trace.points]
                    ys = [p[1] for p in trace.points]
                    ax.plot(xs, ys, "w-", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Magnetic Field Map (max: {bfield.get('max_b_tesla', 0)*1e6:.2f} uT)")
        ax.set_aspect("equal")

        fig.tight_layout()
        self.magnetics_canvas.draw()

    # ---- Utilities ----

    def _set_status(self, text: str):
        """Update the status bar text."""
        self.status_label.config(text=text)

    def _log(self, text: str):
        """Append text to the log."""
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _set_output_dir(self):
        """Set the output directory."""
        path = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir)
        if path:
            self.output_dir = Path(path)
            self.output_label.config(text=f"Output: {self.output_dir}")
            self._log(f"Output directory set to: {self.output_dir}")

    def _open_output_folder(self):
        """Open the output folder in file browser."""
        if self.output_dir.exists():
            webbrowser.open(f"file://{self.output_dir}")
        else:
            messagebox.showinfo("Info", "Output directory does not exist yet.")

    def _open_html_report(self):
        """Open the HTML report in browser."""
        report = self.output_dir / "report.html"
        if report.exists():
            webbrowser.open(f"file://{report}")
        else:
            messagebox.showinfo("Info", "No report generated yet. Run a simulation first.")

    def _export_json(self):
        """Export results to a JSON file."""
        if not self.results:
            messagebox.showinfo("Info", "No results to export. Run a simulation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            data = {
                "design": self.design.name if self.design else "unknown",
                "results": [
                    {
                        "sim_type": r.sim_type.value,
                        "success": r.success,
                        "errors": r.errors,
                        "warnings": r.warnings,
                    }
                    for r in self.results
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._log(f"Results exported to: {path}")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About PCB Simulation Toolkit",
            "PCB Simulation Toolkit v0.1.0\n\n"
            "Open-source PCB design simulation using:\n"
            "- SPICE (ngspice / built-in)\n"
            "- Signal Integrity Analysis\n"
            "- Thermal Simulation\n"
            "- Magnetic Field Analysis\n\n"
            "MIT License",
        )


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = PCBSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
