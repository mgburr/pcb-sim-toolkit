#!/usr/bin/env python3
"""Minimal GUI diagnostic — run each stage to find what freezes."""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def stage1_bare_tkinter():
    """Stage 1: bare Tkinter window with button."""
    print("Stage 1: Bare Tkinter...")
    root = tk.Tk()
    root.title("Stage 1 - Bare Tkinter")
    root.geometry("400x300")
    ttk.Label(root, text="Click the button. Close to continue.").pack(pady=20)
    ttk.Button(root, text="Click Me", command=lambda: print("  Button clicked OK")).pack()
    root.mainloop()
    print("  Stage 1 OK\n")


def stage2_matplotlib_canvas():
    """Stage 2: Tkinter + matplotlib canvas."""
    print("Stage 2: Tkinter + matplotlib canvas...")
    root = tk.Tk()
    root.title("Stage 2 - Matplotlib Canvas")
    root.geometry("600x400")

    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("Test Plot")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    ttk.Button(root, text="Redraw", command=lambda: (ax.clear(), ax.plot([1, 2, 3], [2, 1, 3]), canvas.draw_idle())).pack()
    root.mainloop()
    print("  Stage 2 OK\n")


def stage3_treeview_with_bindings():
    """Stage 3: Treeview with selection bindings + matplotlib."""
    print("Stage 3: Treeview + bindings + matplotlib...")
    root = tk.Tk()
    root.title("Stage 3 - Treeview + Events")
    root.geometry("800x500")

    left = ttk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
    right = ttk.Frame(root)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Treeview
    tree = ttk.Treeview(left, columns=("name",), show="headings", height=10)
    tree.heading("name", text="Component")
    for name in ["R1", "C1", "U1", "R2", "C2"]:
        tree.insert("", tk.END, values=(name,))
    tree.pack(fill=tk.BOTH, expand=True)

    # Matplotlib
    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("No selection")
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, right)
    toolbar.update()

    status = ttk.Label(root, text="Ready")
    status.pack(side=tk.BOTTOM, fill=tk.X)

    def on_select(event):
        sel = tree.selection()
        if not sel:
            return
        name = tree.item(sel[0], "values")[0]
        status.config(text=f"Selected: {name}")
        ax.clear()
        ax.set_title(f"Highlighted: {name}")
        ax.bar([1, 2, 3], [2, 5, 3])
        fig.tight_layout()
        canvas.draw_idle()
        print(f"  Selected {name}")

    def on_click(event):
        if tree.identify_region(event.x, event.y) == "nothing":
            tree.selection_remove(*tree.selection())
            status.config(text="Cleared")
            ax.clear()
            ax.set_title("No selection")
            canvas.draw_idle()
            print("  Cleared selection")

    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Button-1>", on_click, add=True)

    root.mainloop()
    print("  Stage 3 OK\n")


if __name__ == "__main__":
    print("=== GUI Diagnostic ===")
    print("Close each window to proceed to the next stage.\n")
    stage1_bare_tkinter()
    stage2_matplotlib_canvas()
    stage3_treeview_with_bindings()
    print("All stages passed!")
