#!/usr/bin/env python3
"""Generate a PCB Sim Toolkit icon (PNG and ICO)."""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Rectangle, Circle, Polygon as MplPolygon,
)

ASSETS = Path(__file__).parent.parent / "assets"


def draw_icon(size_px=256):
    dpi = 96
    fig_size = size_px / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("none")

    # Board background (rounded green PCB)
    board = FancyBboxPatch(
        (5, 5), 90, 90,
        boxstyle="round,pad=3",
        facecolor="#0a5c0a", edgecolor="#33cc33", linewidth=3,
    )
    ax.add_patch(board)

    # Traces
    trace_style = dict(color="#b87333", linewidth=2.5, solid_capstyle="round")
    ax.plot([15, 40, 40], [75, 75, 55], **trace_style)
    ax.plot([60, 60, 85], [55, 30, 30], **trace_style)
    ax.plot([40, 60], [55, 55], color="#cc8800", linewidth=3, solid_capstyle="round")
    ax.plot([15, 35], [30, 30], **trace_style)
    ax.plot([65, 85], [75, 75], **trace_style)

    # IC chip (center)
    ic = Rectangle((35, 40), 30, 25, facecolor="#222222", edgecolor="#555555", linewidth=1.5)
    ax.add_patch(ic)
    # IC pins
    for y_off in [42, 47, 52, 57, 62]:
        ax.plot([33, 35], [y_off - 1, y_off - 1], color="#ccaa00", linewidth=2)
        ax.plot([65, 67], [y_off - 1, y_off - 1], color="#ccaa00", linewidth=2)

    # Resistor (top-left)
    r = Rectangle((12, 70), 10, 6, facecolor="#4a3728", edgecolor="#886644", linewidth=1)
    ax.add_patch(r)

    # Capacitor (bottom-right)
    c = Rectangle((78, 22), 10, 12, facecolor="#c2a060", edgecolor="#aa8833", linewidth=1)
    ax.add_patch(c)

    # Pads (vias)
    for cx, cy in [(40, 75), (60, 30), (15, 30), (85, 75), (40, 55), (60, 55)]:
        outer = Circle((cx, cy), 3, facecolor="#cc8800", edgecolor="#ffcc00", linewidth=0.8)
        inner = Circle((cx, cy), 1.2, facecolor="#0a5c0a", edgecolor="none")
        ax.add_patch(outer)
        ax.add_patch(inner)

    # Text label
    ax.text(50, 14, "PCB", fontsize=18, color="#33cc33",
            ha="center", va="center", fontweight="bold", fontfamily="monospace")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main():
    ASSETS.mkdir(exist_ok=True)

    # Generate PNG at multiple sizes
    for size in [256, 128, 64, 48, 32, 16]:
        fig = draw_icon(size)
        png_path = ASSETS / f"icon_{size}.png"
        fig.savefig(png_path, transparent=True, dpi=96)
        plt.close(fig)
        print(f"  Created {png_path}")

    # Create ICO with multiple sizes using Pillow
    from PIL import Image
    images = []
    for size in [256, 128, 64, 48, 32, 16]:
        img = Image.open(ASSETS / f"icon_{size}.png")
        img = img.resize((size, size), Image.LANCZOS)
        images.append(img)

    ico_path = ASSETS / "pcb-sim-toolkit.ico"
    images[0].save(
        ico_path, format="ICO",
        sizes=[(img.width, img.height) for img in images],
        append_images=images[1:],
    )
    print(f"  Created {ico_path}")

    # Also save a copy to the Windows-accessible path
    main_png = ASSETS / "icon_256.png"
    print(f"\nIcon files generated in {ASSETS}/")


if __name__ == "__main__":
    main()
