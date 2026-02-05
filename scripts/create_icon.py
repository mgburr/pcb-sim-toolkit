#!/usr/bin/env python3
"""Generate a PCB-themed application icon (.icns) for macOS."""

import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_pcb_icon(output_png: Path, size_px: int = 1024):
    """Draw a PCB-themed icon and save as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(size_px / 100, size_px / 100), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("none")

    # PCB board background (green with rounded corners)
    board = patches.FancyBboxPatch(
        (0.5, 0.5), 9, 9,
        boxstyle="round,pad=0.3",
        facecolor="#1a6b3c",
        edgecolor="#0d4a26",
        linewidth=4,
    )
    ax.add_patch(board)

    # Copper traces
    trace_color = "#d4a017"
    trace_lw = 3.5

    # Horizontal traces
    ax.plot([1.5, 5.0], [7.5, 7.5], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([6.0, 8.5], [7.5, 7.5], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([1.5, 4.0], [2.5, 2.5], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([5.5, 8.5], [2.5, 2.5], color=trace_color, linewidth=trace_lw, solid_capstyle="round")

    # Vertical traces
    ax.plot([5.0, 5.0], [7.5, 6.0], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([5.0, 5.5], [6.0, 5.0], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([5.5, 5.5], [5.0, 2.5], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([4.0, 4.0], [2.5, 4.0], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([4.0, 3.0], [4.0, 5.0], color=trace_color, linewidth=trace_lw, solid_capstyle="round")
    ax.plot([3.0, 3.0], [5.0, 7.0], color=trace_color, linewidth=trace_lw, solid_capstyle="round")

    # IC chip (center)
    ic = patches.FancyBboxPatch(
        (3.5, 4.0), 3.0, 2.0,
        boxstyle="round,pad=0.05",
        facecolor="#2a2a2a",
        edgecolor="#1a1a1a",
        linewidth=2,
    )
    ax.add_patch(ic)

    # IC pins
    for i in range(4):
        x = 3.8 + i * 0.7
        ax.plot([x, x], [3.6, 4.0], color="#c0c0c0", linewidth=2.5)
        ax.plot([x, x], [6.0, 6.4], color="#c0c0c0", linewidth=2.5)

    # IC dot marker
    ax.plot(3.8, 5.7, "o", color="#555555", markersize=4)

    # Vias (drilled holes with copper ring)
    via_positions = [(1.5, 7.5), (8.5, 7.5), (1.5, 2.5), (8.5, 2.5),
                     (6.0, 7.5), (3.0, 7.0), (7.0, 5.0)]
    for vx, vy in via_positions:
        via_outer = plt.Circle((vx, vy), 0.22, color=trace_color, zorder=5)
        via_inner = plt.Circle((vx, vy), 0.10, color="#1a6b3c", zorder=6)
        ax.add_patch(via_outer)
        ax.add_patch(via_inner)

    # SMD components (resistors/capacitors)
    for cx, cy, w in [(7.0, 6.5, 0.8), (2.0, 4.5, 0.6), (7.5, 3.5, 0.7)]:
        smd = patches.FancyBboxPatch(
            (cx - w / 2, cy - 0.15), w, 0.3,
            boxstyle="round,pad=0.02",
            facecolor="#3a3a3a",
            edgecolor="#222222",
            linewidth=1,
        )
        ax.add_patch(smd)
        # Solder pads
        ax.add_patch(patches.Rectangle((cx - w / 2 - 0.05, cy - 0.15), 0.12, 0.3, color="#c0c0c0"))
        ax.add_patch(patches.Rectangle((cx + w / 2 - 0.07, cy - 0.15), 0.12, 0.3, color="#c0c0c0"))

    fig.savefig(output_png, transparent=True, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)


def png_to_icns(png_path: Path, icns_path: Path):
    """Convert a PNG to macOS .icns format using sips and iconutil."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "icon.iconset"
        iconset.mkdir()

        sizes = [16, 32, 64, 128, 256, 512]
        for s in sizes:
            # Standard resolution
            out = iconset / f"icon_{s}x{s}.png"
            subprocess.run(
                ["sips", "-z", str(s), str(s), str(png_path), "--out", str(out)],
                check=True, capture_output=True,
            )
            # Retina (@2x) — uses the next size up
            out_2x = iconset / f"icon_{s}x{s}@2x.png"
            s2 = s * 2
            subprocess.run(
                ["sips", "-z", str(s2), str(s2), str(png_path), "--out", str(out_2x)],
                check=True, capture_output=True,
            )

        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
            check=True, capture_output=True,
        )


def main():
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / "assets"
    assets_dir.mkdir(exist_ok=True)

    png_path = assets_dir / "icon_1024.png"
    icns_path = assets_dir / "icon.icns"

    print("Drawing PCB icon...")
    draw_pcb_icon(png_path, size_px=1024)
    print(f"  Saved PNG: {png_path}")

    print("Converting to .icns...")
    png_to_icns(png_path, icns_path)
    print(f"  Saved ICNS: {icns_path}")
    print("Done!")


if __name__ == "__main__":
    main()
