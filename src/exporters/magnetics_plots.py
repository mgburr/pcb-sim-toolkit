"""Generate magnetic field visualizations for PCB designs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches


def generate_magnetics_plots(
    design_data: dict[str, Any],
    mag_result_data: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate all magnetic field visualizations. Returns list of output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    bfield = mag_result_data.get("bfield_grid", {})
    profiles = mag_result_data.get("trace_profiles", [])
    currents = mag_result_data.get("trace_currents", [])

    if bfield:
        p = _plot_bfield_magnitude(bfield, design_data, output_dir)
        generated.append(p)

        p = _plot_bfield_vector(bfield, design_data, output_dir)
        generated.append(p)

        p = _plot_bfield_contour(bfield, design_data, output_dir)
        generated.append(p)

        p = _plot_bfield_log_heatmap(bfield, design_data, output_dir)
        generated.append(p)

    if profiles:
        p = _plot_trace_profiles(profiles, output_dir)
        generated.append(p)

    if bfield and profiles:
        p = _plot_combined_overview(bfield, profiles, currents, design_data, output_dir)
        generated.append(p)

    return generated


def _get_trace_segments(design_data: dict) -> list[dict]:
    """Extract trace segments from design data for overlay."""
    return design_data.get("traces", [])


def _plot_bfield_magnitude(
    bfield: dict, design_data: dict, output_dir: Path
) -> Path:
    """Full-board B-field magnitude heatmap."""
    bmag = np.array(bfield["bmag_tesla"])
    grid_x = np.array(bfield["grid_x_mm"])
    grid_y = np.array(bfield["grid_y_mm"])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to microtesla for readability
    bmag_ut = bmag * 1e6
    bmag_ut = np.maximum(bmag_ut, 1e-6)  # floor for log scale

    im = ax.pcolormesh(
        grid_x, grid_y, bmag_ut,
        cmap="inferno",
        shading="gouraud",
    )
    cbar = fig.colorbar(im, label="B-field magnitude (uT)")

    # Overlay trace paths
    traces = design_data.get("traces", [])
    for t in traces:
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "w-", linewidth=1.5, alpha=0.8)
            ax.plot(xs[0], ys[0], "wo", markersize=3)
            ax.plot(xs[-1], ys[-1], "ws", markersize=3)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Magnetic Field Magnitude - Board Overview")
    ax.set_aspect("equal")
    ax.set_xlim(grid_x[0], grid_x[-1])
    ax.set_ylim(grid_y[0], grid_y[-1])

    path = output_dir / "mag_field_magnitude.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_bfield_vector(
    bfield: dict, design_data: dict, output_dir: Path
) -> Path:
    """Vector field (quiver) plot of B-field with magnitude background."""
    bx = np.array(bfield["bx_tesla"])
    by = np.array(bfield["by_tesla"])
    bmag = np.array(bfield["bmag_tesla"])
    grid_x = np.array(bfield["grid_x_mm"])
    grid_y = np.array(bfield["grid_y_mm"])

    X, Y = np.meshgrid(grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Background: magnitude heatmap
    bmag_ut = bmag * 1e6
    im = ax.pcolormesh(
        grid_x, grid_y, bmag_ut,
        cmap="YlOrRd",
        shading="gouraud",
        alpha=0.7,
    )
    fig.colorbar(im, label="B-field magnitude (uT)", shrink=0.8)

    # Subsample for quiver (every Nth point)
    step = max(1, min(bx.shape[0], bx.shape[1]) // 25)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Bxs = bx[::step, ::step]
    Bys = by[::step, ::step]
    Bms = bmag[::step, ::step]

    # Normalize arrows for visibility
    Bmax = np.max(Bms) if np.max(Bms) > 0 else 1
    Bxn = Bxs / Bmax
    Byn = Bys / Bmax

    ax.quiver(
        Xs, Ys, Bxn, Byn,
        Bms * 1e6,
        cmap="cool",
        scale=25,
        width=0.003,
        headwidth=4,
        headlength=5,
        alpha=0.9,
    )

    # Overlay traces
    for t in design_data.get("traces", []):
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "k-", linewidth=2.5, alpha=0.9)
            ax.annotate(
                t.get("net", ""),
                xy=(xs[-1], ys[-1]),
                fontsize=7,
                color="black",
                fontweight="bold",
                ha="left",
            )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Magnetic Field Vector Map (arrows show B direction)")
    ax.set_aspect("equal")

    path = output_dir / "mag_field_vectors.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_bfield_contour(
    bfield: dict, design_data: dict, output_dir: Path
) -> Path:
    """Contour line plot of B-field iso-levels."""
    bmag = np.array(bfield["bmag_tesla"])
    grid_x = np.array(bfield["grid_x_mm"])
    grid_y = np.array(bfield["grid_y_mm"])

    fig, ax = plt.subplots(figsize=(12, 8))

    bmag_ut = bmag * 1e6
    bmax = np.max(bmag_ut)
    bmin = max(np.min(bmag_ut), bmax * 0.01)

    # Create contour levels
    levels = np.linspace(bmin, bmax, 20)
    if len(levels) < 2:
        levels = np.linspace(0, max(bmax, 1), 20)

    cs = ax.contourf(
        grid_x, grid_y, bmag_ut,
        levels=levels,
        cmap="plasma",
    )
    fig.colorbar(cs, label="B-field magnitude (uT)")

    # Add contour lines
    cl = ax.contour(
        grid_x, grid_y, bmag_ut,
        levels=levels[::2],
        colors="white",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(cl, inline=True, fontsize=6, fmt="%.2f")

    # Overlay traces
    for t in design_data.get("traces", []):
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "w-", linewidth=2, alpha=0.9)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Magnetic Field Contour Map (iso-B lines in uT)")
    ax.set_aspect("equal")

    path = output_dir / "mag_field_contour.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_bfield_log_heatmap(
    bfield: dict, design_data: dict, output_dir: Path
) -> Path:
    """Logarithmic-scale heatmap to reveal weak-field regions."""
    bmag = np.array(bfield["bmag_tesla"])
    grid_x = np.array(bfield["grid_x_mm"])
    grid_y = np.array(bfield["grid_y_mm"])

    fig, ax = plt.subplots(figsize=(12, 8))

    bmag_ut = bmag * 1e6
    bmag_ut = np.maximum(bmag_ut, 1e-6)

    im = ax.pcolormesh(
        grid_x, grid_y, bmag_ut,
        cmap="magma",
        norm=LogNorm(vmin=np.min(bmag_ut[bmag_ut > 0]), vmax=np.max(bmag_ut)),
        shading="gouraud",
    )
    fig.colorbar(im, label="B-field magnitude (uT) - log scale")

    # Overlay traces
    for t in design_data.get("traces", []):
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color="cyan", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Magnetic Field - Logarithmic Scale (reveals weak field regions)")
    ax.set_aspect("equal")

    path = output_dir / "mag_field_log_scale.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_trace_profiles(profiles: list[dict], output_dir: Path) -> Path:
    """Cross-section B-field profiles for each trace."""
    fig, axes = plt.subplots(
        len(profiles), 1,
        figsize=(10, 3.5 * len(profiles)),
        squeeze=False,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(profiles), 1)))

    for i, prof in enumerate(profiles):
        ax = axes[i, 0]
        offsets = np.array(prof["offset_mm"])
        bmag = np.array(prof["b_magnitude_tesla"]) * 1e6  # to uT

        ax.fill_between(offsets, bmag, alpha=0.3, color=colors[i])
        ax.plot(offsets, bmag, color=colors[i], linewidth=2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Trace center")

        peak = prof.get("peak_b_microtesla", 0)
        ax.set_title(
            f"Net: {prof['net']}  |  I = {prof['current_ma']} mA  |  "
            f"Peak B = {peak} uT"
        )
        ax.set_xlabel("Distance from trace center (mm)")
        ax.set_ylabel("B-field (uT)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    plt.tight_layout()
    path = output_dir / "mag_field_profiles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_combined_overview(
    bfield: dict,
    profiles: list[dict],
    currents: list[dict],
    design_data: dict,
    output_dir: Path,
) -> Path:
    """Combined overview: heatmap + strongest profile + current table."""
    fig = plt.figure(figsize=(16, 10))

    # Layout: top-left = heatmap, top-right = vector detail, bottom = profile
    ax_heat = fig.add_subplot(2, 2, 1)
    ax_vec = fig.add_subplot(2, 2, 2)
    ax_prof = fig.add_subplot(2, 1, 2)

    bmag = np.array(bfield["bmag_tesla"])
    bx = np.array(bfield["bx_tesla"])
    by = np.array(bfield["by_tesla"])
    grid_x = np.array(bfield["grid_x_mm"])
    grid_y = np.array(bfield["grid_y_mm"])
    X, Y = np.meshgrid(grid_x, grid_y)
    bmag_ut = bmag * 1e6

    # -- Heatmap --
    im = ax_heat.pcolormesh(
        grid_x, grid_y, bmag_ut,
        cmap="inferno", shading="gouraud",
    )
    fig.colorbar(im, ax=ax_heat, label="uT", shrink=0.8)
    for t in design_data.get("traces", []):
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax_heat.plot(xs, ys, "w-", linewidth=1)
    ax_heat.set_title("B-Field Magnitude")
    ax_heat.set_xlabel("X (mm)")
    ax_heat.set_ylabel("Y (mm)")
    ax_heat.set_aspect("equal")

    # -- Vector field zoomed on highest-field region --
    peak_idx = np.unravel_index(np.argmax(bmag), bmag.shape)
    cy, cx = grid_y[peak_idx[0]], grid_x[peak_idx[1]]
    zoom = 8  # mm around peak

    ax_vec.pcolormesh(
        grid_x, grid_y, bmag_ut,
        cmap="YlOrRd", shading="gouraud", alpha=0.6,
    )
    step = max(1, min(bx.shape[0], bx.shape[1]) // 20)
    Bmax = np.max(bmag) if np.max(bmag) > 0 else 1
    ax_vec.quiver(
        X[::step, ::step], Y[::step, ::step],
        bx[::step, ::step] / Bmax, by[::step, ::step] / Bmax,
        bmag[::step, ::step] * 1e6,
        cmap="cool", scale=30, width=0.004, alpha=0.9,
    )
    for t in design_data.get("traces", []):
        pts = t.get("points", [])
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax_vec.plot(xs, ys, "k-", linewidth=2)
    ax_vec.set_xlim(cx - zoom, cx + zoom)
    ax_vec.set_ylim(cy - zoom, cy + zoom)
    ax_vec.set_title(f"Vector Detail (around peak at {cx:.0f},{cy:.0f} mm)")
    ax_vec.set_xlabel("X (mm)")
    ax_vec.set_ylabel("Y (mm)")
    ax_vec.set_aspect("equal")

    # -- Profiles overlay --
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(profiles), 1)))
    for i, prof in enumerate(profiles):
        offsets = np.array(prof["offset_mm"])
        b_ut = np.array(prof["b_magnitude_tesla"]) * 1e6
        ax_prof.plot(
            offsets, b_ut,
            color=colors[i], linewidth=2,
            label=f"{prof['net']} ({prof['current_ma']} mA)",
        )
    ax_prof.set_xlabel("Distance from trace center (mm)")
    ax_prof.set_ylabel("B-field (uT)")
    ax_prof.set_title("B-Field Cross-Section Profiles (all traces)")
    ax_prof.legend(loc="upper right", fontsize=8)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        "Magnetic Field Analysis Overview",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    path = output_dir / "mag_field_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
