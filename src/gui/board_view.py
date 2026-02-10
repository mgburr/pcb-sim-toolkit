"""2D and 3D board visualization renderers.

Provides ``plot_board_2d`` for an enhanced top-down PCB view and
``plot_board_3d`` for a 3D perspective view using *mpl_toolkits.mplot3d*.
Both functions accept a matplotlib ``Figure`` and a ``PCBDesign`` instance.
"""

from __future__ import annotations

import math

from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..core.models import PCBDesign, Component, ComponentType, LayerType


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_NET_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9",
]


def _assign_net_colors(design: PCBDesign) -> dict[str, str]:
    """Return a mapping of net-name -> hex colour string."""
    net_names = sorted({t.net for t in design.traces if t.net})
    return {
        name: _NET_PALETTE[i % len(_NET_PALETTE)]
        for i, name in enumerate(net_names)
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _board_outline_polygon(design: PCBDesign) -> list[tuple[float, float]]:
    """Return board outline points, falling back to width/height rectangle."""
    if design.outline:
        return list(design.outline)
    w, h = design.width, design.height
    return [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]


def _offset_segment(
    x0: float, y0: float, x1: float, y1: float, offset: float,
) -> tuple[float, float, float, float]:
    """Offset a line segment perpendicular to its direction."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return x0, y0, x1, y1
    nx, ny = -dy / length, dx / length
    return (
        x0 + nx * offset, y0 + ny * offset,
        x1 + nx * offset, y1 + ny * offset,
    )


def _draw_wide_trace(
    ax,
    points: list[tuple[float, float]],
    width: float,
    color: str,
    alpha: float = 1.0,
) -> None:
    """Render a trace with physical width as a filled polygon strip."""
    if len(points) < 2 or width <= 0:
        return
    hw = width / 2.0
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        lx0, ly0, lx1, ly1 = _offset_segment(x0, y0, x1, y1, hw)
        rx0, ry0, rx1, ry1 = _offset_segment(x0, y0, x1, y1, -hw)
        left.extend([(lx0, ly0), (lx1, ly1)])
        right.extend([(rx0, ry0), (rx1, ry1)])
    poly_pts = left + list(reversed(right))
    patch = MplPolygon(poly_pts, closed=True, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def _component_extent(comp: Component) -> tuple[float, float, float, float]:
    """Return (cx, cy, half_w, half_h) for a component bounding box."""
    if not comp.pads:
        return 0.0, 0.0, 1.0, 0.5
    xs = [p.x for p in comp.pads]
    ys = [p.y for p in comp.pads]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    hw = max((max(xs) - min(xs)) / 2.0 + 0.5, 0.8)
    hh = max((max(ys) - min(ys)) / 2.0 + 0.5, 0.5)
    return cx, cy, hw, hh


def _rotated_rect(
    cx: float, cy: float, hw: float, hh: float, angle_deg: float,
) -> list[tuple[float, float]]:
    """Return 4 corners of a rotated rectangle."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [
        (cx + c[0] * cos_a - c[1] * sin_a,
         cy + c[0] * sin_a + c[1] * cos_a)
        for c in corners
    ]


# ---------------------------------------------------------------------------
# 2D board view
# ---------------------------------------------------------------------------

def plot_board_2d(
    fig: Figure,
    design: PCBDesign,
    highlight_net: str | None = None,
) -> None:
    """Render an enhanced 2D top-down board view.

    Parameters
    ----------
    fig : Figure
        matplotlib figure (will be cleared).
    design : PCBDesign
        The board design to render.
    highlight_net : str or None
        If given, dim all nets except this one.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    net_colors = _assign_net_colors(design)

    # --- Board outline ---
    outline = _board_outline_polygon(design)
    board_patch = MplPolygon(
        outline, closed=True,
        facecolor="#0a3d0a", edgecolor="#33cc33", linewidth=1.5, zorder=1,
    )
    ax.add_patch(board_patch)

    # --- Traces with physical width ---
    for trace in design.traces:
        if len(trace.points) < 2:
            continue
        color = net_colors.get(trace.net, "#cccccc")
        alpha = 1.0
        if highlight_net and trace.net != highlight_net:
            alpha = 0.15
        _draw_wide_trace(ax, trace.points, trace.width, color, alpha=alpha)

    # --- Pads ---
    for comp in design.components:
        for pad in comp.pads:
            alpha = 1.0
            pad_c = Circle(
                (pad.x, pad.y), pad.diameter / 2.0,
                facecolor="#cc8800", edgecolor="#ffcc00",
                linewidth=0.5, zorder=4, alpha=alpha,
            )
            ax.add_patch(pad_c)
            if pad.drill > 0:
                drill_c = Circle(
                    (pad.x, pad.y), pad.drill / 2.0,
                    facecolor="#1a1a2e", edgecolor="none", zorder=5,
                )
                ax.add_patch(drill_c)

    # --- Component outlines ---
    for comp in design.components:
        cx, cy, hw, hh = _component_extent(comp)
        corners = _rotated_rect(cx, cy, hw, hh, comp.rotation)
        comp_patch = MplPolygon(
            corners, closed=True,
            facecolor="none", edgecolor="#dddddd",
            linewidth=0.8, linestyle="--", zorder=6,
        )
        ax.add_patch(comp_patch)
        ax.text(
            cx, cy, comp.reference,
            fontsize=6, color="white", ha="center", va="center",
            zorder=7, fontweight="bold",
        )

    # --- Axes styling ---
    all_xs = [p[0] for p in outline]
    all_ys = [p[1] for p in outline]
    margin = max(design.width, design.height) * 0.05
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", color="white", fontsize=9)
    ax.set_ylabel("Y (mm)", color="white", fontsize=9)
    ax.set_title(f"Board Overview: {design.name}", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, alpha=0.15, color="white")

    # --- Net colour legend ---
    if net_colors and len(net_colors) <= 12:
        handles = [
            MplPolygon(
                [(0, 0)], closed=True, facecolor=c, edgecolor="none", label=n,
            )
            for n, c in net_colors.items()
        ]
        ax.legend(
            handles=handles, loc="upper right", fontsize=6,
            facecolor="#1a1a2e", edgecolor="#444444", labelcolor="white",
        )

    fig.tight_layout()


# ---------------------------------------------------------------------------
# 3D board view
# ---------------------------------------------------------------------------

def _compute_z_scale(design: PCBDesign) -> float:
    """Compute z-axis exaggeration so thin layers are visible."""
    total = design.stackup.total_thickness if design.stackup.layers else 1.6
    board_extent = max(design.width, design.height, 1.0)
    return (board_extent / 5.0) / max(total, 0.01)


def _layer_z_positions(design: PCBDesign, z_scale: float) -> dict[str, float]:
    """Map layer names to their z-centre heights (scaled)."""
    positions: dict[str, float] = {}
    z = 0.0
    for layer in design.stackup.layers:
        mid = z + layer.thickness / 2.0
        positions[layer.name] = mid * z_scale
        z += layer.thickness
    return positions


def _find_trace_z(
    trace_layer: str,
    layer_positions: dict[str, float],
    z_scale: float,
) -> float:
    """Find the z-height for a trace given its layer name."""
    if trace_layer in layer_positions:
        return layer_positions[trace_layer]
    upper = trace_layer.upper()
    for name, z in layer_positions.items():
        if name.upper() == upper:
            return z
    # Default: top surface
    if layer_positions:
        return max(layer_positions.values())
    return 0.0


def _add_box(
    ax,
    x0: float, y0: float, z0: float,
    dx: float, dy: float, dz: float,
    color: str,
    alpha: float = 0.6,
    edgecolor: str = "#333333",
) -> None:
    """Add a rectangular prism (6 faces) to a 3D axes."""
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]
    collection = Poly3DCollection(
        faces, facecolors=color, edgecolors=edgecolor,
        linewidths=0.3, alpha=alpha,
    )
    ax.add_collection3d(collection)


def _component_height(comp: Component) -> float:
    """Estimate component height in mm based on type."""
    heights = {
        ComponentType.IC: 1.5,
        ComponentType.CONNECTOR: 3.0,
        ComponentType.TRANSISTOR_NPN: 1.0,
        ComponentType.TRANSISTOR_PNP: 1.0,
        ComponentType.MOSFET_N: 1.0,
        ComponentType.MOSFET_P: 1.0,
    }
    return heights.get(comp.component_type, 0.5)


def plot_board_3d(fig: Figure, design: PCBDesign) -> None:
    """Render a 3D perspective view of the PCB.

    Parameters
    ----------
    fig : Figure
        matplotlib figure (will be cleared).
    design : PCBDesign
        The board design to render.
    """
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")

    z_scale = _compute_z_scale(design)
    layer_positions = _layer_z_positions(design, z_scale)
    net_colors = _assign_net_colors(design)

    outline = _board_outline_polygon(design)
    ox = min(p[0] for p in outline)
    oy = min(p[1] for p in outline)

    # --- Stackup layers as extruded slabs ---
    z = 0.0
    for layer in design.stackup.layers:
        dz = layer.thickness * z_scale
        is_copper = layer.layer_type in (LayerType.SIGNAL, LayerType.POWER, LayerType.GROUND)
        color = "#b87333" if is_copper else "#d2b48c"
        alpha = 0.7 if is_copper else 0.35
        _add_box(
            ax,
            ox, oy, z,
            design.width, design.height, dz,
            color=color, alpha=alpha,
        )
        z += dz

    total_z = z  # top of the board

    # --- Traces as 3D lines ---
    for trace in design.traces:
        if len(trace.points) < 2:
            continue
        tz = _find_trace_z(trace.layer, layer_positions, z_scale)
        color = net_colors.get(trace.net, "#cccccc")
        xs = [p[0] for p in trace.points]
        ys = [p[1] for p in trace.points]
        zs = [tz] * len(xs)
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, zorder=10)

    # --- Components as 3D prisms ---
    for comp in design.components:
        cx, cy, hw, hh = _component_extent(comp)
        ch = _component_height(comp) * z_scale
        if comp.layer.upper() in ("BOTTOM", "BOT", "BACK", "B.CU"):
            comp_z = -ch
        else:
            comp_z = total_z

        comp_color = "#333399" if comp.component_type == ComponentType.IC else "#336633"
        _add_box(
            ax,
            cx - hw, cy - hh, comp_z,
            hw * 2, hh * 2, ch,
            color=comp_color, alpha=0.8,
        )
        ax.text(
            cx, cy, comp_z + ch,
            comp.reference,
            fontsize=5, color="white", ha="center", va="bottom", zorder=20,
        )

    # --- Axes styling ---
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (scaled)", fontsize=8)
    ax.set_title(f"3D Board View: {design.name}", fontsize=11)
    ax.tick_params(labelsize=7)

    # Set reasonable view angle
    ax.view_init(elev=30, azim=-60)

    fig.tight_layout()
