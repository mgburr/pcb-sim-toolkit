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

from ..core.models import PCBDesign, Component


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
    highlight_component: str | None = None,
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
    highlight_component : str or None
        If given, highlight this component and dim all others.
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

    # --- Copper pours ---
    for pour in getattr(design, "copper_pours", []):
        color = net_colors.get(pour.net, "#b87333")
        pour_patch = MplPolygon(
            pour.outline, closed=True,
            facecolor=color, edgecolor="none",
            alpha=0.2, zorder=2,
        )
        ax.add_patch(pour_patch)

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
        pad_alpha = 1.0
        if highlight_component and comp.reference != highlight_component:
            pad_alpha = 0.3
        for pad in comp.pads:
            p_shape = getattr(pad, "shape", "")
            p_w = getattr(pad, "width", 0.0)
            p_h = getattr(pad, "height", 0.0)
            p_rot = getattr(pad, "rotation", 0.0)

            if p_shape == "rect" and p_w > 0 and p_h > 0:
                corners = _rotated_rect(
                    pad.x, pad.y, p_w / 2.0, p_h / 2.0, p_rot)
                patch = MplPolygon(
                    corners, closed=True,
                    facecolor="#cc8800", edgecolor="#ffcc00",
                    linewidth=0.3, zorder=4, alpha=pad_alpha,
                )
                ax.add_patch(patch)
            elif p_shape == "oval" and p_w > 0 and p_h > 0:
                # Approximate oval as rectangle with rounded appearance
                corners = _rotated_rect(
                    pad.x, pad.y, p_w / 2.0, p_h / 2.0, p_rot)
                patch = MplPolygon(
                    corners, closed=True,
                    facecolor="#cc8800", edgecolor="#ffcc00",
                    linewidth=0.3, zorder=4, alpha=pad_alpha,
                )
                ax.add_patch(patch)
            else:
                pad_c = Circle(
                    (pad.x, pad.y), pad.diameter / 2.0,
                    facecolor="#cc8800", edgecolor="#ffcc00",
                    linewidth=0.5, zorder=4, alpha=pad_alpha,
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
        if highlight_component and comp.reference == highlight_component:
            edge_color = "#00ff00"
            line_width = 2.5
            line_style = "-"
            comp_alpha = 1.0
            text_color = "#00ff00"
            text_size = 7
        elif highlight_component:
            edge_color = "#dddddd"
            line_width = 0.8
            line_style = "--"
            comp_alpha = 0.3
            text_color = "white"
            text_size = 6
        else:
            edge_color = "#dddddd"
            line_width = 0.8
            line_style = "--"
            comp_alpha = 1.0
            text_color = "white"
            text_size = 6
        comp_patch = MplPolygon(
            corners, closed=True,
            facecolor="none", edgecolor=edge_color,
            linewidth=line_width, linestyle=line_style,
            zorder=6, alpha=comp_alpha,
        )
        ax.add_patch(comp_patch)
        ax.text(
            cx, cy, comp.reference,
            fontsize=text_size, color=text_color, ha="center", va="center",
            zorder=7, fontweight="bold", alpha=comp_alpha,
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

# Component height estimates (mm) by refdes prefix
_COMP_HEIGHT: dict[str, float] = {
    "U": 1.5, "IC": 1.5,
    "R": 1.0, "C": 1.0, "L": 1.0,
    "J": 2.5, "P": 2.5, "CN": 2.5,
    "D": 0.8, "LED": 0.8,
    "Q": 1.0, "T": 1.0,
    "SW": 2.0,
    "F": 1.0,
    "Y": 1.2, "X": 1.2,
}

# Component body colour by refdes prefix
_COMP_COLOR: dict[str, str] = {
    "U": "#333333", "IC": "#333333",
    "R": "#4a3728", "C": "#c2a060",
    "L": "#556b2f", "D": "#555555", "LED": "#aa0000",
    "Q": "#333333", "T": "#333333",
    "J": "#777777", "P": "#777777", "CN": "#777777",
    "SW": "#444444",
    "F": "#665544",
    "Y": "#888888", "X": "#888888",
}


def _comp_refdes_prefix(reference: str) -> str:
    """Extract alphabetic prefix from a reference designator."""
    prefix = ""
    for ch in reference:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix


def _get_comp_height(comp: Component) -> float:
    """Estimate component height in mm from refdes prefix."""
    prefix = _comp_refdes_prefix(comp.reference)
    return _COMP_HEIGHT.get(prefix, 0.5)


def _get_comp_color(comp: Component) -> str:
    """Get component body colour from refdes prefix."""
    prefix = _comp_refdes_prefix(comp.reference)
    return _COMP_COLOR.get(prefix, "#555555")


def _extrude_polygon(
    polygon: list[tuple[float, float]],
    z_top: float,
    z_bot: float,
) -> list[list[tuple[float, float, float]]]:
    """Extrude a 2D polygon into a 3D prism (top, bottom, side walls)."""
    # Ensure polygon is clean (at least 3 unique points)
    pts = list(polygon)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []

    faces: list[list[tuple[float, float, float]]] = []
    # Top face
    faces.append([(x, y, z_top) for x, y in pts])
    # Bottom face (reversed winding for correct normals)
    faces.append([(x, y, z_bot) for x, y in reversed(pts)])
    # Side walls
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        faces.append([
            (x0, y0, z_top), (x1, y1, z_top),
            (x1, y1, z_bot), (x0, y0, z_bot),
        ])
    return faces


def _box_faces(
    x0: float, y0: float, z0: float,
    x1: float, y1: float, z1: float,
) -> list[list[tuple[float, float, float]]]:
    """Return 6 faces for an axis-aligned rectangular prism."""
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]


def _rotate_faces(
    faces: list[list[tuple[float, float, float]]],
    cx: float,
    cy: float,
    angle_deg: float,
) -> list[list[tuple[float, float, float]]]:
    """Rotate all face vertices around (cx, cy) by angle_deg in the XY plane."""
    if abs(angle_deg) < 0.01:
        return faces
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rotated: list[list[tuple[float, float, float]]] = []
    for face in faces:
        new_face: list[tuple[float, float, float]] = []
        for x, y, z in face:
            dx, dy = x - cx, y - cy
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            new_face.append((rx, ry, z))
        rotated.append(new_face)
    return rotated


def _component_pad_extent(
    comp: Component,
) -> tuple[float, float, float, float, float, float]:
    """Return (cx, cy, min_px, max_px, min_py, max_py) from pad bounding box.

    Uses pad positions and pad dimensions (diameter as width proxy) with a
    0.3 mm margin, matching the ipc2581-to-kicad approach.
    """
    if not comp.pads:
        return 0.0, 0.0, -1.0, 1.0, -0.5, 0.5

    # Compute pad-relative positions (pads store absolute coords, so we
    # need the component centre first)
    pad_xs = [p.x for p in comp.pads]
    pad_ys = [p.y for p in comp.pads]
    cx = (min(pad_xs) + max(pad_xs)) / 2.0
    cy = (min(pad_ys) + max(pad_ys)) / 2.0

    # Build extents using pad diameters as width/height
    pad_ws = [p.diameter for p in comp.pads]
    pad_hs = [p.diameter for p in comp.pads]

    min_px = min(px - pw / 2 for px, pw in zip(pad_xs, pad_ws)) - cx
    max_px = max(px + pw / 2 for px, pw in zip(pad_xs, pad_ws)) - cx
    min_py = min(py - ph / 2 for py, ph in zip(pad_ys, pad_hs)) - cy
    max_py = max(py + ph / 2 for py, ph in zip(pad_ys, pad_hs)) - cy

    # Add margin
    margin = 0.3
    min_px -= margin
    max_px += margin
    min_py -= margin
    max_py += margin

    return cx, cy, min_px, max_px, min_py, max_py


def _find_trace_z(
    trace_layer: str,
    thickness: float,
) -> float:
    """Return z-offset for a trace: top surface or bottom surface."""
    upper = trace_layer.upper()
    if "BOT" in upper or "BACK" in upper or "B.CU" in upper:
        return -thickness - 0.01
    return 0.01


def plot_board_3d(
    fig: Figure,
    design: PCBDesign,
    highlight_component: str | None = None,
) -> None:
    """Render a 3D perspective view of the PCB.

    Board top surface at z=0, extruded down to z=-thickness.
    Components sit on the appropriate surface. No z-axis exaggeration.

    Parameters
    ----------
    fig : Figure
        matplotlib figure (will be cleared).
    design : PCBDesign
        The board design to render.
    highlight_component : str or None
        If given, highlight this component and dim all others.
    """
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")

    net_colors = _assign_net_colors(design)
    thickness = design.stackup.total_thickness if design.stackup.layers else 1.6
    outline = _board_outline_polygon(design)

    # --- Board body: extrude outline polygon from z=0 to z=-thickness ---
    board_faces = _extrude_polygon(outline, 0.0, -thickness)
    if not board_faces:
        # Fallback to bounding-box
        ox = min(p[0] for p in outline)
        oy = min(p[1] for p in outline)
        board_faces = _box_faces(
            ox, oy, -thickness,
            ox + design.width, oy + design.height, 0.0,
        )
    board_coll = Poly3DCollection(
        board_faces, facecolors="#228B22", edgecolors="#006400",
        linewidths=0.5, alpha=0.3,
    )
    ax.add_collection3d(board_coll)

    # --- Copper zones (flat polygons on surfaces) ---
    # Top copper surface
    if outline:
        pts = list(outline)
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            top_face = [(x, y, 0.005) for x, y in pts]
            top_coll = Poly3DCollection(
                [top_face], facecolors="#b87333", edgecolors="none",
                linewidths=0, alpha=0.25,
            )
            ax.add_collection3d(top_coll)
            bot_face = [(x, y, -thickness - 0.005) for x, y in pts]
            bot_coll = Poly3DCollection(
                [bot_face], facecolors="#b87333", edgecolors="none",
                linewidths=0, alpha=0.25,
            )
            ax.add_collection3d(bot_coll)

    # --- Traces as 3D lines on surface ---
    for trace in design.traces:
        if len(trace.points) < 2:
            continue
        tz = _find_trace_z(trace.layer, thickness)
        color = net_colors.get(trace.net, "#cccccc")
        xs = [p[0] for p in trace.points]
        ys = [p[1] for p in trace.points]
        zs = [tz] * len(xs)
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, zorder=10)

    # --- Components as rotated 3D prisms ---
    for comp in design.components:
        cx, cy, min_px, max_px, min_py, max_py = _component_pad_extent(comp)
        ch = _get_comp_height(comp)
        comp_color = _get_comp_color(comp)

        is_bottom = comp.layer.upper() in ("BOTTOM", "BOT", "BACK", "B.CU")
        if is_bottom:
            z_base = -thickness
            z_top = -thickness - ch
        else:
            z_base = 0.0
            z_top = ch

        faces = _box_faces(
            cx + min_px, cy + min_py, z_base,
            cx + max_px, cy + max_py, z_top,
        )
        faces = _rotate_faces(faces, cx, cy, comp.rotation)

        if highlight_component and comp.reference == highlight_component:
            edge_col = "#00ff00"
            edge_lw = 1.2
            comp_alpha = 0.95
            text_color = "#00ff00"
            text_size = 6
        elif highlight_component:
            edge_col = "#222222"
            edge_lw = 0.4
            comp_alpha = 0.25
            text_color = "white"
            text_size = 5
        else:
            edge_col = "#222222"
            edge_lw = 0.4
            comp_alpha = 0.85
            text_color = "white"
            text_size = 5

        comp_coll = Poly3DCollection(
            faces, facecolors=comp_color, edgecolors=edge_col,
            linewidths=edge_lw, alpha=comp_alpha,
        )
        ax.add_collection3d(comp_coll)

        # Label at top of component
        label_z = z_top if not is_bottom else z_base
        ax.text(
            cx, cy, label_z + 0.1,
            comp.reference,
            fontsize=text_size, color=text_color,
            ha="center", va="bottom", zorder=20,
        )

    # --- Axes styling ---
    all_xs = [p[0] for p in outline]
    all_ys = [p[1] for p in outline]
    margin = max(design.width, design.height) * 0.05
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    z_range = thickness + 5.0  # room for components
    ax.set_zlim(-thickness - z_range * 0.3, z_range * 0.7)

    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.set_title(f"3D Board View: {design.name}", fontsize=11)
    ax.tick_params(labelsize=7)

    ax.view_init(elev=30, azim=-60)
    fig.tight_layout()
