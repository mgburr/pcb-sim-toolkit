"""Mechanical data summary and visualization for PCB designs.

Provides ``compute_board_summary`` for extracting mechanical metrics and
``plot_mechanical_summary`` for rendering a multi-panel overview figure.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from ..core.models import PCBDesign, LayerType


def compute_board_summary(design: PCBDesign) -> dict:
    """Compute a mechanical summary dictionary from a PCBDesign.

    Returns a dict with board-level metrics and a per-component list.
    """
    board_area_mm2 = design.width * design.height
    board_thickness = (
        design.stackup.total_thickness if design.stackup.layers else 1.6
    )

    smd_count = 0
    th_count = 0
    top_count = 0
    bottom_count = 0
    total_pins = 0
    est_comp_weight = 0.0
    unique_packages: set[str] = set()
    type_counts: dict[str, int] = {}
    comp_details: list[dict] = []

    for comp in design.components:
        pkg = comp.package_def
        ctype = comp.component_type.value
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

        pin_count = pkg.pin_count if pkg else len(comp.pads)
        total_pins += pin_count

        is_bottom = comp.layer.upper() in ("BOTTOM", "BOT", "BACK", "B.CU")
        if is_bottom:
            bottom_count += 1
        else:
            top_count += 1

        # SMD vs through-hole detection
        # Prefer package pad_shape info (explicitly parsed drill data)
        is_th = False
        if pkg and pkg.pad_shape:
            is_th = pkg.pad_shape.drill_diameter > 0
        elif comp.pads:
            # Only treat as TH if drill > pad diameter threshold (not default)
            is_th = any(p.drill > 0.5 * p.diameter for p in comp.pads)
        if is_th:
            th_count += 1
        else:
            smd_count += 1

        pkg_name = pkg.name if pkg else comp.footprint or ""
        if pkg_name:
            unique_packages.add(pkg_name)

        body_str = ""
        pad_type = "SMD" if not is_th else "TH"
        height = 0.0
        pitch = 0.0
        if pkg:
            if pkg.body_width > 0 and pkg.body_height > 0:
                body_str = f"{pkg.body_width:.1f}x{pkg.body_height:.1f}"
            height = pkg.height
            pitch = pkg.pin_pitch
            est_comp_weight += pkg.weight_grams

        comp_details.append({
            "refdes": comp.reference,
            "package": pkg_name,
            "body": body_str,
            "pins": pin_count,
            "pitch": f"{pitch:.2f}" if pitch > 0 else "",
            "pad_type": pad_type,
            "height": f"{height:.2f}" if height > 0 else "",
            "layer": comp.layer,
        })

    # Estimate board weight: FR-4 density ~1.85 g/cm3
    # copper layers add ~8.96 g/cm3 * thickness
    board_volume_cm3 = (board_area_mm2 * board_thickness) / 1000.0
    copper_thickness = sum(
        layer.thickness for layer in design.stackup.layers
        if layer.layer_type in (LayerType.SIGNAL, LayerType.POWER, LayerType.GROUND)
    ) if design.stackup.layers else 0.07  # default 2 layers * 35um
    copper_volume_cm3 = (board_area_mm2 * copper_thickness) / 1000.0
    est_board_weight = board_volume_cm3 * 1.85 + copper_volume_cm3 * 8.96

    return {
        "board_area_mm2": board_area_mm2,
        "board_thickness_mm": board_thickness,
        "component_count": len(design.components),
        "component_count_by_type": type_counts,
        "total_pin_count": total_pins,
        "estimated_weight_grams": est_comp_weight,
        "estimated_board_weight_grams": est_board_weight,
        "smd_count": smd_count,
        "through_hole_count": th_count,
        "top_side_count": top_count,
        "bottom_side_count": bottom_count,
        "unique_packages": len(unique_packages),
        "components": comp_details,
    }


def plot_mechanical_summary(fig: Figure, design: PCBDesign) -> None:
    """Render a multi-panel mechanical summary figure.

    Left panel: board-level summary stats as formatted text.
    Right panel: component type distribution pie chart.
    """
    fig.clear()
    summary = compute_board_summary(design)

    ax_text = fig.add_subplot(121)
    ax_pie = fig.add_subplot(122)

    # --- Left panel: summary text ---
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.set_title("Board Mechanical Summary", fontsize=11, fontweight="bold")

    lines = [
        f"Board area:  {summary['board_area_mm2']:.0f} mm2"
        f"  ({design.width:.1f} x {design.height:.1f} mm)",
        f"Thickness:   {summary['board_thickness_mm']:.2f} mm",
        "",
        f"Components:  {summary['component_count']}",
        f"  SMD:         {summary['smd_count']}",
        f"  Through-hole:{summary['through_hole_count']}",
        f"  Top side:    {summary['top_side_count']}",
        f"  Bottom side: {summary['bottom_side_count']}",
        "",
        f"Total pins:  {summary['total_pin_count']}",
        f"Unique pkgs: {summary['unique_packages']}",
        "",
        f"Est. component weight: {summary['estimated_weight_grams']:.2f} g",
        f"Est. board weight:     {summary['estimated_board_weight_grams']:.2f} g",
    ]
    text = "\n".join(lines)
    ax_text.text(
        0.05, 0.95, text,
        transform=ax_text.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="top",
    )
    for spine in ax_text.spines.values():
        spine.set_visible(False)

    # --- Right panel: pie chart ---
    type_counts = summary["component_count_by_type"]
    if type_counts:
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        ax_pie.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
        ax_pie.set_title("Component Types", fontsize=11, fontweight="bold")
    else:
        ax_pie.text(0.5, 0.5, "No components", ha="center", va="center",
                    fontsize=12, color="gray")
        ax_pie.set_xticks([])
        ax_pie.set_yticks([])

    fig.tight_layout()
