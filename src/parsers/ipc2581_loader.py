"""Load PCB designs from IPC-2581 (.cvg / .xml) files.

Supports both IPC-2581B and IPC-2581C revisions. Key format differences handled:

- **Units**: Rev B uses ``CadHeader/@units`` (e.g. ``MILLIMETER``);
  Rev C often uses ``Content/@units``.
- **Layer definitions**: Rev B stores ``<Layer>`` elements under ``<CadData>``
  with ``layerFunction`` attributes that are cross-referenced by the stackup.
  Rev C may embed ``layerFunctionType`` directly on ``<StackupLayer>``.
- **Stackup**: Rev C may use a ``<Dielectric>`` child for ``epsilonR``;
  Rev B stores dielectric info only on the ``<Layer>`` definition.
- **Components**: Rev B places position in a ``<Location>`` child and rotation
  in ``<Xform>``; Rev C merges both into ``<Xform x= y= rotation=>``.
- **Nets**: Rev B has no ``<LogicalNet>`` or ``<PhyNet>`` — net names appear on
  ``<PadStack net=>`` and ``<Set net=>``. Rev C uses ``<LogicalNet>`` at the
  root level or ``<PhyNet>`` inside ``<PhyNetGroup>``.
- **Traces**: Rev B wraps ``<Line>`` inside ``<Features><UserSpecial>``
  containers; Rev C places ``<Line>`` directly under ``<Set>``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import math

from ..core.models import (
    PCBDesign,
    Component,
    ComponentType,
    CopperPour,
    Net,
    Trace,
    Pad,
    PadShape,
    PackageDef,
    Stackup,
    Layer,
    LayerType,
)


def load_ipc2581(path: Path) -> PCBDesign:
    """Parse an IPC-2581 .cvg/.xml file and return a PCBDesign."""
    tree = ET.parse(path)
    root = tree.getroot()

    design = PCBDesign(name=path.stem, width=100, height=80)

    # Determine unit — try CadHeader first (Rev B), then Content (Rev C)
    unit = _get_default_unit(root)

    # Build layer function lookup from <Layer> definitions under CadData
    layer_defs = _parse_layer_defs(root)

    # Parse board outline for dimensions
    _parse_board_outline(root, design, unit)

    # Parse stackup, using layer_defs for cross-referencing
    design.stackup = _parse_stackup(root, unit, layer_defs)

    # Parse components
    design.components = _parse_components(root, unit)

    # Parse nets (LogicalNet, PhyNet, and fallback to PadStack/Set nets)
    design.nets = _parse_nets(root)

    # Parse traces
    design.traces = _parse_traces(root, unit)

    # Parse pad definitions and packages, then link to components
    pad_defs = _parse_pad_definitions(root, unit)
    design.packages = _parse_packages(root, unit, pad_defs)
    design.link_packages()
    holes = _parse_holes(root, unit)
    # Merge PadStack-based holes (LayerHole + PadStackInstance)
    padstack_holes = _parse_padstack_holes(root, unit)
    for k, v in padstack_holes.items():
        holes.setdefault(k, v)
    _synthesize_pads_from_packages(design, holes)
    _compute_fallback_mechanical(design)

    # Parse copper pours
    design.copper_pours = _parse_copper_pours(root, unit)

    return design


# ---- Helpers ----

def _strip_namespace(tag: str) -> str:
    """Remove namespace URI prefix from an XML tag."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _convert_to_mm(value: float, unit: str) -> float:
    """Convert a numeric value to millimeters based on the given unit."""
    unit = unit.upper()
    if unit in ("MM", "MILLIMETER"):
        return value
    elif unit in ("MIL", "THOU"):
        return value * 0.0254
    elif unit in ("INCH", "IN"):
        return value * 25.4
    elif unit in ("UM", "MICRON", "MICROMETER"):
        return value * 0.001
    return value


def _infer_component_type(refdes: str) -> ComponentType:
    """Infer component type from reference designator prefix."""
    for prefix, ct in [
        ("R", ComponentType.RESISTOR),
        ("C", ComponentType.CAPACITOR),
        ("L", ComponentType.INDUCTOR),
        ("D", ComponentType.DIODE),
        ("Q", ComponentType.TRANSISTOR_NPN),
        ("U", ComponentType.IC),
        ("J", ComponentType.CONNECTOR),
    ]:
        if refdes.startswith(prefix) and (len(refdes) == 1 or refdes[1:2].isdigit()):
            return ct
    return ComponentType.IC


def _get_default_unit(root: ET.Element) -> str:
    """Extract the default unit from the IPC-2581 document.

    Checks (in order):
    1. ``<CadHeader units="...">`` (Rev B — always present, authoritative)
    2. ``<DictionaryStandard units="...">`` (Rev B Altium exports)
    3. ``<Content units="...">`` (Rev C)
    4. Root element attributes
    """
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag == "CadHeader":
            for attr in ("units", "unit", "Units", "Unit"):
                if attr in elem.attrib:
                    return elem.attrib[attr]
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag == "DictionaryStandard":
            for attr in ("units", "unit", "Units", "Unit"):
                if attr in elem.attrib:
                    return elem.attrib[attr]
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag == "Content":
            for attr in ("units", "unit", "Units", "Unit"):
                if attr in elem.attrib:
                    return elem.attrib[attr]
    for attr in ("units", "unit", "Units", "Unit"):
        if attr in root.attrib:
            return root.attrib[attr]
    return "MM"


def _find_elements(parent: ET.Element, local_name: str) -> list[ET.Element]:
    """Find all descendant elements matching a local tag name, ignoring namespaces."""
    return [e for e in parent.iter() if _strip_namespace(e.tag) == local_name]


def _find_element(parent: ET.Element, local_name: str) -> ET.Element | None:
    """Find the first descendant element matching a local tag name."""
    for elem in parent.iter():
        if _strip_namespace(elem.tag) == local_name:
            return elem
    return None


def _find_child(parent: ET.Element, local_name: str) -> ET.Element | None:
    """Find a direct child element by local name, ignoring namespaces."""
    for child in parent:
        if _strip_namespace(child.tag) == local_name:
            return child
    return None


def _find_children(parent: ET.Element, local_name: str) -> list[ET.Element]:
    """Find all direct children matching a local tag name."""
    return [ch for ch in parent if _strip_namespace(ch.tag) == local_name]


# ---- Layer definition parsing (Rev B cross-reference) ----

def _parse_layer_defs(root: ET.Element) -> dict[str, str]:
    """Build a mapping of layer name → layerFunction from <Layer> elements.

    Rev B files define ``<Layer name="Top Layer" layerFunction="SIGNAL" ...>``
    under ``<CadData>``.  This mapping is used to determine the type of stackup
    layers, since Rev B ``<StackupLayer>`` elements lack a ``layerFunctionType``
    attribute.
    """
    defs: dict[str, str] = {}
    for elem in _find_elements(root, "Layer"):
        name = elem.attrib.get("name", "")
        func = elem.attrib.get("layerFunction", "")
        if name and func:
            defs[name] = func.upper()
    return defs


# ---- Section parsers ----

def _parse_board_outline(root: ET.Element, design: PCBDesign, unit: str) -> None:
    """Extract board dimensions from Profile polygon and curve points."""
    profile = _find_element(root, "Profile")
    if profile is None:
        return

    # Outer boundary from first Polygon child
    polygon = _find_child(profile, "Polygon")
    if polygon is not None:
        points = _parse_polygon_points(polygon, unit)
    else:
        # Fallback: collect all points from Profile directly
        points = _parse_polygon_points(profile, unit)

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        design.width = max(xs) - min(xs)
        design.height = max(ys) - min(ys)
        design.outline = points

    # Parse cutouts
    for cutout in _find_children(profile, "Cutout"):
        cut_pts = _parse_polygon_points(cutout, unit)
        if len(cut_pts) >= 3:
            design.outline_cutouts.append(cut_pts)


def _parse_stackup(
    root: ET.Element,
    unit: str,
    layer_defs: dict[str, str],
) -> Stackup:
    """Parse stackup layers, cross-referencing Layer definitions for type."""
    layers: list[Layer] = []

    for elem in _find_elements(root, "StackupLayer"):
        name = elem.attrib.get("layerOrGroupRef", elem.attrib.get("name", ""))
        thickness_str = elem.attrib.get("thickness", "")
        try:
            raw_thickness = _convert_to_mm(float(thickness_str), unit) if thickness_str else -1.0
        except ValueError:
            raw_thickness = -1.0

        material = elem.attrib.get("material", "")
        dielectric_constant = 1.0

        # --- Determine layer type ---
        # Strategy 1: Rev C — check layerFunctionType on the StackupLayer itself
        layer_func = elem.attrib.get("layerFunctionType", "").upper()

        # Strategy 2: Rev B — cross-reference the Layer definition by name
        if not layer_func and name in layer_defs:
            layer_func = layer_defs[name]

        # Skip non-physical layers (paste, silkscreen, document, drill, etc.)
        if _is_non_physical_function(layer_func):
            continue

        # Strategy 3: Check for <Dielectric> child (Rev C led_power_board style)
        dielectric_child = _find_child(elem, "Dielectric")
        if dielectric_child is not None:
            layer_type = LayerType.DIELECTRIC
            try:
                dielectric_constant = float(
                    dielectric_child.attrib.get("epsilonR", "4.5"))
            except ValueError:
                dielectric_constant = 4.5
            material = dielectric_child.attrib.get("material", material) or "FR-4"
        elif _is_dielectric_function(layer_func):
            layer_type = LayerType.DIELECTRIC
            material = material or "FR-4"
            try:
                dielectric_constant = float(
                    elem.attrib.get("dielectricConstant", "4.5"))
            except ValueError:
                dielectric_constant = 4.5
        elif _is_power_function(layer_func):
            layer_type = LayerType.POWER
            material = material or "copper"
        elif _is_signal_function(layer_func):
            layer_type = LayerType.SIGNAL
            material = material or "copper"
        else:
            # Unknown function with no real thickness — skip
            if raw_thickness <= 0:
                continue
            layer_type = LayerType.SIGNAL
            material = material or "copper"

        # Apply default thickness for copper layers that report 0
        thickness = raw_thickness if raw_thickness > 0 else 0.035

        layers.append(
            Layer(
                name=name,
                layer_type=layer_type,
                thickness=thickness,
                material=material,
                dielectric_constant=dielectric_constant,
            )
        )

    return Stackup(layers=layers)


def _is_non_physical_function(func: str) -> bool:
    """Return True for layer functions that are not part of the physical stackup."""
    return any(kw in func for kw in (
        "PASTEMASK", "SOLDERMASK", "SOLDER_MASK", "SOLDER_PASTE", "SOLDERPASTE",
        "SILKSCREEN", "SILK_SCREEN", "LEGEND",
        "ASSEMBLY", "ASSEMBLY_DRAWING",
        "DOCUMENT", "DOCUMENTATION",
        "DRILL", "DRILL_FIGURE", "DRILL_DRAWING",
        "BOARD_OUTLINE", "ROUT", "ROUTE",
    ))


def _is_dielectric_function(func: str) -> bool:
    return any(kw in func for kw in (
        "DIELECTRIC", "DIELCORE", "DIELPREPREG", "CORE", "PREPREG",
    ))


def _is_power_function(func: str) -> bool:
    return any(kw in func for kw in ("POWER", "GROUND", "POWER_GROUND", "PLANE"))


def _is_signal_function(func: str) -> bool:
    return any(kw in func for kw in ("SIGNAL", "MIXED", "CONDUCTOR"))


def _parse_components(root: ET.Element, unit: str) -> list[Component]:
    """Parse Component elements into Component objects.

    Handles two position conventions:
    - Rev C: ``<Xform x="..." y="..." rotation="..."/>``
    - Rev B: ``<Xform rotation="..."/>`` + ``<Location x="..." y="..."/>``
    """
    components: list[Component] = []

    for elem in _find_elements(root, "Component"):
        refdes = elem.attrib.get("refDes", elem.attrib.get("name", ""))
        if not refdes:
            continue

        value = elem.attrib.get("value", elem.attrib.get("partNumber", ""))
        if not value:
            value = elem.attrib.get("part", "")
        footprint = elem.attrib.get("packageRef", elem.attrib.get("standardPackageRef", ""))
        comp_type = _infer_component_type(refdes)

        # Determine component position, rotation, and layer
        comp_x, comp_y = 0.0, 0.0
        rotation = 0.0
        comp_layer = "Top"

        # Try Xform with x/y (Rev C style)
        xform = _find_child(elem, "Xform")
        if xform is not None:
            if "x" in xform.attrib:
                try:
                    comp_x = _convert_to_mm(float(xform.attrib.get("x", "0")), unit)
                    comp_y = _convert_to_mm(float(xform.attrib.get("y", "0")), unit)
                except ValueError:
                    pass
            if "rotation" in xform.attrib:
                try:
                    rotation = float(xform.attrib["rotation"])
                except ValueError:
                    pass

        # Try separate Location child (Rev B style, or fallback)
        loc = _find_child(elem, "Location")
        if loc is not None:
            try:
                comp_x = _convert_to_mm(float(loc.attrib.get("x", "0")), unit)
                comp_y = _convert_to_mm(float(loc.attrib.get("y", "0")), unit)
            except ValueError:
                pass

        # Extract layer from layerRef or side attribute
        layer_ref = elem.attrib.get("layerRef", elem.attrib.get("side", ""))
        if layer_ref:
            upper = layer_ref.upper()
            if "BOT" in upper or "BACK" in upper:
                comp_layer = "Bottom"
            else:
                comp_layer = "Top"

        # Parse pads from Pin children of the Component
        # Note: Rev B components don't have inline Pin children — pads come
        # from the Package definition.  We still collect any inline pins.
        pads: list[Pad] = []
        for pin in _find_children(elem, "Pin"):
            pin_name = pin.attrib.get("number", pin.attrib.get("name", ""))
            pin_x, pin_y = comp_x, comp_y
            pin_loc = _find_child(pin, "Location")
            if pin_loc is not None:
                try:
                    pin_x = comp_x + _convert_to_mm(
                        float(pin_loc.attrib.get("x", "0")), unit)
                    pin_y = comp_y + _convert_to_mm(
                        float(pin_loc.attrib.get("y", "0")), unit)
                except ValueError:
                    pass
            pads.append(Pad(name=pin_name, x=pin_x, y=pin_y))

        components.append(
            Component(
                reference=refdes,
                component_type=comp_type,
                value=value,
                footprint=footprint,
                pads=pads,
                rotation=rotation,
                layer=comp_layer,
                properties={"_x": comp_x, "_y": comp_y},
            )
        )

    return components


def _parse_nets(root: ET.Element) -> list[Net]:
    """Parse nets from LogicalNet, PhyNet, and PadStack/Set fallbacks.

    Rev C files use ``<LogicalNet>`` at the root level with ``<PinRef>``
    children.  Rev B files may have no explicit net list — net names appear
    only on ``<PadStack net="...">`` and ``<Set net="...">``.
    """
    nets: list[Net] = []
    seen: set[str] = set()

    # --- LogicalNet (Rev C) ---
    for elem in _find_elements(root, "LogicalNet"):
        name = elem.attrib.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)

        nodes: list[str] = []
        for pin_ref in _find_children(elem, "PinRef"):
            comp_ref = pin_ref.attrib.get("componentRef", pin_ref.attrib.get("compRef", ""))
            pin = pin_ref.attrib.get("pin", pin_ref.attrib.get("pinRef", ""))
            if comp_ref and pin:
                nodes.append(f"{comp_ref}.{pin}")
        nets.append(Net(name=name, nodes=nodes))

    # --- PhyNet inside PhyNetGroup (Rev C alternative) ---
    for elem in _find_elements(root, "PhyNet"):
        name = elem.attrib.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)

        nodes: list[str] = []
        for pin_ref in _find_children(elem, "PinRef"):
            comp_ref = pin_ref.attrib.get("componentRef", pin_ref.attrib.get("compRef", ""))
            pin = pin_ref.attrib.get("pin", pin_ref.attrib.get("pinRef", ""))
            if comp_ref and pin:
                nodes.append(f"{comp_ref}.{pin}")
        nets.append(Net(name=name, nodes=nodes))

    # --- Fallback: collect net names from PadStack and Set attributes (Rev B) ---
    for elem in _find_elements(root, "PadStack"):
        name = elem.attrib.get("net", "")
        if name and name not in seen and name != "No Net":
            seen.add(name)
            nets.append(Net(name=name, nodes=[]))

    for elem in _find_elements(root, "Set"):
        name = elem.attrib.get("net", "")
        if name and name not in seen and name != "No Net":
            seen.add(name)
            nets.append(Net(name=name, nodes=[]))

    return nets


def _parse_traces(root: ET.Element, unit: str) -> list[Trace]:
    """Parse trace geometry from LayerFeature/Set elements.

    Handles two nesting conventions:
    - Rev C: ``<Set><Line .../></Set>``
    - Rev B: ``<Set><Features><UserSpecial><Line .../></UserSpecial></Features></Set>``
    """
    traces: list[Trace] = []

    for lf_elem in _find_elements(root, "LayerFeature"):
        layer_name = lf_elem.attrib.get("layerRef", "")

        for set_elem in _find_children(lf_elem, "Set"):
            net_name = set_elem.attrib.get("net", "")

            # Collect all <Line> elements regardless of nesting depth
            for line_elem in _find_elements(set_elem, "Line"):
                try:
                    sx = _convert_to_mm(float(
                        line_elem.attrib.get("startX",
                            line_elem.attrib.get("x1", "0"))), unit)
                    sy = _convert_to_mm(float(
                        line_elem.attrib.get("startY",
                            line_elem.attrib.get("y1", "0"))), unit)
                    ex = _convert_to_mm(float(
                        line_elem.attrib.get("endX",
                            line_elem.attrib.get("x2", "0"))), unit)
                    ey = _convert_to_mm(float(
                        line_elem.attrib.get("endY",
                            line_elem.attrib.get("y2", "0"))), unit)
                    width_str = line_elem.attrib.get(
                        "lineWidth", line_elem.attrib.get("width", "0.2"))
                    width = _convert_to_mm(float(width_str), unit)
                except ValueError:
                    continue

                traces.append(
                    Trace(
                        net=net_name,
                        width=width,
                        layer=layer_name,
                        points=[(sx, sy), (ex, ey)],
                    )
                )

    return traces


# ---- Pad and package parsing ----

def _parse_pad_definitions(root: ET.Element, unit: str) -> dict[str, PadShape]:
    """Parse pad shapes from DictionaryStandard/DictionaryUser entries."""
    pad_defs: dict[str, PadShape] = {}

    for dict_elem in (_find_elements(root, "DictionaryStandard")
                      + _find_elements(root, "DictionaryUser")):
        dict_unit = dict_elem.attrib.get(
            "units", dict_elem.attrib.get("unit", unit))

        for entry in (_find_children(dict_elem, "EntryStandard")
                      + _find_children(dict_elem, "EntryUser")):
            entry_id = entry.attrib.get("id", "")
            if not entry_id:
                continue

            shape_type = "rect"
            width = 0.0
            height = 0.0
            drill_diameter = 0.0
            plated = True

            for child in entry:
                tag = _strip_namespace(child.tag)
                if tag == "Circle":
                    shape_type = "circle"
                    d = _convert_to_mm(
                        float(child.attrib.get("diameter",
                              child.attrib.get("radius", "0"))), dict_unit)
                    if "radius" in child.attrib:
                        d *= 2
                    width = height = d
                elif tag == "RectCenter":
                    shape_type = "rect"
                    width = _convert_to_mm(
                        float(child.attrib.get("width", "0")), dict_unit)
                    height = _convert_to_mm(
                        float(child.attrib.get("height", "0")), dict_unit)
                elif tag == "Oval":
                    shape_type = "oval"
                    width = _convert_to_mm(
                        float(child.attrib.get("width", "0")), dict_unit)
                    height = _convert_to_mm(
                        float(child.attrib.get("height", "0")), dict_unit)
                elif tag in ("Polygon", "Contour"):
                    shape_type = "polygon"
                    pts_x: list[float] = []
                    pts_y: list[float] = []
                    for pt in child.iter():
                        pt_tag = _strip_namespace(pt.tag)
                        if pt_tag in ("PolyBegin", "PolyStepSegment"):
                            x_s = pt.attrib.get("x", pt.attrib.get("X"))
                            y_s = pt.attrib.get("y", pt.attrib.get("Y"))
                            if x_s is not None and y_s is not None:
                                pts_x.append(_convert_to_mm(float(x_s), dict_unit))
                                pts_y.append(_convert_to_mm(float(y_s), dict_unit))
                    if pts_x:
                        width = max(pts_x) - min(pts_x)
                        height = max(pts_y) - min(pts_y)
                elif tag in ("Drill", "DrillHole"):
                    try:
                        drill_diameter = _convert_to_mm(
                            float(child.attrib.get("diameter", "0")), dict_unit)
                    except ValueError:
                        pass
                    plated_str = child.attrib.get("plated", "true").lower()
                    plated = plated_str in ("true", "yes", "1")

            if width > 0 or height > 0:
                pad_defs[entry_id] = PadShape(
                    shape_type=shape_type,
                    width=width,
                    height=height,
                    drill_diameter=drill_diameter,
                    plated=plated,
                )

    return pad_defs


def _parse_packages(
    root: ET.Element,
    unit: str,
    pad_defs: dict[str, PadShape],
) -> dict[str, PackageDef]:
    """Parse Package elements into PackageDef objects."""
    packages: dict[str, PackageDef] = {}

    for pkg_elem in _find_elements(root, "Package"):
        name = pkg_elem.attrib.get("name", "")
        if not name:
            continue

        pins: list[tuple[str, float, float]] = []
        pad_shape: PadShape | None = None
        pin_shapes: dict[str, PadShape] = {}

        for pin in _find_children(pkg_elem, "Pin"):
            pin_name = pin.attrib.get("number", pin.attrib.get("name", ""))
            px, py = 0.0, 0.0
            loc = _find_child(pin, "Location")
            if loc is not None:
                try:
                    px = _convert_to_mm(float(loc.attrib.get("x", "0")), unit)
                    py = _convert_to_mm(float(loc.attrib.get("y", "0")), unit)
                except ValueError:
                    pass
            pins.append((pin_name, px, py))

            # Resolve pad shape from StandardPrimitiveRef for every pin
            prim_ref = _find_child(pin, "StandardPrimitiveRef")
            if prim_ref is not None:
                ref_id = prim_ref.attrib.get("id", "")
                if ref_id in pad_defs:
                    pin_shapes[pin_name] = pad_defs[ref_id]
            # Keep first-found as package-level fallback
            if pad_shape is None and pin_name in pin_shapes:
                pad_shape = pin_shapes[pin_name]

        # Parse body outline
        body_w, body_h = 0.0, 0.0
        outline = _find_child(pkg_elem, "Outline")
        if outline is not None:
            body_w, body_h = _bounding_box_from_polygon_elem(outline, unit)

        # Parse courtyard
        court_w, court_h = 0.0, 0.0
        courtyard = _find_child(pkg_elem, "Courtyard")
        if courtyard is not None:
            court_w, court_h = _bounding_box_from_polygon_elem(courtyard, unit)

        # Compute pin pitch (minimum non-zero distance between any two pins)
        pin_pitch = 0.0
        if len(pins) >= 2:
            min_dist = float("inf")
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    d = math.hypot(pins[j][1] - pins[i][1],
                                   pins[j][2] - pins[i][2])
                    if d > 1e-6 and d < min_dist:
                        min_dist = d
            if min_dist < float("inf"):
                pin_pitch = min_dist

        packages[name] = PackageDef(
            name=name,
            pin_count=len(pins),
            pins=pins,
            pad_shape=pad_shape,
            pin_shapes=pin_shapes,
            body_width=body_w,
            body_height=body_h,
            courtyard_width=court_w,
            courtyard_height=court_h,
            pin_pitch=pin_pitch,
            source="parsed",
        )

    return packages


def _bounding_box_from_polygon_elem(
    elem: ET.Element, unit: str,
) -> tuple[float, float]:
    """Extract width/height from a polygon element's bounding box."""
    pts_x: list[float] = []
    pts_y: list[float] = []
    for pt in elem.iter():
        tag = _strip_namespace(pt.tag)
        if tag in ("PolyBegin", "PolyStepSegment", "PolyStepCurve",
                    "LineBegin", "LineEnd"):
            x_s = pt.attrib.get("x", pt.attrib.get("X"))
            y_s = pt.attrib.get("y", pt.attrib.get("Y"))
            if x_s is not None and y_s is not None:
                pts_x.append(_convert_to_mm(float(x_s), unit))
                pts_y.append(_convert_to_mm(float(y_s), unit))
    if pts_x:
        return max(pts_x) - min(pts_x), max(pts_y) - min(pts_y)
    return 0.0, 0.0


# ---- Hole parsing and pad synthesis from Package definitions (Rev B) ----

def _parse_holes(root: ET.Element, unit: str) -> dict[tuple[float, float], float]:
    """Parse ``<Hole>`` elements into a position → drill diameter lookup.

    Rounds coordinates to 3 decimal places for fuzzy matching.
    """
    holes: dict[tuple[float, float], float] = {}
    for elem in _find_elements(root, "Hole"):
        try:
            x = round(_convert_to_mm(float(elem.attrib.get("x", "0")), unit), 3)
            y = round(_convert_to_mm(float(elem.attrib.get("y", "0")), unit), 3)
            d = _convert_to_mm(float(elem.attrib.get("diameter", "0")), unit)
        except ValueError:
            continue
        if d > 0:
            holes[(x, y)] = d
    return holes


def _synthesize_pads_from_packages(
    design: PCBDesign,
    holes: dict[tuple[float, float], float],
) -> None:
    """Create Pad objects for components that have no inline pads.

    Rev B IPC-2581 files define pin positions in ``<Package>`` elements
    rather than inline in ``<Component>``.  After ``link_packages()`` has
    cross-referenced packages, this function converts package-relative pin
    positions to absolute board coordinates using each component's stored
    position and rotation.  Drill diameters are looked up from the parsed
    ``<Hole>`` elements; pads with no matching hole are treated as SMD.
    """
    for comp in design.components:
        if comp.pads:
            continue
        pkg = comp.package_def
        if pkg is None or not pkg.pins:
            continue

        cx = comp.properties.get("_x", 0.0)
        cy = comp.properties.get("_y", 0.0)
        rad = math.radians(comp.rotation)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        for pin_name, px, py in pkg.pins:
            # Rotate relative pin position by component rotation
            abs_x = cx + px * cos_a - py * sin_a
            abs_y = cy + px * sin_a + py * cos_a

            # Resolve per-pin shape, falling back to package-level shape
            pin_shape = pkg.pin_shapes.get(pin_name)
            if pin_shape:
                pw, ph = pin_shape.width, pin_shape.height
                shape_type = pin_shape.shape_type
            elif pkg.pad_shape:
                pw, ph = pkg.pad_shape.width, pkg.pad_shape.height
                shape_type = pkg.pad_shape.shape_type
            else:
                pw, ph = 1.0, 1.0
                shape_type = "circle"

            # Look up drill hole at this position
            key = (round(abs_x, 3), round(abs_y, 3))
            drill = holes.get(key, 0.0)

            comp.pads.append(Pad(
                name=pin_name, x=abs_x, y=abs_y,
                diameter=max(pw, ph, 0.5),
                width=pw, height=ph,
                shape=shape_type,
                rotation=comp.rotation,
                drill=drill,
            ))


# ---- PadStack-based hole parsing ----

def _parse_padstack_holes(
    root: ET.Element, unit: str,
) -> dict[tuple[float, float], float]:
    """Parse drill diameters from PadStack/LayerHole + PadStackInstance positions.

    IPC-2581 Rev B stores drill info in ``<LayerHole>`` inside ``<PadStack>``
    definitions, with positions coming from ``<PadStackInstance>`` elements.
    """
    # Step 1: Build padstack_name -> drill_diameter from PadStack/LayerHole
    padstack_drills: dict[str, float] = {}
    for ps_elem in _find_elements(root, "PadStack"):
        ps_name = ps_elem.attrib.get("name", ps_elem.attrib.get("id", ""))
        if not ps_name:
            continue
        for lh in _find_elements(ps_elem, "LayerHole"):
            try:
                d = _convert_to_mm(
                    float(lh.attrib.get("diameter", "0")), unit)
            except ValueError:
                continue
            if d > 0:
                padstack_drills[ps_name] = d
                break  # one drill per padstack

    # Step 2: Build position from PadStackInstance x,y + padStackDefRef
    holes: dict[tuple[float, float], float] = {}
    for psi in _find_elements(root, "PadStackInstance"):
        ps_ref = psi.attrib.get("padstackDefRef",
                                psi.attrib.get("padStackRef", ""))
        if ps_ref not in padstack_drills:
            continue
        loc = _find_child(psi, "Location")
        if loc is None:
            continue
        try:
            x = round(_convert_to_mm(
                float(loc.attrib.get("x", "0")), unit), 3)
            y = round(_convert_to_mm(
                float(loc.attrib.get("y", "0")), unit), 3)
        except ValueError:
            continue
        holes[(x, y)] = padstack_drills[ps_ref]

    return holes


# ---- Copper pour parsing ----

def _parse_polygon_points(
    polygon_elem: ET.Element, unit: str,
) -> list[tuple[float, float]]:
    """Extract vertex list from a Polygon/Cutout element's PolyBegin/PolyStepSegment children."""
    points: list[tuple[float, float]] = []
    for pt in polygon_elem.iter():
        tag = _strip_namespace(pt.tag)
        if tag in ("PolyBegin", "PolyStepSegment", "PolyStepCurve"):
            x_s = pt.attrib.get("x", pt.attrib.get("X"))
            y_s = pt.attrib.get("y", pt.attrib.get("Y"))
            if x_s is not None and y_s is not None:
                try:
                    points.append((
                        _convert_to_mm(float(x_s), unit),
                        _convert_to_mm(float(y_s), unit),
                    ))
                except ValueError:
                    pass
    return points


def _is_copper_layer_name(name: str) -> bool:
    """Return True if the layer name looks like a copper/signal layer."""
    upper = name.upper()
    # Exclude non-copper layers by keyword
    for kw in ("PASTE", "SOLDER", "MASK", "SILK", "OVERLAY", "LEGEND",
               "ASSEMBLY", "DOCUMENT", "DRILL", "MECHANICAL", "MECH",
               "ROUT", "ROUTE", "OUTLINE", "FAB", "COURTYARD"):
        if kw in upper:
            return False
    # Accept common copper layer patterns
    if any(kw in upper for kw in ("LAYER", "CU", "COPPER", "SIGNAL",
                                   "POWER", "GROUND", "PLANE", "TOP",
                                   "BOTTOM", "BOT", "INNER", "MID")):
        return True
    # Accept L<n> patterns (e.g. "L2-CU", "L3")
    stripped = name.strip()
    if stripped and stripped[0].upper() == "L" and len(stripped) > 1 and stripped[1].isdigit():
        return True
    return False


def _parse_copper_pours(
    root: ET.Element, unit: str,
) -> list[CopperPour]:
    """Parse ``<Contour>`` elements inside LayerFeature/Set as copper pours.

    Only includes contours on copper layers, filtering out paste, solder mask,
    silkscreen, mechanical, and other non-copper layers.
    """
    pours: list[CopperPour] = []

    for layer_feat in _find_elements(root, "LayerFeature"):
        layer_name = layer_feat.attrib.get("layerRef", "")

        if not _is_copper_layer_name(layer_name):
            continue

        for set_elem in _find_children(layer_feat, "Set"):
            net_name = set_elem.attrib.get("net", "")

            # Look for Contour elements (may be nested under Features)
            for contour in _find_elements(set_elem, "Contour"):
                polygon = _find_child(contour, "Polygon")
                if polygon is None:
                    continue
                outline = _parse_polygon_points(polygon, unit)
                if len(outline) < 3:
                    continue

                cutouts: list[list[tuple[float, float]]] = []
                for cutout in _find_children(contour, "Cutout"):
                    cut_pts = _parse_polygon_points(cutout, unit)
                    if len(cut_pts) >= 3:
                        cutouts.append(cut_pts)

                pours.append(CopperPour(
                    net=net_name, layer=layer_name,
                    outline=outline, cutouts=cutouts,
                ))

    return pours


# ---- Height / weight heuristics ----

_FOOTPRINT_HEIGHT: dict[str, float] = {
    "0201": 0.25, "0402": 0.35, "0603": 0.45, "0805": 0.55,
    "1206": 0.65, "1210": 0.65,
    "SOT23": 1.1, "SOT223": 1.5,
    "QFP": 1.0, "TQFP": 1.0, "LQFP": 1.0,
    "QFN": 0.8, "DFN": 0.8,
    "BGA": 1.2, "CSP": 0.8,
    "SOP": 1.5, "SOIC": 1.5, "SSOP": 1.3, "TSSOP": 1.1,
    "DIP": 3.5,
}

_PREFIX_HEIGHT: dict[str, float] = {
    "U": 1.5, "R": 1.0, "C": 1.0, "L": 1.0,
    "J": 2.5, "D": 0.8, "Q": 1.0, "SW": 2.0,
}

_CERAMIC_PREFIXES = {"C", "Y", "X"}


def _estimate_height(footprint: str, refdes: str) -> float:
    """Estimate component height from footprint name or refdes prefix."""
    upper = footprint.upper()
    for key, h in _FOOTPRINT_HEIGHT.items():
        if key in upper:
            return h
    # Fall back to refdes prefix
    prefix = ""
    for ch in refdes:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return _PREFIX_HEIGHT.get(prefix, 0.5)


def _estimate_weight(body_w: float, body_h: float, height: float,
                     refdes: str) -> float:
    """Estimate component weight from body volume and material density."""
    if body_w <= 0 or body_h <= 0 or height <= 0:
        return 0.0
    volume_cm3 = (body_w * body_h * height) / 1000.0  # mm³ → cm³
    prefix = ""
    for ch in refdes:
        if ch.isalpha():
            prefix += ch
        else:
            break
    density = 3.0 if prefix in _CERAMIC_PREFIXES else 1.85  # g/cm³
    return volume_cm3 * density


def _compute_fallback_mechanical(design: PCBDesign) -> None:
    """Synthesize PackageDef for components without one."""
    for comp in design.components:
        if comp.package_def is not None:
            # Fill in missing height/weight on parsed packages
            pkg = comp.package_def
            if pkg.height <= 0:
                pkg.height = _estimate_height(comp.footprint, comp.reference)
            if pkg.weight_grams <= 0:
                pkg.weight_grams = _estimate_weight(
                    pkg.body_width, pkg.body_height, pkg.height, comp.reference)
            continue

        # Synthesize from pad extents
        body_w, body_h = 0.0, 0.0
        if comp.pads:
            xs = [p.x for p in comp.pads]
            ys = [p.y for p in comp.pads]
            body_w = (max(xs) - min(xs)) + 0.6  # +0.3mm margin each side
            body_h = (max(ys) - min(ys)) + 0.6

        height = _estimate_height(comp.footprint, comp.reference)
        weight = _estimate_weight(body_w, body_h, height, comp.reference)

        comp.package_def = PackageDef(
            name=comp.footprint or comp.reference,
            pin_count=len(comp.pads),
            body_width=body_w,
            body_height=body_h,
            height=height,
            weight_grams=weight,
            source="computed",
        )
