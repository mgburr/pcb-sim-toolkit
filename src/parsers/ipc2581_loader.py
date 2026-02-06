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

from ..core.models import (
    PCBDesign,
    Component,
    ComponentType,
    Net,
    Trace,
    Pad,
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

    xs: list[float] = []
    ys: list[float] = []

    for elem in profile.iter():
        tag = _strip_namespace(elem.tag)
        if tag in ("PolyBegin", "PolyStepSegment", "PolyStepCurve",
                    "LineBegin", "LineEnd"):
            x_str = elem.attrib.get("x") or elem.attrib.get("X")
            y_str = elem.attrib.get("y") or elem.attrib.get("Y")
            if x_str is not None and y_str is not None:
                xs.append(_convert_to_mm(float(x_str), unit))
                ys.append(_convert_to_mm(float(y_str), unit))

    if xs and ys:
        design.width = max(xs) - min(xs)
        design.height = max(ys) - min(ys)


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

        # Determine component position
        comp_x, comp_y = 0.0, 0.0

        # Try Xform with x/y (Rev C style)
        xform = _find_child(elem, "Xform")
        if xform is not None and "x" in xform.attrib:
            try:
                comp_x = _convert_to_mm(float(xform.attrib.get("x", "0")), unit)
                comp_y = _convert_to_mm(float(xform.attrib.get("y", "0")), unit)
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
