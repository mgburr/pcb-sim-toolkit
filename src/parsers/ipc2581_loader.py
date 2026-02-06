"""Load PCB designs from IPC-2581 (.cvg) XML files.

This parser handles the IPC-2581 XML format, extracting board outline,
stackup, components, nets, and traces into a PCBDesign object.
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
    """Parse an IPC-2581 .cvg file and return a PCBDesign."""
    tree = ET.parse(path)
    root = tree.getroot()

    design = PCBDesign(name=path.stem, width=100, height=80)

    # Determine the default unit from the root or Content element
    unit = _get_default_unit(root)

    # Parse board outline for dimensions
    _parse_board_outline(root, design, unit)

    # Parse stackup
    design.stackup = _parse_stackup(root, unit)

    # Parse components
    design.components = _parse_components(root, unit)

    # Parse nets
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
    """Extract the default unit from the IPC-2581 document."""
    # Check Content element for units attribute
    for elem in root.iter():
        tag = _strip_namespace(elem.tag)
        if tag == "Content":
            for attr in ("units", "unit", "Units", "Unit"):
                if attr in elem.attrib:
                    return elem.attrib[attr]
    # Check root element
    for attr in ("units", "unit", "Units", "Unit"):
        if attr in root.attrib:
            return root.attrib[attr]
    return "MM"


def _find_elements(root: ET.Element, local_name: str) -> list[ET.Element]:
    """Find all elements matching a local tag name, ignoring namespaces."""
    results = []
    for elem in root.iter():
        if _strip_namespace(elem.tag) == local_name:
            results.append(elem)
    return results


def _find_element(root: ET.Element, local_name: str) -> ET.Element | None:
    """Find the first element matching a local tag name, ignoring namespaces."""
    for elem in root.iter():
        if _strip_namespace(elem.tag) == local_name:
            return elem
    return None


# ---- Section parsers ----

def _parse_board_outline(root: ET.Element, design: PCBDesign, unit: str) -> None:
    """Extract board dimensions from Profile/Outline/Polygon."""
    # Look for Profile element containing board outline
    profile = _find_element(root, "Profile")
    if profile is None:
        return

    # Collect all polygon points
    xs: list[float] = []
    ys: list[float] = []

    for elem in profile.iter():
        tag = _strip_namespace(elem.tag)
        if tag in ("PolyBegin", "PolyStepSegment", "LineBegin", "LineEnd"):
            x_str = elem.attrib.get("x") or elem.attrib.get("X")
            y_str = elem.attrib.get("y") or elem.attrib.get("Y")
            if x_str is not None and y_str is not None:
                xs.append(_convert_to_mm(float(x_str), unit))
                ys.append(_convert_to_mm(float(y_str), unit))

    if xs and ys:
        design.width = max(xs) - min(xs)
        design.height = max(ys) - min(ys)


def _parse_stackup(root: ET.Element, unit: str) -> Stackup:
    """Parse stackup layers from StackupLayer elements."""
    layers: list[Layer] = []

    for elem in _find_elements(root, "StackupLayer"):
        name = elem.attrib.get("layerOrGroupRef", elem.attrib.get("name", ""))
        thickness_str = elem.attrib.get("thickness", "0.035")
        thickness = _convert_to_mm(float(thickness_str), unit)
        if thickness <= 0:
            thickness = 0.035

        # Determine layer type from attributes
        layer_type_str = elem.attrib.get("layerFunctionType", "").upper()
        material = elem.attrib.get("material", "copper")
        dielectric_constant = 1.0

        if "DIELECTRIC" in layer_type_str or "CORE" in layer_type_str or "PREPREG" in layer_type_str:
            layer_type = LayerType.DIELECTRIC
            material = material or "FR-4"
            try:
                dielectric_constant = float(elem.attrib.get("dielectricConstant", "4.5"))
            except ValueError:
                dielectric_constant = 4.5
        elif "POWER" in layer_type_str or "PLANE" in layer_type_str:
            layer_type = LayerType.POWER
        else:
            layer_type = LayerType.SIGNAL

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


def _parse_components(root: ET.Element, unit: str) -> list[Component]:
    """Parse Component elements into Component objects."""
    components: list[Component] = []

    for elem in _find_elements(root, "Component"):
        refdes = elem.attrib.get("refDes", elem.attrib.get("name", ""))
        if not refdes:
            continue

        value = elem.attrib.get("value", elem.attrib.get("partNumber", ""))
        footprint = elem.attrib.get("packageRef", elem.attrib.get("standardPackageRef", ""))
        comp_type = _infer_component_type(refdes)

        # Get component position for pin offset computation
        comp_x, comp_y = 0.0, 0.0
        loc = _find_child(elem, "Location")
        if loc is not None:
            try:
                comp_x = _convert_to_mm(float(loc.attrib.get("x", "0")), unit)
                comp_y = _convert_to_mm(float(loc.attrib.get("y", "0")), unit)
            except ValueError:
                pass

        # Parse pads from Pin children
        pads: list[Pad] = []
        for pin in elem:
            if _strip_namespace(pin.tag) == "Pin":
                pin_name = pin.attrib.get("number", pin.attrib.get("name", ""))
                pin_x, pin_y = comp_x, comp_y
                pin_loc = _find_child(pin, "Location")
                if pin_loc is not None:
                    try:
                        pin_x = comp_x + _convert_to_mm(float(pin_loc.attrib.get("x", "0")), unit)
                        pin_y = comp_y + _convert_to_mm(float(pin_loc.attrib.get("y", "0")), unit)
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


def _find_child(parent: ET.Element, local_name: str) -> ET.Element | None:
    """Find a direct child element by local name, ignoring namespaces."""
    for child in parent:
        if _strip_namespace(child.tag) == local_name:
            return child
    return None


def _parse_nets(root: ET.Element) -> list[Net]:
    """Parse PhyNet elements into Net objects."""
    nets: list[Net] = []

    for elem in _find_elements(root, "PhyNet"):
        name = elem.attrib.get("name", "")
        if not name:
            continue

        nodes: list[str] = []
        for pin_ref in elem:
            if _strip_namespace(pin_ref.tag) == "PinRef":
                comp_ref = pin_ref.attrib.get("componentRef", "")
                pin = pin_ref.attrib.get("pin", "")
                if comp_ref and pin:
                    nodes.append(f"{comp_ref}.{pin}")

        nets.append(Net(name=name, nodes=nodes))

    return nets


def _parse_traces(root: ET.Element, unit: str) -> list[Trace]:
    """Parse trace geometry from LayerFeature/Set/Line elements."""
    traces: list[Trace] = []

    for lf_elem in _find_elements(root, "LayerFeature"):
        layer_name = lf_elem.attrib.get("layerRef", "")

        for set_elem in lf_elem:
            if _strip_namespace(set_elem.tag) != "Set":
                continue
            net_name = set_elem.attrib.get("net", "")

            for child in set_elem:
                tag = _strip_namespace(child.tag)
                if tag == "Line":
                    try:
                        sx = _convert_to_mm(float(child.attrib.get("startX", child.attrib.get("x1", "0"))), unit)
                        sy = _convert_to_mm(float(child.attrib.get("startY", child.attrib.get("y1", "0"))), unit)
                        ex = _convert_to_mm(float(child.attrib.get("endX", child.attrib.get("x2", "0"))), unit)
                        ey = _convert_to_mm(float(child.attrib.get("endY", child.attrib.get("y2", "0"))), unit)
                        width_str = child.attrib.get("lineWidth", child.attrib.get("width", "0.2"))
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
