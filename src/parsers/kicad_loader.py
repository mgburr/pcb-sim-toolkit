"""Load PCB designs from KiCad .kicad_pcb files.

This parser handles the S-expression format used by KiCad 6+.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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


def load_kicad_pcb(path: Path) -> PCBDesign:
    """Parse a .kicad_pcb file and return a PCBDesign."""
    content = path.read_text()
    tokens = _tokenize(content)
    tree = _parse_sexp(tokens)

    design = PCBDesign(name=path.stem, width=100, height=80)

    for node in tree:
        if not isinstance(node, list):
            continue
        tag = node[0] if node else None

        if tag == "general":
            _parse_general(node, design)
        elif tag == "layers":
            design.stackup = _parse_layers(node)
        elif tag == "net":
            net = _parse_net_node(node)
            if net:
                design.nets.append(net)
        elif tag == "footprint" or tag == "module":
            comp = _parse_footprint(node)
            if comp:
                design.components.append(comp)
        elif tag == "segment":
            trace = _parse_segment(node)
            if trace:
                design.traces.append(trace)

    return design


# ---- S-expression tokenizer / parser ----

def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in " \t\n\r":
            i += 1
        elif c == "(":
            tokens.append("(")
            i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            tokens.append(text[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < len(text) and text[j] not in " \t\n\r()\"":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_sexp(tokens: list[str]) -> list:
    result: list = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "(":
            sub, idx = _parse_list(tokens, idx + 1)
            result.append(sub)
        else:
            idx += 1
    return result


def _parse_list(tokens: list[str], idx: int) -> tuple[list, int]:
    items: list = []
    while idx < len(tokens):
        token = tokens[idx]
        if token == ")":
            return items, idx + 1
        elif token == "(":
            sub, idx = _parse_list(tokens, idx + 1)
            items.append(sub)
        else:
            items.append(token)
            idx += 1
    return items, idx


# ---- Node parsers ----

def _find_value(node: list, key: str) -> str | None:
    for item in node:
        if isinstance(item, list) and item and item[0] == key and len(item) > 1:
            return str(item[1])
    return None


def _parse_general(node: list, design: PCBDesign) -> None:
    for item in node[1:]:
        if isinstance(item, list) and len(item) >= 2:
            if item[0] == "area" and len(item) >= 5:
                x1, y1, x2, y2 = float(item[1]), float(item[2]), float(item[3]), float(item[4])
                design.width = abs(x2 - x1)
                design.height = abs(y2 - y1)


def _parse_layers(node: list) -> Stackup:
    layers: list[Layer] = []
    for item in node[1:]:
        if isinstance(item, list) and len(item) >= 2:
            name = str(item[1]) if len(item) > 1 else str(item[0])
            ltype_str = str(item[2]) if len(item) > 2 else "signal"
            ltype_map = {"signal": LayerType.SIGNAL, "power": LayerType.POWER, "user": LayerType.SIGNAL}
            layers.append(
                Layer(
                    name=name,
                    layer_type=ltype_map.get(ltype_str, LayerType.SIGNAL),
                    thickness=0.035,
                )
            )
    return Stackup(layers=layers)


def _parse_net_node(node: list) -> Net | None:
    if len(node) >= 3:
        return Net(name=str(node[2]), nodes=[])
    return None


def _parse_footprint(node: list) -> Component | None:
    ref = ""
    value = ""
    pads: list[Pad] = []
    footprint = str(node[1]) if len(node) > 1 else ""

    for item in node[1:]:
        if not isinstance(item, list):
            continue
        if item[0] == "fp_text" and len(item) >= 3:
            if item[1] == "reference":
                ref = str(item[2])
            elif item[1] == "value":
                value = str(item[2])
        elif item[0] == "pad":
            pad = _parse_pad(item)
            if pad:
                pads.append(pad)

    if not ref:
        return None

    # Infer component type from reference prefix
    comp_type = ComponentType.IC
    for prefix, ct in [
        ("R", ComponentType.RESISTOR),
        ("C", ComponentType.CAPACITOR),
        ("L", ComponentType.INDUCTOR),
        ("D", ComponentType.DIODE),
        ("Q", ComponentType.TRANSISTOR_NPN),
        ("U", ComponentType.IC),
        ("J", ComponentType.CONNECTOR),
    ]:
        if ref.startswith(prefix) and (len(ref) == 1 or ref[1:2].isdigit()):
            comp_type = ct
            break

    return Component(
        reference=ref,
        component_type=comp_type,
        value=value,
        footprint=footprint,
        pads=pads,
    )


def _parse_pad(node: list) -> Pad | None:
    if len(node) < 2:
        return None
    name = str(node[1])
    x, y = 0.0, 0.0
    for item in node[2:]:
        if isinstance(item, list) and item[0] == "at" and len(item) >= 3:
            x, y = float(item[1]), float(item[2])
    return Pad(name=name, x=x, y=y)


def _parse_segment(node: list) -> Trace | None:
    net = ""
    width = 0.2
    layer = "F.Cu"
    start = (0.0, 0.0)
    end = (0.0, 0.0)

    for item in node[1:]:
        if not isinstance(item, list):
            continue
        if item[0] == "start" and len(item) >= 3:
            start = (float(item[1]), float(item[2]))
        elif item[0] == "end" and len(item) >= 3:
            end = (float(item[1]), float(item[2]))
        elif item[0] == "width" and len(item) >= 2:
            width = float(item[1])
        elif item[0] == "layer" and len(item) >= 2:
            layer = str(item[1])
        elif item[0] == "net" and len(item) >= 2:
            net = str(item[1])

    return Trace(net=net, width=width, layer=layer, points=[start, end])
