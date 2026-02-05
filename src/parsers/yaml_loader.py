"""Load PCB designs from YAML definition files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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


_COMPONENT_TYPE_MAP = {
    "R": ComponentType.RESISTOR,
    "C": ComponentType.CAPACITOR,
    "L": ComponentType.INDUCTOR,
    "D": ComponentType.DIODE,
    "Q_NPN": ComponentType.TRANSISTOR_NPN,
    "Q_PNP": ComponentType.TRANSISTOR_PNP,
    "M_N": ComponentType.MOSFET_N,
    "M_P": ComponentType.MOSFET_P,
    "U": ComponentType.IC,
    "J": ComponentType.CONNECTOR,
    "V": ComponentType.VOLTAGE_SOURCE,
    "I": ComponentType.CURRENT_SOURCE,
}

_LAYER_TYPE_MAP = {
    "signal": LayerType.SIGNAL,
    "power": LayerType.POWER,
    "ground": LayerType.GROUND,
    "dielectric": LayerType.DIELECTRIC,
}


def load_design(path: Path) -> PCBDesign:
    """Load a PCBDesign from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    design = PCBDesign(
        name=data.get("name", path.stem),
        width=float(data.get("width", 100)),
        height=float(data.get("height", 80)),
    )

    # Stackup
    if "stackup" in data:
        design.stackup = _parse_stackup(data["stackup"])

    # Components
    for cdata in data.get("components", []):
        design.components.append(_parse_component(cdata))

    # Nets
    for ndata in data.get("nets", []):
        design.nets.append(_parse_net(ndata))

    # Traces
    for tdata in data.get("traces", []):
        design.traces.append(_parse_trace(tdata))

    return design


def _parse_stackup(layers_data: list[dict]) -> Stackup:
    layers: list[Layer] = []
    for ld in layers_data:
        layers.append(
            Layer(
                name=ld["name"],
                layer_type=_LAYER_TYPE_MAP.get(
                    ld.get("type", "signal"), LayerType.SIGNAL
                ),
                thickness=float(ld.get("thickness", 0.035)),
                material=ld.get("material", "copper"),
                dielectric_constant=float(ld.get("er", 1.0)),
            )
        )
    return Stackup(layers=layers)


def _parse_component(cdata: dict) -> Component:
    comp_type = _COMPONENT_TYPE_MAP.get(
        cdata.get("type", "R"), ComponentType.RESISTOR
    )
    pads = []
    for pd in cdata.get("pads", []):
        pads.append(
            Pad(
                name=str(pd["name"]),
                x=float(pd.get("x", 0)),
                y=float(pd.get("y", 0)),
                diameter=float(pd.get("diameter", 1.0)),
                drill=float(pd.get("drill", 0.3)),
                layer=pd.get("layer", "F.Cu"),
            )
        )
    return Component(
        reference=cdata["ref"],
        component_type=comp_type,
        value=str(cdata.get("value", "0")),
        footprint=cdata.get("footprint", ""),
        pads=pads,
        properties=cdata.get("properties", {}),
    )


def _parse_net(ndata: dict) -> Net:
    return Net(
        name=ndata["name"],
        nodes=ndata.get("nodes", []),
    )


def _parse_trace(tdata: dict) -> Trace:
    points = [tuple(p) for p in tdata.get("points", [])]
    return Trace(
        net=tdata["net"],
        width=float(tdata.get("width", 0.2)),
        layer=tdata.get("layer", "F.Cu"),
        points=points,
    )
