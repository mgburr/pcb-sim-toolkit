"""Data models for PCB simulation components."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class LayerType(enum.Enum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    DIELECTRIC = "dielectric"


class ComponentType(enum.Enum):
    RESISTOR = "R"
    CAPACITOR = "C"
    INDUCTOR = "L"
    DIODE = "D"
    TRANSISTOR_NPN = "Q_NPN"
    TRANSISTOR_PNP = "Q_PNP"
    MOSFET_N = "M_N"
    MOSFET_P = "M_P"
    IC = "U"
    CONNECTOR = "J"
    VOLTAGE_SOURCE = "V"
    CURRENT_SOURCE = "I"


@dataclass
class Pad:
    name: str
    x: float
    y: float
    diameter: float = 1.0
    drill: float = 0.3
    layer: str = "F.Cu"


@dataclass
class Component:
    reference: str
    component_type: ComponentType
    value: str
    footprint: str = ""
    pads: list[Pad] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def spice_prefix(self) -> str:
        return self.component_type.value[0]


@dataclass
class Net:
    name: str
    nodes: list[str] = field(default_factory=list)


@dataclass
class Trace:
    net: str
    width: float  # mm
    layer: str
    points: list[tuple[float, float]] = field(default_factory=list)
    length: float = 0.0  # computed

    def compute_length(self) -> float:
        total = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i - 1][0]
            dy = self.points[i][1] - self.points[i - 1][1]
            total += (dx**2 + dy**2) ** 0.5
        self.length = total
        return total


@dataclass
class Layer:
    name: str
    layer_type: LayerType
    thickness: float  # mm
    material: str = "copper"
    dielectric_constant: float = 1.0  # for dielectric layers


@dataclass
class Stackup:
    layers: list[Layer] = field(default_factory=list)

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    def copper_layers(self) -> list[Layer]:
        return [
            l
            for l in self.layers
            if l.layer_type in (LayerType.SIGNAL, LayerType.POWER, LayerType.GROUND)
        ]


@dataclass
class PCBDesign:
    name: str
    width: float  # mm
    height: float  # mm
    stackup: Stackup = field(default_factory=Stackup)
    components: list[Component] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)

    def get_component(self, reference: str) -> Component | None:
        for comp in self.components:
            if comp.reference == reference:
                return comp
        return None

    def get_net(self, name: str) -> Net | None:
        for net in self.nets:
            if net.name == name:
                return net
        return None
