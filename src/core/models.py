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
    width: float = 0.0    # 0 = use diameter
    height: float = 0.0   # 0 = use diameter
    shape: str = ""        # "rect", "circle", "oval", "" = circle fallback
    rotation: float = 0.0  # pad rotation in degrees


@dataclass
class PadShape:
    shape_type: str  # circle, rect, oval, polygon
    width: float  # mm
    height: float  # mm
    drill_diameter: float = 0.0  # 0 = SMD
    plated: bool = True


@dataclass
class CopperPour:
    net: str
    layer: str
    outline: list[tuple[float, float]]  # boundary polygon
    cutouts: list[list[tuple[float, float]]] = field(default_factory=list)


@dataclass
class PackageDef:
    name: str
    pin_count: int = 0
    pins: list[tuple[str, float, float]] = field(default_factory=list)  # (name, x, y)
    pad_shape: PadShape | None = None
    pin_shapes: dict[str, PadShape] = field(default_factory=dict)  # pin_name -> PadShape
    body_width: float = 0.0  # mm
    body_height: float = 0.0  # mm
    courtyard_width: float = 0.0  # mm
    courtyard_height: float = 0.0  # mm
    height: float = 0.0  # mm above board
    pin_pitch: float = 0.0  # mm
    weight_grams: float = 0.0
    source: str = "parsed"  # parsed / computed / heuristic


@dataclass
class Component:
    reference: str
    component_type: ComponentType
    value: str
    footprint: str = ""
    pads: list[Pad] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    rotation: float = 0.0
    layer: str = "Top"
    package_def: PackageDef | None = None

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
            layer
            for layer in self.layers
            if layer.layer_type in (LayerType.SIGNAL, LayerType.POWER, LayerType.GROUND)
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
    outline: list[tuple[float, float]] = field(default_factory=list)
    packages: dict[str, PackageDef] = field(default_factory=dict)
    copper_pours: list[CopperPour] = field(default_factory=list)

    def link_packages(self) -> None:
        """Cross-reference component footprints to package definitions."""
        for comp in self.components:
            if comp.footprint and comp.footprint in self.packages:
                comp.package_def = self.packages[comp.footprint]

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
