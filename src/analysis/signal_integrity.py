"""Signal integrity analysis for PCB traces.

Computes characteristic impedance, propagation delay, crosstalk estimates,
and eye-diagram parameters for high-speed signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.models import PCBDesign, Trace, Stackup, Layer, LayerType
from ..core.config import SimulationConfig, SimulationResult, SimulationType


@dataclass
class TraceImpedance:
    reference: str
    net: str
    z0: float  # characteristic impedance (ohms)
    delay_ps_per_mm: float  # propagation delay
    loss_db_per_mm: float  # conductor loss estimate at 1 GHz
    length_mm: float


class SignalIntegrityAnalyzer:
    """Perform signal-integrity analysis on PCB traces."""

    # Copper resistivity (ohm-m)
    COPPER_RESISTIVITY = 1.68e-8
    # Speed of light (m/s)
    C0 = 3e8

    def run(
        self, design: PCBDesign, config: SimulationConfig
    ) -> SimulationResult:
        results: list[dict[str, Any]] = []
        warnings: list[str] = []

        if not design.stackup.layers:
            warnings.append(
                "No stackup defined; using default 4-layer 1.6mm FR4 stackup"
            )
            design.stackup = self._default_stackup()

        for trace in design.traces:
            trace.compute_length()
            imp = self._compute_impedance(trace, design.stackup)
            results.append(
                {
                    "net": imp.net,
                    "z0_ohms": round(imp.z0, 2),
                    "delay_ps_per_mm": round(imp.delay_ps_per_mm, 4),
                    "loss_db_per_mm_1ghz": round(imp.loss_db_per_mm, 6),
                    "length_mm": round(imp.length_mm, 3),
                    "total_delay_ps": round(
                        imp.delay_ps_per_mm * imp.length_mm, 2
                    ),
                }
            )

        crosstalk = self._estimate_crosstalk(design.traces, design.stackup)
        eye = self._estimate_eye_opening(results, config)

        data = {
            "traces": results,
            "crosstalk": crosstalk,
            "eye_diagram": eye,
            "analysis": "signal_integrity",
        }
        return SimulationResult(
            success=True,
            sim_type=SimulationType.SIGNAL_INTEGRITY,
            data=data,
            warnings=warnings,
        )

    def _compute_impedance(self, trace: Trace, stackup: Stackup) -> TraceImpedance:
        """Microstrip impedance approximation (IPC-2141 / Wadell)."""
        w = trace.width  # mm
        layer = self._find_dielectric_below(trace.layer, stackup)
        h = layer.thickness if layer else 0.2  # mm
        er = layer.dielectric_constant if layer else 4.4
        t = 0.035  # copper thickness mm (1 oz)

        # Effective width accounting for thickness
        w_eff = w + (t / math.pi) * (1 + math.log(2 * h / t)) if t > 0 else w

        # Microstrip Z0 formula (Hammerstad-Jensen)
        u = w_eff / h
        er_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 / u) ** (-0.5)

        if u <= 1:
            z0 = (60 / math.sqrt(er_eff)) * math.log(8 / u + u / 4)
        else:
            z0 = (120 * math.pi) / (
                math.sqrt(er_eff) * (u + 1.393 + 0.667 * math.log(u + 1.444))
            )

        # Propagation delay
        delay = math.sqrt(er_eff) / self.C0 * 1e9  # ns/mm -> ps/mm * 1000
        delay_ps_per_mm = math.sqrt(er_eff) / (self.C0 * 1e-3) * 1e12 / 1e6

        # Conductor loss at 1 GHz (approximate)
        rs = math.sqrt(
            math.pi * 1e9 * 4 * math.pi * 1e-7 * self.COPPER_RESISTIVITY
        )
        alpha_c = rs / (z0 * w_eff * 1e-3)  # Np/m
        loss_db_per_m = alpha_c * 8.686
        loss_db_per_mm = loss_db_per_m / 1000

        return TraceImpedance(
            reference=trace.net,
            net=trace.net,
            z0=z0,
            delay_ps_per_mm=delay_ps_per_mm,
            loss_db_per_mm=loss_db_per_mm,
            length_mm=trace.length,
        )

    def _find_dielectric_below(
        self, layer_name: str, stackup: Stackup
    ) -> Layer | None:
        """Find the dielectric layer below the given signal layer."""
        found = False
        for layer in stackup.layers:
            if found and layer.layer_type == LayerType.DIELECTRIC:
                return layer
            if layer.name == layer_name:
                found = True
        # Fallback: return first dielectric
        for layer in stackup.layers:
            if layer.layer_type == LayerType.DIELECTRIC:
                return layer
        return None

    def _estimate_crosstalk(
        self, traces: list[Trace], stackup: Stackup
    ) -> list[dict[str, Any]]:
        """Estimate near-end crosstalk (NEXT) between adjacent trace pairs."""
        crosstalk_results: list[dict[str, Any]] = []
        for i, t1 in enumerate(traces):
            for t2 in traces[i + 1 :]:
                if t1.layer != t2.layer:
                    continue
                spacing = self._min_spacing(t1, t2)
                if spacing is None or spacing > 5.0:  # skip if > 5mm apart
                    continue

                layer = self._find_dielectric_below(t1.layer, stackup)
                h = layer.thickness if layer else 0.2

                # Empirical NEXT approximation
                # NEXT ~ 1 / (1 + (spacing/h)^2)
                next_coeff = 1.0 / (1.0 + (spacing / h) ** 2)
                crosstalk_results.append(
                    {
                        "aggressor": t1.net,
                        "victim": t2.net,
                        "spacing_mm": round(spacing, 3),
                        "next_coefficient": round(next_coeff, 4),
                        "next_db": round(20 * math.log10(next_coeff + 1e-12), 2),
                    }
                )
        return crosstalk_results

    def _estimate_eye_opening(
        self, trace_results: list[dict], config: SimulationConfig
    ) -> dict[str, Any]:
        """Rough eye-diagram estimation for digital signals."""
        max_freq = config.frequency_range[1]
        bit_period_ps = 1e12 / (2 * max_freq)  # ps

        eye_results: list[dict[str, Any]] = []
        for tr in trace_results:
            total_delay = tr["total_delay_ps"]
            total_loss_db = tr["loss_db_per_mm_1ghz"] * tr["length_mm"]

            # Eye height reduction from loss
            eye_height = 10 ** (-total_loss_db / 20)  # normalized to 1.0
            # Eye width reduction from delay
            eye_width_ps = max(0, bit_period_ps - total_delay)

            eye_results.append(
                {
                    "net": tr["net"],
                    "eye_height_normalized": round(eye_height, 4),
                    "eye_width_ps": round(eye_width_ps, 2),
                    "bit_period_ps": round(bit_period_ps, 2),
                    "margin_pct": round(eye_width_ps / bit_period_ps * 100, 1)
                    if bit_period_ps > 0
                    else 0,
                }
            )
        return {"bit_rate_gbps": 2 * max_freq / 1e9, "traces": eye_results}

    @staticmethod
    def _min_spacing(t1: Trace, t2: Trace) -> float | None:
        """Find minimum distance between two traces."""
        if not t1.points or not t2.points:
            return None
        min_d = float("inf")
        for p1 in t1.points:
            for p2 in t2.points:
                d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d < min_d:
                    min_d = d
        return min_d if min_d < float("inf") else None

    @staticmethod
    def _default_stackup() -> Stackup:
        """Standard 4-layer FR4 stackup."""
        from ..core.models import Stackup, Layer, LayerType

        return Stackup(
            layers=[
                Layer("F.Cu", LayerType.SIGNAL, 0.035, "copper"),
                Layer(
                    "Prepreg1",
                    LayerType.DIELECTRIC,
                    0.2,
                    "FR4",
                    dielectric_constant=4.4,
                ),
                Layer("In1.Cu", LayerType.GROUND, 0.035, "copper"),
                Layer(
                    "Core",
                    LayerType.DIELECTRIC,
                    1.0,
                    "FR4",
                    dielectric_constant=4.4,
                ),
                Layer("In2.Cu", LayerType.POWER, 0.035, "copper"),
                Layer(
                    "Prepreg2",
                    LayerType.DIELECTRIC,
                    0.2,
                    "FR4",
                    dielectric_constant=4.4,
                ),
                Layer("B.Cu", LayerType.SIGNAL, 0.035, "copper"),
            ]
        )
