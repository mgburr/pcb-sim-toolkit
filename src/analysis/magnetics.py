"""Magnetic field analysis for PCB traces.

Computes the magnetic field (B-field) distribution around current-carrying
traces using the Biot-Savart law for finite-length conductors.

For a straight conductor segment carrying current I, the magnetic field at
a perpendicular distance r is:
    B = (mu_0 * I) / (2 * pi * r) * [geometric factor for finite length]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.models import PCBDesign, Trace, ComponentType
from ..core.config import SimulationConfig, SimulationResult, SimulationType


# Permeability of free space (T*m/A)
MU_0 = 4 * math.pi * 1e-7


@dataclass
class TraceCurrentEstimate:
    net: str
    current_a: float
    frequency_hz: float


class MagneticsAnalyzer:
    """Compute magnetic field maps around PCB traces using Biot-Savart."""

    def run(
        self,
        design: PCBDesign,
        config: SimulationConfig,
        resolution_mm: float = 0.25,
    ) -> SimulationResult:
        warnings: list[str] = []
        data: dict[str, Any] = {}

        # Estimate currents in each trace
        currents = self._estimate_currents(design)
        data["trace_currents"] = [
            {"net": tc.net, "current_ma": round(tc.current_a * 1000, 3),
             "frequency_hz": tc.frequency_hz}
            for tc in currents
        ]

        # Build current map: net_name -> TraceCurrentEstimate
        current_map = {tc.net: tc for tc in currents}

        # Compute 2D B-field magnitude on a grid at board surface
        bx, by, bmag, grid_x, grid_y = self._compute_bfield_grid(
            design, current_map, resolution_mm
        )
        data["bfield_grid"] = {
            "resolution_mm": resolution_mm,
            "width_cells": bmag.shape[1],
            "height_cells": bmag.shape[0],
            "bx_tesla": bx.tolist(),
            "by_tesla": by.tolist(),
            "bmag_tesla": bmag.tolist(),
            "min_b_tesla": float(np.min(bmag)),
            "max_b_tesla": float(np.max(bmag)),
            "grid_x_mm": grid_x.tolist(),
            "grid_y_mm": grid_y.tolist(),
        }

        # Compute per-trace field profiles (cross-section)
        profiles = self._compute_trace_profiles(design, current_map)
        data["trace_profiles"] = profiles
        data["analysis"] = "magnetics"

        return SimulationResult(
            success=True,
            sim_type=SimulationType.SIGNAL_INTEGRITY,  # grouped under SI
            data=data,
            warnings=warnings,
        )

    def _estimate_currents(self, design: PCBDesign) -> list[TraceCurrentEstimate]:
        """Estimate current flowing through each trace from the circuit."""
        currents: list[TraceCurrentEstimate] = []
        seen_nets: set[str] = set()

        for trace in design.traces:
            if trace.net in seen_nets:
                continue
            seen_nets.add(trace.net)

            current = 0.0
            freq = 0.0

            # Find components on this net and estimate current
            for comp in design.components:
                nodes = []
                for pad in comp.pads:
                    pad_ref = f"{comp.reference}.{pad.name}"
                    for net in design.nets:
                        if net.name == trace.net and pad_ref in net.nodes:
                            nodes.append(pad.name)

                if not nodes:
                    continue

                if comp.component_type == ComponentType.VOLTAGE_SOURCE:
                    # Estimate current from voltage / total resistance in path
                    try:
                        val = comp.value.strip().upper()
                        if "SIN" in val:
                            import re
                            m = re.search(
                                r"SIN\(\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)",
                                val,
                            )
                            if m:
                                amp = float(m.group(2))
                                freq = float(m.group(3))
                                # Estimate impedance from nearby R
                                z = self._estimate_path_impedance(design, trace.net)
                                current = max(current, amp / z if z > 0 else amp)
                        else:
                            v = float(val.replace("DC", "").strip())
                            z = self._estimate_path_impedance(design, trace.net)
                            current = max(current, v / z if z > 0 else 0)
                    except (ValueError, AttributeError):
                        pass

                elif comp.component_type == ComponentType.RESISTOR:
                    try:
                        r = self._parse_value(comp.value)
                        v = float(comp.properties.get("voltage", 3.3))
                        current = max(current, v / r if r > 0 else 0)
                    except (ValueError, ZeroDivisionError):
                        pass

            if current == 0:
                current = 0.01  # 10mA default for unknown traces

            currents.append(TraceCurrentEstimate(
                net=trace.net, current_a=current, frequency_hz=freq
            ))

        return currents

    def _estimate_path_impedance(self, design: PCBDesign, net_name: str) -> float:
        """Estimate total impedance in a net path from resistors."""
        total_r = 0.0
        for comp in design.components:
            if comp.component_type != ComponentType.RESISTOR:
                continue
            for pad in comp.pads:
                pad_ref = f"{comp.reference}.{pad.name}"
                for net in design.nets:
                    if net.name == net_name and pad_ref in net.nodes:
                        try:
                            total_r += self._parse_value(comp.value)
                        except ValueError:
                            pass
        return total_r if total_r > 0 else 50.0  # default 50 ohm

    def _compute_bfield_grid(
        self,
        design: PCBDesign,
        current_map: dict[str, TraceCurrentEstimate],
        resolution_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D magnetic field on a grid across the board."""
        nx = max(int(design.width / resolution_mm), 20)
        ny = max(int(design.height / resolution_mm), 20)

        grid_x = np.linspace(0, design.width, nx)
        grid_y = np.linspace(0, design.height, ny)
        X, Y = np.meshgrid(grid_x, grid_y)

        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)

        # Height above the trace plane where we observe the field (mm)
        obs_height = 0.5

        for trace in design.traces:
            tc = current_map.get(trace.net)
            if not tc or len(trace.points) < 2:
                continue

            I = tc.current_a

            # For each segment of the trace, apply Biot-Savart
            for k in range(len(trace.points) - 1):
                x1, y1 = trace.points[k]
                x2, y2 = trace.points[k + 1]

                bx_seg, by_seg = self._biot_savart_segment(
                    x1, y1, x2, y2, I, X, Y, obs_height
                )
                Bx += bx_seg
                By += by_seg

        Bmag = np.sqrt(Bx**2 + By**2)
        return Bx, By, Bmag, grid_x, grid_y

    @staticmethod
    def _biot_savart_segment(
        x1: float, y1: float, x2: float, y2: float,
        I: float,
        X: np.ndarray, Y: np.ndarray,
        z_obs: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Magnetic field from a finite straight conductor segment at z=0
        observed at height z_obs, using the Biot-Savart integral.

        The segment runs from (x1,y1,0) to (x2,y2,0).
        Observation points are at (X, Y, z_obs).

        Returns (Bx, By) in Tesla. Coordinates in mm, converted to meters.
        """
        # Convert to meters
        scale = 1e-3
        x1m, y1m = x1 * scale, y1 * scale
        x2m, y2m = x2 * scale, y2 * scale
        z_m = z_obs * scale

        Xm = X * scale
        Ym = Y * scale

        # Direction vector of the segment
        dlx = x2m - x1m
        dly = y2m - y1m
        seg_len = math.sqrt(dlx**2 + dly**2)
        if seg_len < 1e-12:
            return np.zeros_like(X), np.zeros_like(X)

        # Unit vector along segment
        ux = dlx / seg_len
        uy = dly / seg_len

        # Vector from start of segment to each observation point
        rx = Xm - x1m
        ry = Ym - y1m

        # Component along the segment direction
        dot = rx * ux + ry * uy

        # Perpendicular distance from the line (in the z=0 plane)
        perp_x = rx - dot * ux
        perp_y = ry - dot * uy

        # Full 3D perpendicular distance including z offset
        d_perp = np.sqrt(perp_x**2 + perp_y**2 + z_m**2)
        d_perp = np.maximum(d_perp, 1e-10)  # avoid division by zero

        # Angles from the two ends of the segment
        # cos(theta) = dot / sqrt(dot^2 + d_perp^2)
        r1 = np.sqrt(dot**2 + d_perp**2)
        r2 = np.sqrt((dot - seg_len)**2 + d_perp**2)

        cos_theta1 = dot / np.maximum(r1, 1e-10)
        cos_theta2 = (dot - seg_len) / np.maximum(r2, 1e-10)

        # B magnitude from finite wire: B = (mu0*I)/(4*pi*d) * (cos1 - cos2)
        B_scalar = (MU_0 * I) / (4 * math.pi * d_perp) * (cos_theta1 - cos_theta2)

        # Direction: B is perpendicular to both the wire direction and the
        # displacement vector. In the observation plane:
        # B direction ~ dl x r_hat (cross product gives the curl direction)
        # For a wire along (ux, uy, 0), the B-field curls around it.
        # At observation point, the perpendicular component in the xy-plane:
        Bx = B_scalar * (-uy)  # perpendicular to wire in xy
        By = B_scalar * (ux)

        return Bx, By

    def _compute_trace_profiles(
        self,
        design: PCBDesign,
        current_map: dict[str, TraceCurrentEstimate],
    ) -> list[dict[str, Any]]:
        """Compute B-field cross-section profile perpendicular to each trace."""
        profiles = []
        for trace in design.traces:
            tc = current_map.get(trace.net)
            if not tc or len(trace.points) < 2:
                continue

            # Take the midpoint of the trace, compute field perpendicular
            mid_idx = len(trace.points) // 2
            x1, y1 = trace.points[max(0, mid_idx - 1)]
            x2, y2 = trace.points[min(len(trace.points) - 1, mid_idx)]

            # Perpendicular direction
            dx = x2 - x1
            dy = y2 - y1
            seg_len = math.sqrt(dx**2 + dy**2)
            if seg_len < 1e-6:
                continue
            # Normal to the trace
            nx = -dy / seg_len
            ny = dx / seg_len

            # Sample points along the perpendicular, +/- 5mm
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            offsets = np.linspace(-5, 5, 200)
            sample_x = mid_x + offsets * nx
            sample_y = mid_y + offsets * ny

            X = sample_x.reshape(1, -1)
            Y = sample_y.reshape(1, -1)

            bx_total = np.zeros_like(X)
            by_total = np.zeros_like(Y)

            for k in range(len(trace.points) - 1):
                px1, py1 = trace.points[k]
                px2, py2 = trace.points[k + 1]
                bx_seg, by_seg = self._biot_savart_segment(
                    px1, py1, px2, py2, tc.current_a, X, Y, 0.5
                )
                bx_total += bx_seg
                by_total += by_seg

            bmag = np.sqrt(bx_total**2 + by_total**2).flatten()

            profiles.append({
                "net": trace.net,
                "current_ma": round(tc.current_a * 1000, 3),
                "offset_mm": offsets.tolist(),
                "b_magnitude_tesla": bmag.tolist(),
                "peak_b_tesla": float(np.max(bmag)),
                "peak_b_microtesla": round(float(np.max(bmag)) * 1e6, 3),
            })

        return profiles

    @staticmethod
    def _parse_value(val_str: str) -> float:
        val_str = val_str.strip().upper()
        multipliers = [
            ("MEG", 1e6), ("T", 1e12), ("G", 1e9), ("K", 1e3),
            ("M", 1e-3), ("U", 1e-6), ("N", 1e-9), ("P", 1e-12), ("F", 1e-15),
        ]
        for suffix, mult in multipliers:
            if val_str.endswith(suffix):
                return float(val_str[: -len(suffix)]) * mult
        return float(val_str)
