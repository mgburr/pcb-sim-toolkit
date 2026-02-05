"""Thermal analysis for PCB designs.

Estimates steady-state temperature distribution using a simplified
finite-difference thermal model for the copper/FR4 stackup.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.models import PCBDesign, Component, ComponentType, LayerType
from ..core.config import SimulationConfig, SimulationResult, SimulationType


class ThermalAnalyzer:
    """Estimate thermal behavior of the PCB under operating conditions."""

    # Thermal conductivity W/(m*K)
    K_COPPER = 385.0
    K_FR4 = 0.3
    # Convective heat transfer coefficient W/(m^2*K) - natural convection
    H_CONV = 10.0

    def run(
        self, design: PCBDesign, config: SimulationConfig
    ) -> SimulationResult:
        warnings: list[str] = []
        data: dict[str, Any] = {}

        # Estimate power dissipation per component
        power_map = self._estimate_power(design)
        data["component_power_mw"] = {
            ref: round(p * 1000, 2) for ref, p in power_map.items()
        }

        total_power = sum(power_map.values())
        data["total_power_w"] = round(total_power, 4)

        # Board-level thermal resistance
        r_th_board = self._board_thermal_resistance(design)
        data["board_thermal_resistance_c_per_w"] = round(r_th_board, 2)

        # Steady-state temperature rise (lumped)
        delta_t = total_power * r_th_board
        data["avg_temp_rise_c"] = round(delta_t, 2)
        data["avg_board_temp_c"] = round(config.ambient_temp + delta_t, 2)

        # Per-component temperature estimation
        comp_temps = self._per_component_temps(
            design, power_map, config.ambient_temp, r_th_board
        )
        data["component_temperatures_c"] = comp_temps

        # Thermal grid (simplified 2D)
        grid = self._thermal_grid(design, power_map, config.ambient_temp)
        data["thermal_grid"] = {
            "resolution_mm": 1.0,
            "width_cells": grid.shape[1],
            "height_cells": grid.shape[0],
            "min_temp_c": round(float(np.min(grid)), 2),
            "max_temp_c": round(float(np.max(grid)), 2),
            "grid": grid.tolist(),
        }

        # Warnings for hot spots
        max_temp = float(np.max(grid))
        if max_temp > 100:
            warnings.append(
                f"Hot spot detected: {max_temp:.1f}C exceeds 100C threshold"
            )
        if max_temp > 125:
            warnings.append(
                f"CRITICAL: {max_temp:.1f}C exceeds FR4 Tg of ~130C"
            )

        data["analysis"] = "thermal"
        return SimulationResult(
            success=True,
            sim_type=SimulationType.THERMAL,
            data=data,
            warnings=warnings,
        )

    def _estimate_power(self, design: PCBDesign) -> dict[str, float]:
        """Estimate power dissipation (watts) for each component."""
        power: dict[str, float] = {}

        for comp in design.components:
            p = comp.properties.get("power_w")
            if p is not None:
                power[comp.reference] = float(p)
                continue

            if comp.component_type == ComponentType.RESISTOR:
                # P = V^2 / R -- estimate from nearby voltage source
                try:
                    r = self._parse_value(comp.value)
                    v = comp.properties.get("voltage", 3.3)
                    power[comp.reference] = (float(v) ** 2) / r
                except (ValueError, ZeroDivisionError):
                    power[comp.reference] = 0.0
            elif comp.component_type == ComponentType.IC:
                # Default IC power from property or 0.5W typical
                power[comp.reference] = float(
                    comp.properties.get("typical_power_w", 0.5)
                )
            else:
                power[comp.reference] = 0.0

        return power

    def _board_thermal_resistance(self, design: PCBDesign) -> float:
        """Compute board-level thermal resistance (C/W) using stackup."""
        area = (design.width * 1e-3) * (design.height * 1e-3)  # m^2
        if area <= 0:
            return 100.0  # fallback

        # Conduction through board + convection from top/bottom
        # R_cond = thickness / (k * A), R_conv = 1 / (h * A)
        total_thickness = 0.0
        effective_k = 0.0

        if design.stackup.layers:
            for layer in design.stackup.layers:
                t = layer.thickness * 1e-3  # to meters
                total_thickness += t
                k = (
                    self.K_COPPER
                    if layer.layer_type != LayerType.DIELECTRIC
                    else self.K_FR4
                )
                effective_k += t / k
        else:
            total_thickness = 1.6e-3
            effective_k = total_thickness / self.K_FR4

        r_cond = effective_k / area if area > 0 else 100
        r_conv = 1.0 / (self.H_CONV * area * 2)  # both sides

        return r_cond + r_conv

    def _per_component_temps(
        self,
        design: PCBDesign,
        power_map: dict[str, float],
        t_ambient: float,
        r_th_board: float,
    ) -> dict[str, float]:
        """Estimate junction/case temperature for each component."""
        temps: dict[str, float] = {}
        for comp in design.components:
            p = power_map.get(comp.reference, 0)
            # Use component-specific theta_ja if available
            theta_ja = float(comp.properties.get("theta_ja", r_th_board * 0.5))
            temps[comp.reference] = round(t_ambient + p * theta_ja, 2)
        return temps

    def _thermal_grid(
        self,
        design: PCBDesign,
        power_map: dict[str, float],
        t_ambient: float,
        resolution_mm: float = 1.0,
        iterations: int = 500,
    ) -> np.ndarray:
        """2D finite-difference steady-state thermal solver."""
        nx = max(int(design.width / resolution_mm), 10)
        ny = max(int(design.height / resolution_mm), 10)

        # Initialize grid at ambient
        T = np.full((ny, nx), t_ambient, dtype=np.float64)

        # Build heat source matrix (W/m^2)
        Q = np.zeros((ny, nx), dtype=np.float64)
        cell_area = (resolution_mm * 1e-3) ** 2

        for comp in design.components:
            p = power_map.get(comp.reference, 0)
            if p <= 0 or not comp.pads:
                continue
            # Place heat at component center
            cx = sum(pad.x for pad in comp.pads) / len(comp.pads)
            cy = sum(pad.y for pad in comp.pads) / len(comp.pads)
            ix = int(cx / resolution_mm)
            iy = int(cy / resolution_mm)
            ix = max(0, min(ix, nx - 1))
            iy = max(0, min(iy, ny - 1))
            # Spread over a small area
            spread = max(1, int(2 / resolution_mm))
            for dy in range(-spread, spread + 1):
                for dx in range(-spread, spread + 1):
                    jx = max(0, min(ix + dx, nx - 1))
                    jy = max(0, min(iy + dy, ny - 1))
                    n_cells = (2 * spread + 1) ** 2
                    Q[jy, jx] += p / (n_cells * cell_area)

        # Effective thermal conductivity and thickness
        k_eff = self.K_FR4
        if design.stackup.layers:
            total_t = sum(l.thickness for l in design.stackup.layers)
            # Weighted average conductivity
            k_sum = sum(
                l.thickness
                * (
                    self.K_COPPER
                    if l.layer_type != LayerType.DIELECTRIC
                    else self.K_FR4
                )
                for l in design.stackup.layers
            )
            k_eff = k_sum / total_t if total_t > 0 else self.K_FR4
            board_thickness = total_t * 1e-3
        else:
            board_thickness = 1.6e-3

        dx_m = resolution_mm * 1e-3

        # Gauss-Seidel iteration
        for _ in range(iterations):
            T_old = T.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    neighbors = (
                        T[j - 1, i] + T[j + 1, i] + T[j, i - 1] + T[j, i + 1]
                    )
                    # Heat equation with convection loss
                    conv_loss = (
                        self.H_CONV
                        * dx_m**2
                        / (k_eff * board_thickness)
                        * (T[j, i] - t_ambient)
                    )
                    source = Q[j, i] * dx_m**2 / (k_eff * board_thickness)
                    T[j, i] = (neighbors + source - conv_loss) / 4.0

            # Boundary: convective (Neumann-ish)
            T[0, :] = T[1, :]
            T[-1, :] = T[-2, :]
            T[:, 0] = T[:, 1]
            T[:, -1] = T[:, -2]

            # Check convergence
            if np.max(np.abs(T - T_old)) < 0.01:
                break

        return np.round(T, 2)

    @staticmethod
    def _parse_value(val_str: str) -> float:
        val_str = val_str.strip().upper()
        multipliers = {
            "T": 1e12, "G": 1e9, "MEG": 1e6, "K": 1e3,
            "M": 1e-3, "U": 1e-6, "N": 1e-9, "P": 1e-12, "F": 1e-15,
        }
        for suffix, mult in multipliers.items():
            if val_str.endswith(suffix):
                return float(val_str[: -len(suffix)]) * mult
        return float(val_str)
