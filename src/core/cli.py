"""Command-line interface for pcb-sim-toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SimulationConfig, SimulationType
from .simulator import PCBSimulator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcb-sim",
        description="Open-source PCB design simulation toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # --- simulate ---
    sim_parser = sub.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument("design", type=Path, help="Path to PCB design YAML file")
    sim_parser.add_argument(
        "--config", type=Path, default=None, help="Simulation config YAML"
    )
    sim_parser.add_argument(
        "--type",
        choices=[t.value for t in SimulationType],
        default="full",
        help="Simulation type (default: full)",
    )
    sim_parser.add_argument(
        "-o", "--output", type=Path, default=Path("./sim_output"), help="Output dir"
    )

    # --- check ---
    sub.add_parser("check", help="Check installed dependencies")

    # --- netlist ---
    nl_parser = sub.add_parser("netlist", help="Export SPICE netlist from design")
    nl_parser.add_argument("design", type=Path, help="Path to PCB design YAML file")
    nl_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output netlist path"
    )

    args = parser.parse_args(argv)

    if args.command == "check":
        return _cmd_check()
    elif args.command == "simulate":
        return _cmd_simulate(args)
    elif args.command == "netlist":
        return _cmd_netlist(args)
    else:
        parser.print_help()
        return 0


def _cmd_check() -> int:
    deps = PCBSimulator.check_dependencies()
    print("Dependency status:")
    for name, ok in deps.items():
        status = "OK" if ok else "MISSING"
        print(f"  {name:20s} [{status}]")
    return 0 if all(deps.values()) else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    from ..parsers.yaml_loader import load_design

    design = load_design(args.design)

    if args.config:
        config = SimulationConfig.from_yaml(args.config)
    else:
        config = SimulationConfig()

    config.sim_type = SimulationType(args.type)
    config.output_dir = args.output

    sim = PCBSimulator(design, config)
    results = sim.run()

    failed = [r for r in results if not r.success]
    if failed:
        for r in failed:
            for err in r.errors:
                print(f"[ERROR] {r.sim_type.value}: {err}", file=sys.stderr)
        return 1
    print(f"[pcb-sim] All stages complete. Results in {config.output_dir}")
    return 0


def _cmd_netlist(args: argparse.Namespace) -> int:
    from ..parsers.yaml_loader import load_design
    from ..analysis.spice import SpiceSimulator

    design = load_design(args.design)
    spice = SpiceSimulator()
    netlist = spice.generate_netlist(design)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(netlist)
        print(f"[pcb-sim] Netlist written to {args.output}")
    else:
        print(netlist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
