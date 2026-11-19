"""Required end-to-end Lab 10 entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config import (
    HARDWARE_CONFIG_PATH,
    SIMULATION_CONFIG_PATH,
    load_yaml,
    require_hardware_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated Lab 10 perception")
    parser.add_argument("--mode", choices=("simulation", "hardware"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "hardware":
        config = load_yaml(args.config or HARDWARE_CONFIG_PATH)
        require_hardware_runtime(config)
        if args.check_config:
            return 0
        raise NotImplementedError("Real-car integration is scheduled for stage B4")

    config = load_yaml(args.config or SIMULATION_CONFIG_PATH)
    if config.get("mode") != "simulation":
        raise ValueError("simulation mode requires a simulation configuration")
    if args.check_config:
        return 0
    from simulation.offline_pipeline import DEFAULT_MODEL, DEFAULT_OUTPUT, run_offline_pipeline

    report = run_offline_pipeline(
        config,
        model_path=(args.model or DEFAULT_MODEL).resolve(),
        output_dir=(args.output_dir or DEFAULT_OUTPUT).resolve(),
        frame_count=args.frames,
        provider=args.provider,
    )
    summary = {
        "scope": report["scope"],
        "frame_count": report["frame_count"],
        "association_success_count": report["association_success_count"],
        "distance_error_m": report["distance_error_m"],
        "outputs": report["outputs"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

