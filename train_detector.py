"""Command-line entry point for local F1/10 RT-DETRv2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training import DEFAULT_OUTPUT, DEFAULT_PRETRAINED, TrainConfig, train
from training_data import DEFAULT_DATA_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6150)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = TrainConfig(
        data_root=arguments.data_root.resolve(),
        pretrained_checkpoint=arguments.pretrained.resolve(),
        output_dir=arguments.output_dir.resolve(),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        gradient_accumulation=arguments.gradient_accumulation,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        warmup_steps=arguments.warmup_steps,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        amp=not arguments.no_amp,
    )
    summary = train(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
