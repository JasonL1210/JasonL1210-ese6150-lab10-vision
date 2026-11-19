"""Required Lab 10 camera entry point.

The V4L2 implementation belongs to hardware stage B1.  This A0 file defines a
safe CLI boundary and refuses hardware execution while measurements are absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from config import HARDWARE_CONFIG_PATH, load_yaml, require_hardware_runtime
from interfaces import FramePacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 10 camera capture")
    parser.add_argument("--mode", choices=("simulation", "hardware"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, nargs="*")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate mode configuration without opening a camera",
    )
    return parser


class LocalImageSource:
    """Finite deterministic image source for offline integration."""

    def __init__(self, paths: Sequence[Path], fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("at least one image path is required")
        self.period_s = 1.0 / fps
        self.index = 0

    def read(self) -> FramePacket | None:
        if self.index >= len(self.paths):
            return None
        import cv2

        path = self.paths[self.index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        packet = FramePacket(
            image=image,
            timestamp_s=self.index * self.period_s,
            frame_id=f"offline_camera/{path.name}",
        )
        self.index += 1
        return packet


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "hardware":
        config = load_yaml(args.config or HARDWARE_CONFIG_PATH)
        require_hardware_runtime(config)
        if args.check_config:
            return 0
        raise NotImplementedError("V4L2 capture is scheduled for hardware stage B1")

    if args.check_config:
        return 0
    if not args.input:
        raise ValueError("simulation mode requires one or more --input image paths")
    source = LocalImageSource(args.input, args.fps)
    count = 0
    while source.read() is not None:
        count += 1
    print(f"offline_frames={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

