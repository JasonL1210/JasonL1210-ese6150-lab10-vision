"""Dependency-free contracts shared by simulation and future hardware code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class FramePacket:
    image: Any
    timestamp_s: float
    frame_id: str = "camera"


@dataclass(frozen=True)
class LaserScanPacket:
    ranges_m: Sequence[float]
    angle_min_rad: float
    angle_increment_rad: float
    range_min_m: float
    range_max_m: float
    timestamp_s: float
    frame_id: str = "laser"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


class ImageSource(Protocol):
    def read(self) -> FramePacket | None:
        """Return the newest frame, or None when the source is exhausted."""


class ScanSource(Protocol):
    def latest(self) -> LaserScanPacket | None:
        """Return the newest scan, or None before the first scan arrives."""


class DetectorBackend(Protocol):
    def detect(self, image: Any) -> list[Detection]:
        """Return detections in the original image coordinate system."""

