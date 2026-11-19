"""Deterministic 2D LaserScan generator for vertical planar surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from interfaces import LaserScanPacket


@dataclass(frozen=True)
class VerticalSurface:
    x_m: float
    y_min_m: float
    y_max_m: float
    label: str = "surface"

    def __post_init__(self) -> None:
        if self.x_m <= 0 or self.y_max_m <= self.y_min_m:
            raise ValueError("surface must be in front of the LiDAR and have positive width")


def generate_scan(
    surfaces: Sequence[VerticalSurface],
    *,
    angle_min_rad: float = -np.pi / 3,
    angle_max_rad: float = np.pi / 3,
    beam_count: int = 721,
    range_min_m: float = 0.05,
    range_max_m: float = 10.0,
    noise_std_m: float = 0.0,
    dropout_probability: float = 0.0,
    seed: int = 6150,
    timestamp_s: float = 0.0,
) -> LaserScanPacket:
    if beam_count < 2 or angle_max_rad <= angle_min_rad:
        raise ValueError("scan needs at least two beams and an increasing angle interval")
    if noise_std_m < 0 or not 0 <= dropout_probability < 1:
        raise ValueError("invalid noise or dropout setting")
    angles = np.linspace(angle_min_rad, angle_max_rad, beam_count)
    ranges = np.full(beam_count, np.inf, dtype=float)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    for surface in surfaces:
        positive = cosine > 1e-9
        candidate = np.full(beam_count, np.inf, dtype=float)
        candidate[positive] = surface.x_m / cosine[positive]
        y = candidate * sine
        hit = (
            positive
            & (y >= surface.y_min_m)
            & (y <= surface.y_max_m)
            & (candidate >= range_min_m)
            & (candidate <= range_max_m)
        )
        ranges[hit] = np.minimum(ranges[hit], candidate[hit])
    rng = np.random.default_rng(seed)
    finite = np.isfinite(ranges)
    if noise_std_m:
        ranges[finite] += rng.normal(0.0, noise_std_m, finite.sum())
        ranges[finite] = np.clip(ranges[finite], range_min_m, range_max_m)
    if dropout_probability:
        ranges[finite & (rng.random(beam_count) < dropout_probability)] = np.inf
    increment = (angle_max_rad - angle_min_rad) / (beam_count - 1)
    return LaserScanPacket(
        ranges_m=tuple(float(value) for value in ranges),
        angle_min_rad=float(angle_min_rad),
        angle_increment_rad=float(increment),
        range_min_m=float(range_min_m),
        range_max_m=float(range_max_m),
        timestamp_s=float(timestamp_s),
        frame_id="synthetic_laser",
    )


def default_demo_scene() -> tuple[VerticalSurface, ...]:
    """Target at 2 m, background wall at 5 m, and a closer off-axis distractor."""

    return (
        VerticalSurface(2.0, -0.35, 0.35, "target_car"),
        VerticalSurface(5.0, -6.0, 6.0, "background_wall"),
        VerticalSurface(1.2, 0.75, 1.15, "off_axis_obstacle"),
    )


__all__ = ["VerticalSurface", "default_demo_scene", "generate_scan"]

