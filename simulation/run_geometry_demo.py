"""Render an explicitly synthetic LiDAR/camera association demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lidar_camera import (  # noqa: E402
    AssociationConfig,
    CameraIntrinsics,
    associate_detection_to_scan,
    make_lidar_to_camera_transform,
    project_laser_scan,
)
from simulation.synthetic_scan import default_demo_scene, generate_scan  # noqa: E402


def main() -> int:
    output_dir = ROOT / "artifacts" / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = CameraIntrinsics(600.0, 600.0, 480.0, 270.0, 960, 540)
    transform = make_lidar_to_camera_transform((0.08, 0.0, -0.06))
    scan = generate_scan(default_demo_scene(), noise_std_m=0.002, seed=6150)
    projected = project_laser_scan(scan, intrinsics, transform)
    box = (360.0, 150.0, 600.0, 430.0)
    config = AssociationConfig(column_half_width_px=80.0)
    result = associate_detection_to_scan(box, projected, intrinsics, transform, config)
    if result is None:
        raise RuntimeError("synthetic target was not associated")

    canvas = np.full((intrinsics.height, intrinsics.width, 3), (32, 32, 32), np.uint8)
    cv2.line(canvas, (0, int(intrinsics.cy)), (intrinsics.width - 1, int(intrinsics.cy)),
             (80, 80, 80), 1)
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 110, 180), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 255), 3)
    cv2.putText(canvas, "SYNTHETIC f1tenth_car", (x0, y0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    depths = projected.points_camera[:, 2]
    normalized = np.clip((depths - 1.0) / 5.0, 0.0, 1.0)
    colours = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
    )
    selected_beams = set(int(value) for value in result.cluster_beam_indices)
    for index, (pixel, beam) in enumerate(zip(projected.pixels_uv, projected.beam_indices)):
        u, v = np.rint(pixel).astype(int)
        if int(beam) in selected_beams:
            colour, radius = (255, 255, 255), 4
        else:
            colour, radius = tuple(int(value) for value in colours[index, 0]), 2
        cv2.circle(canvas, (u, v), radius, colour, -1, cv2.LINE_AA)

    position = result.position_camera
    lines = (
        "SYNTHETIC GEOMETRY ONLY - NOT A HARDWARE RESULT",
        f"P_C = ({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}) m",
        f"distance = {result.distance_camera_m:.3f} m",
        f"selected cluster = {len(result.cluster_beam_indices)} beams",
    )
    for row, text in enumerate(lines):
        colour = (0, 80, 255) if row == 0 else (240, 240, 240)
        cv2.putText(canvas, text, (20, 32 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, colour, 2, cv2.LINE_AA)
    image_path = output_dir / "lidar_projection_demo.png"
    cv2.imwrite(str(image_path), canvas)

    payload = {
        "scope": "synthetic_geometry_test_only",
        "box_xyxy": list(box),
        "hit_lidar_m": result.hit_lidar.tolist(),
        "hit_camera_m": result.hit_camera.tolist(),
        "position_camera_m": result.position_camera.tolist(),
        "distance_camera_m": result.distance_camera_m,
        "selected_beams": len(result.cluster_beam_indices),
    }
    (output_dir / "lidar_projection_demo.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"image={image_path.resolve()}")
    print(f"distance_camera_m={result.distance_camera_m:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

