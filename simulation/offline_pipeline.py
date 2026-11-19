"""Deterministic offline image-detection and synthetic-LiDAR integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from camera import LocalImageSource
from detection import OnnxDetectorBackend
from lidar_camera import (
    AssociationConfig,
    CameraIntrinsics,
    backproject_pixel,
    associate_detection_to_scan,
    make_lidar_to_camera_transform,
    project_laser_scan,
)
from simulation.synthetic_scan import VerticalSurface, generate_scan
from training_data import block_split, load_dataset_index


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "onnx" / "rtdetrv2_f1tenth_640.onnx"
DEFAULT_OUTPUT = ROOT / "artifacts" / "simulation" / "offline_integration"


def dotted(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


def make_intrinsics(config: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        float(dotted(config, "camera.intrinsics.fx")),
        float(dotted(config, "camera.intrinsics.fy")),
        float(dotted(config, "camera.intrinsics.cx")),
        float(dotted(config, "camera.intrinsics.cy")),
        int(dotted(config, "camera.width")),
        int(dotted(config, "camera.height")),
    )


def aligned_synthetic_scan(
    box_xyxy: Sequence[float],
    target_depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_center_lidar: np.ndarray,
    timestamp_s: float,
    seed: int,
):
    """Place a synthetic vertical target behind the detected box center."""

    x0, _, x1, _ = map(float, box_xyxy)
    center_u = (x0 + x1) / 2
    target_x = float(camera_center_lidar[0] + target_depth_m)
    target_y = float(
        camera_center_lidar[1]
        - (center_u - intrinsics.cx) * target_depth_m / intrinsics.fx
    )
    half_width = max(0.08, 0.42 * (x1 - x0) * target_depth_m / intrinsics.fx)
    direction = 1.0 if target_y <= 0 else -1.0
    distractor_y = target_y + direction * max(0.8, half_width + 0.55)
    surfaces = (
        VerticalSurface(target_x, target_y - half_width, target_y + half_width, "target"),
        VerticalSurface(5.5, -7.0, 7.0, "background"),
        VerticalSurface(1.15, distractor_y - 0.18, distractor_y + 0.18, "distractor"),
    )
    scan = generate_scan(
        surfaces,
        angle_min_rad=-np.deg2rad(80),
        angle_max_rad=np.deg2rad(80),
        beam_count=1081,
        noise_std_m=0.0015,
        dropout_probability=0.01,
        seed=seed,
        timestamp_s=timestamp_s,
    )
    return scan, target_x


def draw_result(
    image: np.ndarray,
    detection,
    projected,
    association,
    expected_distance_m: float,
    frame_number: int,
) -> np.ndarray:
    canvas = image.copy()
    depths = projected.points_camera[:, 2]
    normalized = np.clip((depths - 0.5) / 5.0, 0, 1)
    colours = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
    )
    chosen = set(int(value) for value in association.cluster_beam_indices)
    for index, (pixel, beam) in enumerate(zip(projected.pixels_uv, projected.beam_indices)):
        u, v = np.rint(pixel).astype(int)
        if int(beam) in chosen:
            colour, radius = (255, 255, 255), 3
        else:
            colour, radius = tuple(int(value) for value in colours[index, 0]), 1
        cv2.circle(canvas, (u, v), radius, colour, -1, cv2.LINE_AA)
    x0, y0, x1, y1 = map(int, detection.box_xyxy)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
    center = tuple(np.rint(association.box_center_uv).astype(int))
    cv2.drawMarker(canvas, center, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    panel = canvas.copy()
    cv2.rectangle(panel, (0, 0), (430, 94), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.72, canvas, 0.28, 0, canvas)
    lines = (
        "OFFLINE: REAL IMAGE + SYNTHETIC LIDAR",
        f"frame {frame_number} | ONNX score {detection.score:.3f}",
        f"distance {association.distance_camera_m:.3f} m | expected {expected_distance_m:.3f} m",
        f"cluster {len(association.cluster_beam_indices)} beams",
    )
    for row, text in enumerate(lines):
        y = 22 + row * 22
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        colour = (0, 100, 255) if row == 0 else (245, 245, 245)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    return canvas


def run_offline_pipeline(
    config: Mapping[str, Any],
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT,
    frame_count: int = 8,
    provider: str = "cpu",
) -> dict[str, Any]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = make_intrinsics(config)
    center_config = dotted(config, "lidar.camera_center_in_lidar_m")
    camera_center = np.array(
        [center_config["x"], center_config["y"], center_config["z"]], dtype=float
    )
    rotation_config = dotted(config, "lidar.residual_rotation_rpy_deg")
    residual_rpy = np.deg2rad(
        [rotation_config["roll"], rotation_config["pitch"], rotation_config["yaw"]]
    )
    transform = make_lidar_to_camera_transform(camera_center, residual_rpy)
    association_values = dotted(config, "lidar.association")
    association_config = AssociationConfig(
        trim_fraction=float(association_values["trim_fraction"]),
        column_half_width_px=float(association_values["column_half_width_px"]),
        vertical_margin_px=float(association_values.get("vertical_margin_px", 4.0)),
        absolute_range_jump_m=float(association_values["absolute_range_jump_m"]),
        relative_range_jump=float(association_values["relative_range_jump"]),
        minimum_cluster_size=int(association_values["minimum_cluster_size"]),
    )
    paths, _ = load_dataset_index()
    split = block_split(len(paths), 0.15, 50, 6150)
    positions = np.linspace(0, len(split.validation_indices) - 1, frame_count, dtype=int)
    indices = [split.validation_indices[int(position)] for position in positions]
    source = LocalImageSource([paths[index] for index in indices], fps=20.0)
    detector = OnnxDetectorBackend(
        model_path,
        float(dotted(config, "detector.confidence_threshold")),
        provider,
    )
    max_skew_s = float(dotted(config, "simulation.max_sync_skew_s"))
    seed = int(dotted(config, "simulation.random_seed"))
    depths = np.linspace(1.45, 3.25, frame_count)
    records: list[dict[str, Any]] = []
    rendered: list[np.ndarray] = []
    for sequence in range(frame_count):
        frame = source.read()
        assert frame is not None
        detections = detector.detect(frame.image)
        if not detections:
            raise RuntimeError(f"no detection for held-out dataset index {indices[sequence]}")
        detection = detections[0]
        scan_timestamp = frame.timestamp_s + (0.0, 0.004, -0.004)[sequence % 3]
        scan, target_x = aligned_synthetic_scan(
            detection.box_xyxy,
            float(depths[sequence]),
            intrinsics,
            camera_center,
            scan_timestamp,
            seed + sequence,
        )
        skew = abs(frame.timestamp_s - scan.timestamp_s)
        if skew > max_skew_s:
            raise RuntimeError(f"camera/LiDAR skew {skew:.6f}s exceeds {max_skew_s:.6f}s")
        projected = project_laser_scan(scan, intrinsics, transform)
        association = associate_detection_to_scan(
            detection.box_xyxy, projected, intrinsics, transform, association_config
        )
        if association is None:
            raise RuntimeError(f"association failed for held-out dataset index {indices[sequence]}")
        expected_position = backproject_pixel(
            association.box_center_uv, float(depths[sequence]), intrinsics
        )
        expected_distance = float(np.linalg.norm(expected_position))
        distance_error = abs(association.distance_camera_m - expected_distance)
        records.append(
            {
                "sequence": sequence,
                "dataset_index": int(indices[sequence]),
                "image_path": str(paths[indices[sequence]].resolve()),
                "frame_timestamp_s": frame.timestamp_s,
                "scan_timestamp_s": scan.timestamp_s,
                "sync_skew_s": skew,
                "detection": {
                    "label": detection.label,
                    "score": detection.score,
                    "box_xyxy": list(detection.box_xyxy),
                },
                "synthetic_target_x_lidar_m": target_x,
                "expected_distance_camera_m": expected_distance,
                "estimated_distance_camera_m": association.distance_camera_m,
                "absolute_distance_error_m": distance_error,
                "position_camera_m": association.position_camera.tolist(),
                "selected_beam_count": len(association.cluster_beam_indices),
            }
        )
        rendered.append(
            draw_result(
                frame.image, detection, projected, association, expected_distance, sequence
            )
        )
    montage_rows = []
    columns = 4
    for start in range(0, len(rendered), columns):
        row = rendered[start : start + columns]
        row.extend([np.zeros_like(rendered[0])] * (columns - len(row)))
        montage_rows.append(np.hstack(row))
    montage_path = output_dir / "offline_integration_montage.png"
    cv2.imwrite(str(montage_path), np.vstack(montage_rows))
    video_path = output_dir / "offline_integration.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0,
        (intrinsics.width, intrinsics.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the offline integration video")
    for image in rendered:
        writer.write(image)
    writer.release()
    errors = [record["absolute_distance_error_m"] for record in records]
    report = {
        "scope": "offline_real_images_with_synthetic_lidar_only",
        "hardware_submission_evidence": False,
        "detector_backend": "onnxruntime",
        "detector_provider": detector.active_provider,
        "model": str(model_path.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "frame_count": frame_count,
        "detection_success_count": len(records),
        "association_success_count": len(records),
        "maximum_sync_skew_s": max(record["sync_skew_s"] for record in records),
        "distance_error_m": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
        },
        "outputs": {
            "montage": str(montage_path.resolve()),
            "video": str(video_path.resolve()),
        },
        "frames": records,
    }
    report_path = output_dir / "offline_integration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_MODEL", "DEFAULT_OUTPUT", "aligned_synthetic_scan", "make_intrinsics",
    "run_offline_pipeline",
]

