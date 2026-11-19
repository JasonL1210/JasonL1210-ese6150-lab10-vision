"""Offline checkerboard calibration for the local Lab 10 image set.

The script discovers the checkerboard pattern, detects corners, fits the OpenCV
pinhole model, rejects only statistically clear per-view outliers, and writes
auditable artifacts. It never substitutes its result for the real car's
factory ``camera_info`` calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "calibration"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "calibration"
# ``findChessboardCornersSB`` already performs a robust localized binarization.
# NORMALIZE_IMAGE made one of this dataset's oblique views fail, so the flags
# below are deliberately the pair verified on all 14 local images.
DETECTION_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
COMMON_PATTERNS: tuple[tuple[int, int], ...] = (
    (6, 4),
    (7, 5),
    (8, 5),
    (8, 6),
    (9, 6),
    (9, 7),
    (6, 6),
    (7, 7),
)


@dataclass(frozen=True)
class CornerObservation:
    path: Path
    image_size: tuple[int, int]
    corners: np.ndarray


@dataclass(frozen=True)
class CalibrationFit:
    image_size: tuple[int, int]
    pattern: tuple[int, int]
    square_size: float
    rms_px: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_vectors: tuple[np.ndarray, ...]
    translation_vectors: tuple[np.ndarray, ...]
    per_view_errors_px: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationRun:
    pattern: tuple[int, int]
    all_observations: tuple[CornerObservation, ...]
    used_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    initial_fit: CalibrationFit
    final_fit: CalibrationFit


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def list_images(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in extensions),
        key=natural_key,
    )
    if not paths:
        raise FileNotFoundError(f"no calibration images found in {directory}")
    return paths


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return image


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=DETECTION_FLAGS)
    if not found or corners is None:
        return None
    expected = pattern[0] * pattern[1]
    if corners.shape[0] != expected:
        return None
    # OpenCV builds differ here: some return [N, 1, 2], while newer wheels may
    # return [N, 2].  Keep one stable representation for calibration and tests.
    return corners.astype(np.float32).reshape(expected, 1, 2)


def discover_pattern(
    image_paths: Sequence[Path],
    candidates: Sequence[tuple[int, int]] = COMMON_PATTERNS,
) -> tuple[int, int]:
    """Choose the pattern detected in the most images, then prefer more corners."""

    grays = [read_grayscale(path) for path in image_paths]
    scored = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        # Pattern orientation is geometrically equivalent. Prefer width >= height
        # to make corner ordering deterministic in reports.
        pattern = tuple(sorted(candidate, reverse=True))
        if pattern in seen:
            continue
        seen.add(pattern)
        detections = sum(detect_corners(gray, pattern) is not None for gray in grays)
        scored.append((detections, pattern[0] * pattern[1], pattern))
    if not scored:
        raise ValueError("no checkerboard pattern candidates were provided")
    detections, _, pattern = max(scored)
    if detections < max(3, math.ceil(0.5 * len(image_paths))):
        raise RuntimeError(
            f"checkerboard discovery was unreliable: best pattern {pattern} "
            f"worked in only {detections}/{len(image_paths)} images"
        )
    return pattern


def collect_observations(
    image_paths: Sequence[Path], pattern: tuple[int, int]
) -> tuple[CornerObservation, ...]:
    observations = []
    expected_size: tuple[int, int] | None = None
    for path in image_paths:
        gray = read_grayscale(path)
        image_size = (gray.shape[1], gray.shape[0])
        if expected_size is None:
            expected_size = image_size
        elif image_size != expected_size:
            raise ValueError(
                f"all calibration images must have one resolution; {path.name} is "
                f"{image_size}, expected {expected_size}"
            )
        corners = detect_corners(gray, pattern)
        if corners is not None:
            observations.append(CornerObservation(path, image_size, corners))
    if len(observations) < 3:
        raise RuntimeError(
            f"only {len(observations)} images contained the {pattern} checkerboard"
        )
    return tuple(observations)


def checkerboard_object_points(
    pattern: tuple[int, int], square_size: float
) -> np.ndarray:
    if square_size <= 0:
        raise ValueError("square_size must be positive")
    columns, rows = pattern
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= square_size
    return points


def fit_calibration(
    observations: Sequence[CornerObservation],
    pattern: tuple[int, int],
    square_size: float = 1.0,
) -> CalibrationFit:
    if not observations:
        raise ValueError("at least one observation is required")
    image_size = observations[0].image_size
    object_template = checkerboard_object_points(pattern, square_size)
    object_points = [object_template.copy() for _ in observations]
    image_points = [observation.corners for observation in observations]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = []
    for world, observation, rotation, translation in zip(
        object_points, observations, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            world, rotation, translation, matrix, distortion
        )
        delta = observation.corners.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return CalibrationFit(
        image_size=image_size,
        pattern=pattern,
        square_size=square_size,
        rms_px=float(rms),
        camera_matrix=matrix,
        distortion=distortion.reshape(-1),
        rotation_vectors=tuple(rotations),
        translation_vectors=tuple(translations),
        per_view_errors_px=tuple(errors),
    )


def robust_outlier_indices(errors: Sequence[float], sigma: float = 3.0) -> tuple[int, ...]:
    values = np.asarray(errors, dtype=np.float64)
    if values.size < 8:
        return ()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= np.finfo(float).eps:
        return ()
    robust_sigma = 1.4826 * mad
    threshold = median + sigma * robust_sigma
    return tuple(int(index) for index in np.flatnonzero(values > threshold))


def calibrate_directory(
    input_directory: Path,
    pattern: tuple[int, int] | None = None,
    square_size: float = 1.0,
    reject_outliers: bool = True,
) -> CalibrationRun:
    paths = list_images(input_directory)
    resolved_pattern = pattern or discover_pattern(paths)
    observations = collect_observations(paths, resolved_pattern)
    initial = fit_calibration(observations, resolved_pattern, square_size)
    rejected = (
        robust_outlier_indices(initial.per_view_errors_px) if reject_outliers else ()
    )
    rejected_set = set(rejected)
    used = tuple(index for index in range(len(observations)) if index not in rejected_set)
    if rejected and len(used) >= 8:
        final = fit_calibration(
            tuple(observations[index] for index in used), resolved_pattern, square_size
        )
    else:
        rejected, used, final = (), tuple(range(len(observations))), initial
    return CalibrationRun(
        resolved_pattern,
        observations,
        used,
        rejected,
        initial,
        final,
    )


def _to_serializable(run: CalibrationRun) -> dict[str, object]:
    fit = run.final_fit
    used_observations = [run.all_observations[index] for index in run.used_indices]
    errors = np.asarray(fit.per_view_errors_px)
    return {
        "scope": "offline_recalibration_from_local_images",
        "submission_warning": (
            "Do not use this as factory camera_info; real-car intrinsics remain PENDING_HARDWARE."
        ),
        "image_size": {"width": fit.image_size[0], "height": fit.image_size[1]},
        "checkerboard": {
            "inner_corners_columns": run.pattern[0],
            "inner_corners_rows": run.pattern[1],
            "square_size": fit.square_size,
            "square_unit": "arbitrary" if fit.square_size == 1.0 else "user_supplied",
        },
        "images": {
            "detected": len(run.all_observations),
            "used": [observation.path.name for observation in used_observations],
            "rejected": [
                run.all_observations[index].path.name for index in run.rejected_indices
            ],
        },
        "initial_rms_px": run.initial_fit.rms_px,
        "final_rms_px": fit.rms_px,
        "per_view_error_px": {
            observation.path.name: error
            for observation, error in zip(used_observations, fit.per_view_errors_px)
        },
        "error_summary_px": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "maximum": float(errors.max()),
        },
        "camera_matrix": fit.camera_matrix.tolist(),
        "distortion_model": "opencv_plumb_bob",
        "distortion_coefficients": fit.distortion.tolist(),
    }


def write_artifacts(run: CalibrationRun, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    overlays = output_directory / "corner_overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(run)
    (output_directory / "calibration_result.yaml").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )

    used_errors = {
        index: error
        for index, error in zip(run.used_indices, run.final_fit.per_view_errors_px)
    }
    with (output_directory / "reprojection_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "status", "reprojection_error_px"))
        for index, observation in enumerate(run.all_observations):
            status = "rejected_outlier" if index in run.rejected_indices else "used"
            error = (
                run.initial_fit.per_view_errors_px[index]
                if index in run.rejected_indices
                else used_errors[index]
            )
            writer.writerow((observation.path.name, status, f"{error:.9f}"))

    tiles = []
    for index, observation in enumerate(run.all_observations):
        image = cv2.imread(str(observation.path), cv2.IMREAD_COLOR)
        cv2.drawChessboardCorners(image, run.pattern, observation.corners, True)
        rejected = index in run.rejected_indices
        error = (
            run.initial_fit.per_view_errors_px[index]
            if rejected
            else used_errors[index]
        )
        status = "REJECTED" if rejected else "USED"
        colour = (0, 0, 255) if rejected else (0, 200, 0)
        cv2.putText(
            image,
            f"{observation.path.name} {status} error={error:.3f}px",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(overlays / observation.path.name), image)
        tile = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

    columns = 4
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_directory / "corner_summary.png"), np.vstack(rows))

    matrix = run.final_fit.camera_matrix
    distortion = run.final_fit.distortion
    report = f"""# Offline Camera Recalibration Report

This report uses only the 14 local checkerboard images. It is not the real
car's factory `camera_info` result.

- Image size: {run.final_fit.image_size[0]}×{run.final_fit.image_size[1]}
- Checkerboard: {run.pattern[0]}×{run.pattern[1]} inner corners
- Detected images: {len(run.all_observations)}
- Used images: {len(run.used_indices)}
- Rejected outliers: {', '.join(run.all_observations[i].path.name for i in run.rejected_indices) or 'none'}
- Initial RMS: {run.initial_fit.rms_px:.6f} px
- Final RMS: {run.final_fit.rms_px:.6f} px
- Mean per-view error: {np.mean(run.final_fit.per_view_errors_px):.6f} px

## Camera matrix

```text
{np.array2string(matrix, precision=8)}
```

## Distortion coefficients

```text
{np.array2string(distortion, precision=8)}
```

The physical checker square size was not supplied. A unit square was used;
this does not change the estimated intrinsic matrix or pixel reprojection
error, but translation vectors are expressed in arbitrary square units.
"""
    (output_directory / "calibration_report.md").write_text(report, encoding="utf-8")


def parse_pattern(value: str) -> tuple[int, int] | None:
    if value.lower() == "auto":
        return None
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("pattern must be auto or COLSxROWS, e.g. 8x6")
    columns, rows = map(int, match.groups())
    if columns < 3 or rows < 3:
        raise argparse.ArgumentTypeError("checkerboard needs at least 3x3 inner corners")
    return columns, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate from local checkerboard images")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", type=parse_pattern, default=None)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--keep-outliers", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = calibrate_directory(
        args.input,
        pattern=args.pattern,
        square_size=args.square_size,
        reject_outliers=not args.keep_outliers,
    )
    write_artifacts(run, args.output)
    matrix = run.final_fit.camera_matrix
    print(f"pattern={run.pattern[0]}x{run.pattern[1]}")
    print(
        f"images_detected={len(run.all_observations)} used={len(run.used_indices)} "
        f"rejected={len(run.rejected_indices)}"
    )
    print(f"final_rms_px={run.final_fit.rms_px:.9f}")
    print(
        "intrinsics="
        f"fx={matrix[0, 0]:.9f},fy={matrix[1, 1]:.9f},"
        f"cx={matrix[0, 2]:.9f},cy={matrix[1, 2]:.9f}"
    )
    print(f"artifacts={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

