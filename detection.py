"""Detector backends and a local image inference entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from interfaces import Detection


SUPPORTED_BACKENDS = ("pytorch", "onnx", "tensorrt", "mock")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 10 object detection")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, required=True)
    parser.add_argument("--input", type=Path, required=True, help="local image path")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser


class OnnxDetectorBackend:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
        provider: str = "cpu",
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0,1]")
        import numpy as np
        import onnxruntime as ort

        if os.name == "nt" and provider == "cuda" and hasattr(os, "add_dll_directory"):
            import torch

            self._dll_handle = os.add_dll_directory(
                str(Path(torch.__file__).resolve().parent / "lib")
            )
        requested = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda" else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=requested)
        expected = requested[0]
        if self.session.get_providers()[0] != expected:
            raise RuntimeError(
                f"requested {expected}, activated {self.session.get_providers()[0]}"
            )
        input_metadata = self.session.get_inputs()[0]
        if tuple(input_metadata.shape) != (1, 3, 640, 640):
            raise ValueError(f"unexpected detector input: {input_metadata.shape}")
        self.input_name = input_metadata.name
        self.confidence_threshold = confidence_threshold
        self.np = np

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    def detect(self, image: Any) -> list[Detection]:
        import cv2

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR HxWx3 array")
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = self.np.ascontiguousarray(
            resized.transpose(2, 0, 1)[None], dtype=self.np.float32
        ) / 255.0
        logits, boxes = self.session.run(None, {self.input_name: tensor})
        scores = 1 / (1 + self.np.exp(-logits[0, :, 0]))
        query = int(scores.argmax())
        score = float(scores[query])
        if score < self.confidence_threshold:
            return []
        cx, cy, box_width, box_height = map(float, boxes[0, query])
        x0 = max(0.0, (cx - box_width / 2) * width)
        y0 = max(0.0, (cy - box_height / 2) * height)
        x1 = min(float(width - 1), (cx + box_width / 2) * width)
        y1 = min(float(height - 1), (cy + box_height / 2) * height)
        if x1 <= x0 or y1 <= y0:
            return []
        return [Detection("f1tenth_car", score, (x0, y0, x1, y1))]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend != "onnx":
        raise NotImplementedError(f"The {args.backend} CLI backend is not implemented yet")
    import cv2

    model = args.model or (
        Path(__file__).resolve().parent / "models" / "onnx" / "rtdetrv2_f1tenth_640.onnx"
    )
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(args.input)
    detector = OnnxDetectorBackend(model, args.confidence, args.provider)
    detections = detector.detect(image)
    payload = [
        {"label": item.label, "score": item.score, "box_xyxy": list(item.box_xyxy)}
        for item in detections
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        annotated = image.copy()
        for item in detections:
            x0, y0, x1, y1 = map(int, item.box_xyxy)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(annotated, f"{item.label} {item.score:.3f}", (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output), annotated)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

