"""Export the independently trained RT-DETRv2 model and validate ONNX output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from rtdetrv2_model import build_rtdetrv2
from training_data import F110DetectionDataset, block_split, load_dataset_index


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "training" / "selected" / "model.pt"
DEFAULT_ONNX = ROOT / "models" / "onnx" / "rtdetrv2_f1tenth_640.onnx"
DEFAULT_REPORT = ROOT / "artifacts" / "benchmark" / "onnx_validation.json"
_DLL_DIRECTORY_HANDLES: list[Any] = []


class ExportWrapper(nn.Module):
    """Expose tensor-only outputs with stable names for ONNX and TensorRT."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def load_export_model(checkpoint_path: Path) -> tuple[ExportWrapper, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_rtdetrv2()
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.deploy()
    return ExportWrapper(model).eval(), checkpoint


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    image_size: int = 640,
    opset: int = 17,
) -> dict[str, Any]:
    wrapper, checkpoint = load_export_model(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            sample,
            output_path,
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = onnx.load(output_path)
    onnx.checker.check_model(graph, full_check=True)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "onnx": str(output_path.resolve()),
        "opset": opset,
        "input_shape": [1, 3, image_size, image_size],
        "output_shapes": {"pred_logits": [1, 300, 1], "pred_boxes": [1, 300, 4]},
        "onnx_bytes": output_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "graph_nodes": len(graph.graph.node),
        "graph_initializers": len(graph.graph.initializer),
    }


def choose_provider(requested: str) -> tuple[list[Any], str]:
    available = ort.get_available_providers()
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers={available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
    if requested == "cpu":
        return ["CPUExecutionProvider"], "CPUExecutionProvider"
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
    return ["CPUExecutionProvider"], "CPUExecutionProvider"


def expose_torch_cuda_libraries() -> None:
    """Let ONNX Runtime reuse the CUDA/cuDNN DLLs bundled with CUDA PyTorch."""

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        torch_libraries = Path(torch.__file__).resolve().parent / "lib"
        if torch_libraries.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(torch_libraries)))


def cxcywh_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_xyxy = np.array(
        [first[0] - first[2] / 2, first[1] - first[3] / 2,
         first[0] + first[2] / 2, first[1] + first[3] / 2]
    )
    second_xyxy = np.array(
        [second[0] - second[2] / 2, second[1] - second[3] / 2,
         second[0] + second[2] / 2, second[1] + second[3] / 2]
    )
    intersection_wh = np.maximum(
        0, np.minimum(first_xyxy[2:], second_xyxy[2:])
        - np.maximum(first_xyxy[:2], second_xyxy[:2])
    )
    intersection = float(np.prod(intersection_wh))
    first_area = float(np.prod(first_xyxy[2:] - first_xyxy[:2]))
    second_area = float(np.prod(second_xyxy[2:] - second_xyxy[:2]))
    return intersection / (first_area + second_area - intersection)


def validate_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    provider: str = "auto",
    sample_count: int = 8,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, Any]:
    expose_torch_cuda_libraries()
    providers, intended_provider = choose_provider(provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0]
    if active_provider != intended_provider:
        raise RuntimeError(
            f"requested {intended_provider}, but ONNX Runtime activated {active_provider}; "
            "refusing to report a silently-fallback benchmark"
        )
    wrapper, _ = load_export_model(checkpoint_path)
    paths, labels = load_dataset_index()
    split = block_split(len(paths), 0.15, 50, 6150)
    positions = np.linspace(0, len(split.validation_indices) - 1, sample_count, dtype=int)
    dataset = F110DetectionDataset(
        paths,
        labels,
        [split.validation_indices[int(position)] for position in positions],
        image_size=640,
        augment=False,
    )
    comparisons = []
    benchmark_input: np.ndarray | None = None
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            array = image.unsqueeze(0).numpy()
            torch_logits, torch_boxes = wrapper(torch.from_numpy(array))
            ort_logits, ort_boxes = session.run(None, {"images": array})
            torch_logits_array = torch_logits.numpy()
            torch_boxes_array = torch_boxes.numpy()
            torch_scores = 1 / (1 + np.exp(-torch_logits_array[0, :, 0]))
            ort_scores = 1 / (1 + np.exp(-ort_logits[0, :, 0]))
            torch_query = int(torch_scores.argmax())
            ort_query = int(ort_scores.argmax())
            torch_top_box = torch_boxes_array[0, torch_query]
            ort_top_box = ort_boxes[0, ort_query]
            comparisons.append(
                {
                    "dataset_index": int(target["index"]),
                    "all_queries_logits_max_abs": float(
                        np.max(np.abs(torch_logits_array - ort_logits))
                    ),
                    "all_queries_boxes_max_abs": float(
                        np.max(np.abs(torch_boxes_array - ort_boxes))
                    ),
                    "top_score_abs": float(
                        abs(torch_scores[torch_query] - ort_scores[ort_query])
                    ),
                    "top_box_max_abs": float(np.max(np.abs(torch_top_box - ort_top_box))),
                    "top_box_iou": cxcywh_iou(torch_top_box, ort_top_box),
                }
            )
            benchmark_input = array
    assert benchmark_input is not None
    for _ in range(warmup if iterations else 0):
        session.run(None, {"images": benchmark_input})
    timings_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {"images": benchmark_input})
        timings_ms.append((time.perf_counter() - start) * 1000)
    logits_max = max(item["all_queries_logits_max_abs"] for item in comparisons)
    boxes_max = max(item["all_queries_boxes_max_abs"] for item in comparisons)
    top_score_max = max(item["top_score_abs"] for item in comparisons)
    top_box_max = max(item["top_box_max_abs"] for item in comparisons)
    top_iou_min = min(item["top_box_iou"] for item in comparisons)
    if active_provider == "CPUExecutionProvider":
        if logits_max > 1e-3 or boxes_max > 1e-3:
            raise RuntimeError(
                f"CPU ONNX elementwise mismatch: logits={logits_max}, boxes={boxes_max}"
            )
        comparison_policy = "all query tensors elementwise"
    else:
        if top_score_max > 0.01 or top_box_max > 0.005 or top_iou_min < 0.99:
            raise RuntimeError(
                "accelerator top-detection mismatch: "
                f"score={top_score_max}, box={top_box_max}, iou={top_iou_min}"
            )
        comparison_policy = (
            "top detection semantics; full low-confidence query differences are recorded "
            "because accelerator reductions may reorder queries"
        )
    return {
        "requested_provider": provider,
        "intended_provider": intended_provider,
        "active_provider": active_provider,
        "available_providers": ort.get_available_providers(),
        "onnxruntime_version": ort.__version__,
        "sample_count": sample_count,
        "comparison_policy": comparison_policy,
        "comparisons": comparisons,
        "all_queries_max_abs": {"pred_logits": logits_max, "pred_boxes": boxes_max},
        "top_detection_worst_case": {
            "score_abs": top_score_max,
            "box_max_abs": top_box_max,
            "box_iou_min": top_iou_min,
        },
        "benchmark_scope": "local development PC only; not valid as Jetson submission timing",
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_ms": (
            {
                "mean": statistics.fmean(timings_ms),
                "median": statistics.median(timings_ms),
                "min": min(timings_ms),
                "max": max(timings_ms),
            }
            if timings_ms else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--provider", choices=("both", "auto", "cpu", "cuda"), default="both"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--cuda-iterations", type=int, default=0,
        help="local CUDA timing is disabled by default because provider fallback is not Jetson data",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_report = export_onnx(args.checkpoint.resolve(), args.output.resolve())
    if args.provider == "both":
        validation_report = {
            selected: validate_onnx(
                args.checkpoint.resolve(), args.output.resolve(), selected, args.samples,
                iterations=(args.cuda_iterations if selected == "cuda" else args.iterations),
            )
            for selected in ("cpu", "cuda")
        }
    else:
        validation_report = validate_onnx(
            args.checkpoint.resolve(), args.output.resolve(), args.provider, args.samples,
            iterations=args.iterations,
        )
    report = {"export": export_report, "validation": validation_report}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
