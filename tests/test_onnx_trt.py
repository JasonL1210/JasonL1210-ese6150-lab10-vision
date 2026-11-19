"""Regression tests for the A5 ONNX and TensorRT handoff."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch

from benchmark_trt import benchmark
from convert_trt import build_engine, build_parser
from onnx_export import DEFAULT_ONNX, DEFAULT_REPORT, ExportWrapper, cxcywh_iou


class OnnxArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_ONNX.is_file() and DEFAULT_REPORT.is_file(),
        "local exported ONNX model and validation report are required",
    )
    def test_onnx_artifact_is_valid_and_report_hash_matches(self) -> None:
        self.assertTrue(DEFAULT_ONNX.is_file())
        graph = onnx.load(DEFAULT_ONNX)
        onnx.checker.check_model(graph)
        self.assertEqual(graph.graph.input[0].name, "images")
        self.assertEqual([output.name for output in graph.graph.output], ["pred_logits", "pred_boxes"])
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        digest = hashlib.sha256(DEFAULT_ONNX.read_bytes()).hexdigest()
        self.assertEqual(report["export"]["onnx_sha256"], digest)
        self.assertEqual(report["validation"]["cpu"]["active_provider"], "CPUExecutionProvider")
        self.assertEqual(report["validation"]["cuda"]["active_provider"], "CUDAExecutionProvider")
        self.assertGreater(report["validation"]["cuda"]["top_detection_worst_case"]["box_iou_min"], 0.99)

    def test_export_wrapper_has_stable_tensor_outputs(self) -> None:
        class Tiny(torch.nn.Module):
            def forward(self, images):
                return {"pred_logits": images.mean((2, 3))[:, None, :1], "pred_boxes": torch.zeros(len(images), 1, 4)}

        logits, boxes = ExportWrapper(Tiny())(torch.ones(2, 3, 4, 4))
        self.assertEqual(tuple(logits.shape), (2, 1, 1))
        self.assertEqual(tuple(boxes.shape), (2, 1, 4))

    def test_box_iou_comparison(self) -> None:
        box = np.array([0.5, 0.5, 0.2, 0.3], dtype=np.float32)
        self.assertAlmostEqual(cxcywh_iou(box, box), 1.0, places=6)


class TensorRTGuardTests(unittest.TestCase):
    def test_conversion_parser(self) -> None:
        args = build_parser().parse_args(
            ["--onnx", "model.onnx", "--engine", "model.engine", "--precision", "fp16"]
        )
        self.assertEqual(args.precision, "fp16")
        self.assertEqual(args.workspace_mib, 2048)

    def test_missing_tensorrt_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "target Jetson"):
                build_engine(Path(directory) / "x.onnx", Path(directory) / "x.engine", "fp16")

    def test_benchmark_requires_one_hundred_iterations(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 100"):
            benchmark(Path("missing.engine"), iterations=99)


if __name__ == "__main__":
    unittest.main()
