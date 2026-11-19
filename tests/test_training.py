"""Tests for the local-only training data and optimization pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rtdetrv2_model import build_rtdetrv2
from training import (
    COCO_CAR_INDEX,
    DEFAULT_PRETRAINED,
    HungarianDetectionLoss,
    average_precision,
    load_coco_pretrained_car_head,
)
from training_data import DEFAULT_DATA_ROOT, F110DetectionDataset, block_split, load_dataset_index


class DatasetTests(unittest.TestCase):
    @unittest.skipUnless(
        (DEFAULT_DATA_ROOT / "labels.npy").is_file(), "local F1/10 dataset is required"
    )
    def test_downloaded_dataset_and_sample(self) -> None:
        paths, labels = load_dataset_index(DEFAULT_DATA_ROOT)
        self.assertEqual(len(paths), 1792)
        self.assertEqual(labels.shape, (1792, 4))
        dataset = F110DetectionDataset(paths, labels, [0], image_size=160)
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 160, 160))
        self.assertTrue(torch.all((target["boxes"] >= 0) & (target["boxes"] <= 1)))
        self.assertEqual(target["labels"].item(), 0)

    def test_block_split_is_disjoint_and_reproducible(self) -> None:
        first = block_split(1792, 0.15, 50, 6150)
        second = block_split(1792, 0.15, 50, 6150)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_indices) & set(first.validation_indices))
        self.assertEqual(len(first.train_indices) + len(first.validation_indices), 1792)
        for block in range((1792 + 49) // 50):
            members = set(range(block * 50, min((block + 1) * 50, 1792)))
            self.assertTrue(members <= set(first.train_indices) or members <= set(first.validation_indices))


class TrainingMathTests(unittest.TestCase):
    def test_average_precision_perfect_and_failed(self) -> None:
        scores = np.array([0.9, 0.8, 0.7])
        self.assertEqual(average_precision(scores, np.ones(3)), 1.0)
        self.assertEqual(average_precision(scores, np.zeros(3)), 0.0)

    def test_hungarian_loss_is_finite_and_differentiable(self) -> None:
        logits = torch.randn(2, 8, 1, requires_grad=True)
        boxes = torch.sigmoid(torch.randn(2, 8, 4, requires_grad=True))
        targets = [
            {"labels": torch.zeros(1, dtype=torch.long), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
            {"labels": torch.zeros(1, dtype=torch.long), "boxes": torch.tensor([[0.4, 0.6, 0.3, 0.2]])},
        ]
        losses = HungarianDetectionLoss()({"pred_logits": logits, "pred_boxes": boxes}, targets)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(logits.grad)

    @unittest.skipUnless(DEFAULT_PRETRAINED.is_file(), "local COCO checkpoint is required")
    def test_pretrained_car_head_mapping(self) -> None:
        self.assertTrue(DEFAULT_PRETRAINED.is_file())
        model = build_rtdetrv2()
        report = load_coco_pretrained_car_head(model, DEFAULT_PRETRAINED)
        self.assertGreater(report.exact_parameter_fraction, 0.999)
        self.assertEqual(len(report.copied_class_heads), 8)
        checkpoint = torch.load(DEFAULT_PRETRAINED, map_location="cpu", weights_only=False)
        source = checkpoint["ema"]["module"]
        loaded = model.state_dict()
        for key in report.copied_class_heads:
            torch.testing.assert_close(loaded[key][0], source[key][COCO_CAR_INDEX])


if __name__ == "__main__":
    unittest.main()
