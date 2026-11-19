"""Local F1/10 dataset loader with leakage-resistant block splitting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "f110_dataset_20220209"
    / "f110_dataset_20220209"
)


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_blocks: tuple[int, ...]
    block_size: int
    seed: int

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": "consecutive_frame_blocks",
            "seed": self.seed,
            "block_size": self.block_size,
            "validation_blocks": list(self.validation_blocks),
            "train_count": len(self.train_indices),
            "validation_count": len(self.validation_indices),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def block_split(
    sample_count: int,
    validation_fraction: float = 0.15,
    block_size: int = 50,
    seed: int = 6150,
) -> DatasetSplit:
    if sample_count < 2 or block_size < 1:
        raise ValueError("sample_count must exceed one and block_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    blocks = [
        tuple(range(start, min(start + block_size, sample_count)))
        for start in range(0, sample_count, block_size)
    ]
    validation_block_count = max(1, min(len(blocks) - 1, round(len(blocks) * validation_fraction)))
    rng = random.Random(seed)
    validation_blocks = tuple(sorted(rng.sample(range(len(blocks)), validation_block_count)))
    validation_set = set(validation_blocks)
    train = tuple(index for block, indices in enumerate(blocks) if block not in validation_set for index in indices)
    validation = tuple(index for block, indices in enumerate(blocks) if block in validation_set for index in indices)
    return DatasetSplit(train, validation, validation_blocks, block_size, seed)


def load_dataset_index(root: Path = DEFAULT_DATA_ROOT) -> tuple[list[Path], np.ndarray]:
    labels_path = root / "labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing labels file: {labels_path}")
    labels = np.load(labels_path).astype(np.float32)
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError(f"expected labels [N,4], found {labels.shape}")
    paths = [root / f"{index}.jpg" for index in range(len(labels))]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images; first is {missing[0]}")
    return paths, labels


def clip_box_cxcywh(box: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, box_width, box_height = map(float, box)
    x0 = np.clip(cx - box_width / 2, 0, width)
    y0 = np.clip(cy - box_height / 2, 0, height)
    x1 = np.clip(cx + box_width / 2, 0, width)
    y1 = np.clip(cy + box_height / 2, 0, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("augmentation produced a degenerate box")
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=np.float32)


def augment_image_and_box(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    box = box.copy()
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        box[0] = width - box[0]
    if random.random() < 0.8:
        contrast = random.uniform(0.82, 1.18)
        brightness = random.uniform(-20, 20)
        image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.75, 1.25)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, clip_box_cxcywh(box, width, height)


class F110DetectionDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        boxes_cxcywh_px: np.ndarray,
        indices: Sequence[int],
        image_size: int = 640,
        augment: bool = False,
    ) -> None:
        self.paths = list(paths)
        self.boxes = np.asarray(boxes_cxcywh_px, dtype=np.float32)
        self.indices = tuple(int(index) for index in indices)
        self.image_size = int(image_size)
        self.augment = augment
        if len(self.paths) != len(self.boxes):
            raise ValueError("image and label counts differ")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, dict[str, Tensor]]:
        index = self.indices[item]
        bgr = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {self.paths[index]}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = self.boxes[index].copy()
        height, width = image.shape[:2]
        if self.augment:
            image, box = augment_image_and_box(image, box)
        normalized_box = box / np.array([width, height, width, height], dtype=np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(normalized_box[None]),
            "labels": torch.zeros(1, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
            "original_size_wh": torch.tensor([width, height], dtype=torch.float32),
        }
        return image_tensor, target


def detection_collate(
    batch: Sequence[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DatasetSplit",
    "F110DetectionDataset",
    "augment_image_and_box",
    "block_split",
    "clip_box_cxcywh",
    "detection_collate",
    "load_dataset_index",
]

