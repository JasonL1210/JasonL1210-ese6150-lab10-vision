"""Independent single-class RT-DETRv2 training and evaluation pipeline."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import DataLoader

from rtdetrv2_model import RTDETR, build_rtdetrv2
from rtdetrv2_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from training_data import (
    DEFAULT_DATA_ROOT,
    F110DetectionDataset,
    block_split,
    detection_collate,
    load_dataset_index,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED = (
    PROJECT_ROOT / "models" / "pretrained" / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "training"
COCO_CAR_INDEX = 2


@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    pretrained_checkpoint: Path = DEFAULT_PRETRAINED
    output_dir: Path = DEFAULT_OUTPUT
    image_size: int = 640
    epochs: int = 12
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    clip_max_norm: float = 0.1
    validation_fraction: float = 0.15
    split_block_size: int = 50
    seed: int = 6150
    num_workers: int = 2
    amp: bool = True
    ema_decay: float = 0.9998
    confidence_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result


@dataclass(frozen=True)
class PretrainedLoadReport:
    exact_tensor_count: int
    exact_parameter_fraction: float
    copied_class_heads: tuple[str, ...]
    missing_model_keys: tuple[str, ...]
    unused_checkpoint_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / 2000))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco_pretrained_car_head(
    model: RTDETR, checkpoint_path: Path
) -> PretrainedLoadReport:
    """Load shared COCO weights and initialize the one-class head from COCO car."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    own = model.state_dict()
    updated = {key: value.clone() for key, value in own.items()}
    exact = []
    class_heads = []
    class_head_prefixes = (
        "decoder.enc_score_head.",
        "decoder.dec_score_head.",
    )
    for key, value in own.items():
        source = state.get(key)
        if source is not None and source.shape == value.shape:
            updated[key] = source
            exact.append(key)
            continue
        if key.startswith(class_head_prefixes) and source is not None:
            if source.ndim >= 1 and source.shape[0] > COCO_CAR_INDEX and value.shape[0] == 1:
                updated[key] = source[COCO_CAR_INDEX : COCO_CAR_INDEX + 1].clone()
                class_heads.append(key)
    model.load_state_dict(updated, strict=True)
    exact_parameters = sum(own[key].numel() for key in exact)
    all_parameters = sum(value.numel() for value in own.values())
    missing = tuple(sorted(key for key in own if key not in exact and key not in class_heads))
    unused = tuple(sorted(key for key in state if key not in own))
    report = PretrainedLoadReport(
        len(exact),
        exact_parameters / all_parameters,
        tuple(sorted(class_heads)),
        missing,
        unused,
    )
    if report.exact_parameter_fraction < 0.99:
        raise RuntimeError(
            f"pretrained architecture coverage is unexpectedly low: "
            f"{report.exact_parameter_fraction:.3%}"
        )
    if len(report.copied_class_heads) != 8:
        raise RuntimeError(
            f"expected 8 one-class head tensors, copied {len(report.copied_class_heads)}"
        )
    return report


class HungarianDetectionLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def match(
        self, outputs: dict[str, Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        matches = []
        probabilities = outputs["pred_logits"].sigmoid()
        boxes = outputs["pred_boxes"]
        for batch_index, target in enumerate(targets):
            labels = target["labels"]
            target_boxes = target["boxes"]
            class_cost = -probabilities[batch_index][:, labels]
            bbox_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(boxes[batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            cost = class_cost + 5.0 * bbox_cost + 2.0 * giou_cost
            prediction_indices, target_indices = linear_sum_assignment(cost.detach().cpu())
            matches.append(
                (
                    torch.as_tensor(prediction_indices, device=boxes.device, dtype=torch.long),
                    torch.as_tensor(target_indices, device=boxes.device, dtype=torch.long),
                )
            )
        return matches

    def one_output(
        self, outputs: dict[str, Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        matches = self.match(outputs, targets)
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        target_scores = torch.zeros_like(logits)
        selected_boxes, selected_targets = [], []
        total_targets = max(1, sum(len(target["boxes"]) for target in targets))
        for batch_index, ((prediction_ids, target_ids), target) in enumerate(
            zip(matches, targets)
        ):
            prediction_boxes = boxes[batch_index, prediction_ids]
            target_boxes = target["boxes"][target_ids]
            selected_boxes.append(prediction_boxes)
            selected_targets.append(target_boxes)
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(prediction_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
            iou_values = torch.diag(ious).detach().to(target_scores.dtype)
            target_scores[batch_index, prediction_ids, target["labels"][target_ids]] = iou_values
        selected_boxes_tensor = torch.cat(selected_boxes)
        selected_targets_tensor = torch.cat(selected_targets)
        predicted_score = logits.sigmoid().detach()
        weight = self.alpha * predicted_score.pow(self.gamma) * (target_scores == 0) + target_scores
        loss_vfl = F.binary_cross_entropy_with_logits(
            logits, target_scores, weight=weight, reduction="sum"
        ) / total_targets
        loss_bbox = F.l1_loss(
            selected_boxes_tensor, selected_targets_tensor, reduction="sum"
        ) / total_targets
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(selected_boxes_tensor),
            box_cxcywh_to_xyxy(selected_targets_tensor),
        )
        loss_giou = (1 - torch.diag(giou)).sum() / total_targets
        return {
            "loss_vfl": self.classification_weight * loss_vfl,
            "loss_bbox": self.bbox_weight * loss_bbox,
            "loss_giou": self.giou_weight * loss_giou,
        }

    def forward(
        self, outputs: dict[str, Any], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        losses = self.one_output(outputs, targets)
        for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            for name, value in self.one_output(auxiliary, targets).items():
                losses[f"{name}_aux_{layer_index}"] = value
        return losses


def move_targets(
    targets: Sequence[dict[str, Tensor]], device: torch.device
) -> list[dict[str, Tensor]]:
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]


def set_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def average_precision(scores: np.ndarray, ious: np.ndarray, threshold: float = 0.5) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    true_positive = (ious[order] >= threshold).astype(np.float64)
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / len(scores)
    precision = np.cumsum(true_positive) / np.maximum(
        np.cumsum(true_positive) + np.cumsum(false_positive), 1
    )
    return float(
        np.mean(
            [
                precision[recall >= recall_level].max()
                if np.any(recall >= recall_level)
                else 0.0
                for recall_level in np.linspace(0, 1, 101)
            ]
        )
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: HungarianDetectionLoss,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    scores, ious, losses = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        outputs = model(images)
        loss_dict = criterion.one_output(outputs, targets)
        losses.append(float(sum(loss_dict.values()).item()))
        probabilities = outputs["pred_logits"].sigmoid().squeeze(-1)
        top_scores, top_indices = probabilities.max(dim=1)
        predictions = outputs["pred_boxes"].gather(
            1, top_indices[:, None, None].expand(-1, 1, 4)
        ).squeeze(1)
        ground_truth = torch.cat([target["boxes"] for target in targets])
        for prediction, truth, score in zip(predictions, ground_truth, top_scores):
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(prediction[None]),
                box_cxcywh_to_xyxy(truth[None]),
            )
            scores.append(float(score.item()))
            ious.append(float(iou.item()))
    score_array = np.asarray(scores)
    iou_array = np.asarray(ious)
    correct = (score_array >= confidence_threshold) & (iou_array >= 0.5)
    return {
        "validation_loss": float(np.mean(losses)),
        "mean_iou": float(iou_array.mean()),
        "median_iou": float(np.median(iou_array)),
        "accuracy_50": float(correct.mean()),
        "ap50": average_precision(score_array, iou_array, 0.5),
        "mean_score": float(score_array.mean()),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float],
    pretrained_report: PretrainedLoadReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_updates": ema.updates,
            "class_names": ["f1tenth_car"],
            "image_size": config.image_size,
            "epoch": epoch,
            "metrics": metrics,
            "config": config.to_dict(),
            "pretrained_load": pretrained_report.to_dict(),
        },
        path,
    )


def plot_history(history: Sequence[dict[str, float]], output_path: Path) -> None:
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [record["train_loss"] for record in history], label="train")
    axes[0].plot(epochs, [record["validation_loss"] for record in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="weighted loss", title="Loss")
    axes[0].legend()
    for key in ("ap50", "accuracy_50", "mean_iou"):
        axes[1].plot(epochs, [record[key] for record in history], label=key)
    axes[1].set(xlabel="epoch", ylabel="metric", ylim=(0, 1.02), title="Validation")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_validation_montage(
    model: nn.Module,
    dataset: F110DetectionDataset,
    output_path: Path,
    device: torch.device,
    count: int = 12,
) -> None:
    model.eval()
    selected = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    tiles = []
    for item in selected:
        image_tensor, target = dataset[int(item)]
        output = model(image_tensor[None].to(device))
        scores = output["pred_logits"].sigmoid()[0, :, 0]
        query = int(scores.argmax())
        prediction = output["pred_boxes"][0, query].cpu().numpy()
        truth = target["boxes"][0].numpy()
        original_index = int(target["index"])
        image = cv2.imread(str(dataset.paths[original_index]))
        height, width = image.shape[:2]

        def corners(box: np.ndarray) -> tuple[int, int, int, int]:
            cx, cy, box_width, box_height = box
            return (
                int((cx - box_width / 2) * width),
                int((cy - box_height / 2) * height),
                int((cx + box_width / 2) * width),
                int((cy + box_height / 2) * height),
            )

        px0, py0, px1, py1 = corners(prediction)
        gx0, gy0, gx1, gy1 = corners(truth)
        cv2.rectangle(image, (gx0, gy0), (gx1, gy1), (0, 0, 255), 2)
        cv2.rectangle(image, (px0, py0), (px1, py1), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"pred {float(scores[query]):.3f} | red=GT green=pred",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(image, (480, 270)))
    columns = 3
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.hstack(row))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels = load_dataset_index(config.data_root)
    split = block_split(
        len(paths), config.validation_fraction, config.split_block_size, config.seed
    )
    split.write_manifest(config.output_dir / "split_manifest.json")
    train_dataset = F110DetectionDataset(
        paths, labels, split.train_indices, config.image_size, augment=True
    )
    validation_dataset = F110DetectionDataset(
        paths, labels, split.validation_indices, config.image_size, augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        persistent_workers=config.num_workers > 0,
    )

    model = build_rtdetrv2()
    pretrained_report = load_coco_pretrained_car_head(model, config.pretrained_checkpoint)
    (config.output_dir / "pretrained_load_report.json").write_text(
        json.dumps(pretrained_report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    model.to(device)
    ema = ModelEMA(model, config.ema_decay)
    criterion = HungarianDetectionLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float]] = []
    # AP50 can saturate early on this one-object dataset.  Break ties with the
    # thresholded success rate, localization quality, and validation loss.
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        set_batch_norm_eval(model)
        start = time.perf_counter()
        running_loss = 0.0
        batches = 0
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            if global_step < config.warmup_steps:
                scale = (global_step + 1) / config.warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * scale
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                scaled_loss = loss / config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % config.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
            running_loss += float(loss.detach().item())
            batches += 1
        metrics = evaluate(
            ema.module,
            validation_loader,
            criterion,
            device,
            config.confidence_threshold,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(batches, 1),
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - start,
            "peak_cuda_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            ),
        }
        history.append(record)
        print(" ".join(
            f"{key}={value:.5g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in record.items()
        ), flush=True)
        score = (
            record["ap50"],
            record["accuracy_50"],
            record["mean_iou"],
            -record["validation_loss"],
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                config.output_dir / "best" / "model.pt",
                model,
                ema,
                config,
                epoch + 1,
                metrics,
                pretrained_report,
            )
    final_metrics = history[-1]
    save_checkpoint(
        config.output_dir / "last" / "model.pt",
        model,
        ema,
        config,
        config.epochs,
        final_metrics,
        pretrained_report,
    )
    (config.output_dir / "history.yaml").write_text(
        yaml.safe_dump(history, sort_keys=False), encoding="utf-8"
    )
    plot_history(history, config.output_dir / "training_history.png")
    best_checkpoint = torch.load(
        config.output_dir / "best" / "model.pt", map_location="cpu", weights_only=False
    )
    best_model = build_rtdetrv2()
    best_model.load_state_dict(best_checkpoint["ema"])
    best_model.to(device).eval()
    save_validation_montage(
        best_model,
        validation_dataset,
        config.output_dir / "validation_predictions.png",
        device,
    )
    selection_key = lambda item: (
        item["ap50"], item["accuracy_50"], item["mean_iou"], -item["validation_loss"]
    )
    best_record = max(history, key=selection_key)
    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
        "best_ap50": best_record["ap50"],
        "best_epoch": best_record["epoch"],
        "selection_policy": "ap50, accuracy_50, mean_iou, then lower validation_loss",
        "final": final_metrics,
        "config": config.to_dict(),
        "pretrained_load": pretrained_report.to_dict(),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "COCO_CAR_INDEX",
    "DEFAULT_OUTPUT",
    "DEFAULT_PRETRAINED",
    "HungarianDetectionLoss",
    "ModelEMA",
    "PretrainedLoadReport",
    "TrainConfig",
    "average_precision",
    "evaluate",
    "load_coco_pretrained_car_head",
    "train",
]

