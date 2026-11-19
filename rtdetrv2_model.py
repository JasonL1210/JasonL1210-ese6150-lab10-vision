"""Independent single-class RT-DETRv2-R18vd model for ESE 6150 Lab 10.

The non-student architecture follows the course specification. Student-owned
operations live in :mod:`rtdetrv2_ops` and are tested independently. No
teammate code, checkpoint, or metric is used.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

from rtdetrv2_ops import (
    ConvNormLayer,
    MLP,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    bias_init_with_prob,
    box_cxcywh_to_xyxy,
    get_activation,
    inverse_sigmoid,
)


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        variance = self.running_var.reshape(1, -1, 1, 1)
        mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (variance + self.eps).rsqrt()
        return x * scale + (bias - mean * scale)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int,
        shortcut: bool,
        act: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        (
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        )
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut else self.short(x)
        return self.act(self.branch2b(self.branch2a(x)) + residual)


class Blocks(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        count: int,
        stage_num: int,
        act: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        layers = []
        for index in range(count):
            layers.append(
                BasicBlock(
                    ch_in,
                    ch_out,
                    stride=2 if index == 0 and stage_num != 2 else 1,
                    shortcut=index != 0,
                    act=act,
                    variant=variant,
                )
            )
            ch_in = ch_out
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class PResNet(nn.Module):
    """ResNet18-vd backbone returning stride 8, 16, and 32 feature maps."""

    def __init__(
        self,
        return_idx: Sequence[int] = (1, 2, 3),
        freeze_at: int = -1,
        freeze_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            OrderedDict(
                (
                    ("conv1_1", ConvNormLayer(3, 32, 3, 2, act="relu")),
                    ("conv1_2", ConvNormLayer(32, 32, 3, 1, act="relu")),
                    ("conv1_3", ConvNormLayer(32, 64, 3, 1, act="relu")),
                )
            )
        )
        channels = (64, 128, 256, 512)
        self.res_layers = nn.ModuleList()
        ch_in = 64
        for stage, ch_out in enumerate(channels, start=2):
            self.res_layers.append(Blocks(ch_in, ch_out, 2, stage))
            ch_in = ch_out
        self.return_idx = tuple(return_idx)
        self.out_channels = [channels[index] for index in self.return_idx]
        self.out_strides = [(4, 8, 16, 32)[index] for index in self.return_idx]
        if freeze_at >= 0:
            self._freeze(self.conv1)
            for index in range(min(freeze_at, len(self.res_layers))):
                self._freeze(self.res_layers[index])
        if freeze_norm:
            self._replace_batch_norm(self)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def _replace_batch_norm(cls, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                frozen = FrozenBatchNorm2d(child.num_features, child.eps)
                frozen.weight.copy_(child.weight.detach())
                frozen.bias.copy_(child.bias.detach())
                frozen.running_mean.copy_(child.running_mean.detach())
                frozen.running_var.copy_(child.running_var.detach())
                setattr(module, name, frozen)
            else:
                cls._replace_batch_norm(child)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = F.max_pool2d(self.conv1(x), 3, 2, 1)
        outputs = []
        for index, stage in enumerate(self.res_layers):
            x = stage(x)
            if index in self.return_idx:
                outputs.append(x)
        return outputs


class RepVggBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act: str = "relu") -> None:
        super().__init__()
        self.ch_in, self.ch_out = ch_in, ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    @staticmethod
    def _fuse(branch: ConvNormLayer) -> tuple[Tensor, Tensor]:
        norm = branch.norm
        std = (norm.running_var + norm.eps).sqrt()
        scale = (norm.weight / std).reshape(-1, 1, 1, 1)
        kernel = branch.conv.weight * scale
        bias = norm.bias - norm.running_mean * norm.weight / std
        return kernel, bias

    def convert_to_deploy(self) -> None:
        if hasattr(self, "conv"):
            return
        kernel3, bias3 = self._fuse(self.conv1)
        kernel1, bias1 = self._fuse(self.conv2)
        self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, 1, bias=True)
        self.conv.weight.data.copy_(kernel3 + F.pad(kernel1, (1, 1, 1, 1)))
        self.conv.bias.data.copy_(bias3 + bias1)
        del self.conv1
        del self.conv2


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 1.0,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden, 1, 1, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden, 1, 1, act=act)
        self.bottlenecks = nn.Sequential(
            *(RepVggBlock(hidden, hidden, act) for _ in range(num_blocks))
        )
        self.conv3 = (
            ConvNormLayer(hidden, out_channels, 1, 1, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512),
        feat_strides: Sequence[int] = (8, 16, 32),
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        use_encoder_idx: Sequence[int] = (2,),
        num_encoder_layers: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.feat_strides = tuple(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = tuple(use_encoder_idx)
        self.input_proj = nn.ModuleList(
            nn.Sequential(
                OrderedDict(
                    (
                        ("conv", nn.Conv2d(channels, hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    )
                )
            )
            for channels in in_channels
        )
        layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, 0.0, activation="gelu"
        )
        self.encoder = nn.ModuleList(
            TransformerEncoder(copy.deepcopy(layer), num_encoder_layers)
            for _ in self.use_encoder_idx
        )
        levels = len(in_channels)
        self.lateral_convs = nn.ModuleList(
            ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act="silu")
            for _ in range(levels - 1)
        )
        self.fpn_blocks = nn.ModuleList(
            CSPRepLayer(hidden_dim * 2, hidden_dim, 3, expansion)
            for _ in range(levels - 1)
        )
        self.downsample_convs = nn.ModuleList(
            ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act="silu")
            for _ in range(levels - 1)
        )
        self.pan_blocks = nn.ModuleList(
            CSPRepLayer(hidden_dim * 2, hidden_dim, 3, expansion)
            for _ in range(levels - 1)
        )

    @staticmethod
    def position_embedding(
        width: int,
        height: int,
        embed_dim: int,
        device: torch.device,
        temperature: float = 10000.0,
    ) -> Tensor:
        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by four")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        quarter = embed_dim // 4
        omega = torch.arange(quarter, dtype=torch.float32, device=device) / quarter
        omega = 1 / temperature**omega
        y = grid_y.flatten()[:, None] @ omega[None]
        x = grid_x.flatten()[:, None] @ omega[None]
        return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=1)[None]

    def forward(self, feats: Sequence[Tensor]) -> list[Tensor]:
        if len(feats) != len(self.in_channels):
            raise ValueError("unexpected backbone feature count")
        projected = [projection(feat) for projection, feat in zip(self.input_proj, feats)]
        for encoder_index, feature_index in enumerate(self.use_encoder_idx):
            height, width = projected[feature_index].shape[-2:]
            tokens = projected[feature_index].flatten(2).permute(0, 2, 1)
            position = self.position_embedding(
                width, height, self.hidden_dim, tokens.device
            ).to(tokens.dtype)
            tokens = self.encoder[encoder_index](tokens, position)
            projected[feature_index] = tokens.permute(0, 2, 1).reshape(
                -1, self.hidden_dim, height, width
            )

        inner = [projected[-1]]
        for index in range(len(projected) - 1, 0, -1):
            block_index = len(projected) - 1 - index
            high = self.lateral_convs[block_index](inner[0])
            inner[0] = high
            high = F.interpolate(high, size=projected[index - 1].shape[-2:], mode="nearest")
            inner.insert(
                0,
                self.fpn_blocks[block_index](torch.cat((high, projected[index - 1]), 1)),
            )
        outputs = [inner[0]]
        for index in range(len(inner) - 1):
            low = self.downsample_convs[index](outputs[-1])
            outputs.append(self.pan_blocks[index](torch.cat((low, inner[index + 1]), 1)))
        return outputs


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(
        self,
        target: Tensor,
        reference_unactivated: Tensor,
        memory: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
        box_heads: nn.ModuleList,
        score_heads: nn.ModuleList,
        query_pos_head: MLP,
    ) -> tuple[Tensor, Tensor]:
        boxes, logits = [], []
        references = reference_unactivated.sigmoid()
        for index, layer in enumerate(self.layers):
            target = layer(
                target,
                references.unsqueeze(2),
                memory,
                spatial_shapes,
                query_pos=query_pos_head(references),
            )
            references_next = (
                box_heads[index](target) + inverse_sigmoid(references)
            ).sigmoid()
            boxes.append(references_next)
            logits.append(score_heads[index](target))
            references = references_next.detach()
        return torch.stack(boxes), torch.stack(logits)


class RTDETRTransformerv2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        feat_channels: Sequence[int] = (256, 256, 256),
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 3,
        nhead: int = 8,
        num_points: int | Sequence[int] = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_levels = len(feat_channels)
        self.input_proj = nn.ModuleList(
            nn.Sequential(
                OrderedDict(
                    (
                        ("conv", nn.Conv2d(channels, hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    )
                )
            )
            for channels in feat_channels
        )
        layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            hidden_dim * 4,
            n_levels=self.num_levels,
            n_points=num_points,
        )
        self.decoder = TransformerDecoder(layer, num_layers)
        self.query_pos_head = MLP(4, hidden_dim * 2, hidden_dim, 2)
        self.enc_output = nn.Sequential(
            OrderedDict(
                (("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim)))
            )
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_score_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        )
        self.dec_bbox_head = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for score, box in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(score.bias, bias)
            init.constant_(box.layers[-1].weight, 0)
            init.constant_(box.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        for projection in self.input_proj:
            init.xavier_uniform_(projection[0].weight)

    def _encoder_input(self, feats: Sequence[Tensor]) -> tuple[Tensor, list[list[int]]]:
        flattened, shapes = [], []
        for projection, feat in zip(self.input_proj, feats):
            feat = projection(feat)
            height, width = feat.shape[-2:]
            shapes.append([height, width])
            flattened.append(feat.flatten(2).permute(0, 2, 1))
        return torch.cat(flattened, 1), shapes

    @staticmethod
    def _anchors(
        shapes: Sequence[Sequence[int]], dtype: torch.dtype, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        anchors = []
        for level, (height, width) in enumerate(shapes):
            y, x = torch.meshgrid(
                torch.arange(height, dtype=dtype, device=device),
                torch.arange(width, dtype=dtype, device=device),
                indexing="ij",
            )
            centers = (torch.stack((x, y), -1) + 0.5) / torch.tensor(
                [width, height], dtype=dtype, device=device
            )
            sizes = torch.ones_like(centers) * 0.05 * (2**level)
            anchors.append(torch.cat((centers, sizes), -1).reshape(1, -1, 4))
        anchors = torch.cat(anchors, 1)
        valid = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return torch.where(valid, anchors, torch.inf), valid

    def forward(self, feats: Sequence[Tensor]) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        memory, shapes = self._encoder_input(feats)
        anchors, valid = self._anchors(shapes, memory.dtype, memory.device)
        memory = memory * valid.to(memory.dtype)
        output_memory = self.enc_output(memory)
        encoder_logits = self.enc_score_head(output_memory)
        encoder_boxes = self.enc_bbox_head(output_memory) + anchors
        count = min(self.num_queries, memory.shape[1])
        indices = encoder_logits.amax(-1).topk(count, dim=1).indices
        content = memory.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        ).detach()
        reference = encoder_boxes.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, 4)
        ).detach()
        boxes, logits = self.decoder(
            content,
            reference,
            memory,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
        )
        output: dict[str, Tensor | list[dict[str, Tensor]]] = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
        }
        if self.training and len(logits) > 1:
            output["aux_outputs"] = [
                {"pred_logits": layer_logits, "pred_boxes": layer_boxes}
                for layer_logits, layer_boxes in zip(logits[:-1], boxes[:-1])
            ]
        return output


class RTDETR(nn.Module):
    def __init__(
        self, backbone: PResNet, encoder: HybridEncoder, decoder: RTDETRTransformerv2
    ) -> None:
        super().__init__()
        self.backbone, self.encoder, self.decoder = backbone, encoder, decoder

    def forward(self, images: Tensor) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        return self.decoder(self.encoder(self.backbone(images)))

    def deploy(self) -> "RTDETR":
        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self


class RTDETRPostProcessor(nn.Module):
    def __init__(self, num_classes: int = 1, num_top_queries: int = 300) -> None:
        super().__init__()
        self.num_classes, self.num_top_queries = num_classes, num_top_queries

    def forward(
        self, outputs: dict[str, Tensor], original_sizes_wh: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
        boxes = boxes * original_sizes_wh.repeat(1, 2).unsqueeze(1)
        flat_scores = outputs["pred_logits"].sigmoid().flatten(1)
        count = min(self.num_top_queries, flat_scores.shape[1])
        scores, indices = flat_scores.topk(count, -1)
        labels = indices.remainder(self.num_classes)
        queries = indices.div(self.num_classes, rounding_mode="floor")
        boxes = boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
        return labels, boxes, scores


def build_rtdetrv2_r18vd(
    *,
    num_classes: int = 1,
    hidden_dim: int = 256,
    num_queries: int = 300,
    num_decoder_layers: int = 3,
    nhead: int = 8,
    num_points: int | Sequence[int] = 4,
) -> RTDETR:
    if num_classes != 1:
        raise ValueError("the independent Lab 10 model is intentionally single-class")
    backbone = PResNet()
    encoder = HybridEncoder(hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4)
    decoder = RTDETRTransformerv2(
        num_classes=num_classes,
        feat_channels=(hidden_dim, hidden_dim, hidden_dim),
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=num_decoder_layers,
        nhead=nhead,
        num_points=num_points,
    )
    model = RTDETR(backbone, encoder, decoder)
    model.class_names = ["f1tenth_car"]
    return model


def build_rtdetrv2(*args, **kwargs) -> RTDETR:
    return build_rtdetrv2_r18vd(*args, **kwargs)


__all__ = [
    "ConvNormLayer",
    "HybridEncoder",
    "MLP",
    "PResNet",
    "RTDETR",
    "RTDETRPostProcessor",
    "RTDETRTransformerv2",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "build_rtdetrv2",
    "build_rtdetrv2_r18vd",
]

