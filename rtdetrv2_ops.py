"""Independent RT-DETRv2 tensor operations and transformer building blocks."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    if not 0.0 < prior_prob < 1.0:
        raise ValueError("prior_prob must lie strictly between zero and one")
    return float(-math.log((1.0 - prior_prob) / prior_prob))


def get_activation(act: str | nn.Module | None, inplace: bool = True) -> nn.Module:
    """Build an activation used by the course model."""

    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    if not isinstance(act, str):
        raise TypeError("activation must be a string, module, or None")
    factories: dict[str, type[nn.Module]] = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "hardsigmoid": nn.Hardsigmoid,
    }
    name = act.lower()
    if name not in factories:
        raise ValueError(f"unsupported activation: {act!r}")
    module = factories[name]()
    if hasattr(module, "inplace"):
        module.inplace = inplace
    return module


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    sides = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return sides[..., 0] * sides[..., 1]


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1, area2 = box_area(boxes1), box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    sides = (right_bottom - left_top).clamp(min=0)
    intersection = sides[..., 0] * sides[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError("boxes1 contains invalid xyxy boxes")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError("boxes2 contains invalid xyxy boxes")
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    sides = (right_bottom - left_top).clamp(min=0)
    enclosing = sides[..., 0] * sides[..., 1]
    penalty = torch.where(
        enclosing > 0, (enclosing - union) / enclosing, torch.zeros_like(enclosing)
    )
    return iou - penalty


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        bias: bool = False,
        act: str | nn.Module | None = None,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        pos_embed: Tensor | None = None,
    ) -> Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        query = key = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(query, key, src, attn_mask=src_mask, need_weights=False)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        import copy

        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))

    def forward(self, src: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            src = layer(src, pos_embed=pos_embed)
        return src


def deformable_attention_core(
    value: Tensor,
    spatial_shapes: Sequence[Sequence[int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
    points_per_level: Sequence[int],
) -> Tensor:
    batch, _, heads, head_dim = value.shape
    query_length = sampling_locations.shape[1]
    split_sizes = [int(height * width) for height, width in spatial_shapes]
    values = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids = grids.split(tuple(points_per_level), dim=-2)
    sampled = []
    for (height, width), level_value, level_grid in zip(spatial_shapes, values, grids):
        level_value = level_value.reshape(batch * heads, head_dim, int(height), int(width))
        sampled.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch * heads, 1, query_length, sum(points_per_level)
    )
    output = (torch.cat(sampled, dim=-1) * weights).sum(-1)
    return output.reshape(batch, heads * head_dim, query_length).permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int | Sequence[int] = 4,
        offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if isinstance(num_points, int):
            points = [num_points] * num_levels
        else:
            points = list(num_points)
            if len(points) != num_levels:
                raise ValueError("num_points must have one value per level")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.points_per_level = points
        self.offset_scale = offset_scale
        self.head_dim = embed_dim // num_heads
        total_points = num_heads * sum(points)
        self.sampling_offsets = nn.Linear(embed_dim, total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        point_scale = [1.0 / count for count in points for _ in range(count)]
        self.register_buffer("num_points_scale", torch.tensor(point_scale, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init.constant_(self.sampling_offsets.weight, 0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().amax(-1, keepdim=True)
        grid = grid.reshape(self.num_heads, 1, 2).tile((1, sum(self.points_per_level), 1))
        radii = torch.cat(
            tuple(torch.arange(1, count + 1) for count in self.points_per_level)
        ).reshape(1, -1, 1)
        self.sampling_offsets.bias.data.copy_((grid * radii).flatten())
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
        value_mask: Tensor | None = None,
    ) -> Tensor:
        batch, queries = query.shape[:2]
        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(batch, value.shape[1], self.num_heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(
            batch, queries, self.num_heads, sum(self.points_per_level), 2
        )
        weights = F.softmax(
            self.attention_weights(query).reshape(
                batch, queries, self.num_heads, sum(self.points_per_level)
            ),
            dim=-1,
        )
        if reference_points.shape[-1] == 4:
            if reference_points.shape[2] == 1:
                reference_points = reference_points.expand(-1, -1, self.num_levels, -1)
            elif reference_points.shape[2] != self.num_levels:
                raise ValueError(
                    "four-coordinate references need one shared box or one box per level"
                )
            centers = torch.cat(
                tuple(
                    reference_points[:, :, level : level + 1, :2].expand(
                        -1, -1, count, -1
                    )
                    for level, count in enumerate(self.points_per_level)
                ),
                dim=2,
            )
            sizes = torch.cat(
                tuple(
                    reference_points[:, :, level : level + 1, 2:].expand(
                        -1, -1, count, -1
                    )
                    for level, count in enumerate(self.points_per_level)
                ),
                dim=2,
            )
            scale = self.num_points_scale.to(query.dtype).reshape(1, 1, 1, -1, 1)
            locations = centers[:, :, None] + (
                offsets * scale * sizes[:, :, None] * self.offset_scale
            )
        elif reference_points.shape[-1] == 2:
            normalizer = torch.as_tensor(
                spatial_shapes, device=query.device, dtype=query.dtype
            ).flip(-1)
            normalizer = normalizer.reshape(1, 1, 1, self.num_levels, 1, 2)
            offsets = offsets.reshape(
                batch, queries, self.num_heads, self.num_levels, -1, 2
            )
            locations = (
                reference_points.reshape(batch, queries, 1, self.num_levels, 1, 2)
                + offsets / normalizer
            ).flatten(3, 4)
        else:
            raise ValueError("reference points must end in two or four coordinates")
        output = deformable_attention_core(
            value, spatial_shapes, locations, weights, self.points_per_level
        )
        return self.output_proj(output)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip([input_dim, *hidden], [*hidden, output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < self.num_layers - 1:
                x = self.act(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 3,
        n_points: int | Sequence[int] = 4,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1, self.norm1 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2, self.norm2 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4, self.norm3 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(
        self,
        target: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
        attn_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        query = key = target if query_pos is None else target + query_pos
        attended, _ = self.self_attn(
            query, key, target, attn_mask=attn_mask, need_weights=False
        )
        target = self.norm1(target + self.dropout1(attended))
        cross_query = target if query_pos is None else target + query_pos
        attended = self.cross_attn(
            cross_query, reference_points, memory, spatial_shapes, memory_mask
        )
        target = self.norm2(target + self.dropout2(attended))
        feedforward = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout4(feedforward))

