from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "run ML tests in the local WSL PyTorch environment")
class RTDETRComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from rtdetrv2_model import (
            RTDETRPostProcessor,
            build_rtdetrv2_r18vd,
        )
        from rtdetrv2_ops import (
            ConvNormLayer,
            MLP,
            MSDeformableAttention,
            TransformerDecoderLayer,
            TransformerEncoderLayer,
            box_cxcywh_to_xyxy,
            box_iou,
            box_xyxy_to_cxcywh,
            generalized_box_iou,
            get_activation,
        )

        cls.ConvNormLayer = ConvNormLayer
        cls.MLP = MLP
        cls.MSDeformableAttention = MSDeformableAttention
        cls.PostProcessor = RTDETRPostProcessor
        cls.DecoderLayer = TransformerDecoderLayer
        cls.EncoderLayer = TransformerEncoderLayer
        cls.box_cxcywh_to_xyxy = staticmethod(box_cxcywh_to_xyxy)
        cls.box_iou = staticmethod(box_iou)
        cls.box_xyxy_to_cxcywh = staticmethod(box_xyxy_to_cxcywh)
        cls.build_model = staticmethod(build_rtdetrv2_r18vd)
        cls.generalized_box_iou = staticmethod(generalized_box_iou)
        cls.get_activation = staticmethod(get_activation)

    def setUp(self) -> None:
        torch.manual_seed(6150)

    def test_activation_factory(self) -> None:
        self.assertIsInstance(self.get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(self.get_activation("swish"), torch.nn.SiLU)
        self.assertIsInstance(self.get_activation(None), torch.nn.Identity)
        with self.assertRaises(ValueError):
            self.get_activation("unknown")

    def test_box_conversions_round_trip(self) -> None:
        boxes = torch.tensor([[0.5, 0.4, 0.2, 0.6], [0.2, 0.3, 0.1, 0.1]])
        recovered = self.box_xyxy_to_cxcywh(self.box_cxcywh_to_xyxy(boxes))
        torch.testing.assert_close(recovered, boxes)

    def test_pairwise_iou_and_giou(self) -> None:
        first = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
        second = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        iou, union = self.box_iou(first, second)
        torch.testing.assert_close(iou[:, 0], torch.tensor([1.0 / 7.0, 0.0]))
        torch.testing.assert_close(union[:, 0], torch.tensor([7.0, 5.0]))
        giou = self.generalized_box_iou(first, second)
        self.assertEqual(tuple(giou.shape), (2, 1))
        self.assertTrue(torch.isfinite(giou).all())

    def test_conv_norm_layer_shape_and_gradient(self) -> None:
        layer = self.ConvNormLayer(3, 8, 3, 2, act="relu")
        image = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = layer(image)
        self.assertEqual(tuple(output.shape), (2, 8, 16, 16))
        output.mean().backward()
        self.assertIsNotNone(image.grad)

    def test_transformer_encoder_shape_and_gradient(self) -> None:
        layer = self.EncoderLayer(32, 4, 64, 0.0, "gelu")
        source = torch.randn(2, 20, 32, requires_grad=True)
        output = layer(source, pos_embed=torch.randn(1, 20, 32))
        self.assertEqual(tuple(output.shape), (2, 20, 32))
        output.square().mean().backward()
        self.assertIsNotNone(source.grad)

    def test_mlp_shape(self) -> None:
        output = self.MLP(4, 16, 8, 3)(torch.randn(2, 10, 4))
        self.assertEqual(tuple(output.shape), (2, 10, 8))

    def test_deformable_attention_shape_and_gradient(self) -> None:
        layer = self.MSDeformableAttention(32, 4, 2, 2)
        query = torch.randn(2, 5, 32, requires_grad=True)
        value = torch.randn(2, 20, 32, requires_grad=True)
        references = torch.rand(2, 5, 2, 4)
        output = layer(query, references, value, ((4, 4), (2, 2)))
        self.assertEqual(tuple(output.shape), (2, 5, 32))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(value.grad)

    def test_decoder_layer_shape(self) -> None:
        layer = self.DecoderLayer(32, 4, 64, n_levels=2, n_points=2)
        output = layer(
            torch.randn(1, 6, 32),
            torch.rand(1, 6, 1, 4),
            torch.randn(1, 20, 32),
            ((4, 4), (2, 2)),
        )
        self.assertEqual(tuple(output.shape), (1, 6, 32))

    def test_small_end_to_end_model_and_postprocessor(self) -> None:
        model = self.build_model(
            hidden_dim=32,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            num_points=2,
        ).eval()
        with torch.no_grad():
            output = model(torch.rand(1, 3, 64, 64))
        self.assertEqual(tuple(output["pred_logits"].shape), (1, 10, 1))
        self.assertEqual(tuple(output["pred_boxes"].shape), (1, 10, 4))
        self.assertTrue(torch.isfinite(output["pred_logits"]).all())
        self.assertTrue(((output["pred_boxes"] >= 0) & (output["pred_boxes"] <= 1)).all())

        labels, boxes, scores = self.PostProcessor(1, 5)(
            output, torch.tensor([[960.0, 540.0]])
        )
        self.assertEqual(tuple(labels.shape), (1, 5))
        self.assertEqual(tuple(boxes.shape), (1, 5, 4))
        self.assertEqual(tuple(scores.shape), (1, 5))

    def test_small_model_training_backward_and_auxiliary_output(self) -> None:
        model = self.build_model(
            hidden_dim=32,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            num_points=2,
        ).train()
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(len(output["aux_outputs"]), 1)
        loss = output["pred_logits"].square().mean() + output["pred_boxes"].mean()
        loss.backward()
        self.assertIsNotNone(model.decoder.dec_score_head[-1].weight.grad)
        self.assertIsNotNone(model.backbone.conv1.conv1_1.conv.weight.grad)

    def test_deploy_fusion_preserves_predictions(self) -> None:
        model = self.build_model(
            hidden_dim=32,
            num_queries=10,
            num_decoder_layers=2,
            nhead=4,
            num_points=2,
        ).eval()
        image = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            before = model(image)
            model.deploy()
            after = model(image)
        torch.testing.assert_close(
            after["pred_logits"], before["pred_logits"], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            after["pred_boxes"], before["pred_boxes"], rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

