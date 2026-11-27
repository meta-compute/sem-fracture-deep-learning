import unittest

import torch
import torch.nn as nn

try:
    from sem_fracture.models import (
        MODEL_NAMES,
        TinyViTDualAttentionModel,
        class_probabilities,
        configure_trainable_parameters,
        set_training_mode,
    )
    from sem_fracture.losses import WeightedBinaryEvidenceLoss
except ImportError:
    MODEL_NAMES = None
    TinyViTDualAttentionModel = None
    class_probabilities = None
    configure_trainable_parameters = None
    set_training_mode = None
    WeightedBinaryEvidenceLoss = None


class FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.fc = nn.Linear(2, 3)


class FakeDeiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(4))
        self.head = nn.Linear(2, 3)


class FakeTinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)) for _ in range(4)
        )
        self.head = nn.Module()
        self.head.fc = nn.Linear(2, 3)
        self.stages[-1].blocks = nn.ModuleList(
            [nn.Module() for _ in range(2)]
        )
        for block in self.stages[-1].blocks:
            block.attn = nn.Linear(2, 2)


class ModelConfigurationTests(unittest.TestCase):
    def test_registry_contains_supported_model_identifiers(self):
        if MODEL_NAMES is None:
            self.fail("sem_fracture.models is not implemented")

        self.assertEqual(
            MODEL_NAMES,
            {
                "resnet18": None,
                "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
                "tinyvit_11m": "tiny_vit_11m_224.in1k",
                "tinyvit_21m": "tiny_vit_21m_224.in1k",
                "tinyvit_21m_dual_attention": "tiny_vit_21m_224.in1k",
            },
        )

    def test_resnet_finetuning_unfreezes_last_block_and_classifier(self):
        if configure_trainable_parameters is None:
            self.fail("trainable parameter configuration is not implemented")

        model = FakeResNet()

        groups = configure_trainable_parameters(
            model, "resnet18", "finetune", backbone_lr=1e-5, head_lr=1e-4
        )

        self.assertTrue(all(parameter.requires_grad for parameter in model.layer4[1].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.fc.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.layer4[0].parameters()))
        self.assertEqual([group["lr"] for group in groups], [1e-5, 1e-4])

    def test_deit_finetuning_unfreezes_last_two_blocks_and_head(self):
        if configure_trainable_parameters is None:
            self.fail("trainable parameter configuration is not implemented")

        model = FakeDeiT()

        configure_trainable_parameters(
            model, "deit_tiny", "finetune", backbone_lr=1e-5, head_lr=1e-4
        )

        self.assertFalse(any(p.requires_grad for p in model.blocks[:2].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.blocks[2:].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_tinyvit_training_mode_reactivates_only_final_stage_attention(self):
        if set_training_mode is None:
            self.fail("TinyViT training mode helper is not implemented")

        model = FakeTinyViT()

        set_training_mode(model, "tinyvit_11m")

        self.assertFalse(model.training)
        self.assertTrue(all(block.attn.training for block in model.stages[-1].blocks))

    def test_dual_attention_mil_trains_only_mil_head_and_computes_evidence_loss(self):
        backbone = FakeTinyViT()
        model = TinyViTDualAttentionModel(backbone=backbone, feature_dim=8, attention_hidden=4)

        groups = configure_trainable_parameters(
            model, "tinyvit_21m_dual_attention", "finetune", backbone_lr=1e-5, head_lr=1e-4
        )
        set_training_mode(model, "tinyvit_21m_dual_attention")

        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.mil_head.parameters()))
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.mil_head.training)
        self.assertEqual([g["lr"] for g in groups], [1e-4])

        bags = torch.randn(2, 10, 8)
        logits = model(bags)
        probs = class_probabilities(logits)
        loss_fn = WeightedBinaryEvidenceLoss(torch.tensor([1.0, 1.0, 2.0]))
        loss = loss_fn(logits, torch.tensor([0, 2]))
        loss.backward()

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(probs.shape, (2, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
