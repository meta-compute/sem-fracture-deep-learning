import unittest

import torch.nn as nn

try:
    from sem_fracture.models import (
        MODEL_NAMES,
        configure_trainable_parameters,
        set_training_mode,
    )
except ImportError:
    MODEL_NAMES = None
    configure_trainable_parameters = None
    set_training_mode = None


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


if __name__ == "__main__":
    unittest.main()
