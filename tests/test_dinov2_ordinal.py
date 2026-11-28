import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from sem_fracture.dinov2_ordinal import (
    CLASS_INDEX_ORDINAL_TARGETS,
    DinoV2OrdinalClassifier,
    OrdinalBCEWithLogitsLoss,
    PadToSquare,
    build_adamw_optimizer,
    build_dinov2_transforms,
    class_index_from_ordinal_logits,
    configure_trainable_parameters,
    decode_ordinal_logits,
    encode_ordinal_targets,
    evaluate_dinov2_ordinal,
    set_training_mode,
    train_dinov2_ordinal,
)
from sem_fracture.training import _evaluate
from sem_fracture.models import MODEL_NAMES


class FakeDino(nn.Module):
    def __init__(self, embed_dim=8, n_blocks=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(n_blocks))
        self.stem = nn.Conv2d(3, embed_dim, kernel_size=1)

    def forward(self, images):
        hidden = self.stem(images).mean(dim=(2, 3))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _write_ordinal_split(root: Path) -> None:
    for split in ("train", "val"):
        for name, color in (("ductile", 20), ("brittle", 200), ("mixed", 100)):
            folder = root / split / name
            folder.mkdir(parents=True)
            Image.new("L", (12, 30), color=color).save(folder / "sample.png")


class OrdinalTargetTests(unittest.TestCase):
    def test_targets_are_brittle_ductile_evidence_bits(self):
        encoded = encode_ordinal_targets(torch.tensor([0, 1, 2]))
        self.assertTrue(
            torch.equal(
                encoded,
                torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            )
        )

    def test_loss_matches_bce_with_logits_on_evidence_codes(self):
        logits = torch.tensor([[0.2, -0.4], [1.5, 0.1], [-0.3, 0.8]])
        labels = torch.tensor([0, 1, 2])
        expected = nn.BCEWithLogitsLoss()(
            logits,
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        actual = OrdinalBCEWithLogitsLoss()(logits, labels)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_decode_uses_evidence_presence_rules(self):
        logits = torch.tensor(
            [
                [10.0, -10.0],
                [10.0, 10.0],
                [-10.0, 10.0],
            ]
        )
        # ImageFolder indices: brittle, mixed, ductile
        self.assertTrue(
            torch.equal(decode_ordinal_logits(logits), torch.tensor([0, 1, 2]))
        )
        # CLASSES indices: Brittle, Ductile, Mixed
        self.assertTrue(
            torch.equal(
                class_index_from_ordinal_logits(logits),
                torch.tensor([0, 2, 1]),
            )
        )
        both_high = torch.tensor([[2.0, 2.0]])
        self.assertEqual(
            int(class_index_from_ordinal_logits(both_high, threshold=0.6).item()),
            2,
        )

    def test_pipeline_evidence_targets_match_class_order(self):
        loss = OrdinalBCEWithLogitsLoss(CLASS_INDEX_ORDINAL_TARGETS)(
            torch.zeros(3, 2),
            torch.tensor([0, 1, 2]),
        )
        expected = nn.BCEWithLogitsLoss()(
            torch.zeros(3, 2),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        )
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_shared_evaluator_uses_evidence_decoder_for_dinov2(self):
        class _Fixed(nn.Module):
            def forward(self, images):
                return torch.tensor([[10.0, 10.0]]).expand(images.shape[0], 2)

        images = torch.zeros(2, 3, 8, 8)
        labels = torch.tensor([2, 2])  # Mixed
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels),
            batch_size=2,
        )
        loss, metrics = _evaluate(
            _Fixed(),
            loader,
            OrdinalBCEWithLogitsLoss(CLASS_INDEX_ORDINAL_TARGETS),
            torch.device("cpu"),
            "dinov2_vitb14",
            prediction_threshold=0.6,
        )
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))


class PreprocessAndHeadTests(unittest.TestCase):
    def test_registry_of_existing_models_is_unchanged(self):
        self.assertEqual(MODEL_NAMES["dinov2_vitb14"], "dinov2_vitb14")

    def test_image_size_must_match_patch_grid(self):
        with self.assertRaises(ValueError):
            build_dinov2_transforms(image_size=225)

    def test_pad_then_resize_keeps_aspect_and_rgb(self):
        _, eval_transform = build_dinov2_transforms(image_size=28)
        image = Image.new("L", (12, 30), color=80)
        padded = PadToSquare()(image)
        self.assertEqual(padded.size[0], padded.size[1])
        tensor = eval_transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 28, 28))

    def test_folder_order_is_ordinal_not_alphabetical(self):
        from sem_fracture.dinov2_ordinal import RGBOrdinalImageFolder, build_dinov2_transforms

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_ordinal_split(root)
            _, eval_transform = build_dinov2_transforms(image_size=28)
            dataset = RGBOrdinalImageFolder(root / "train", transform=eval_transform)
            self.assertEqual(
                dataset.class_to_idx,
                {"brittle": 0, "mixed": 1, "ductile": 2},
            )
            ductile_targets = [
                target
                for path, target in dataset.samples
                if Path(path).parent.name == "ductile"
            ]
            self.assertEqual(ductile_targets, [2])
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 28, 28))
            self.assertIn(target, (0, 1, 2))

    def test_head_and_partial_unfreeze(self):
        model = DinoV2OrdinalClassifier(FakeDino(), dropout=0.3)
        linear, norm, gelu, dropout, output = model.head
        self.assertEqual(linear.out_features, 256)
        self.assertIsInstance(norm, nn.LayerNorm)
        self.assertIsInstance(gelu, nn.GELU)
        self.assertEqual(dropout.p, 0.3)
        self.assertEqual(output.out_features, 2)

        groups = configure_trainable_parameters(
            model, freeze_backbone=True, unfreeze_last_blocks=2
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.blocks[0].parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.blocks[1].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.backbone.blocks[-1].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.backbone.blocks[-2].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.stem.parameters()))

        optimizer = build_adamw_optimizer(groups, head_lr=1e-3, backbone_lr=1e-5)
        learning_rates = {group["lr"] for group in optimizer.param_groups}
        self.assertEqual(learning_rates, {1e-3, 1e-5})

        set_training_mode(model)
        self.assertFalse(model.backbone.blocks[0].training)
        self.assertTrue(model.backbone.blocks[-1].training)
        self.assertTrue(model.head.training)

        full = DinoV2OrdinalClassifier(FakeDino())
        with self.assertRaises(ValueError):
            configure_trainable_parameters(
                full, freeze_backbone=False, unfreeze_last_blocks=2
            )


class TrainLoopTests(unittest.TestCase):
    def test_one_epoch_saves_best_macro_f1_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_ordinal_split(root)
            checkpoint = root / "best.pth"
            result = train_dinov2_ordinal(
                data_dir=root,
                checkpoint_path=checkpoint,
                device="cpu",
                epochs=1,
                batch_size=2,
                image_size=28,
                model=DinoV2OrdinalClassifier(FakeDino()),
                seed=0,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(result["best_checkpoint"]["epoch"], 1)
            self.assertIn("val_macro_f1", result["history"])
            self.assertEqual(len(result["history"]["val_class_f1"][0]), 3)
            self.assertEqual(
                result["class_to_idx"],
                {"brittle": 0, "mixed": 1, "ductile": 2},
            )

            evaluation = evaluate_dinov2_ordinal(
                data_dir=root,
                checkpoint_path=checkpoint,
                device="cpu",
                split="val",
                batch_size=2,
                model=DinoV2OrdinalClassifier(FakeDino()),
            )
            self.assertEqual(len(evaluation["confusion_matrix"]), 3)
            self.assertIn("macro_f1", evaluation["metrics"])

            with self.assertRaises(FileExistsError):
                train_dinov2_ordinal(
                    data_dir=root,
                    checkpoint_path=checkpoint,
                    device="cpu",
                    epochs=1,
                    batch_size=2,
                    image_size=28,
                    model=DinoV2OrdinalClassifier(FakeDino()),
                )
