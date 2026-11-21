import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sem_fracture.training import train_model
except ImportError:
    train_model = None


class SharedTrainingTests(unittest.TestCase):
    def test_training_reports_epoch_start_and_metrics(self):
        if train_model is None:
            self.fail("sem_fracture.training.train_model is not implemented")

        model = nn.Linear(2, 3)
        images = torch.ones(3, 2)
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        class FlushTrackingStream(io.StringIO):
            flush_count = 0

            def flush(self):
                self.flush_count += 1
                super().flush()

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pth"
            output = FlushTrackingStream()

            with contextlib.redirect_stdout(output):
                train_model(
                    model=model,
                    train_loader=loader,
                    val_loader=loader,
                    optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(),
                    device=torch.device("cpu"),
                    model_key="resnet18",
                    epochs=1,
                    checkpoint_path=checkpoint_path,
                )

        progress = output.getvalue()
        self.assertIn("Epoch 1/1 starting", progress)
        self.assertIn("Epoch 1/1 |", progress)
        for metric in (
            "train_loss=",
            "val_loss=",
            "val_accuracy=",
            "val_macro_f1=",
            "elapsed=",
            "best_checkpoint=True",
        ):
            self.assertIn(metric, progress)
        self.assertRegex(progress, r"elapsed=\d+\.\d+s")
        self.assertGreaterEqual(output.flush_count, 2)

    def test_training_refuses_to_overwrite_an_existing_checkpoint(self):
        if train_model is None:
            self.fail("sem_fracture.training.train_model is not implemented")

        model = nn.Linear(2, 3)
        images = torch.ones(3, 2)
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pth"
            checkpoint_path.write_bytes(b"existing checkpoint")

            with self.assertRaises(FileExistsError):
                train_model(
                    model=model,
                    train_loader=loader,
                    val_loader=loader,
                    optimizer=optimizer,
                    criterion=nn.CrossEntropyLoss(),
                    device=torch.device("cpu"),
                    model_key="resnet18",
                    epochs=1,
                    checkpoint_path=checkpoint_path,
                )

            self.assertEqual(checkpoint_path.read_bytes(), b"existing checkpoint")

    def test_training_saves_best_macro_f1_checkpoint(self):
        if train_model is None:
            self.fail("sem_fracture.training.train_model is not implemented")

        model = nn.Linear(2, 3)
        images = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [-0.8, -0.8],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pth"

            result = train_model(
                model=model,
                train_loader=loader,
                val_loader=loader,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                model_key="resnet18",
                epochs=1,
                checkpoint_path=checkpoint_path,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            for parameter in model.parameters():
                parameter.data.zero_()

        self.assertEqual(len(result["history"]["train_loss"]), 1)
        self.assertEqual(checkpoint["model_name"], "resnet18")
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("val_macro_f1", checkpoint)
        self.assertGreaterEqual(checkpoint["val_macro_f1"], 0.0)
        self.assertLessEqual(checkpoint["val_macro_f1"], 1.0)
        for key, saved_parameter in checkpoint["model_state_dict"].items():
            self.assertTrue(
                torch.equal(
                    result["best_checkpoint"]["model_state_dict"][key],
                    saved_parameter,
                )
            )


if __name__ == "__main__":
    unittest.main()
