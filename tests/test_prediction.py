import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sem_fracture.prediction import (
        evaluate_predictions,
        load_checkpoint,
        predict,
    )
except ImportError:
    evaluate_predictions = None
    load_checkpoint = None
    predict = None


class SharedPredictionTests(unittest.TestCase):
    def test_checkpoint_loading_and_prediction_preserve_sample_order(self):
        if load_checkpoint is None or predict is None:
            self.fail("sem_fracture.prediction is not implemented")

        model = nn.Linear(3, 3)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3))
            model.bias.zero_()

        images = torch.eye(3)
        labels = torch.tensor([2, 0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pth"
            torch.save(
                {"model_state_dict": model.state_dict()},
                checkpoint_path,
            )
            restored_model = nn.Linear(3, 3)
            checkpoint = load_checkpoint(
                restored_model, checkpoint_path, torch.device("cpu")
            )
            actual, predicted = predict(
                restored_model,
                loader,
                torch.device("cpu"),
            )

        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(actual, [2, 0, 1])
        self.assertEqual(predicted, [0, 1, 2])
        self.assertFalse(restored_model.training)

    def test_evaluation_reports_three_class_confusion_and_macro_metrics(self):
        if evaluate_predictions is None:
            self.fail("prediction metric helper is not implemented")

        metrics = evaluate_predictions([0, 1, 2], [0, 1, 1])

        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(len(metrics["class_f1"]), 3)
        self.assertAlmostEqual(metrics["macro_f1"], sum(metrics["class_f1"]) / 3)


if __name__ == "__main__":
    unittest.main()
