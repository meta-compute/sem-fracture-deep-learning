import ast
import contextlib
import io
import json
import unittest
from pathlib import Path


class PredictionNotebookTests(unittest.TestCase):
    def test_modelling_notebook_sections_preserve_validation_and_gate_test(self):
        repository_root = Path(__file__).resolve().parents[1]
        notebook_path = repository_root / "modelling.ipynb"
        self.assertTrue(
            notebook_path.is_file(),
            "modelling.ipynb must be the canonical notebook",
        )
        self.assertFalse((repository_root / "train_model.ipynb").exists())
        self.assertFalse((repository_root / "predict_model.ipynb").exists())
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

        section_ids = ("train-section", "validate-section", "test-section")
        positions = {}
        for section_id in section_ids:
            self.assertIn(section_id, cells)
            positions[section_id] = next(
                index
                for index, cell in enumerate(notebook["cells"])
                if cell.get("id") == section_id
            )
        self.assertEqual(
            [cells[section_id].splitlines()[0] for section_id in section_ids],
            ["## Train", "## Validate", "## Test"],
        )
        self.assertEqual(
            [positions[section_id] for section_id in section_ids],
            sorted(positions.values()),
        )

        self.assertIn("training-config", cells)
        self.assertIn("validation-run", cells)
        self.assertIn("final-test-run", cells)
        config = ast.parse(cells["training-config"])
        test_flag = [
            node.value
            for node in config.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "RUN_FINAL_TEST_EVALUATION"
                for target in node.targets
            )
        ]
        self.assertEqual(len(test_flag), 1)
        self.assertIsInstance(test_flag[0], ast.Constant)
        self.assertIs(test_flag[0].value, False)

        calls = []

        def fake_run_prediction(**kwargs):
            calls.append(kwargs)
            return {"metrics": {"accuracy": 1.0}, "predictions": []}

        class FakePandas:
            @staticmethod
            def DataFrame(rows):
                return rows

        namespace = {
            "RUN_FINAL_TEST_EVALUATION": False,
            "run_prediction": fake_run_prediction,
            "DATA_DIR": Path("."),
            "MODEL_KEY": "resnet18",
            "CHECKPOINT_PATH": Path("checkpoint.pth"),
            "DEVICE": "cpu",
            "pd": FakePandas,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cells["validation-run"], namespace)
        self.assertEqual(calls[-1]["split"], "validation")
        self.assertEqual(calls[-1]["split_file"], "dataset_split.csv")

        with contextlib.redirect_stdout(io.StringIO()):
            exec(cells["final-test-run"], namespace)
        self.assertEqual(len(calls), 1)

        namespace["RUN_FINAL_TEST_EVALUATION"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cells["final-test-run"], namespace)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["split"], "test")
        self.assertEqual(calls[1]["split_file"], "dataset_split.csv")


if __name__ == "__main__":
    unittest.main()
