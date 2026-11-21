import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    from sem_fracture.data import (
        CLASS_TO_INDEX,
        SEMDataset,
        compute_class_weights,
        load_split,
        prepare_training_split,
    )
except ImportError:
    CLASS_TO_INDEX = None
    SEMDataset = None
    compute_class_weights = None
    load_split = None
    prepare_training_split = None


class SharedDataTests(unittest.TestCase):
    def test_class_weights_use_training_rows_and_shared_class_order(self):
        if compute_class_weights is None:
            self.fail("class-weight calculation is not implemented")

        weights = compute_class_weights(
            [
                {"label": "Brittle", "split": "train"},
                {"label": "Brittle", "split": "train"},
                {"label": "Ductile", "split": "train"},
                {"label": "Ductile", "split": "train"},
                {"label": "Ductile", "split": "train"},
                {"label": "Mixed", "split": "train"},
                {"label": "Mixed", "split": "validation"},
            ]
        )

        for actual, expected in zip(weights.tolist(), [1.0, 2 / 3, 2.0]):
            self.assertAlmostEqual(actual, expected)

    def test_dataset_loads_augmented_relative_path_with_shared_class_index(self):
        if SEMDataset is None:
            self.fail("sem_fracture.data.SEMDataset is not implemented")

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            image_path = data_dir / "Mixed" / "augmented" / "sample.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), color=(1, 2, 3)).save(image_path)
            records = [
                {
                    "filename": "augmented/sample.png",
                    "label": "Mixed",
                    "group": "group-1",
                    "split": "train",
                }
            ]
            dataset = SEMDataset(records, data_dir, transform=lambda image: image.size)

            image_size, label = dataset[0]

            self.assertEqual(image_size, (8, 6))
            self.assertEqual(label, CLASS_TO_INDEX["Mixed"])

    def test_training_split_augmentation_is_reused_and_preserves_original(self):
        if prepare_training_split is None or load_split is None:
            self.fail("shared training split preparation is not implemented")

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            mixed_dir = data_dir / "Mixed"
            mixed_dir.mkdir()
            rows = [
                {
                    "filename": "mixed1.png",
                    "label": "Mixed",
                    "group": "m1",
                    "split": "train",
                },
                {
                    "filename": "mixed2.png",
                    "label": "Mixed",
                    "group": "m2",
                    "split": "train",
                },
                {
                    "filename": "mixed_val.png",
                    "label": "Mixed",
                    "group": "m3",
                    "split": "validation",
                },
                {
                    "filename": "ductile1.png",
                    "label": "Ductile",
                    "group": "d1",
                    "split": "train",
                },
                {
                    "filename": "ductile2.png",
                    "label": "Ductile",
                    "group": "d2",
                    "split": "train",
                },
                {
                    "filename": "ductile3.png",
                    "label": "Ductile",
                    "group": "d3",
                    "split": "train",
                },
            ]
            for name in ("mixed1.png", "mixed2.png", "mixed_val.png"):
                Image.new("RGB", (12, 12), color=(30, 60, 90)).save(mixed_dir / name)

            split_file = data_dir / "dataset_split.csv"
            with split_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            original_bytes = split_file.read_bytes()

            manifest_path = prepare_training_split(data_dir, seed=11)
            manifest_bytes = manifest_path.read_bytes()
            repeated_path = prepare_training_split(data_dir, seed=11)

            self.assertEqual(repeated_path, manifest_path)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            self.assertEqual(split_file.read_bytes(), original_bytes)
            augmented_rows = load_split(data_dir, manifest_path)
            new_rows = augmented_rows[len(rows):]
            self.assertEqual(len(new_rows), 1)
            self.assertEqual(new_rows[0]["label"], "Mixed")
            self.assertEqual(new_rows[0]["split"], "train")
            generated_path = mixed_dir / new_rows[0]["filename"]
            self.assertTrue(generated_path.is_file())

            generated_path.unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_training_split(data_dir, seed=11)


if __name__ == "__main__":
    unittest.main()
