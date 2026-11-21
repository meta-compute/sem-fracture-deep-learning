import csv
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from augmentation import augment_image, augment_mixed_training_set


class MixedTrainingAugmentationTests(unittest.TestCase):
    def test_right_angle_rotation_preserves_non_square_image_extent(self):
        image = Image.new("RGB", (16, 32), color=(100, 120, 140))

        augmented = augment_image(image, random.Random(1))

        self.assertEqual(augmented.size, (32, 16))

    def test_balances_training_mixed_images_without_touching_original_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            mixed_dir = data_dir / "Mixed"
            mixed_dir.mkdir()
            rows = [
                {
                    "filename": "brittle.png",
                    "label": "Brittle",
                    "group": "b1",
                    "split": "train",
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
                {
                    "filename": "mixed_train.png",
                    "label": "Mixed",
                    "group": "m1",
                    "split": "train",
                },
                {
                    "filename": "mixed_val.png",
                    "label": "Mixed",
                    "group": "m2",
                    "split": "validation",
                },
                {
                    "filename": "mixed_test.png",
                    "label": "Mixed",
                    "group": "m3",
                    "split": "test",
                },
            ]
            for name in ("mixed_train.png", "mixed_val.png", "mixed_test.png"):
                Image.new("RGB", (16, 16), color=(100, 120, 140)).save(mixed_dir / name)

            split_path = data_dir / "dataset_split.csv"
            with split_path.open("w", newline="", encoding="utf-8") as split_file:
                writer = csv.DictWriter(split_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            original_split = split_path.read_bytes()

            augmented_manifest = augment_mixed_training_set(data_dir, seed=7)

            self.assertEqual(split_path.read_bytes(), original_split)
            with augmented_manifest.open(newline="", encoding="utf-8") as manifest_file:
                augmented_rows = list(csv.DictReader(manifest_file))
            new_rows = augmented_rows[len(rows):]
            self.assertEqual(len(new_rows), 2)
            self.assertTrue(all(row["label"] == "Mixed" for row in new_rows))
            self.assertTrue(all(row["split"] == "train" for row in new_rows))
            self.assertTrue(all(row["group"] == "m1" for row in new_rows))
            self.assertTrue(
                all((mixed_dir / row["filename"]).is_file() for row in new_rows)
            )

    def test_rejects_target_smaller_than_existing_mixed_training_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "Mixed").mkdir()
            split_path = data_dir / "dataset_split.csv"
            split_path.write_text(
                "filename,label,group,split\nmixed.png,Mixed,m1,train\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                augment_mixed_training_set(data_dir, target_count=0)

    def test_missing_source_leaves_no_partial_augmentation_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            mixed_dir = data_dir / "Mixed"
            mixed_dir.mkdir()
            Image.new("RGB", (16, 16), color=(100, 120, 140)).save(
                mixed_dir / "present.png"
            )
            rows = [
                {
                    "filename": "present.png",
                    "label": "Mixed",
                    "group": "m1",
                    "split": "train",
                },
                {
                    "filename": "missing.png",
                    "label": "Mixed",
                    "group": "m2",
                    "split": "train",
                },
                {
                    "filename": "ductile.png",
                    "label": "Ductile",
                    "group": "d1",
                    "split": "train",
                },
            ]
            split_path = data_dir / "dataset_split.csv"
            with split_path.open("w", newline="", encoding="utf-8") as split_file:
                writer = csv.DictWriter(split_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(FileNotFoundError):
                augment_mixed_training_set(data_dir, target_count=4)

            self.assertFalse((mixed_dir / "augmented").exists())
            self.assertFalse((data_dir / "dataset_split_augmented.csv").exists())


if __name__ == "__main__":
    unittest.main()
