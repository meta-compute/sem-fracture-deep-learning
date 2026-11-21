import argparse
import csv
import random
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def augment_image(image, rng):
    """Return a morphology-preserving randomized view of an SEM image."""
    augmented = image.rotate(rng.choice((90, 180, 270)), expand=True)
    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)
    if rng.random() < 0.5:
        augmented = ImageOps.flip(augmented)
    augmented = ImageEnhance.Contrast(augmented).enhance(rng.uniform(0.9, 1.1))
    return ImageEnhance.Brightness(augmented).enhance(rng.uniform(0.9, 1.1))


def augment_mixed_training_set(data_dir, target_count=None, seed=42):
    """Augment only Mixed training images and write a separate split manifest."""
    data_dir = Path(data_dir)
    split_path = data_dir / "dataset_split.csv"
    with split_path.open(newline="", encoding="utf-8-sig") as split_file:
        reader = csv.DictReader(split_file)
        fieldnames = reader.fieldnames
        required_columns = {"filename", "label", "group", "split"}
        if not fieldnames or not required_columns.issubset(fieldnames):
            raise ValueError(
                "dataset_split.csv must contain filename, label, group, and split columns"
            )
        records = list(reader)

    training_records = [row for row in records if row["split"] == "train"]
    training_counts = Counter(row["label"] for row in training_records)
    mixed_records = [row for row in training_records if row["label"] == "Mixed"]
    if not mixed_records:
        raise ValueError("dataset_split.csv contains no Mixed training images")

    if target_count is None:
        target_count = max(training_counts.values())
    if target_count < len(mixed_records):
        raise ValueError("target_count cannot be smaller than the existing Mixed training count")

    mixed_dir = data_dir / "Mixed"
    augmented_dir = mixed_dir / "augmented"
    manifest_path = data_dir / "dataset_split_augmented.csv"
    if augmented_dir.exists():
        raise FileExistsError(f"Augmentation output directory already exists: {augmented_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"Augmented split manifest already exists: {manifest_path}")

    source_paths = [
        data_dir / row["label"] / row["filename"] for row in mixed_records
    ]
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    needed = target_count - len(mixed_records)
    per_image, remainder = divmod(needed, len(mixed_records))
    rng = random.Random(seed)
    new_records = []

    with tempfile.TemporaryDirectory(
        prefix=".augmentation-", dir=mixed_dir
    ) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        staged_images_dir = temporary_dir / "augmented"
        staged_images_dir.mkdir()

        for image_index, (row, source_path) in enumerate(
            zip(mixed_records, source_paths)
        ):
            count = per_image + (image_index < remainder)
            for variant_index in range(count):
                output_name = (
                    f"{image_index:04d}_{Path(row['filename']).stem}"
                    f"_aug_{variant_index:03d}.png"
                )
                with Image.open(source_path) as source_image:
                    augmented = augment_image(source_image.convert("RGB"), rng)
                augmented.save(staged_images_dir / output_name)

                augmented_record = dict(row)
                augmented_record["filename"] = f"augmented/{output_name}"
                new_records.append(augmented_record)

        staged_manifest = temporary_dir / manifest_path.name
        with staged_manifest.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records + new_records)

        staged_manifest.replace(manifest_path)
        try:
            staged_images_dir.replace(augmented_dir)
        except OSError:
            manifest_path.unlink()
            raise

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Balance Mixed training images without changing the original split."
    )
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = augment_mixed_training_set(
        args.data_dir,
        target_count=args.target_count,
        seed=args.seed,
    )
    print(f"Augmented split written to: {manifest_path}")


if __name__ == "__main__":
    main()
