import csv
from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from augmentation import augment_mixed_training_set


CLASSES = ("Brittle", "Ductile", "Mixed")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
SPLIT_COLUMNS = {"filename", "label", "group", "split"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(data_dir, split_file):
    split_file = Path(split_file)
    if not split_file.is_absolute():
        split_file = Path(data_dir) / split_file

    with split_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not SPLIT_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                "Split file must contain filename, label, group, and split columns"
            )
        return list(reader)


def compute_class_weights(records):
    training_counts = Counter(
        row["label"] for row in records if row["split"] == "train"
    )
    sample_count = sum(training_counts.values())
    if not sample_count or any(training_counts[label] == 0 for label in CLASSES):
        raise ValueError("Training records must contain every class")
    return torch.tensor(
        [
            sample_count / (len(CLASSES) * training_counts[label])
            for label in CLASSES
        ],
        dtype=torch.float32,
    )


def prepare_training_split(data_dir, augment_mixed=True, target_count=None, seed=42):
    data_dir = Path(data_dir)
    original_split = data_dir / "dataset_split.csv"
    if not augment_mixed:
        return original_split

    augmented_split = data_dir / "dataset_split_augmented.csv"
    augmented_dir = data_dir / "Mixed" / "augmented"
    if augmented_split.exists() and augmented_dir.is_dir():
        original_records = load_split(data_dir, original_split)
        records = load_split(data_dir, augmented_split)
        if records[: len(original_records)] != original_records:
            raise ValueError("Augmented manifest does not preserve the original split")

        generated = records[len(original_records) :]
        training_counts = Counter(
            row["label"] for row in original_records if row["split"] == "train"
        )
        mixed_groups = {
            row["group"]
            for row in original_records
            if row["label"] == "Mixed" and row["split"] == "train"
        }
        expected_count = (
            target_count if target_count is not None else max(training_counts.values())
        )
        if training_counts["Mixed"] + len(generated) != expected_count:
            raise ValueError("Existing augmented manifest does not match target_count")

        generated_names = set()
        for row in generated:
            relative_path = Path(row["filename"])
            if (
                row["label"] != "Mixed"
                or row["split"] != "train"
                or row["group"] not in mixed_groups
                or relative_path.parts[:1] != ("augmented",)
                or len(relative_path.parts) != 2
            ):
                raise ValueError("Augmented manifest contains an invalid generated row")
            generated_names.add(relative_path.name)

        if len(generated_names) != len(generated):
            raise ValueError("Augmented manifest contains duplicate generated images")
        for name in generated_names:
            if not (augmented_dir / name).is_file():
                raise FileNotFoundError(augmented_dir / name)

        actual_names = {
            path.name for path in augmented_dir.iterdir() if path.is_file()
        }
        if actual_names != generated_names:
            raise ValueError("Augmentation directory does not match its manifest")
        return augmented_split

    if augmented_split.exists() or augmented_dir.exists():
        raise FileExistsError(
            "Augmented split and image directory must either both exist or both be absent"
        )

    return augment_mixed_training_set(
        data_dir,
        target_count=target_count,
        seed=seed,
    )


class SEMDataset(Dataset):
    def __init__(self, records, data_dir, transform=None):
        self.records = records
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if row["label"] not in CLASS_TO_INDEX:
            raise ValueError(f"Unknown class label: {row['label']}")

        image_path = self.data_dir / row["label"] / row["filename"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, CLASS_TO_INDEX[row["label"]]


def build_transforms():
    from torchvision import transforms

    resize = transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    train_transform = transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform
