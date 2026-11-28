"""DINOv2 ordinal classifier for brittle / mixed / ductile fracture surfaces.

This module is intentionally separate from ``sem_fracture.models`` and
``sem_fracture.training``. The shared trainer treats every 2-logit output as
dual-attention evidence (Brittle, Ductile, Mixed). Ordinal ductility uses a
different class order and a different decoder, so it cannot share that path
without changing existing checkpoints.

Class index order here is the physical ductility progression:

    0 Brittle -> [0, 0]
    1 Mixed   -> [1, 0]
    2 Ductile -> [1, 1]

That is not ``sem_fracture.data.CLASSES`` (Brittle, Ductile, Mixed).
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sem_fracture.data import IMAGENET_MEAN, IMAGENET_STD
from sem_fracture.training import classification_metrics


ORDINAL_CLASS_NAMES: tuple[str, ...] = ("brittle", "mixed", "ductile")
DISPLAY_CLASS_NAMES: tuple[str, ...] = ("Brittle", "Mixed", "Ductile")
ORDINAL_TARGETS = (
    (0.0, 0.0),  # brittle
    (1.0, 0.0),  # mixed
    (1.0, 1.0),  # ductile
)
# sem_fracture.data.CLASSES index order: Brittle, Ductile, Mixed.
CLASS_INDEX_ORDINAL_TARGETS = (
    (0.0, 0.0),  # 0 Brittle
    (1.0, 1.0),  # 1 Ductile
    (1.0, 0.0),  # 2 Mixed
)
# Bit-sum rank (brittle, mixed, ductile) -> CLASSES index.
RANK_TO_CLASS_INDEX = (0, 2, 1)
IMAGE_SIZE = 518
PATCH_SIZE = 14
HUB_REPO = "facebookresearch/dinov2"
HUB_MODEL = "dinov2_vitb14"
EMBED_DIM = 768


def encode_ordinal_targets(class_indices: torch.Tensor) -> torch.Tensor:
    """Map class indices 0, 1, 2 onto the 2-bit ductility codes."""
    table = class_indices.new_tensor(ORDINAL_TARGETS, dtype=torch.float32)
    if class_indices.dtype != torch.long:
        class_indices = class_indices.long()
    if class_indices.numel() and (
        int(class_indices.min()) < 0 or int(class_indices.max()) > 2
    ):
        raise ValueError("Ordinal class indices must be 0 (brittle), 1 (mixed), or 2 (ductile)")
    return table[class_indices]


def decode_ordinal_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Recover a ductility rank by summing sigmoid outputs that exceed ``threshold``.

    Rank 0 is brittle, 1 is mixed, and 2 is ductile.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits of shape (N, 2), got {tuple(logits.shape)}")
    return (torch.sigmoid(logits) > threshold).sum(dim=1)


def class_index_from_ordinal_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Map an ordinal rank onto ``CLASSES`` indices: Brittle 0, Ductile 1, Mixed 2."""
    rank = decode_ordinal_logits(logits, threshold)
    mapping = rank.new_tensor(RANK_TO_CLASS_INDEX)
    return mapping[rank]


class OrdinalBCEWithLogitsLoss(nn.Module):
    """``BCEWithLogitsLoss`` on ordinal codes looked up from class indices.

    The default table matches ImageFolder order (brittle, mixed, ductile).
    ``run_experiment`` passes ``CLASS_INDEX_ORDINAL_TARGETS`` so label 0, 1, 2
    stay Brittle, Ductile, Mixed.
    """

    def __init__(self, targets: Sequence[Sequence[float]] = ORDINAL_TARGETS) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.register_buffer(
            "ordinal_targets",
            torch.tensor(targets, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        if class_indices.dtype != torch.long:
            class_indices = class_indices.long()
        return self.bce(logits, self.ordinal_targets[class_indices])


class RandomRotate90:
    """Lossless rotation by 0, 90, 180, or 270 degrees."""

    def __call__(self, image: Image.Image) -> Image.Image:
        turns = random.randrange(4)
        if turns == 0:
            return image
        transpose = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )[turns - 1]
        return image.transpose(transpose)


class PadToSquare:
    """Pad the shorter side so a later square resize does not stretch the image."""

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        if width == height == side:
            return image
        canvas = Image.new("RGB", (side, side), (self.fill, self.fill, self.fill))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


def _require_patch_compatible_size(image_size: int) -> int:
    if image_size <= 0 or image_size % PATCH_SIZE != 0:
        raise ValueError(
            f"image_size must be a positive multiple of DINOv2 patch size {PATCH_SIZE}, got {image_size}"
        )
    return image_size


def build_dinov2_transforms(image_size: int = IMAGE_SIZE):
    """Train and eval transforms. Geometry only; no color jitter."""
    image_size = _require_patch_compatible_size(image_size)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    geometry = [
        PadToSquare(fill=0),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90(),
            *geometry,
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            *geometry,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


class RGBOrdinalImageFolder(ImageFolder):
    """ImageFolder fixed to brittle=0, mixed=1, ductile=2, with RGB conversion.

    ``ImageFolder`` would otherwise sort folders alphabetically
    (brittle, ductile, mixed), which does not match the ordinal codes.
    Grayscale SEM images are converted to 3-channel RGB before transforms.
    """

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        folder = Path(directory)
        discovered = {
            entry.name.lower(): entry.name
            for entry in folder.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        }
        missing = [name for name in ORDINAL_CLASS_NAMES if name not in discovered]
        if missing:
            raise FileNotFoundError(
                f"Missing class folders {missing} in {folder}. "
                f"Expected {list(ORDINAL_CLASS_NAMES)} (any capitalization)."
            )
        ordered = [discovered[name] for name in ORDINAL_CLASS_NAMES]
        class_to_idx = {name: index for index, name in enumerate(ordered)}
        return ordered, class_to_idx

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def load_dinov2_backbone(model_name: str = HUB_MODEL) -> nn.Module:
    """Load ``facebookresearch/dinov2`` ``dinov2_vitb14`` (or another hub entry)."""
    return torch.hub.load(HUB_REPO, model_name, trust_repo=True)


def _embed_dim(backbone: nn.Module) -> int:
    embed_dim = getattr(backbone, "embed_dim", None) or getattr(backbone, "num_features", None)
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError("DINOv2 backbone must expose a positive embed_dim or num_features")
    return embed_dim


class DinoV2OrdinalClassifier(nn.Module):
    """Pretrained DINOv2 backbone plus a classification head.

    ``num_outputs=2`` is the ordinal binary head. ``num_outputs=3`` is the
    standard three-class head used with cross-entropy losses.
    """

    def __init__(self, backbone: nn.Module, dropout: float = 0.3, num_outputs: int = 2) -> None:
        super().__init__()
        if num_outputs not in (2, 3):
            raise ValueError(f"DINOv2 head outputs must be 2 or 3, got {num_outputs}")
        self.backbone = backbone
        embed_dim = _embed_dim(backbone)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.head(features)


def build_dinov2_ordinal_model(
    backbone: nn.Module | None = None,
    dropout: float = 0.3,
    model_name: str = HUB_MODEL,
    num_outputs: int = 2,
) -> DinoV2OrdinalClassifier:
    if backbone is None:
        backbone = load_dinov2_backbone(model_name)
    return DinoV2OrdinalClassifier(backbone, dropout=dropout, num_outputs=num_outputs)


def configure_trainable_parameters(
    model: DinoV2OrdinalClassifier,
    freeze_backbone: bool = True,
    unfreeze_last_blocks: int = 0,
) -> list[dict]:
    """Freeze the backbone, optionally train its last transformer blocks.

    Returns AdamW parameter groups: head at the caller's head learning rate,
    and unfrozen backbone blocks at the backbone learning rate. Learning rates
    are filled in by ``build_adamw_optimizer``.
    """
    if unfreeze_last_blocks < 0:
        raise ValueError("unfreeze_last_blocks must be >= 0")
    if unfreeze_last_blocks and not freeze_backbone:
        raise ValueError(
            "unfreeze_last_blocks only applies when the backbone starts frozen"
        )
    blocks = list(getattr(model.backbone, "blocks", []))
    if unfreeze_last_blocks > len(blocks):
        raise ValueError(
            f"Cannot unfreeze {unfreeze_last_blocks} blocks; backbone has {len(blocks)}"
        )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = not freeze_backbone
    if freeze_backbone and unfreeze_last_blocks:
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    model.freeze_backbone = freeze_backbone
    model.unfreeze_last_blocks = unfreeze_last_blocks if freeze_backbone else 0

    groups: list[dict] = [{"params": [parameter for parameter in model.head.parameters() if parameter.requires_grad], "group": "head"}]
    backbone_params = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if backbone_params:
        groups.append({"params": backbone_params, "group": "backbone"})
    return groups


def set_training_mode(model: DinoV2OrdinalClassifier) -> None:
    """Keep frozen DINOv2 blocks in eval; train the head and any unfrozen blocks."""
    freeze_backbone = getattr(model, "freeze_backbone", True)
    unfreeze_last_blocks = getattr(model, "unfreeze_last_blocks", 0)
    if freeze_backbone:
        model.backbone.eval()
        blocks = list(getattr(model.backbone, "blocks", []))
        for block in blocks[-unfreeze_last_blocks:]:
            block.train()
    else:
        model.backbone.train()
    model.head.train()


def build_adamw_optimizer(
    parameter_groups: Sequence[dict],
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    optimizer_groups = []
    for group in parameter_groups:
        name = group.get("group", "head")
        learning_rate = head_lr if name == "head" else backbone_lr
        optimizer_groups.append({"params": group["params"], "lr": learning_rate})
    return torch.optim.AdamW(optimizer_groups, weight_decay=weight_decay)


def make_ordinal_loaders(
    data_dir: Path | str,
    batch_size: int,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    train_transform, eval_transform = build_dinov2_transforms(image_size)
    datasets = {
        "train": RGBOrdinalImageFolder(data_dir / "train", transform=train_transform),
        "val": RGBOrdinalImageFolder(data_dir / "val", transform=eval_transform),
    }
    if datasets["train"].class_to_idx != datasets["val"].class_to_idx:
        raise ValueError("Train and val class_to_idx do not match")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "class_to_idx": datasets["train"].class_to_idx,
    }


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 3,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(y_true, y_pred):
        if actual not in range(num_classes) or predicted not in range(num_classes):
            raise ValueError(
                f"Labels must be in 0..{num_classes - 1}, got actual={actual} predicted={predicted}"
            )
        matrix[actual][predicted] += 1
    return matrix


def format_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str] = DISPLAY_CLASS_NAMES,
) -> str:
    width = max(len(name) for name in class_names)
    header = " " * (width + 2) + " ".join(f"{name:>{width}}" for name in class_names)
    lines = ["Validation confusion matrix (rows=actual, cols=predicted)", header]
    for name, row in zip(class_names, matrix):
        cells = " ".join(f"{count:>{width}d}" for count in row)
        lines.append(f"{name:<{width}}  {cells}")
    return "\n".join(lines)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict, list[int], list[int]]:
    training = optimizer is not None
    if training:
        set_training_mode(model)
    else:
        model.eval()

    loss_total = 0.0
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            sample_count += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(decode_ordinal_logits(logits.detach()).cpu().tolist())

    if not sample_count:
        raise ValueError("Loader is empty")
    metrics = classification_metrics(y_true, y_pred, num_classes=3)
    return loss_total / sample_count, metrics, y_true, y_pred


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / path.name
        torch.save(payload, temporary_path)
        temporary_path.replace(path)


def train_dinov2_ordinal(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    device: torch.device | str,
    epochs: int = 10,
    batch_size: int = 8,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    min_lr: float = 1e-7,
    freeze_backbone: bool = True,
    unfreeze_last_blocks: int = 0,
    dropout: float = 0.3,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = 42,
    model: DinoV2OrdinalClassifier | None = None,
    model_name: str = HUB_MODEL,
) -> dict:
    """Train and validate. Saves the best checkpoint by validation macro-F1."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    if model is None:
        model = build_dinov2_ordinal_model(dropout=dropout, model_name=model_name)
    model = model.to(device)
    parameter_groups = configure_trainable_parameters(
        model,
        freeze_backbone=freeze_backbone,
        unfreeze_last_blocks=unfreeze_last_blocks,
    )
    optimizer = build_adamw_optimizer(
        parameter_groups,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )
    criterion = OrdinalBCEWithLogitsLoss().to(device)
    loaders = make_ordinal_loaders(
        data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )
    print(f"class_to_idx={loaders['class_to_idx']}", flush=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_class_f1": [],
    }
    best_f1 = -1.0
    best_checkpoint = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} starting", flush=True)
        train_loss, train_metrics, _, _ = _run_epoch(
            model, loaders["train"], criterion, device, optimizer=optimizer
        )
        val_loss, val_metrics, y_true, y_pred = _run_epoch(
            model, loaders["val"], criterion, device, optimizer=None
        )
        scheduler.step()
        matrix = confusion_matrix(y_true, y_pred)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_class_f1"].append(val_metrics["class_f1"])

        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            best_checkpoint = {
                "model_name": model_name,
                "model_key": "dinov2_vitb14_ordinal",
                "model_state_dict": {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                },
                "epoch": epoch + 1,
                "val_macro_f1": best_f1,
                "val_accuracy": val_metrics["accuracy"],
                "class_names": list(DISPLAY_CLASS_NAMES),
                "class_to_idx": dict(loaders["class_to_idx"]),
                "ordinal_targets": list(ORDINAL_TARGETS),
                "image_size": image_size,
                "freeze_backbone": freeze_backbone,
                "unfreeze_last_blocks": unfreeze_last_blocks,
                "confusion_matrix": matrix,
            }
            _save_checkpoint(checkpoint_path, best_checkpoint)

        class_f1 = ", ".join(
            f"{name}={score:.4f}"
            for name, score in zip(DISPLAY_CLASS_NAMES, val_metrics["class_f1"])
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_accuracy={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_f1[{class_f1}] | "
            f"best_checkpoint={is_best}",
            flush=True,
        )
        print(format_confusion_matrix(matrix), flush=True)

    return {
        "history": history,
        "best_checkpoint": best_checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "class_to_idx": dict(loaders["class_to_idx"]),
    }


def evaluate_dinov2_ordinal(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    device: torch.device | str,
    split: str = "val",
    batch_size: int = 8,
    image_size: int | None = None,
    num_workers: int = 0,
    model: DinoV2OrdinalClassifier | None = None,
    model_name: str = HUB_MODEL,
) -> dict:
    """Load a checkpoint and report loss, accuracy, per-class F1, and confusion."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved_image_size = image_size or int(checkpoint.get("image_size", IMAGE_SIZE))
    if model is None:
        model = build_dinov2_ordinal_model(
            dropout=0.3,
            model_name=checkpoint.get("model_name", model_name),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_transform = build_dinov2_transforms(resolved_image_size)
    dataset = RGBOrdinalImageFolder(Path(data_dir) / split, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    criterion = OrdinalBCEWithLogitsLoss().to(device)
    loss, metrics, y_true, y_pred = _run_epoch(
        model, loader, criterion, device, optimizer=None
    )
    matrix = confusion_matrix(y_true, y_pred)
    print(
        f"split={split} | loss={loss:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"class_f1={metrics['class_f1']}",
        flush=True,
    )
    print(format_confusion_matrix(matrix), flush=True)
    return {"loss": loss, "metrics": metrics, "confusion_matrix": matrix, "checkpoint": checkpoint}
