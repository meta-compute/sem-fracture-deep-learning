from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sem_fracture.data import (
    CLASSES,
    SEMDataset,
    build_transforms,
    compute_class_weights,
    load_split,
    prepare_training_split,
)
from sem_fracture.models import (
    MODEL_NAMES,
    configure_trainable_parameters,
    create_model,
)
from sem_fracture.prediction import (
    evaluate_predictions,
    load_checkpoint,
    predict,
)
from sem_fracture.training import train_model


def _resolve_path(data_dir, path):
    path = Path(path)
    return path if path.is_absolute() else Path(data_dir) / path


def _split_datasets(records, data_dir):
    train_transform, eval_transform = build_transforms()
    return {
        "train": SEMDataset(
            [row for row in records if row["split"] == "train"],
            data_dir,
            train_transform,
        ),
        "validation": SEMDataset(
            [row for row in records if row["split"] == "validation"],
            data_dir,
            eval_transform,
        ),
        "test": SEMDataset(
            [row for row in records if row["split"] == "test"],
            data_dir,
            eval_transform,
        ),
    }


def run_experiment(
    data_dir,
    model_key,
    mode,
    checkpoint_path,
    device,
    head_lr=1e-3,
    backbone_lr=None,
    weight_decay=1e-2,
    epochs=10,
    patience=None,
    grad_clip_norm=None,
    initial_checkpoint=None,
    augment_mixed=True,
    target_count=None,
    seed=42,
    batch_size=16,
    lr_scheduler="cosine",
    min_lr=1e-7,
):
    data_dir = Path(data_dir).resolve()
    checkpoint_path = _resolve_path(data_dir, checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

    model = create_model(model_key, pretrained=initial_checkpoint is None)
    if initial_checkpoint is not None:
        initial_checkpoint = _resolve_path(data_dir, initial_checkpoint)
        load_checkpoint(model, initial_checkpoint, device)
    model = model.to(device)

    parameter_groups = configure_trainable_parameters(
        model,
        model_key,
        mode,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )

    split_file = prepare_training_split(
        data_dir,
        augment_mixed=augment_mixed,
        target_count=target_count,
        seed=seed,
    )
    records = load_split(data_dir, split_file)
    datasets = _split_datasets(records, data_dir)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    training_records = [row for row in records if row["split"] == "train"]
    class_weights = compute_class_weights(records)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )

    result = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_key=model_key,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        patience=patience,
        grad_clip_norm=grad_clip_norm,
        scheduler=scheduler,
    )
    result.update(
        {
            "class_weights": class_weights.tolist(),
            "training_images": len(training_records),
            "split_file": str(split_file),
            "test_loader": loaders["test"],
        }
    )
    return result


def run_prediction(
    data_dir,
    model_key,
    checkpoint_path,
    split="test",
    split_file="dataset_split.csv",
    device="cpu",
    batch_size=16,
):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}")
    data_dir = Path(data_dir).resolve()
    split_path = _resolve_path(data_dir, split_file)
    checkpoint_path = _resolve_path(data_dir, checkpoint_path)
    records = [
        row for row in load_split(data_dir, split_path) if row["split"] == split
    ]
    if not records:
        raise ValueError(f"No records found for split: {split}")

    _, eval_transform = build_transforms()
    dataset = SEMDataset(records, data_dir, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = create_model(model_key, pretrained=False)
    checkpoint = load_checkpoint(model, checkpoint_path, device)

    checkpoint_model = checkpoint.get("model_key")
    if checkpoint_model is not None and checkpoint_model != model_key:
        raise ValueError(
            f"Checkpoint model {checkpoint_model} does not match {model_key}"
        )
    expected_model_name = MODEL_NAMES[model_key] or model_key
    checkpoint_model_name = checkpoint.get("model_name")
    if (
        checkpoint_model_name is not None
        and checkpoint_model_name != expected_model_name
    ):
        raise ValueError(
            f"Checkpoint model {checkpoint_model_name} does not match "
            f"{expected_model_name}"
        )

    y_true, y_pred = predict(model, loader, device)
    predictions = [
        {
            "filename": row["filename"],
            "actual": row["label"],
            "predicted": CLASSES[predicted],
        }
        for row, predicted in zip(records, y_pred)
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "predictions": predictions,
        "metrics": evaluate_predictions(y_true, y_pred),
    }
