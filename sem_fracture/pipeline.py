from collections import Counter
import csv
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from sem_fracture.data import (
    BAG_TRAIN_VARIANTS,
    CLASSES,
    CLASS_TO_INDEX,
    CachedBagDataset,
    SEMDataset,
    SEMBagDataset,
    build_transforms,
    compute_class_weights,
    load_split,
    make_bag,
    prepare_training_split,
)
from sem_fracture.losses import (
    ClassBalancedCrossEntropyLoss,
    WeightedBinaryEvidenceLoss,
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


def _save_training_artifacts(checkpoint_path, history, best_checkpoint, class_weights):
    artifact_dir = Path(checkpoint_path).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training history CSV
    history_csv = artifact_dir / "training_history.csv"
    epochs = len(history.get("train_loss", []))
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        for ep in range(epochs):
            writer.writerow([
                ep + 1,
                history["train_loss"][ep],
                history["val_loss"][ep],
                history["val_accuracy"][ep],
                history["val_macro_f1"][ep],
            ])

    # 2. Experiment summary JSON
    summary_json = artifact_dir / "experiment_summary.json"
    summary_data = {
        "best_epoch": best_checkpoint.get("epoch") if best_checkpoint else None,
        "val_macro_f1": best_checkpoint.get("val_macro_f1") if best_checkpoint else None,
        "val_accuracy": best_checkpoint.get("val_accuracy") if best_checkpoint else None,
        "class_weights": class_weights,
        "history": history,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)


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


def _extract_bag_features(model, records, data_dir, variants, device, patch_batch_size=16):
    model.eval()
    variant_tensors = []
    for variant in variants:
        features = []
        for row in records:
            image_path = Path(data_dir) / row["label"] / row["filename"]
            with Image.open(image_path) as source:
                bag = make_bag(source, variant=variant)
            patch_vectors = []
            for batch in bag.split(patch_batch_size, dim=0):
                z = model.extract_embeddings(batch.to(device)).cpu().float()
                patch_vectors.append(z)
            features.append(torch.cat(patch_vectors, dim=0))
        variant_tensors.append(torch.stack(features, dim=0).contiguous())
    return torch.stack(variant_tensors, dim=1)


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
    loss_type="cross_entropy",
    cb_beta=0.999,
):
    data_dir = Path(data_dir).resolve()
    checkpoint_path = _resolve_path(data_dir, checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

    model = create_model(model_key, pretrained=initial_checkpoint is None)
    if initial_checkpoint is not None:
        initial_checkpoint = _resolve_path(data_dir, initial_checkpoint)
        target_module = (
            model.backbone
            if model_key == "tinyvit_21m_dual_attention"
            else model
        )
        load_checkpoint(target_module, initial_checkpoint, device)
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
    training_records = [row for row in records if row["split"] == "train"]
    val_records = [row for row in records if row["split"] == "validation"]
    test_records = [row for row in records if row["split"] == "test"]

    if model_key == "tinyvit_21m_dual_attention":
        train_features = _extract_bag_features(
            model, training_records, data_dir, BAG_TRAIN_VARIANTS, device
        )
        val_features = _extract_bag_features(
            model, val_records, data_dir, ("none",), device
        )
        datasets = {
            "train": CachedBagDataset(
                train_features,
                [CLASS_TO_INDEX[row["label"]] for row in training_records],
                augment=True,
            ),
            "validation": CachedBagDataset(
                val_features,
                [CLASS_TO_INDEX[row["label"]] for row in val_records],
                augment=False,
            ),
            "test": SEMBagDataset(test_records, data_dir, variants=("none",)),
        }
    else:
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

    class_weights = compute_class_weights(records)
    if model_key == "tinyvit_21m_dual_attention":
        criterion = WeightedBinaryEvidenceLoss(class_weights).to(device)
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif loss_type == "class_balanced":
        training_counts = Counter(row["label"] for row in training_records)
        class_counts = [training_counts[label] for label in CLASSES]
        criterion = ClassBalancedCrossEntropyLoss(
            class_counts=class_counts,
            beta=cb_beta,
        ).to(device)
        class_weights = criterion.weight.detach().cpu()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

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
    _save_training_artifacts(
        checkpoint_path=checkpoint_path,
        history=result.get("history", {}),
        best_checkpoint=result.get("best_checkpoint", {}),
        class_weights=class_weights.tolist(),
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

    if model_key == "tinyvit_21m_dual_attention":
        dataset = SEMBagDataset(records, data_dir, variants=("none",))
    else:
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
    metrics = evaluate_predictions(y_true, y_pred)
    predictions = [
        {
            "filename": row["filename"],
            "actual": row["label"],
            "predicted": CLASSES[predicted],
            "correct": row["label"] == CLASSES[predicted],
        }
        for row, predicted in zip(records, y_pred)
    ]

    artifact_dir = checkpoint_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction audit CSV for this split
    predictions_csv = artifact_dir / f"{split}_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "group", "actual", "predicted", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row, pred in zip(records, predictions):
            writer.writerow({
                "filename": row.get("filename", ""),
                "group": row.get("group", ""),
                "actual": pred["actual"],
                "predicted": pred["predicted"],
                "correct": pred["correct"],
            })

    # 2. Split metrics CSV
    metrics_csv = artifact_dir / f"{split}_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "class", "value"])
        writer.writerow(["accuracy", "overall", metrics["accuracy"]])
        writer.writerow(["macro_f1", "overall", metrics["macro_f1"]])
        for idx, cls_name in enumerate(CLASSES):
            writer.writerow(["precision", cls_name, metrics["precision"][idx]])
            writer.writerow(["recall", cls_name, metrics["recall"][idx]])
            writer.writerow(["class_f1", cls_name, metrics["class_f1"][idx]])

    # 3. Split evaluation JSON
    eval_json = artifact_dir / f"{split}_evaluation.json"
    eval_data = {
        "model_key": model_key,
        "checkpoint": str(checkpoint_path),
        "split": split,
        "split_file": str(split_path),
        "metrics": metrics,
    }
    with eval_json.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4)

    return {
        "checkpoint": str(checkpoint_path),
        "predictions": predictions,
        "metrics": metrics,
    }
