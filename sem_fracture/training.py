from pathlib import Path
import tempfile
import time

import torch

from sem_fracture.models import MODEL_NAMES, class_probabilities, set_training_mode


def classification_metrics(y_true, y_pred, num_classes=3):
    support = [0] * num_classes
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes

    for actual, predicted in zip(y_true, y_pred):
        support[actual] += 1
        if actual == predicted:
            true_positive[actual] += 1
        else:
            false_positive[predicted] += 1
            false_negative[actual] += 1

    f1_scores = []
    for class_index in range(num_classes):
        denominator = (
            2 * true_positive[class_index]
            + false_positive[class_index]
            + false_negative[class_index]
        )
        f1_scores.append(
            2 * true_positive[class_index] / denominator if denominator else 0.0
        )

    accuracy = sum(true_positive) / len(y_true) if y_true else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / num_classes,
        "class_f1": f1_scores,
        "support": support,
    }


def _predicted_classes(outputs, model_key):
    if model_key == "dinov2_vitb14" and outputs.shape[1] == 2:
        from sem_fracture.dinov2_ordinal import class_index_from_ordinal_logits

        return class_index_from_ordinal_logits(outputs)
    if outputs.shape[1] == 2:
        return class_probabilities(outputs).argmax(dim=1)
    return outputs.argmax(dim=1)


def _evaluate(model, loader, criterion, device, model_key):
    model.eval()
    loss_total = 0.0
    sample_count = 0
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            sample_count += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(_predicted_classes(outputs, model_key).cpu().tolist())

    if not sample_count:
        raise ValueError("Evaluation loader is empty")
    metrics = classification_metrics(y_true, y_pred, num_classes=3)
    return loss_total / sample_count, metrics


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    model_key,
    epochs,
    checkpoint_path,
    patience=None,
    grad_clip_norm=None,
    scheduler=None,
):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    best_f1 = -1.0
    epochs_without_improvement = 0
    best_checkpoint = None

    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        print(f"Epoch {epoch + 1}/{epochs} starting", flush=True)
        set_training_mode(model, model_key)
        loss_total = 0.0
        sample_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=grad_clip_norm,
                )
            optimizer.step()

            batch_size = labels.size(0)
            loss_total += loss.item() * batch_size
            sample_count += batch_size

        if not sample_count:
            raise ValueError("Training loader is empty")
        train_loss = loss_total / sample_count
        val_loss, val_metrics = _evaluate(
            model, val_loader, criterion, device, model_key
        )
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            best_checkpoint = {
                "model_name": MODEL_NAMES[model_key] or model_key,
                "model_key": model_key,
                "model_state_dict": {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                },
                "epoch": epoch + 1,
                "val_macro_f1": best_f1,
                "val_accuracy": val_metrics["accuracy"],
            }
            with tempfile.TemporaryDirectory(
                prefix=".checkpoint-", dir=checkpoint_path.parent
            ) as temporary_dir:
                temporary_path = Path(temporary_dir) / checkpoint_path.name
                torch.save(best_checkpoint, temporary_path)
                temporary_path.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1

        elapsed = time.perf_counter() - epoch_started_at
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"elapsed={elapsed:.1f}s | "
            f"best_checkpoint={is_best}",
            flush=True,
        )

        if patience is not None and epochs_without_improvement >= patience:
            break

    return {"history": history, "best_checkpoint": best_checkpoint}
