from pathlib import Path

import torch


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def predict(model, loader, device, model_key=None):
    model.eval()
    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_true.extend(labels.tolist())
            if model_key == "dinov2_vitb14" and outputs.shape[1] == 2:
                from sem_fracture.dinov2_ordinal import class_index_from_ordinal_logits

                predicted = class_index_from_ordinal_logits(outputs)
            else:
                predicted = outputs.argmax(dim=1)
            y_pred.extend(predicted.cpu().tolist())

    return y_true, y_pred


def evaluate_predictions(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of labels")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set")

    confusion = [[0] * 3 for _ in range(3)]
    for actual, predicted in zip(y_true, y_pred):
        if actual not in range(3) or predicted not in range(3):
            raise ValueError("Labels must be class indices 0, 1, or 2")
        confusion[actual][predicted] += 1

    precision = []
    recall = []
    class_f1 = []
    for class_index in range(3):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row][class_index]
            for row in range(3)
            if row != class_index
        )
        false_negative = sum(
            confusion[class_index][column]
            for column in range(3)
            if column != class_index
        )
        precision_value = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall_value = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_value = (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )
        precision.append(precision_value)
        recall.append(recall_value)
        class_f1.append(f1_value)

    return {
        "confusion_matrix": confusion,
        "accuracy": sum(confusion[index][index] for index in range(3)) / len(y_true),
        "precision": precision,
        "recall": recall,
        "class_f1": class_f1,
        "macro_f1": sum(class_f1) / 3,
    }
