import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def compute_effective_number_weights(
    class_counts: Sequence[int],
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute normalized class weights using the Effective Number of Samples.

    Implements Cui et al. (2019):
        E_n = (1 - beta^n) / (1 - beta)
        w_c = (1 - beta) / (1 - beta^n_c)
        w_hat_c = w_c / mean(w)

    Args:
        class_counts: Positive integer sample counts per class from the training split.
        beta: Hyperparameter in [0, 1) controlling diminishing returns.

    Returns:
        1D float32 tensor of normalized class weights with mean 1.0.
    """
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("beta must be a real number in [0, 1)")
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")

    counts = list(class_counts)
    if not counts:
        raise ValueError("class_counts must not be empty")

    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"Each class count must be an integer, got {type(count).__name__}")
        if count <= 0:
            raise ValueError(f"Each class count must be strictly positive, got {count}")

    if beta == 0.0:
        return torch.ones(len(counts), dtype=torch.float32)

    # Use log1p / expm1 for numerical stability when beta is close to 1 or n is large:
    # 1 - beta^n = -expm1(n * log1p(beta - 1))
    log_beta = math.log1p(beta - 1.0)
    one_minus_beta = 1.0 - beta
    raw_weights = [
        one_minus_beta / (-math.expm1(count * log_beta))
        for count in counts
    ]

    mean_weight = sum(raw_weights) / len(raw_weights)
    normalized = [weight / mean_weight for weight in raw_weights]
    return torch.tensor(normalized, dtype=torch.float32)


class ClassBalancedCrossEntropyLoss(nn.Module):
    """Class-Balanced Cross-Entropy Loss based on Effective Number of Samples."""

    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.999,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        weights = compute_effective_number_weights(class_counts, beta=beta)
        self.register_buffer("weight", weights)
        self.beta = float(beta)
        self.reduction = reduction
        self._loss = nn.CrossEntropyLoss(weight=self.weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(logits, targets)


class BinaryCrossEntropyLoss(nn.Module):
    """BCE-with-logits on one-hot labels. Targets stay integer class indices."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"Expected logits with {self.num_classes} classes, got {tuple(logits.shape)}"
            )
        one_hot = torch.nn.functional.one_hot(targets.long(), self.num_classes).to(logits.dtype)
        return self.bce(logits, one_hot)


class WeightedBinaryEvidenceLoss(nn.Module):
    """Class-weighted binary cross-entropy over [Brittle, Ductile] evidence logits."""

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "binary_targets",
            torch.tensor(
                [
                    [1.0, 0.0],  # Brittle only
                    [0.0, 1.0],  # Ductile only
                    [1.0, 1.0],  # Mixed: both
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "class_weights",
            torch.as_tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = self.binary_targets[labels]
        two_losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ).mean(dim=1)
        weights = self.class_weights[labels]
        return (two_losses * weights).sum() / weights.sum().clamp_min(1e-8)
