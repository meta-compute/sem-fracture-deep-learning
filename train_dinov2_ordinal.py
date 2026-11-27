"""Train or evaluate the DINOv2 ordinal fracture-surface classifier.

Dataset layout (folder names are matched case-insensitively):

    data_dir/
      train/brittle/  train/mixed/  train/ductile/
      val/brittle/    val/mixed/    val/ductile/

Class indices follow ductility, not alphabetical ImageFolder order and not
the Brittle/Ductile/Mixed order used by modelling-sem.ipynb:

    brittle -> 0 -> [0, 0]
    mixed   -> 1 -> [1, 0]
    ductile -> 2 -> [1, 1]

``modelling-sem.ipynb`` trains this model through ``run_experiment`` by setting
``MODEL_KEY = "dinov2_vitb14"``. ``head_only`` freezes the backbone.
``finetune`` trains the last two transformer blocks and expects the head-only
checkpoint listed in ``BASELINE_CHECKPOINTS``. This script is the ImageFolder
entry point for a ``train/`` and ``val/`` tree.

Examples:

    python train_dinov2_ordinal.py --data-dir data/ordinal
    python train_dinov2_ordinal.py --data-dir data/ordinal --unfreeze-last-blocks 2
    python train_dinov2_ordinal.py --data-dir data/ordinal --evaluate-only --checkpoint results/dinov2_vitb14_ordinal/best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sem_fracture.dinov2_ordinal import evaluate_dinov2_ordinal, train_dinov2_ordinal


CONFIG = {
    "data_dir": "data/ordinal",
    "checkpoint_path": "results/dinov2_vitb14_ordinal/dinov2_vitb14_ordinal_best.pth",
    "epochs": 10,
    "batch_size": 8,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "weight_decay": 1e-4,
    "min_lr": 1e-7,
    "freeze_backbone": True,
    "unfreeze_last_blocks": 0,
    "dropout": 0.3,
    "image_size": 518,
    "num_workers": 0,
    "seed": 42,
    "split": "val",
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a DINOv2 ordinal brittle/mixed/ductile classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(CONFIG["data_dir"]))
    parser.add_argument("--checkpoint", type=Path, default=Path(CONFIG["checkpoint_path"]))
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--head-lr", type=float, default=CONFIG["head_lr"])
    parser.add_argument("--backbone-lr", type=float, default=CONFIG["backbone_lr"])
    parser.add_argument("--weight-decay", type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--min-lr", type=float, default=CONFIG["min_lr"])
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["freeze_backbone"],
        help="Freeze DINOv2 weights. Use --no-freeze-backbone to train the full backbone.",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=CONFIG["unfreeze_last_blocks"],
        help="When the backbone is frozen, also train this many final transformer blocks (use 2 to unfreeze the last two).",
    )
    parser.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    parser.add_argument("--image-size", type=int, default=CONFIG["image_size"])
    parser.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate an existing checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=CONFIG["split"],
        help="Split used by --evaluate-only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = _device()
    print(f"device={device}", flush=True)
    if args.evaluate_only:
        return evaluate_dinov2_ordinal(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            device=device,
            split=args.split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )
    result = train_dinov2_ordinal(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        dropout=args.dropout,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    best = result["best_checkpoint"]
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"Best validation macro-F1: {best['val_macro_f1']:.4f}", flush=True)
    print(f"Checkpoint: {result['checkpoint_path']}", flush=True)
    return result


if __name__ == "__main__":
    main()
