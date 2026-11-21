import torch.nn as nn


MODEL_NAMES = {
    "resnet18": None,
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "tinyvit_11m": "tiny_vit_11m_224.in1k",
    "tinyvit_21m": "tiny_vit_21m_224.in1k",
}


def create_model(model_key, pretrained=True, num_classes=3):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}")

    if model_key == "resnet18":
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    import timm

    return timm.create_model(
        MODEL_NAMES[model_key],
        pretrained=pretrained,
        num_classes=num_classes,
    )


def _classifier(model, model_key):
    if model_key == "resnet18":
        return model.fc
    if model_key in ("tinyvit_11m", "tinyvit_21m"):
        return model.head.fc
    return model.head


def _fine_tune_modules(model, model_key):
    if model_key == "resnet18":
        return [model.layer4[-1]]
    if model_key == "deit_tiny":
        return list(model.blocks[-2:])
    return [model.stages[-1]]


def configure_trainable_parameters(
    model, model_key, mode, backbone_lr=None, head_lr=1e-4
):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key: {model_key}")
    if mode not in ("head_only", "finetune"):
        raise ValueError(f"Unknown training mode: {mode}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = _classifier(model, model_key)
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    groups = []
    if mode == "finetune":
        if backbone_lr is None:
            raise ValueError("backbone_lr is required for fine-tuning")
        feature_modules = _fine_tune_modules(model, model_key)
        for module in feature_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        groups.extend(
            {
                "params": list(module.parameters()),
                "lr": backbone_lr,
            }
            for module in feature_modules
        )

    groups.append(
        {
            "params": list(classifier.parameters()),
            "lr": head_lr,
        }
    )
    model.eval()
    return groups


def set_training_mode(model, model_key):
    model.eval()
    if model_key in ("tinyvit_11m", "tinyvit_21m"):
        for block in model.stages[-1].blocks:
            block.attn.train()
