import torch
import torch.nn as nn


MODEL_NAMES = {
"resnet18": None,
"deit_tiny": "deit_tiny_patch16_224.fb_in1k",
"tinyvit_11m": "tiny_vit_11m_224.in1k",
"tinyvit_21m": "tiny_vit_21m_224.in1k",
"tinyvit_21m_dual_attention": "tiny_vit_21m_224.in1k",
}


class DualAttentionMIL(nn.Module):
    """Dual-branch attention MIL head predicting Brittle and Ductile evidence logits."""

    def __init__(self, feature_dim=576, attention_hidden=128):
        super().__init__()
        self.brittle_attention = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.ductile_attention = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.brittle_head = nn.Linear(feature_dim, 1)
        self.ductile_head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        a_b = torch.softmax(self.brittle_attention(x).squeeze(-1), dim=1)
        a_d = torch.softmax(self.ductile_attention(x).squeeze(-1), dim=1)
        b_features = (x * a_b.unsqueeze(-1)).sum(dim=1)
        d_features = (x * a_d.unsqueeze(-1)).sum(dim=1)
        b_logit = self.brittle_head(b_features)
        d_logit = self.ductile_head(d_features)
        logits = torch.cat([b_logit, d_logit], dim=1)
        attention = torch.stack([a_b, a_d], dim=1)
        return logits, attention


def class_probabilities(binary_logits):
    """Convert [Brittle, Ductile] evidence logits to normalized 3-class probabilities."""
    p = torch.sigmoid(binary_logits)
    pb = p[:, 0]
    pd = p[:, 1]
    joint = torch.stack(
        [
            pb * (1.0 - pd),
            (1.0 - pb) * pd,
            pb * pd,
        ],
        dim=1,
    )
    return joint / joint.sum(dim=1, keepdim=True).clamp_min(1e-8)


class TinyViTDualAttentionModel(nn.Module):
    """Frozen TinyViT-21M backbone paired with a trainable DualAttentionMIL head."""

    def __init__(self, backbone, feature_dim=576, attention_hidden=128):
        super().__init__()
        self.backbone = backbone
        self.mil_head = DualAttentionMIL(
            feature_dim=feature_dim,
            attention_hidden=attention_hidden,
        )

    @torch.no_grad()
    def extract_embeddings(self, image_batch):
        self.backbone.eval()
        spatial = self.backbone.forward_features(image_batch)
        return self.backbone.forward_head(spatial, pre_logits=True)

    def forward(self, x):
        if x.ndim == 3:
            logits, _ = self.mil_head(x)
            return logits
        if x.ndim == 5:
            batch_size, n_patches, channels, height, width = x.shape
            flat = x.view(batch_size * n_patches, channels, height, width)
            embeddings = self.extract_embeddings(flat).view(batch_size, n_patches, -1)
            logits, _ = self.mil_head(embeddings)
            return class_probabilities(logits)
        raise ValueError(f"Expected 3D embeddings or 5D image bags, got shape {tuple(x.shape)}")


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

    backbone = timm.create_model(
        MODEL_NAMES[model_key],
        pretrained=pretrained,
        num_classes=num_classes,
    )
    if model_key == "tinyvit_21m_dual_attention":
        return TinyViTDualAttentionModel(
            backbone=backbone,
            feature_dim=backbone.head.fc.in_features,
            attention_hidden=128,
        )
    return backbone


def _classifier(model, model_key):
    if model_key == "resnet18":
        return model.fc
    if model_key in ("tinyvit_11m", "tinyvit_21m"):
        return model.head.fc
    if model_key == "tinyvit_21m_dual_attention":
        return model.mil_head
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
    if mode == "finetune" and model_key != "tinyvit_21m_dual_attention":
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
    elif model_key == "tinyvit_21m_dual_attention":
        model.mil_head.train()
