import math

import torch
import torch.nn as nn
from torchvision import models


class ECALayer(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECAResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.eca1 = ECALayer(256)
        self.layer2 = backbone.layer2
        self.eca2 = ECALayer(512)
        self.layer3 = backbone.layer3
        self.eca3 = ECALayer(1024)
        self.layer4 = backbone.layer4
        self.eca4 = ECALayer(2048)
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.eca1(self.layer1(x))
        x = self.eca2(self.layer2(x))
        x = self.eca3(self.layer3(x))
        x = self.eca4(self.layer4(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class BaselineResNet50(nn.Module):
    """Plain ResNet50 without ECA attention."""

    def __init__(self, num_classes=7):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def create_model(arch_name, num_classes=7):
    arch = str(arch_name).strip().lower()
    if arch in ("eca_resnet50", "eca", "ours"):
        return ECAResNet50(num_classes=num_classes)
    if arch in ("resnet50", "baseline_resnet50", "baseline"):
        return BaselineResNet50(num_classes=num_classes)
    raise ValueError(f"Unsupported model arch: {arch_name}")


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be dict")
    for key in ("state_dict", "model_state_dict", "model", "net"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


def _adapt_legacy_keys(state_dict):
    """
    Compatible with historical checkpoints:
    - remove DataParallel `module.` prefix
    - map legacy `resnet.*` keys to flattened keys (conv1/layer1/...)
    - ignore historical `resnet.fc.*` keys
    """
    adapted = {}
    legacy_mapped = 0
    ignored_keys = []

    for raw_key, tensor in state_dict.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("resnet.fc."):
            ignored_keys.append(raw_key)
            continue

        if key.startswith("resnet."):
            suffix = key[len("resnet."):]
            if suffix.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "avgpool.")):
                mapped_key = suffix
                if mapped_key not in adapted:
                    adapted[mapped_key] = tensor
                legacy_mapped += 1
                continue

        adapted[key] = tensor

    return adapted, legacy_mapped, ignored_keys


def load_state_dict_compatible(model, checkpoint_payload, strict=False):
    state_dict = _extract_state_dict(checkpoint_payload)
    adapted_state_dict, legacy_mapped, ignored_keys = _adapt_legacy_keys(state_dict)
    incompatible = model.load_state_dict(adapted_state_dict, strict=strict)
    return {
        "legacy_mapped": legacy_mapped,
        "ignored_keys": ignored_keys,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }
