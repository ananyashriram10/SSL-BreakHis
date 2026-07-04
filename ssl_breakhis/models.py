from __future__ import annotations

import copy
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False, include_pool: bool = True):
        super().__init__()
        if name != "resnet18":
            raise ValueError("Only resnet18 is currently wired for this project.")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        children = list(resnet.children())[:-1 if include_pool else -2]
        self.features = nn.Sequential(*children)
        self.out_dim = resnet.fc.in_features
        self.include_pool = include_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1) if self.include_pool else x


class SimMIMResNet(nn.Module):
    def __init__(self, mask_ratio: float = 0.4, patch_size: int = 32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder = ResNetEncoder(include_pool=False)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def create_mask(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        patch = self.patch_size
        grid_h, grid_w = height // patch, width // patch
        num_patches = grid_h * grid_w
        num_masked = max(1, int(num_patches * self.mask_ratio))
        patch_mask = torch.ones(batch_size, num_patches, device=device)
        noise = torch.rand(batch_size, num_patches, device=device)
        masked_indices = noise.argsort(dim=1)[:, :num_masked]
        patch_mask.scatter_(1, masked_indices, 0.0)
        mask = patch_mask.view(batch_size, 1, grid_h, grid_w)
        return F.interpolate(mask, size=(height, width), mode="nearest")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = x.shape
        mask = self.create_mask(x.size(0), height, width, x.device)
        masked_x = x * mask
        features = self.encoder(masked_x)
        reconstruction = self.decoder(features)
        return reconstruction, x, mask


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_region = (1.0 - mask).expand_as(target)
    loss = (pred - target).abs() * masked_region
    return loss.sum() / masked_region.sum().clamp_min(1.0)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLModel(nn.Module):
    def __init__(self, encoder: ResNetEncoder):
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = MLPHead(encoder.out_dim)
        self.predictor = MLPHead(256, 1024, 256)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.online_projector(self.online_encoder(x)))

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_projector(self.target_encoder(x)).detach()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def byol_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target, dim=1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained, include_pool=True)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def load_simmim_encoder(encoder: ResNetEncoder, checkpoint_path: str, map_location="cpu") -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model", checkpoint)
    encoder_state = OrderedDict()
    for key, value in state.items():
        if key.startswith("encoder.features."):
            encoder_state[key.replace("encoder.", "", 1)] = value
        elif key.startswith("encoder."):
            encoder_state[key.replace("encoder.", "features.", 1)] = value
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    critical_missing = [key for key in missing if key.startswith("features.")]
    if critical_missing or unexpected:
        raise RuntimeError(
            f"Could not load SimMIM encoder cleanly. Missing={critical_missing}, unexpected={unexpected}"
        )


def load_byol_encoder(classifier: ResNetClassifier, checkpoint_path: str, map_location="cpu") -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model", checkpoint)
    encoder_state = OrderedDict()
    for key, value in state.items():
        if key.startswith("online_encoder."):
            encoder_state[key.replace("online_encoder.", "", 1)] = value
        elif key.startswith("encoder."):
            encoder_state[key.replace("encoder.", "", 1)] = value
        elif key.startswith("features."):
            encoder_state[key] = value
        elif key.split(".", 1)[0].isdigit():
            encoder_state[f"features.{key}"] = value
    missing, unexpected = classifier.encoder.load_state_dict(encoder_state, strict=False)
    critical_missing = [key for key in missing if key.startswith("features.")]
    if critical_missing or unexpected:
        raise RuntimeError(f"Could not load BYOL encoder. Missing={critical_missing}, unexpected={unexpected}")
