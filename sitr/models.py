"""
Stage 1 base segmentation models, and the Stage 2 reliability estimator.

Three encoders are available. They are NOT interchangeable in what they are for:

  unet11         TernausNet-11 (U-Net + ImageNet VGG11). The 2018 EndoVis
                 baseline, cited as [22] in the report. Use this when the point
                 is comparability with the report's Baseline A — a stronger
                 backbone makes any gain from selective recovery unattributable.
  resnet34_unet  DEFAULT. Best accuracy-per-FLOP here: five clean skip levels
                 including a /2 skip, which is what keeps 2-4 px clasper tips
                 alive. ~24M params, faster than unet11 despite being stronger.
  convnext_unet  Strongest, ~30M params, ~1.7x the cost. ConvNeXt's /4 stem
                 discards full-resolution detail, so a separate full-res stem
                 branch is added to recover it; without that it loses to
                 resnet34_unet on thin structures despite better ImageNet
                 numbers.

Every model returns (logits, features) where `features` is the final decoder map
at input resolution. Stage 2 needs it for the instrument embedding E_t of
eq. (9), so exposing it here avoids a second backbone pass at inference: the
embedding is free, which is what keeps c_rel negligible in eq. (2).
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURES = ("unet11", "resnet34_unet", "convnext_unet")


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class ConvRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bn: bool = False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample by 2, concatenate the skip, then two convs.

    Bilinear upsample + conv rather than a transposed conv: transposed convs
    leave checkerboard artefacts along thin structures, which is exactly where
    this task is decided.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bn: bool = True):
        super().__init__()
        self.reduce = ConvRelu(in_ch, out_ch, bn)
        self.fuse = nn.Sequential(ConvRelu(out_ch + skip_ch, out_ch, bn),
                                  ConvRelu(out_ch, out_ch, bn))

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


def _init_decoder(*modules: nn.Module) -> None:
    for m in modules:
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)


def _pretrained(fn, weights_enum_name: str, pretrained: bool):
    """Load a torchvision backbone, degrading gracefully with no network."""
    import torchvision

    if not pretrained:
        return fn(weights=None)
    try:
        enum = getattr(torchvision.models, weights_enum_name)
        return fn(weights=enum.DEFAULT)
    except Exception as exc_new:  # noqa: BLE001
        try:
            return fn(pretrained=True)
        except Exception as exc_old:  # noqa: BLE001
            warnings.warn(
                f"could not load pretrained weights ({exc_new!r} / {exc_old!r}); "
                "falling back to random initialisation. Expect a large drop in "
                "Stage 1 IoU — 2235 frames is far too few to train an encoder "
                "from scratch.", RuntimeWarning)
            return fn(weights=None)


# --------------------------------------------------------------------------- #
# TernausNet-11 (report baseline)
# --------------------------------------------------------------------------- #
class UNet11(nn.Module):
    """U-Net with an ImageNet VGG11 encoder (Shvets et al.)."""

    feature_dim = 32

    def __init__(self, num_classes: int, num_filters: int = 32,
                 pretrained: bool = True):
        super().__init__()
        import torchvision
        enc = _pretrained(torchvision.models.vgg11, "VGG11_Weights", pretrained).features

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1, self.conv2 = enc[0], enc[3]
        self.conv3s, self.conv3 = enc[6], enc[8]
        self.conv4s, self.conv4 = enc[11], enc[13]
        self.conv5s, self.conv5 = enc[16], enc[18]

        nf = num_filters
        self.center = UpBlock(nf * 16, nf * 16, nf * 8, bn=False)
        self.dec5 = UpBlock(nf * 8, nf * 16, nf * 8, bn=False)
        self.dec4 = UpBlock(nf * 8, nf * 8, nf * 4, bn=False)
        self.dec3 = UpBlock(nf * 4, nf * 4, nf * 2, bn=False)
        self.dec2 = UpBlock(nf * 2, nf * 2, nf, bn=False)
        self.dec1 = ConvRelu(nf, nf)
        self.final = nn.Conv2d(nf, num_classes, 1)
        _init_decoder(self.center, self.dec5, self.dec4, self.dec3, self.dec2,
                      self.dec1, self.final)

    def forward(self, x):
        c1 = self.relu(self.conv1(x))                      # 64,  H
        c2 = self.relu(self.conv2(self.pool(c1)))          # 128, H/2
        c3 = self.relu(self.conv3(self.relu(self.conv3s(self.pool(c2)))))   # 256, H/4
        c4 = self.relu(self.conv4(self.relu(self.conv4s(self.pool(c3)))))   # 512, H/8
        c5 = self.relu(self.conv5(self.relu(self.conv5s(self.pool(c4)))))   # 512, H/16

        d = self.center(self.pool(c5), c5)                 # 256, H/16
        d = self.dec5(d, c4)                               # 256, H/8
        d = self.dec4(d, c3)                               # 128, H/4
        d = self.dec3(d, c2)                               # 64,  H/2
        d = self.dec2(d, c1)                               # 32,  H
        feat = self.dec1(d)
        return self.final(feat), feat


# --------------------------------------------------------------------------- #
# ResNet-34 U-Net (default)
# --------------------------------------------------------------------------- #
class ResNetUNet(nn.Module):
    """U-Net with a ResNet-34 encoder and five skip levels (/2 … /32).

    The /2 skip is the reason this is the default: clasper tips are 2-4 px wide
    at 256x320, and a decoder that only reaches /4 before the final upsample
    cannot put them back.
    """

    feature_dim = 32

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        import torchvision
        net = _pretrained(torchvision.models.resnet34, "ResNet34_Weights", pretrained)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # 64,  /2
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2         # 64 /4, 128 /8
        self.layer3, self.layer4 = net.layer3, net.layer4         # 256 /16, 512 /32

        self.dec4 = UpBlock(512, 256, 256)     # /32 -> /16
        self.dec3 = UpBlock(256, 128, 128)     # /16 -> /8
        self.dec2 = UpBlock(128, 64, 64)       # /8  -> /4
        self.dec1 = UpBlock(64, 64, 48)        # /4  -> /2
        self.dec0 = UpBlock(48, 0, 32)         # /2  -> /1
        self.final = nn.Conv2d(32, num_classes, 1)
        _init_decoder(self.dec4, self.dec3, self.dec2, self.dec1, self.dec0,
                      self.final)

    def forward(self, x):
        s0 = self.stem(x)                 # 64,  H/2
        s1 = self.layer1(self.pool(s0))   # 64,  H/4
        s2 = self.layer2(s1)              # 128, H/8
        s3 = self.layer3(s2)              # 256, H/16
        s4 = self.layer4(s3)              # 512, H/32

        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        feat = self.dec0(d, None)         # 32, H
        return self.final(feat), feat


# --------------------------------------------------------------------------- #
# ConvNeXt-Tiny U-Net (strongest)
# --------------------------------------------------------------------------- #
class ConvNeXtUNet(nn.Module):
    """U-Net with a ConvNeXt-Tiny encoder plus a full-resolution stem branch.

    ConvNeXt's patchify stem downsamples by 4 immediately, so the encoder never
    sees full-resolution detail. The extra `hi` branch (two cheap convs on the
    raw image) supplies the /1 and /2 skips the decoder needs; without it this
    model loses to resnet34_unet on thin instruments despite stronger ImageNet
    features.
    """

    feature_dim = 32

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        import torchvision
        net = _pretrained(torchvision.models.convnext_tiny,
                          "ConvNeXt_Tiny_Weights", pretrained)
        f = net.features
        self.stage1 = nn.Sequential(f[0], f[1])    # 96,  /4
        self.stage2 = nn.Sequential(f[2], f[3])    # 192, /8
        self.stage3 = nn.Sequential(f[4], f[5])    # 384, /16
        self.stage4 = nn.Sequential(f[6], f[7])    # 768, /32

        self.hi1 = ConvRelu(3, 32, bn=True)                        # /1
        self.hi2 = nn.Sequential(nn.MaxPool2d(2, 2), ConvRelu(32, 48, bn=True))  # /2

        self.dec3 = UpBlock(768, 384, 256)    # /32 -> /16
        self.dec2 = UpBlock(256, 192, 160)    # /16 -> /8
        self.dec1 = UpBlock(160, 96, 96)      # /8  -> /4
        self.dec0 = UpBlock(96, 48, 48)       # /4  -> /2
        self.dec_full = UpBlock(48, 32, 32)   # /2  -> /1
        self.final = nn.Conv2d(32, num_classes, 1)
        _init_decoder(self.hi1, self.hi2, self.dec3, self.dec2, self.dec1,
                      self.dec0, self.dec_full, self.final)

    def forward(self, x):
        h1 = self.hi1(x)                  # 32, H
        h2 = self.hi2(h1)                 # 48, H/2
        s1 = self.stage1(x)               # 96,  H/4
        s2 = self.stage2(s1)              # 192, H/8
        s3 = self.stage3(s2)              # 384, H/16
        s4 = self.stage4(s3)              # 768, H/32

        d = self.dec3(s4, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        d = self.dec0(d, h2)
        feat = self.dec_full(d, h1)       # 32, H
        return self.final(feat), feat


# --------------------------------------------------------------------------- #
def build_model(num_classes: int, arch: str = "resnet34_unet",
                pretrained: bool = True, num_filters: int = 32) -> nn.Module:
    arch = (arch or "resnet34_unet").lower()
    if arch == "unet11":
        return UNet11(num_classes, num_filters, pretrained)
    if arch == "resnet34_unet":
        return ResNetUNet(num_classes, pretrained)
    if arch == "convnext_unet":
        return ConvNeXtUNet(num_classes, pretrained)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHITECTURES}")


def encoder_param_names(arch: str) -> Tuple[str, ...]:
    """Top-level attribute names holding PRETRAINED weights, for lr scaling."""
    arch = (arch or "resnet34_unet").lower()
    if arch == "unet11":
        return ("conv1", "conv2", "conv3s", "conv3", "conv4s", "conv4",
                "conv5s", "conv5")
    if arch == "resnet34_unet":
        return ("stem", "layer1", "layer2", "layer3", "layer4")
    if arch == "convnext_unet":
        return ("stage1", "stage2", "stage3", "stage4")
    raise ValueError(f"unknown arch {arch!r}")


@torch.no_grad()
def predict_with_tta(model: nn.Module, images: torch.Tensor,
                     hflip: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal-flip test-time augmentation, averaged in probability space.

    Worth ~1-2 IoU for free accuracy, but it DOUBLES c_seg in eq. (2), which is
    the cost the whole selectivity argument is trying to protect. Off by default
    and never used when timing the system.
    """
    logits, feat = model(images)
    probs = torch.softmax(logits.float(), dim=1)
    if hflip:
        flip_logits, _ = model(torch.flip(images, dims=[3]))
        probs = 0.5 * (probs + torch.flip(torch.softmax(flip_logits.float(), 1),
                                          dims=[3]))
    return probs, feat


# --------------------------------------------------------------------------- #
# Stage 2 reliability estimator
# --------------------------------------------------------------------------- #
class ReliabilityMLP(nn.Module):
    """r_t = sigma(f_theta(indicators)) — eq. (11), generalised.

    Deliberately tiny: its cost c_rel is paid on EVERY frame (eq. 2), so it must
    stay negligible next to c_seg. With the extended 9-indicator set this is
    still ~5k parameters over 9 scalars — microseconds per frame.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)

    @torch.no_grad()
    def predict(self, feats: Sequence[float]) -> float:
        x = torch.as_tensor(feats, dtype=torch.float32).view(1, -1)
        return float(self.forward(x).item())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
