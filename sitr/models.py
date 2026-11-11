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

ARCHITECTURES = ("unet11", "resnet34_unet", "convnext_unet",
                 "caformer_surgenet", "caformer_imagenet",
                 # Task P.1 baselines, trained under the identical protocol as the main model
                 "unet_classic", "deeplabv3p_r18", "segformer_b0")


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
# CAFormer-S18 + FPN (+ full-resolution refinement)          -- v2 default
# --------------------------------------------------------------------------- #
def _gn(ch: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class GNConv(nn.Module):
    """3x3 conv + GroupNorm + ReLU.

    GroupNorm, not BatchNorm: on an 8 GB GPU at 512x640 the batch is 2, and
    BatchNorm statistics over two images are noise.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                   _gn(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class CaFormerFPN(nn.Module):
    """CAFormer-S18 encoder, FPN decoder, and a full-resolution refinement path.

    Encoder weights:
      caformer_surgenet  SurgeNetXL: DINO self-supervised on 4.7M SURGICAL
                         frames (Jaspers et al., Medical Image Analysis 2025).
                         The authors report +14.4% on semantic segmentation
                         over ImageNet initialisation of the same network.
      caformer_imagenet  identical network, ImageNet-1k weights -- the controlled
                         comparison that isolates the effect of surgical
                         pretraining.

    Decoder: the FPN is the one SurgeNet was evaluated with (4 pyramid levels,
    256 ch, merged by concatenation at 1/4 resolution). An FPN alone outputs
    at 1/4 resolution, which blurs 2-4 px needle, thread and clasper tips, so
    two extra stages bring it back to full resolution with skips from a cheap
    full-resolution branch on the raw image -- the same fix convnext_unet uses.

    Returns (logits, features) with features = 32 channels at input resolution,
    exactly like every other model here, so Stages 2-5 are unchanged.
    """

    feature_dim = 32

    def __init__(self, num_classes: int, weights: str = None,
                 variant: str = "surgenet", pretrained_path: str = None,
                 drop_path_rate: float = 0.1, grad_checkpoint: bool = False):
        super().__init__()
        from .third_party.metaformer import caformer_s18
        from .pretrained import VARIANT_OF, load_encoder_weights
        if weights:
            variant = VARIANT_OF[weights]
        self.variant = variant
        self.encoder = caformer_s18(variant, drop_path_rate=drop_path_rate)
        self.encoder.head = nn.Identity()   # classifier: unused for segmentation
        self.encoder.norm = nn.Identity()   # pooled-output norm: unused too
        if weights:
            load_encoder_weights(self.encoder, weights, pretrained_path)
        self.grad_checkpoint = bool(grad_checkpoint)

        enc_ch, P, S = (64, 128, 320, 512), 256, 128
        self.lateral = nn.ModuleList([nn.Conv2d(c, P, 1) for c in enc_ch])
        # level i lives at 1/(4*2^i); bring each to 1/4 with i upsampling convs
        self.seg_blocks = nn.ModuleList()
        for i in range(4):
            blocks = [GNConv(P, S)]
            blocks += [GNConv(S, S) for _ in range(max(0, i - 1))]
            self.seg_blocks.append(nn.ModuleList(blocks))
        self.fuse = GNConv(4 * S, 128)                    # 1/4

        self.hi1 = GNConv(3, 32)                           # 1/1
        self.hi2 = nn.Sequential(nn.MaxPool2d(2, 2), GNConv(32, 48))  # 1/2
        self.up2 = GNConv(128 + 48, 64)                    # 1/2
        self.up1 = GNConv(64 + 32, 32)                     # 1/1 -> features
        self.final = nn.Conv2d(32, num_classes, 1)
        for m in [self.lateral, self.seg_blocks, self.fuse, self.hi1, self.hi2,
                  self.up2, self.up1, self.final]:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)

    def _encode(self, x):
        enc, feats, y = self.encoder, [], x
        for i in range(enc.num_stage):
            y = enc.downsample_layers[i](y)
            if self.grad_checkpoint and self.training:
                y = torch.utils.checkpoint.checkpoint(enc.stages[i], y,
                                                      use_reentrant=False)
            else:
                y = enc.stages[i](y)
            feats.append(y.permute(0, 3, 1, 2).contiguous())   # B,H,W,C -> B,C,H,W
        return feats

    def forward(self, x):
        H, W = x.shape[-2:]
        if H % 32 or W % 32:
            raise ValueError(f"input {H}x{W} must be a multiple of 32")
        c = self._encode(x)
        p = [None] * 4
        p[3] = self.lateral[3](c[3])
        for i in (2, 1, 0):
            p[i] = self.lateral[i](c[i]) + F.interpolate(
                p[i + 1], size=c[i].shape[-2:], mode="nearest")
        q_size = c[0].shape[-2:]
        outs = []
        for i in range(4):
            y = p[i]
            for j, blk in enumerate(self.seg_blocks[i]):
                y = blk(y)
                if i > 0 and j < i:          # upsample once per level step
                    y = F.interpolate(y, scale_factor=2, mode="bilinear",
                                      align_corners=False)
            if y.shape[-2:] != q_size:
                y = F.interpolate(y, size=q_size, mode="bilinear",
                                  align_corners=False)
            outs.append(y)
        d = self.fuse(torch.cat(outs, 1))                               # 1/4
        h1 = self.hi1(x)
        h2 = self.hi2(h1)
        d = F.interpolate(d, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.up2(torch.cat([d, h2], 1))                              # 1/2
        d = F.interpolate(d, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.up1(torch.cat([d, h1], 1))                           # 1/1
        return self.final(feat), feat

# --------------------------------------------------------------------------- #
# Task P.1 baselines. Every one returns (logits at input resolution, a feature map) like every other model here,
# so the same training, EMA, loss, optimiser and evaluation code runs on all of them unchanged. GroupNorm replaces
# BatchNorm wherever a block is built here, for the same reason as GNConv above (batch 1 with gradient accumulation).
# --------------------------------------------------------------------------- #
class _DoubleGN(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), _gn(o), nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1, bias=False), _gn(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class ClassicUNet(nn.Module):
    """The classic encoder-decoder U-Net (Ronneberger et al. 2015): four 2x downsamplings with doubled width, four
    upsamplings with skip concatenation, double 3x3 convolutions, 1x1 classifier. Base width 32.
    Trained FROM SCRATCH (no pretrained weights): it is the floor of the comparison."""

    feature_dim = 32

    def __init__(self, num_classes: int, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_DoubleGN(3, c[0])] + [_DoubleGN(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([_DoubleGN(c[i] * 2, c[i]) for i in range(4)])
        self.final = nn.Conv2d(c[0], num_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skips, y = [], x
        for i, blk in enumerate(self.enc):
            y = blk(y if i == 0 else F.max_pool2d(y, 2))
            skips.append(y)
        for i in (3, 2, 1, 0):
            y = self.up[i](y)
            y = self.dec[i](torch.cat([y, skips[i]], 1))
        return self.final(y), y


class DeepLabV3PlusR18(nn.Module):
    """DeepLabV3+ (Chen et al. 2018) with a ResNet-18 encoder, the backbone of STswinCL: ImageNet-pretrained ResNet-18 at output stride 16
    (layer4 dilated), ASPP (1x1, three 3x3 atrous branches at rates 6/12/18, image pooling; 256 channels), and the DeepLabV3+ decoder that
    fuses the 1/4-resolution layer1 features (reduced to 48 channels) with the upsampled ASPP output."""

    feature_dim = 256

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # output stride 16: layer4 keeps resolution and dilates its 3x3 convolutions
        for blk in r.layer4:
            for m in blk.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    m.dilation, m.padding = (2, 2), (2, 2)
        r.layer4[0].conv1.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        P = 256

        def br(rate):
            k = 1 if rate == 1 else 3
            return nn.Sequential(nn.Conv2d(512, P, k, padding=0 if rate == 1 else rate, dilation=rate, bias=False), _gn(P), nn.ReLU(inplace=True))

        self.aspp = nn.ModuleList([br(1), br(6), br(12), br(18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, P, 1, bias=False), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(5 * P, P, 1, bias=False), _gn(P), nn.ReLU(inplace=True))
        self.low = nn.Sequential(nn.Conv2d(64, 48, 1, bias=False), _gn(48), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(P + 48, P, 3, padding=1, bias=False), _gn(P), nn.ReLU(inplace=True),
                                  nn.Conv2d(P, P, 3, padding=1, bias=False), _gn(P), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(P, num_classes, 1)
        for m in [self.aspp, self.proj, self.low, self.fuse, self.final]:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        y = self.stem(x)
        l1 = self.layer1(y)
        y = self.layer4(self.layer3(self.layer2(l1)))
        a = [b(y) for b in self.aspp] + [F.interpolate(self.pool(y), size=y.shape[-2:], mode="bilinear", align_corners=False)]
        a = F.interpolate(self.proj(torch.cat(a, 1)), size=l1.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([a, self.low(l1)], 1))
        return F.interpolate(self.final(f), size=(H, W), mode="bilinear", align_corners=False), f


class SegFormerB0(nn.Module):
    """SegFormer-B0 (Xie et al. 2021; 3.7 M parameters), the MiT-B0 hierarchical transformer encoder with the all-MLP decoder, initialised from
    the ImageNet-1k encoder weights (nvidia/mit-b0). Logits come out at 1/4 resolution and are upsampled bilinearly to the input size."""

    feature_dim = 256

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=num_classes))
        self.encoder = net.segformer
        self.decode_head = net.decode_head

    def forward(self, x):
        H, W = x.shape[-2:]
        hs = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        lg = self.decode_head(hs)
        return F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False), lg


# --------------------------------------------------------------------------- #
def build_model(num_classes: int, arch: str = "resnet34_unet",
                pretrained: bool = True, num_filters: int = 32,
                pretrained_path: str = None, grad_checkpoint: bool = False,
                drop_path_rate: float = 0.1) -> nn.Module:
    """`pretrained=False` builds the architecture only (used when loading a
    trained checkpoint); it never downloads anything."""
    arch = (arch or "resnet34_unet").lower()
    if arch in ("caformer_surgenet", "caformer_imagenet"):
        src = "surgenetxl" if arch == "caformer_surgenet" else "imagenet"
        variant = "surgenet" if arch == "caformer_surgenet" else "imagenet"
        return CaFormerFPN(num_classes, weights=src if pretrained else None,
                           variant=variant, pretrained_path=pretrained_path,
                           grad_checkpoint=grad_checkpoint,
                           drop_path_rate=drop_path_rate)
    if arch == "unet_classic":
        return ClassicUNet(num_classes)
    if arch == "deeplabv3p_r18":
        return DeepLabV3PlusR18(num_classes, pretrained)
    if arch == "segformer_b0":
        return SegFormerB0(num_classes, pretrained)
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
    if arch in ("caformer_surgenet", "caformer_imagenet"):
        return ("encoder",)
    if arch == "unet_classic":
        return ()               # trained from scratch: no pretrained part, so no reduced encoder learning rate
    if arch == "deeplabv3p_r18":
        return ("stem", "layer1", "layer2", "layer3", "layer4")
    if arch == "segformer_b0":
        return ("encoder",)
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
