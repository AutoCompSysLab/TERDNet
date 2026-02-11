"""
TERDNet model wrapper (formerly DeepLabV3.py in a C-3PO-derived codebase).

This file defines:
- Combined convolution–interpolation upsampler (paper: "Combined Upsampler")
- TERDNet model wrapper that takes TERDCore outputs (iterative 512-ch features)
  and produces full-resolution change masks for each iteration.

Recommended filename:
  - terdnet.py
(You can keep the old filename temporarily and rename later.)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Be robust to your ongoing refactor:
# - If you named the new backbone module "backbone.py", import from model.backbone
# - If you kept the old "Backbone.py", import from model.Backbone
try:
    from model.backbone import build_terdnet_vit_core
except Exception:  # pragma: no cover
    from model.Backbone import build_terdnet_vit_core  # type: ignore


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class CombinedUpsampler(nn.Module):
    """
    Combined convolution–interpolation upsampler (TERDNet).

    Paper description:
      - Starting from (H/16, W/16, 512), apply 4 blocks:
        conv (reduce channels) + bilinear interpolation (increase resolution)
      - Channel reduction: 512 -> 256 -> 128 -> 64 -> num_classes

    This module outputs logits at full input resolution (H, W).
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, padding=1)        # 512 -> 256
        self.conv2 = nn.Conv2d(input_dim // 2, input_dim // 4, kernel_size=3, padding=1)   # 256 -> 128
        self.conv3 = nn.Conv2d(input_dim // 4, input_dim // 8, kernel_size=3, padding=1)   # 128 -> 64
        self.conv4 = nn.Conv2d(input_dim // 8, num_classes, kernel_size=3, padding=1)      # 64 -> C

        # NOTE: The original code applied BN+ReLU on the final logits.
        # This is kept to preserve behavior until you verify training reproducibility.
        self.bn = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, out_size: Tuple[int, int]) -> Tensor:
        """
        Args:
            x: (B, 512, H/16, W/16)
            out_size: (H, W) target resolution (input image size)

        Returns:
            logits: (B, num_classes, H, W)
        """
        H, W = int(out_size[0]), int(out_size[1])

        # Target intermediate sizes (robust to non-multiples, though your input is typically /16)
        s8 = (_ceil_div(H, 8), _ceil_div(W, 8))
        s4 = (_ceil_div(H, 4), _ceil_div(W, 4))
        s2 = (_ceil_div(H, 2), _ceil_div(W, 2))

        x = self.conv1(x)
        x = F.interpolate(x, size=s8, mode="bilinear", align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, size=s4, mode="bilinear", align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, size=s2, mode="bilinear", align_corners=False)

        x = self.conv4(x)
        x = self.relu(self.bn(x))

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


class TERDNetModel(nn.Module):
    """
    Wrapper: TERDCore (encoder+fusion+recurrent) + CombinedUpsampler.

    To keep compatibility with the current train.py:
      - exposes .backbone and .classifier attributes
      - returns OrderedDict({'out': list_of_logits})
    """

    def __init__(self, backbone: nn.Module, classifier: nn.Module, iters: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.iters = int(iters)

    def forward(self, x: Tensor) -> "OrderedDict[str, List[Tensor]]":
        input_shape = x.shape[-2:]  # (H, W)

        features = self.backbone(x)  # OrderedDict with key 'out' -> list of (B,512,H/16,W/16)
        if not isinstance(features, OrderedDict) or "out" not in features:
            raise RuntimeError("Backbone must return OrderedDict with key 'out' (list of features).")

        feat_list = features["out"]
        if not isinstance(feat_list, (list, tuple)):
            raise RuntimeError("features['out'] must be a list/tuple of feature tensors.")

        if len(feat_list) != self.iters:
            # Be strict: your loss/eval code indexes by args.iter
            raise RuntimeError(f"Expected {self.iters} iterations, got {len(feat_list)} outputs from backbone.")

        logits_list: List[Tensor] = []
        for i in range(self.iters):
            logits = self.classifier(feat_list[i], out_size=input_shape)  # (B,C,H,W)
            logits_list.append(logits)

        out: "OrderedDict[str, List[Tensor]]" = OrderedDict()
        out["out"] = logits_list
        return out


def vit_b_terdnet(args) -> nn.Module:
    """
    New preferred factory name (TERDNet).
    """
    # args.msf existed in legacy code (number of pyramid levels).
    # TERDNet core here expects 4 levels (paper uses 4).
    if hasattr(args, "msf") and int(args.msf) != 4:
        raise ValueError(f"TERDNet expects args.msf==4 (4 pyramid levels). Got msf={args.msf}")

    backbone = build_terdnet_vit_core(
        vit_variant="vit_b",
        fusion_type=getattr(args, "mtf", "ours"),  # accepts legacy: corrconv/conv/corr/globalcorr
        corr_radius=4,
        proj_channels=64,
        decoder_iters=int(getattr(args, "iter", 5)),
        freeze_encoder=True,
        sam_checkpoint=getattr(args, "sam_checkpoint", None),
        layer_indices=None,
    )

    classifier = CombinedUpsampler(input_dim=512, num_classes=int(args.num_classes))
    model = TERDNetModel(backbone=backbone, classifier=classifier, iters=int(getattr(args, "iter", 5)))
    return model


# ---------------------------------------------------------------------
# Backward-compatible alias (so your current --model name keeps working
# until you update model_dict/train.py in step 4)
# ---------------------------------------------------------------------
def vit_b_mtf_msf_deeplabv3(args) -> nn.Module:
    """
    Legacy name kept as an alias.
    You can remove this after train.py/model_dict are updated.
    """
    return vit_b_terdnet(args)
