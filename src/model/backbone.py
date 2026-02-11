"""
TERDNet backbone components.

This file contains:
- SAM ViT image encoder wrapper (frozen by default)
- Feature Fusion Module (per-level fusion + multi-level aggregation)
- Recurrent 3-gate-GRU decoder hook (implemented in model.update)

It is adapted from a C-3PO-based codebase but has been renamed and simplified
to match the TERDNet paper terminology.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import sys
sys.path.append('/home/main/workspace/jiae1/segment-anything-main/segment_anything')
from build_sam import *
#from build_sam import sam_model_registry

# Keep compatibility during the refactor step order:
# - Later you will rename ThreeGateGRU -> ThreeGateGRUDecoder in update.py.
try:
    from model.update import ThreeGateGRUDecoder  # type: ignore
except Exception:  # pragma: no cover
    from model.update import ThreeGateGRU as ThreeGateGRUDecoder  # type: ignore


def _normalize_fusion_type(name: str) -> str:
    """
    Normalize legacy / short names to TERDNet paper-friendly names.

    Supported normalized names:
      - 'ours'       : feature maps + local correlation (TERDNet default)
      - 'feature'    : feature maps only
      - 'local_corr' : local correlation only
      - 'global_corr': global correlation only (very memory heavy)

    Legacy aliases:
      - 'corrconv'   -> 'ours'
      - 'conv'       -> 'feature'
      - 'corr'       -> 'local_corr'
      - 'globalcorr' -> 'global_corr'
    """
    name = (name or "").lower().strip()

    legacy_map = {
        "corrconv": "ours",
        "conv": "feature",
        "corr": "local_corr",
        "globalcorr": "global_corr",
    }
    name = legacy_map.get(name, name)

    if name not in {"ours", "feature", "local_corr", "global_corr"}:
        raise ValueError(
            f"Unknown fusion_type='{name}'. "
            f"Use one of: ours, feature, local_corr, global_corr "
            f"(or legacy: corrconv/conv/corr/globalcorr)."
        )
    return name


class GlobalCorrelationVolume(nn.Module):
    """
    Global (all-pairs) correlation volume.

    Output shape:
      - (B, H*W, H, W)

    Warning:
      - This is extremely memory intensive for H=W=64 (4096 channels),
        and the intermediate matrix multiplication is O((HW)^2).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_a.size()

        # Reshape for matrix multiplication
        a = feature_a.transpose(2, 3).contiguous().view(b, c, h * w)           # (B, C, HW)
        b_ = feature_b.view(b, c, h * w).transpose(1, 2)                        # (B, HW, C)
        corr = torch.bmm(b_, a)                                                 # (B, HW, HW)

        corr = corr.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)        # (B, HW, H, W)
        return corr


class LocalCorrelationVolume(nn.Module):
    """
    Local correlation volume within a (2r+1)x(2r+1) window.

    The output channels are (2r+1)^2, with each channel representing
    the channel-wise mean dot-product at a specific displacement.
    (Mean is a scaled dot product.)
    """

    def __init__(self, radius: int = 4) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("radius must be >= 0")
        self.radius = int(radius)
        self.pad = nn.ConstantPad2d(self.radius, 0)

        # Precompute offsets (dy, dx) in the padded feature space
        size = 2 * self.radius + 1
        self.offsets: List[Tuple[int, int]] = [(dy, dx) for dy in range(size) for dx in range(size)]

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: (B, C, H, W)
            f1: (B, C, H, W)

        Returns:
            corr: (B, (2r+1)^2, H, W)
        """
        f1_pad = self.pad(f1)
        h, w = f0.shape[-2], f0.shape[-1]

        corr = torch.cat(
            [torch.mean(f0 * f1_pad[:, :, dy : dy + h, dx : dx + w], dim=1, keepdim=True)
             for (dy, dx) in self.offsets],
            dim=1,
        )
        return corr


class CorrFeatureFusionBlock(nn.Module):
    """
    Per-level feature fusion block.

    For each pyramid level, TERDNet fuses:
      - image features at t0 and t1 (after channel reduction conv)
      - correlation volume between t0 and t1 features (optional)

    Supported fusion_type:
      - ours       : local corr + reduced features (t0, t1)
      - feature    : reduced features only
      - local_corr : local corr only
      - global_corr: global corr only
    """

    def __init__(
        self,
        in_channels: int,
        fusion_type: str = "ours",
        corr_radius: int = 4,
        proj_channels: int = 64,
    ) -> None:
        super().__init__()
        self.fusion_type = _normalize_fusion_type(fusion_type)
        self.corr_radius = int(corr_radius)
        self.proj_channels = int(proj_channels)

        if self.fusion_type in {"ours", "local_corr"}:
            self.local_corr = LocalCorrelationVolume(radius=self.corr_radius)
        else:
            self.local_corr = None

        if self.fusion_type == "global_corr":
            self.global_corr = GlobalCorrelationVolume()
        else:
            self.global_corr = None

        if self.fusion_type in {"ours", "feature"}:
            self.proj = nn.Conv2d(in_channels, self.proj_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.proj = None

        self.act = nn.ReLU(inplace=True)

    @property
    def out_channels(self) -> Optional[int]:
        """Static output channels (None for dynamic/global_corr)."""
        r = self.corr_radius
        corr_ch = (2 * r + 1) ** 2

        if self.fusion_type == "ours":
            return corr_ch + 2 * self.proj_channels
        if self.fusion_type == "feature":
            return 2 * self.proj_channels
        if self.fusion_type == "local_corr":
            return corr_ch
        return None  # global_corr is dynamic: H*W channels

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        pieces: List[torch.Tensor] = []

        if self.fusion_type == "global_corr":
            assert self.global_corr is not None
            pieces.append(self.global_corr(f0, f1))

        if self.fusion_type in {"ours", "local_corr"}:
            assert self.local_corr is not None
            pieces.append(self.local_corr(f0, f1))

        if self.fusion_type in {"ours", "feature"}:
            assert self.proj is not None
            pieces.append(self.proj(f0))
            pieces.append(self.proj(f1))

        x = torch.cat(pieces, dim=1)
        return self.act(x)


class FeatureFusionModule(nn.Module):
    """
    Multi-level feature fusion module.

    It applies CorrFeatureFusionBlock per level, concatenates all levels,
    then reduces channels to `out_channels` (default 512).

    Notes:
      - With SAM ViT encoders, each level has the same spatial resolution.
      - For TERDNet (as implemented here), we keep `out_channels=512`
        to match the recurrent decoder input_dim.
    """

    def __init__(
        self,
        channels: Sequence[int],
        fusion_type: str = "ours",
        corr_radius: int = 4,
        proj_channels: int = 64,
        out_channels: int = 512,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must be non-empty")

        self.fusion_type = _normalize_fusion_type(fusion_type)
        self.num_levels = len(channels)
        self.out_channels_ = int(out_channels)

        self.level_blocks = nn.ModuleList(
            [
                CorrFeatureFusionBlock(
                    in_channels=int(ch),
                    fusion_type=self.fusion_type,
                    corr_radius=int(corr_radius),
                    proj_channels=int(proj_channels),
                )
                for ch in channels
            ]
        )

        # Channel reduction after concatenation across levels
        if self.fusion_type == "global_corr":
            # Dynamic in_channels (depends on H*W), so use a lazy conv.
            self.reduce = nn.LazyConv2d(self.out_channels_, kernel_size=1, stride=1, bias=False)
        else:
            # Static in_channels
            per_level = self.level_blocks[0].out_channels
            assert per_level is not None
            in_channels = per_level * self.num_levels
            self.reduce = nn.Conv2d(in_channels, self.out_channels_, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm2d(self.out_channels_) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, t0_pyramid: Sequence[torch.Tensor], t1_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(t0_pyramid) != self.num_levels or len(t1_pyramid) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} pyramid levels, got "
                f"{len(t0_pyramid)} (t0) and {len(t1_pyramid)} (t1)."
            )

        fused_levels: List[torch.Tensor] = [
            self.level_blocks[i](t0_pyramid[i], t1_pyramid[i]) for i in range(self.num_levels)
        ]
        x = torch.cat(fused_levels, dim=1)
        x = self.reduce(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def _default_sam_checkpoint(vit_variant: str) -> str:
    """
    Default checkpoint paths for SAM image encoders.

    Recommended open-source layout:
      - ./checkpoints/sam_vit_b_01ec64.pth
      - ./checkpoints/sam_vit_l_0b3195.pth
      - ./checkpoints/sam_vit_h_4b8939.pth

    You can override via environment variables:
      - SAM_CKPT_VIT_B / SAM_CKPT_VIT_L / SAM_CKPT_VIT_H
    """
    vit_variant = vit_variant.lower().strip()
    env_map = {
        "vit_b": "SAM_CKPT_VIT_B",
        "vit_l": "SAM_CKPT_VIT_L",
        "vit_h": "SAM_CKPT_VIT_H",
    }
    default_map = {
        "vit_b": '/home/main/workspace/jiae1/new_scd_model/src/model/sam_vit_b_01ec64.pth',
        "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
        "vit_h": "checkpoints/sam_vit_h_4b8939.pth",
    }
    if vit_variant not in default_map:
        raise ValueError(f"Unsupported vit_variant='{vit_variant}'. Use vit_b/vit_l/vit_h.")
    return os.getenv(env_map[vit_variant], default_map[vit_variant])


def _vit_embed_dim(vit_variant: str) -> int:
    vit_variant = vit_variant.lower().strip()
    dims = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
    if vit_variant not in dims:
        raise ValueError(f"Unsupported vit_variant='{vit_variant}'. Use vit_b/vit_l/vit_h.")
    return dims[vit_variant]


def _default_layer_indices(vit_variant: str) -> List[int]:
    """
    Layer indices used to extract 4-level pyramid features.

    Matches the TERDNet paper description:
      - ViT-b (12 blocks): [2, 5, 8, 11]
      - ViT-l (24 blocks): [5, 11, 17, 23]
      - ViT-h (32 blocks): [7, 15, 23, 31]
    """
    vit_variant = vit_variant.lower().strip()
    layer_map = {
        "vit_b": [2, 5, 8, 11],
        "vit_l": [5, 11, 17, 23],
        "vit_h": [7, 15, 23, 31],
    }
    if vit_variant not in layer_map:
        raise ValueError(f"Unsupported vit_variant='{vit_variant}'. Use vit_b/vit_l/vit_h.")
    return layer_map[vit_variant]


def build_sam_image_encoder(vit_variant: str, checkpoint: Optional[str] = None) -> nn.Module:
    """
    Build SAM image encoder (ViT) given a variant and checkpoint path.
    """
    vit_variant = vit_variant.lower().strip()
    ckpt = checkpoint or _default_sam_checkpoint(vit_variant)
    sam = sam_model_registry[vit_variant](checkpoint=ckpt)
    return sam.image_encoder


class TERDCore(nn.Module):
    """
    TERDNet core: (Encoder -> Feature Fusion -> Recurrent 3-gate-GRU)

    Input:
      - x: (B, 6, H, W) where channels are concatenated (t0_rgb, t1_rgb)

    Output:
      - dict with key 'out'
        'out' is a list of length `decoder_iters`, each element:
          (B, 512, H/16, W/16)
    """

    def __init__(
        self,
        encoder: nn.Module,
        layer_indices: Sequence[int],
        fusion: FeatureFusionModule,
        decoder: nn.Module,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_indices = list(layer_indices)
        self.fusion = fusion
        self.decoder = decoder
        self.freeze_encoder = bool(freeze_encoder)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    @staticmethod
    def _to_bchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(f"Expected a 4D tensor feature, got shape={tuple(feat.shape)}")

        # Heuristic: SAM intermediates are typically (B, H, W, C).
        # If the 2nd dim is smaller than the last dim, treat as BHWC.
        if feat.shape[1] < feat.shape[-1]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _extract_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(img)
        else:
            out = self.encoder(img)

        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise RuntimeError(
                "SAM image encoder is expected to return (output, intermediates). "
                f"Got type={type(out)}."
            )

        _, intermediates = out
        if not isinstance(intermediates, (list, tuple)):
            raise RuntimeError("Expected intermediates to be a list/tuple of features.")

        pyramid: List[torch.Tensor] = []
        for idx in self.layer_indices:
            if idx < 0 or idx >= len(intermediates):
                raise IndexError(
                    f"layer index {idx} out of range for intermediates length {len(intermediates)}"
                )
            pyramid.append(self._to_bchw(intermediates[idx]))
        return pyramid

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, List[torch.Tensor]]":
        out: "OrderedDict[str, List[torch.Tensor]]" = OrderedDict()

        t0_img, t1_img = torch.split(x, 3, dim=1)

        t0_pyramid = self._extract_pyramid(t0_img)
        t1_pyramid = self._extract_pyramid(t1_img)

        fused = self.fusion(t0_pyramid, t1_pyramid)  # (B, 512, H/16, W/16)

        # Recurrent refinement
        refined_list: List[torch.Tensor] = self.decoder(t0_pyramid, t1_pyramid, fused)

        out["out"] = refined_list
        return out


def build_terdnet_vit_core(
    vit_variant: str = "vit_b",
    fusion_type: str = "ours",
    corr_radius: int = 4,
    proj_channels: int = 64,
    decoder_iters: int = 5,
    freeze_encoder: bool = True,
    sam_checkpoint: Optional[str] = None,
    layer_indices: Optional[Sequence[int]] = None,
) -> TERDCore:
    """
    Factory to build TERDNet core with a SAM ViT image encoder.

    Args:
        vit_variant: 'vit_b' | 'vit_l' | 'vit_h'
        fusion_type: ours | feature | local_corr | global_corr
                     (legacy aliases: corrconv/conv/corr/globalcorr)
        corr_radius: local correlation radius (r)
        proj_channels: channel reduction dim for each image feature before fusion
        decoder_iters: number of recurrent refinement iterations
        freeze_encoder: freeze SAM encoder params (paper uses frozen encoder)
        sam_checkpoint: optional checkpoint path (default: ./checkpoints/*.pth)
        layer_indices: optional list of indices to pick intermediate features

    Returns:
        TERDCore
    """
    vit_variant = vit_variant.lower().strip()
    encoder = build_sam_image_encoder(vit_variant, checkpoint=sam_checkpoint)

    if layer_indices is None:
        layer_indices = _default_layer_indices(vit_variant)

    # The current ThreeGateGRU implementation (update.py) is hard-coded for 4 levels.
    if len(layer_indices) != 4:
        raise ValueError(f"TERDNet core currently expects 4 pyramid levels, got {len(layer_indices)}.")

    embed_dim = _vit_embed_dim(vit_variant)

    fusion = FeatureFusionModule(
        channels=[embed_dim] * len(layer_indices),
        fusion_type=fusion_type,
        corr_radius=corr_radius,
        proj_channels=proj_channels,
        out_channels=512,
        use_bn=True,
    )

    decoder = ThreeGateGRUDecoder(hidden_dim=128, input_dim=512, iter=int(decoder_iters))

    core = TERDCore(
        encoder=encoder,
        layer_indices=layer_indices,
        fusion=fusion,
        decoder=decoder,
        freeze_encoder=freeze_encoder,
    )
    return core
