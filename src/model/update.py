"""
TERDNet recurrent decoder: 3-gate-GRU

This file contains only the modules used by TERDNet:
- ThreeGateGRUCell: 3-gate GRU cell that integrates feature pyramid information
- ThreeGateGRUDecoder: iterative decoder that returns a list of refined features

Legacy / unused modules from the original codebase were removed.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class ThreeGateGRUCell(nn.Module):
    """
    3-gate-GRU cell used in TERDNet.

    It extends a GRU-like update with an additional "feature gate" that injects
    feature pyramid information (f) into the candidate state computation.

    This implementation follows the original structure:
    - horizontal separable conv (1x5)
    - vertical separable conv (5x1)
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)

        # Project the concatenated pyramid feature difference f into hidden_dim.
        # Use LazyConv2d so it works for vit_b/vit_l/vit_h etc. without hard-coding channels.
        self.f_proj_h = nn.LazyConv2d(self.hidden_dim, kernel_size=(1, 5), padding=(0, 2), bias=True)
        self.f_proj_w = nn.LazyConv2d(self.hidden_dim, kernel_size=(5, 1), padding=(2, 0), bias=True)

        gate_in = self.hidden_dim + self.input_dim + self.hidden_dim  # [h, x, fx]

        # Horizontal (1x5)
        self.convz_h = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convr_h = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convp_h = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convq_h = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(1, 5), padding=(0, 2))

        # Vertical (5x1)
        self.convz_w = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.convr_w = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.convp_w = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.convq_w = nn.Conv2d(gate_in, self.hidden_dim, kernel_size=(5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, hidden_dim, H, W) previous hidden state
            x: (B, input_dim, H, W) fused feature input (constant across iterations in this design)
            f: (B, sum(pyramid_channels), H, W) concatenated pyramid difference features

        Returns:
            h_next: (B, hidden_dim, H, W)
        """
        # Horizontal update
        fx = torch.sigmoid(self.f_proj_h(f))  # (B, hidden_dim, H, W)
        hx = torch.cat([h, x, fx], dim=1)

        z = torch.sigmoid(self.convz_h(hx))
        r = torch.sigmoid(self.convr_h(hx))
        p = torch.sigmoid(self.convp_h(hx))
        q = torch.tanh(self.convq_h(torch.cat([r * h, x, p * fx], dim=1)))

        h = (1 - z) * h + z * q

        # Vertical update
        fx = torch.sigmoid(self.f_proj_w(f))
        hx = torch.cat([h, x, fx], dim=1)

        z = torch.sigmoid(self.convz_w(hx))
        r = torch.sigmoid(self.convr_w(hx))
        p = torch.sigmoid(self.convp_w(hx))
        q = torch.tanh(self.convq_w(torch.cat([r * h, x, p * fx], dim=1)))

        h = (1 - z) * h + z * q
        return h


class ThreeGateGRUDecoder(nn.Module):
    """
    Iterative recurrent decoder (TERDNet).

    Input:
      - h0_pyr, h1_pyr: feature pyramid lists from t0/t1 (each element: BxCxHxW)
      - x: fused feature input (B, input_dim, H, W)

    Output:
      - list of length `iters`, each element is (B, input_dim, H, W)
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 512, iter: int = 5) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.iters = int(iter)

        self.cell = ThreeGateGRUCell(hidden_dim=self.hidden_dim, input_dim=self.input_dim)

        # Reduce the "deepest" pyramid difference feature into the hidden state.
        # Use LazyConv2d so it works for different ViT embed dims.
        self.h_reduce = nn.LazyConv2d(self.hidden_dim, kernel_size=1, bias=False)
        self.bn_h = nn.BatchNorm2d(self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Map hidden state back to the feature space that the upsampler expects (input_dim=512).
        self.h_to_feat = nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1, bias=False)
        self.bn_feat = nn.BatchNorm2d(self.input_dim)

    def forward(
        self,
        h0_pyr: Sequence[torch.Tensor],
        h1_pyr: Sequence[torch.Tensor],
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        if len(h0_pyr) == 0 or len(h1_pyr) == 0:
            raise ValueError("h0_pyr/h1_pyr must be non-empty pyramid feature lists.")
        if len(h0_pyr) != len(h1_pyr):
            raise ValueError(f"Mismatch pyramid levels: len(h0_pyr)={len(h0_pyr)} vs len(h1_pyr)={len(h1_pyr)}")
        if self.iters <= 0:
            raise ValueError(f"iter must be >= 1, got {self.iters}")

        # Initial hidden state: deepest level difference
        h = h0_pyr[-1] - h1_pyr[-1]
        h = self.relu(self.bn_h(self.h_reduce(h)))  # -> (B, hidden_dim, H, W)

        # Pyramid difference feature (concatenated across levels)
        f = torch.cat([h0_pyr[i] - h1_pyr[i] for i in range(len(h0_pyr))], dim=1)

        out_list: List[torch.Tensor] = []
        for _ in range(self.iters):
            h = self.cell(h, x, f)  # (B, hidden_dim, H, W)
            feat = self.relu(self.bn_feat(self.h_to_feat(h)))  # (B, input_dim, H, W)
            out_list.append(feat)

        return out_list


# Optional backward-compatible alias.
# You can remove this later once all imports in the repo are updated.
ThreeGateGRU = ThreeGateGRUDecoder
