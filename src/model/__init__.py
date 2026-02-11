"""
Model registry (TERDNet-only).
"""

from .terdnet import vit_b_terdnet

model_dict = {
    "vit_b_terdnet": vit_b_terdnet,
}

__all__ = ["vit_b_terdnet", "model_dict"]
