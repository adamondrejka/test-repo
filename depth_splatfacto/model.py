"""
Depth-supervised Splatfacto model.

Inherits from SplatfactoModel and adds L1 depth loss comparing
rendered depth to ground-truth sensor depth (LiDAR).
"""

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Config for depth-supervised splatfacto."""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_mult: float = 0.2
    """Multiplier for depth loss. Set to 0 to disable."""


class DepthSplatfactoModel(SplatfactoModel):
    """Splatfacto with depth supervision from sensor depth maps."""

    config: DepthSplatfactoModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.config.depth_loss_mult > 0 and "depth_image" in batch:
            pred_depth = outputs["depth"]
            gt_depth = batch["depth_image"].to(pred_depth.device)

            # Resize GT depth to match rendered depth if needed
            if gt_depth.shape[:2] != pred_depth.shape[:2]:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),
                    size=pred_depth.shape[:2],
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)

            # Only supervise valid (non-zero) depth pixels
            valid = gt_depth > 0
            if valid.any():
                depth_loss = torch.abs(pred_depth[valid] - gt_depth[valid]).mean()
                loss_dict["depth_loss"] = self.config.depth_loss_mult * depth_loss

        return loss_dict
