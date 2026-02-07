"""
Depth-supervised Splatfacto — minimal nerfstudio plugin.

Extends splatfacto with L1 depth loss from LiDAR sensor depth maps.
Registers as 'depth-splatfacto' method via nerfstudio entry points.
"""

from depth_splatfacto.config import DepthSplatfactoSpecification

__all__ = ["DepthSplatfactoSpecification"]
