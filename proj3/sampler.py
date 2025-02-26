import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth, self.max_depth, self.n_pts_per_ray, device=ray_bundle.origins.device
        )  # (n_pts_per_ray,)

        # Expand z_vals to match (H*W, n_pts_per_ray)
        z_vals = z_vals.unsqueeze(0).expand(ray_bundle.origins.shape[0], self.n_pts_per_ray)  #(H*W, n_pts_per_ray)

        # TODO (Q1.4): Sample points from z values
        sample_points = ray_bundle.origins.unsqueeze(1) + z_vals.unsqueeze(-1) * ray_bundle.directions.unsqueeze(1) # (H*W, 1, 3) + (H*W, n_pts_per_ray, 1) * (H*W, 1, 3) -> #(H*W, n_pts_per_ray, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(-1) * torch.ones_like(sample_points[..., :1]), #(H*W, n_pts_per_ray, 1)
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}