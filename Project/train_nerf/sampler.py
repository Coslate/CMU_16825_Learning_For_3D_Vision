import math
from typing import List

import torch
from train_nerf.ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg #cfg.sampler
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.use_fine_sampling = False

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
            sample_points=sample_points, #(H*W, n_pts_per_ray, 3)
            sample_lengths=z_vals.unsqueeze(-1) * torch.ones_like(sample_points[..., :1]), #(H*W, n_pts_per_ray, 1)
        )


class CoarseFineRaysampler(torch.nn.Module):
    def __init__(self, cfg): #cfg.sampler
        super().__init__()
        self.n_coarse = cfg.n_coarse_pts_per_ray
        self.n_fine = cfg.n_fine_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.use_fine_sampling = cfg.get("use_fine_sampling", False)  # Option to revert to simple sampling

    def sample_coarse(self, ray_bundle):
        #Coarse uniform sampling along rays.
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_coarse, 
                                device=ray_bundle.origins.device)
        z_vals = z_vals.unsqueeze(0).expand(ray_bundle.origins.shape[0], self.n_coarse)  # (H*W, n_coarse)
        sample_points = ray_bundle.origins.unsqueeze(1) + z_vals.unsqueeze(-1) * ray_bundle.directions.unsqueeze(1)
        return z_vals, sample_points  # (H*W, n_coarse), (H*W, n_coarse, 3)

    def sample_fine(self, ray_bundle, z_vals, weights):
        #Fine importance sampling based on coarse density.
        n_rays, n_coarse = z_vals.shape
        #weights = weights + 1e-5  # Avoid zero weights
        
        # Compute PDF & CDF
        pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
        pdf = torch.where(torch.isnan(pdf), torch.zeros_like(pdf), pdf)  # Remove NaNs

        cdf = torch.cumsum(pdf, dim=-1)
        cdf_min = cdf.min(dim=-1, keepdim=True)[0]
        cdf_max = cdf.max(dim=-1, keepdim=True)[0]
        cdf_range = cdf_max - cdf_min + 1e-5  # Avoid division by zero
        cdf = (cdf - cdf_min) / cdf_range        

        #cdf = (cdf - cdf.min(dim=-1, keepdim=True)[0]) / (cdf.max(dim=-1, keepdim=True)[0] - cdf.min(dim=-1, keepdim=True)[0])

        with open("debug_log.txt", 'a') as f:
            print(f"pdf min: {pdf.min()}, max: {pdf.max()}, mean: {pdf.mean()}", file=f)
            print(f"cdf min: {cdf.min()}, max: {cdf.max()}, mean: {cdf.mean()}", file=f)

        # Invert CDF for importance sampling
        u = torch.rand(n_rays, self.n_fine, device=ray_bundle.origins.device)
        #indices = torch.searchsorted(cdf, u, right=True)
        #indices = torch.clamp(torch.searchsorted(cdf, u, right=True), max=cdf.shape[-1] - 1)
        indices = torch.clamp(torch.searchsorted(cdf, u, right=False), max=cdf.shape[-1] - 1)

        below = torch.clamp(indices - 1, min=0)
        above = torch.clamp(indices, max=cdf.shape[-1] - 1)
        indices_g = torch.stack([below, above], dim=-1)

        # Linear interpolation
        cdf_g_0 = torch.gather(cdf, -1, indices_g[..., 0])  # Extract first set of indices
        cdf_g_1 = torch.gather(cdf, -1, indices_g[..., 1])  # Extract second set of indices
        cdf_g = torch.stack([cdf_g_0, cdf_g_1], dim=-1)  # Recombine properly

        bins_g_0 = torch.gather(z_vals, -1, indices_g[..., 0])
        bins_g_1 = torch.gather(z_vals, -1, indices_g[..., 1])
        bins_g = torch.stack([bins_g_0, bins_g_1], dim=-1)  # Recombine properly

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        jitter = torch.rand_like(denom) * 1e-3  # Add jitter to avoid duplicate samples
        z_fine = torch.where(denom < 1e-5, bins_g[..., 0], bins_g[..., 0] + ((u - cdf_g[..., 0]) / (denom + jitter)) * (bins_g[..., 1] - bins_g[..., 0]))        
        z_fine = torch.clamp(z_fine, min=self.min_depth, max=self.max_depth)
        #denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        #z_fine = bins_g[..., 0] + ((u - cdf_g[..., 0]) / denom) * (bins_g[..., 1] - bins_g[..., 0])
        #z_fine = torch.where(denom < 1e-5, bins_g[..., 0], bins_g[..., 0] + ((u - cdf_g[..., 0]) / denom) * (bins_g[..., 1] - bins_g[..., 0]))
        #z_fine = torch.clamp(z_fine, min=self.min_depth, max=self.max_depth)

        with open("debug_log.txt", 'a') as f:
            print(f"z_fine min: {z_fine.min()}, max: {z_fine.max()}, mean: {z_fine.mean()}", file=f)
            print(f"z_vals min: {z_vals.min()}, max: {z_vals.max()}, mean: {z_vals.mean()}", file=f)
            print(f"z_fine.shape: {z_fine.shape}", file=f)
            print(f"z_vals.shape: {z_vals.shape}", file=f)

        # Merge with coarse samples
        z_vals_final, _ = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)
        sample_points_fine = ray_bundle.origins.unsqueeze(1) + z_vals_final.unsqueeze(-1) * ray_bundle.directions.unsqueeze(1) #(H*W, 1, 3) + (H*W, n_coarse+n_fine, 1)*(H*W, 1, 3) = (H*W, n_coarse+n_fine, 3)

        with open("debug_log.txt", 'a') as f:
            print(f"sample_points_fine min: {sample_points_fine.min()}, max: {sample_points_fine.max()}, mean: {sample_points_fine.mean()}", file=f)
        return z_vals_final, sample_points_fine #(H*W, n_coarse+n_fine), #(H*W, n_coarse+n_fine, 3)

    def forward(self, ray_bundle, weights_coarse=None):
        if not self.use_fine_sampling or weights_coarse is None:
            z_vals, sample_points = self.sample_coarse(ray_bundle)
            return ray_bundle._replace(
                sample_points=sample_points, #(H*W, n_pts_per_ray, 3)
                sample_lengths=z_vals.unsqueeze(-1) * torch.ones_like(sample_points[..., :1]), #(H*W, n_pts_per_ray, 1)
            )

        z_vals_final, sample_points_fine = self.sample_fine(ray_bundle, ray_bundle.sample_lengths.squeeze(-1), weights_coarse.squeeze(-1))
        with open("debug_log.txt", 'a') as f:
            print(f"sample_points_fine.shape = {sample_points_fine.shape}", file=f)
            print(f"z_vals_final.shape = {z_vals_final.shape}", file=f)
            print(f"z_vals_final min: {z_vals_final.min()}, max: {z_vals_final.max()}, mean: {z_vals_final.mean()}", file=f)

        return ray_bundle._replace(
            sample_points=sample_points_fine, #(H*W, n_pts_per_ray, 3)
            sample_lengths=z_vals_final.unsqueeze(-1) * torch.ones_like(sample_points_fine[..., :1]), #(H*W, n_pts_per_ray, 1)
        )


# Update the sampler dictionary to allow switching
sampler_dict = {
    'stratified': StratifiedRaysampler, 
    'coarse_fine': CoarseFineRaysampler  # New coarse-to-fine sampler
}
