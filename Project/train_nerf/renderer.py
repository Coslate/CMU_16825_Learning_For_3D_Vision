import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        # Compute prob to hit
        prob = 1.0 - torch.exp(-rays_density * deltas)  # (H*W, n_pts_per_ray, 1)

        # Compute survival probability (1 - prob) with numerical stability
        survival_prob = 1.0 - prob + eps  # (H*W, n_pts_per_ray, 1)

        # Initialize transmittance with first value as 1
        T_init = torch.ones_like(prob[..., :1, :])  # (H*W, 1, 1)

        # Concatenate T_init with survival probabilities
        T_full = torch.cat([T_init, survival_prob], dim=-2)  # (H*W, n_pts_per_ray + 1, 1)

        # Compute cumulative product to get transmittance
        transmittance = torch.cumprod(T_full, dim=-2)[..., :-1, :]  # (H*W, n_pts_per_ray, 1)

        weights = transmittance * prob  # (H*W, n_pts_per_ray, 1)

        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor, #(H*W, n_pts_per_ray, 1)
        rays_feature: torch.Tensor #(H*W, n_pts_per_ray, C)
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=-2)  # (H*W, C)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs_fine = []
        chunk_outputs_coarse = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            #Coarse sampling
            cur_ray_bundle_coarse = sampler(cur_ray_bundle)
            n_pts_coarse = cur_ray_bundle_coarse.sample_points.shape[1]

            #Coarse pass through NeRF
            implicit_output = implicit_fn(cur_ray_bundle_coarse)
            density_coarse = implicit_output['density']
            feature_coarse = implicit_output['feature']
            #print(f"")
            #print(f"density_coarse min: {density_coarse.min()}, max: {density_coarse.max()}, mean: {density_coarse.mean()}")

            # Compute length of each ray segment
            depth_values_coarse = cur_ray_bundle_coarse.sample_lengths[..., 0]
            deltas_coarse = torch.cat(
                (
                    depth_values_coarse[..., 1:] - depth_values_coarse[..., :-1], #(H*W, n_pts_per_ray - 1)
                    1e10 * torch.ones_like(depth_values_coarse[..., :1]), # enshure it does not contribute to volume rendering
                ),
                dim=-1, #(H*W, n_pts_per_ray)
            )[..., None] #(H*W, n_pts_per_ray, 1)

            # Compute aggregation weights
            weights_coarse = self._compute_weights(
                deltas_coarse.view(-1, n_pts_coarse, 1),
                density_coarse.view(-1, n_pts_coarse, 1)
            )  #(H*W, n_pts_per_ray, 1)

            # TODO (1.5): Render (color) features using weights_coarse
            feature_coarse = self._aggregate(weights_coarse, feature_coarse.reshape(-1, n_pts_coarse, 3))  # (H*W, C)

            # TODO (1.5): Render depth_coarse map
            depth_coarse = self._aggregate(weights_coarse, depth_values_coarse.reshape(-1, n_pts_coarse, 1))  # (H*W, 1)
            # Return

            cur_out = {
                'feature': feature_coarse,
                'depth': depth_coarse,
            }
            chunk_outputs_coarse.append(cur_out)

            if sampler.use_fine_sampling:
                # Fine sampling with importance-based resampling
                cur_ray_bundle_fine = sampler(cur_ray_bundle_coarse, weights_coarse)
                implicit_output = implicit_fn(cur_ray_bundle_fine)
                density_fine = implicit_output['density']
                feature_fine = implicit_output['feature']
                n_pts_fine = cur_ray_bundle_fine.sample_points.shape[1]
                #print(f"density_fine min: {density_fine.min()}, max: {density_fine.max()}, mean: {density_fine.mean()}")

                # Compute length of each ray segment
                depth_values_fine = cur_ray_bundle_fine.sample_lengths[..., 0]
                deltas_fine = torch.cat(
                    (
                        depth_values_fine[..., 1:] - depth_values_fine[..., :-1], #(H*W, n_pts_per_ray - 1)
                        1e10 * torch.ones_like(depth_values_fine[..., :1]), # enshure it does not contribute to volume rendering
                    ),
                    dim=-1, #(H*W, n_pts_per_ray)
                )[..., None] #(H*W, n_pts_per_ray, 1)

                # Compute aggregation weights
                weights_fine = self._compute_weights(
                    deltas_fine.view(-1, n_pts_fine, 1),
                    density_fine.view(-1, n_pts_fine, 1)
                )  #(H*W, n_pts_per_ray, 1)

                # TODO (1.5): Render (color) features using weights_fine
                feature_fine = self._aggregate(weights_fine, feature_fine.reshape(-1, n_pts_fine, 3))  # (H*W, C)

                # TODO (1.5): Render depth_fine map
                depth_fine = self._aggregate(weights_fine, depth_values_fine.reshape(-1, n_pts_fine, 1))  # (H*W, 1)

                # Return
                cur_out = {
                    'feature': feature_fine,
                    'depth': depth_fine,
                }
                chunk_outputs_fine.append(cur_out)

        if sampler.use_fine_sampling:
            # Concatenate chunk outputs
            out = {
                k: torch.cat(
                    [chunk_out[k] for chunk_out in chunk_outputs_coarse], dim=0
                ) for k in chunk_outputs_coarse[0].keys()
            }

            out_fine = {
                k: torch.cat(
                    [chunk_out[k] for chunk_out in chunk_outputs_fine], dim=0
                ) for k in chunk_outputs_fine[0].keys()
            }

            # Rename keys for clarity
            out = {f"{k}_coarse": v for k, v in out.items()}
            out_fine_rendered = {f"{k}": v for k, v in out_fine.items()} #out['feature'] would call out_fine items.
            out_fine = {f"{k}_fine": v for k, v in out_fine.items()}

            # Merge both dictionaries
            out.update(out_fine)
            out.update(out_fine_rendered)
        else:
            # Concatenate chunk outputs
            out = {
                k: torch.cat(
                [chunk_out[k] for chunk_out in chunk_outputs_coarse],
                dim=0
                ) for k in chunk_outputs_coarse[0].keys()
            }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
        self.epsilon = 1e-7
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface, the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not

        # Initialize points at the ray origins
        points = origins.clone()
        mask = torch.zeros((origins.shape[0], 1), dtype=torch.bool, device=origins.device)
    
        for _ in range(self.max_iters):
            # Compute SDF value at the current points
            sdf_values = implicit_fn(points).squeeze(-1)  # (N_rays,)

            # Check for convergence (rays that hit the surface)
            converged = sdf_values < self.epsilon
            mask[converged] = True  # Mark rays that hit the surface

            # Stop updating rays that have converged
            if converged.all():
                break

            # Move points along the ray direction by the SDF value
            points = points + directions * sdf_values.unsqueeze(-1)

        return points, mask        

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins) #(H*W, 3)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out

def laplace_cdf(signed_distance, beta):
    return 0.5 * torch.exp(signed_distance / beta) * (signed_distance <= 0).float() + (1 - 0.5 * torch.exp(-signed_distance / beta)) * (signed_distance > 0).float()

def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    """
    Convert Signed Distance Function (SDF) to volume density using VolSDF formulation.

    Args:
        signed_distance (torch.Tensor): Signed distance function values (N, 1).
        alpha (float): Learnable parameter controlling density scaling.
        beta (float): Learnable parameter controlling density smoothness.

    Returns:
        density (torch.Tensor): Converted volume density (N, 1).
    """
    psi_beta = laplace_cdf(-signed_distance, beta)
    density = alpha * psi_beta  # Apply scaling
    return density    

def sdf_to_density_neus(sdf, s):
    """
    Convert SDF to density using NeuS's logistic density function.

    Args:
        sdf (torch.Tensor): Signed distance function values.
        s (float): Trainable sharpness parameter.

    Returns:
        torch.Tensor: Converted volume density.
    """
    exp_sdf = torch.exp(-s * sdf)
    density = (s * exp_sdf) / ((1 + exp_sdf) ** 2)
    return density

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            # Q8.3
            if self.cfg.get("neus_sdf", False):
                density = sdf_to_density_neus(distance, implicit_fn.s)
            else:
                density = sdf_to_density(distance, self.alpha, self.beta)# TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
