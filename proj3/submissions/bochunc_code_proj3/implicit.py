import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SceneSphereSDF(torch.nn.Module):
    def __init__(
        self,
        center_pt,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(center_pt).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius

class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

class SceneSDF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect GPU
        self.primitives = [
            SceneSphereSDF(torch.tensor([x, y, z], device=self.device), cfg)
            for x in range(-3, 3, 2)
            for y in range(-3, 3, 2)
            for z in range(-3, 3, 2)
        ]  # 27 spheres in a grid

        # Compute Scene Center as the mean of all sphere centers
        centers = torch.stack([primitive.center for primitive in self.primitives], dim=0)
        self.center = torch.mean(centers, dim=0)  # (1, 3)        

    def forward(self, points):
        sdf_values = [primitive(points) for primitive in self.primitives]
        return torch.min(torch.stack(sdf_values), dim=0)[0]  # Take minimum SDF value        

sdf_dict = {
    'scene': SceneSDF,
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.n_layers_xyz = cfg.n_layers_xyz
        self.use_views    = cfg.get("use_views", False)

        # Positional Encoding using Harmonic Embedding
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        if self.use_views:
            self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)
            embedding_dim_dir = self.harmonic_embedding_dir.output_dim
            self.hidden_dim_dir = cfg.n_hidden_neurons_dir
        else:
            embedding_dim_dir = 0
            self.hidden_dim_dir = 0


        # 8-layer MLP with a skip connection at layer 4, from NeRF paper
        self.hidden_dim = cfg.n_hidden_neurons_xyz
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim_xyz, self.hidden_dim))  # First layer

        for i in range(1, self.n_layers_xyz):  # 8 layers total
            if i == 6:
                self.layers.append(nn.Linear(self.hidden_dim + embedding_dim_xyz, self.hidden_dim))  # Skip connection
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Density (σ) output branch, from NeRF paper
        self.density_layer = nn.Linear(self.hidden_dim, 1)  # Predicts volume density (σ)
        
        # Feature vector branch (for RGB prediction), from NeRF paper
        self.feature_layer = nn.Linear(self.hidden_dim, self.hidden_dim)  # Intermediate feature output

        # RGB color prediction branch, from NeRF paper
        if self.use_views:
            # MLP to generate FiLM parameters (scale and shift)
            self.view_mlp = nn.Sequential(
                nn.Linear(embedding_dim_dir, self.hidden_dim_dir),  # Transform view dir
                nn.ReLU(),
                nn.Linear(self.hidden_dim_dir, self.hidden_dim * 2)  # Outputs scale & shift
            )

            # Color prediction MLP
            self.color_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3),  # RGB output
                nn.Sigmoid(),  # Ensures output in [0,1]
            )
        else:
            self.color_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Sigmoid(),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """ Xavier Initialization for MLPs (following NeRF paper) """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier Uniform Initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero Bias Initialization        

    def forward(self, ray_bundle):
        """
        Inputs:
          - ray_bundle: contains (xyz, view direction, sample_points, sample_lengths) per ray

        Outputs:
          - density (sigma) for volume rendering
          - color (RGB)
        """
        # Positional encoding
        _, n_points_sampled, _ = ray_bundle.sample_points.shape
        xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points)  # (H*W, n, embedding_dim_xyz)
        if self.use_views:
            view_dirs = self.harmonic_embedding_dir(ray_bundle.directions)  # (H*W, embedding_dim_dir)
        else:
            view_dirs = None

        # Forward pass through MLP
        x = xyz
        for i, layer in enumerate(self.layers):
            if i == 6: #Move to layer6 for sharper results
                x = torch.cat([x, xyz], dim=-1)  # Skip connection at layer 6
            x = layer(x)  # No activation for the last (8th) layer

            if i < (self.n_layers_xyz-1):  # Apply ReLU only to the first 7 layers, not at the last layer
                x = F.relu(x)            

        # Density output (σ) with small bias
        sigma = F.relu(self.density_layer(x) + 0.1)  

        # Feature vector (for RGB) with non-linearity
        feature = F.relu(self.feature_layer(x))  

        # RGB color prediction
        if self.use_views:
            # Compute FiLM scaling & shifting factors
            film_params = self.view_mlp(view_dirs)  # (batch_size, hidden_dim * 2)
            film_params = film_params.unsqueeze(1).expand(-1, n_points_sampled, -1)  # (H*W, n_points_sampled, feature_dim * 2)
            scale, shift = torch.chunk(film_params, 2, dim=-1)  # Split into (batch_size, n_points_sampled, hidden_dim)

            # Apply FiLM conditioning
            film_feature = feature * (1 + scale) + shift  # Modulate feature vector

            # Pass through color layer
            rgb = self.color_layer(film_feature)
        else:
            rgb = self.color_layer(feature)
        
        return {'density': sigma, 'feature': rgb}   

class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        #Q8.3
        if cfg.get("neus_s_trainable", False):
            self.s = torch.nn.Parameter(torch.tensor(40.0))  # Trainable sharpness for NeuS SDF to Density function: logistic density distribution

        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        self.n_layers_distance = cfg.get("n_layers_distance", 6)  # Use 6 layers (default)
        self.n_layers_color = cfg.get("n_layers_color", 3)  # Use 6 layers (default)
        self.hidden_dim = cfg.get("n_hidden_neurons_distance", 128)  # Set hidden neurons
        self.color_hidden_dim = cfg.get("n_hidden_neurons_color", 128)  # Set hidden neurons

        # Positional Encoding (optional)
        self.use_positional_encoding = True if int(cfg.get("n_harmonic_functions_xyz", 0)) > 0 else False
        if self.use_positional_encoding:
            self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
            embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        else:
            embedding_dim_xyz = 3  # No encoding, input is (x, y, z)        

        # MLP Architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim_xyz, self.hidden_dim))  # First Layer

        for i in range(1, self.n_layers_distance):
            if i == 3:  # Skip connection at layer 3
                self.layers.append(nn.Linear(self.hidden_dim + embedding_dim_xyz, self.hidden_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Output Layer (Predicts signed distance)
        self.sdf_layer = nn.Linear(self.hidden_dim, 1)  # SDF can be positive or negative

        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        # MLP for Color Prediction
        self.color_layers = nn.ModuleList()
        self.color_layers.append(nn.Linear(embedding_dim_xyz + self.hidden_dim, self.color_hidden_dim))  # First Layer
        self.color_layers.append(nn.ReLU())

        for i in range(1, self.n_layers_color):
            if i == 1:  # Skip connection at layer 1
                self.color_layers.append(nn.Linear(self.color_hidden_dim + embedding_dim_xyz, self.color_hidden_dim))
            else:
                self.color_layers.append(nn.Linear(self.color_hidden_dim, self.color_hidden_dim))
            self.color_layers.append(nn.ReLU())

        self.color_layers.append(nn.Linear(self.color_hidden_dim, 3))
        self.color_layers.append(nn.Sigmoid())
        self._initialize_weights()        

    def _initialize_weights(self):
        """ Xavier Initialization for MLPs (following NeRF/SDF papers) """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier Uniform Initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero Bias Initialization

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Forward pass to predict SDF values.

        Args:
            points: Tensor of shape (N, 3) representing input 3D points.
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)  # Ensure correct shape
        
        if self.use_positional_encoding:
            points = self.harmonic_embedding_xyz(points)  # Apply positional encoding

        x = points
        for i, layer in enumerate(self.layers):
            if i == 3:
                x = torch.cat([x, points], dim=-1)  # Skip connection at layer 3
            x = layer(x)
            if i < self.n_layers_distance - 1:  # Apply ReLU to all but last layer
                x = F.relu(x)

        distance = self.sdf_layer(x)  # No ReLU, output can be positive/negative
        return distance        
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Predict per-point color.

        Args:
            points: Tensor of shape (N, 3) representing input 3D points.
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)

        if self.use_positional_encoding:
            points = self.harmonic_embedding_xyz(points)

        x_points = points
        x = points
        for i, layer in enumerate(self.layers):
            if i == 3:
                x = torch.cat([x, points], dim=-1)  # Skip connection
            x = layer(x)
            if i < self.n_layers_distance - 1:
                x = F.relu(x)
        
        # Predict RGB color
        x_dist = x
        for i, layer in enumerate(self.color_layers):
            if i == 0:
                x = torch.cat([x_dist, x_points], dim=-1)  # Skip connection
            elif i == 1:
                x = torch.cat([x, x_points], dim=-1)  # Skip connection
            x = layer(x)

        return x

    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Args:
            points: Tensor of shape (N, 3) representing input 3D points.

        Output:
            (distance, points: N X 1, N X 3 Tensors, where N is number of input points)
            distance: (N, 1) tensor of SDF values.
            color: (N, 3) tensor of RGB values.
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)

        if self.use_positional_encoding:
            points = self.harmonic_embedding_xyz(points)

        x = points
        x_points = points
        for i, layer in enumerate(self.layers):
            if i == 3:
                x = torch.cat([x, points], dim=-1)  # Skip connection
            x = layer(x)
            if i < self.n_layers_distance - 1:
                x = F.relu(x)

        distance = self.sdf_layer(x)  # Signed Distance Function

        # Predict RGB color
        x_dist = x
        for i, layer in enumerate(self.color_layers):
            if i == 0:
                x = torch.cat([x_dist, x_points], dim=-1)  # Skip connection
            elif i == 1:
                x = torch.cat([x, x_points], dim=-1)  # Skip connection
            x = layer(x)

        color = x  # RGB Color Output
        return distance, color        
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
