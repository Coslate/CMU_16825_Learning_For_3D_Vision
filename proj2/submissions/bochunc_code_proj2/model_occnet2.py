from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return x + residual  # Residual connection

class ImageConditionedOccupancyDecoder(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=512):
        super().__init__()
        self.fc_in = nn.Linear(feature_dim + 3, hidden_dim)
        self.resblocks = nn.Sequential(
            ResNetBlock(hidden_dim),
            ResNetBlock(hidden_dim),
            ResNetBlock(hidden_dim),
            ResNetBlock(hidden_dim),
            ResNetBlock(hidden_dim),
        )
        self.fc_out = nn.Linear(hidden_dim, 1)  # Logits output

    def forward(self, z, coords):
        z = z.repeat(coords.shape[0] // z.shape[0], 1)  # Expand latent space
        x = torch.cat([z, coords], dim=-1)  # Concatenate latent and 3D coordinates
        x = F.leaky_relu(self.fc_in(x), negative_slope=0.2)
        x = self.resblocks(x)
        return self.fc_out(x)  # Return logits

class ImageConditionedOccupancyNetwork2(nn.Module):
    def __init__(self, args, feature_dim=512, hidden_dim=256):
        super(ImageConditionedOccupancyNetwork2, self).__init__()
        self.device = args.device
        self.hidden_dim = hidden_dim
        
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = nn.Sequential(*list(vision_model.children())[:-1])
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.decoder = ImageConditionedOccupancyDecoder(feature_dim, hidden_dim)
    
    def forward(self, images, coords, args):
        """
        Args:
        coords: Tensor of shape (B * num_samples, 3) - 3D points in (-1,1)^3
        images: Tensor (B, H, W, C) - Original Images

        Returns:
        occupancy probabilities: (B * num_samples, 1)
        """
        B = images.shape[0]
        num_samples = coords.shape[0] // B  # B

        if not args.load_feat:
            images_normalized = self.normalize(images.permute(0, 3, 1, 2))
            encoded_feat = self.encoder(images_normalized).squeeze(-1).squeeze(-1)  # (B, 512)
        else:
            encoded_feat = images  # Pretrained ResNet18 features
        
        return self.decoder(encoded_feat, coords)
