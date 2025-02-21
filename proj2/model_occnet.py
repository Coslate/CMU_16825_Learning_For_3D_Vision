from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import torch.nn.functional as F

class ImageConditionedOccupancyDecoder(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        self.alpha = 0.2
        super(ImageConditionedOccupancyDecoder, self).__init__()
        self.fc1 = nn.Linear(feature_dim + 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  # Output occupancy probability
        
    def forward(self, z, coords):
        """
        z: (B, 512) - Latent feature from ResNet18
        coords: (B * args.n_sample_pt, 3) - Query points in 3D space
        Returns: (B * args.n_sample_pt, 1) - Occupancy probabilities
        """
        z = z.repeat(coords.shape[0] // z.shape[0], 1)  # Expand latent to match coords
        x = torch.cat([z, coords], dim=-1)  # Concatenate latent and coords
        
        x      = F.leaky_relu(self.fc1(x), negative_slope=self.alpha) #1024
        x      = F.leaky_relu(self.fc2(x), negative_slope=self.alpha) #512
        x_res  = F.leaky_relu(self.fc3(x), negative_slope=self.alpha) #256
        x      = F.leaky_relu(self.fc4(x_res), negative_slope=self.alpha) #256
        x_res2 = x + x_res
        x      = F.leaky_relu(self.fc5(x_res2), negative_slope=self.alpha) #256
        x      = x + x_res2
        #x = torch.sigmoid(self.fc_out(x))  # Predict occupancy probability
        x = self.fc_out(x)  # Predict occupancy probability
        return x

class ImageConditionedOccupancyNetwork(nn.Module):
    def __init__(self, args, feature_dim=512, hidden_dim=256):
        super(ImageConditionedOccupancyNetwork, self).__init__()
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
