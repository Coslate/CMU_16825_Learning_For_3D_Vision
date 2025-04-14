import os
import json
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pytorch3d.renderer.cameras import PerspectiveCameras
from data_utils_harder_scene import load_nerfsynthetic_camera  # Update this import path
import torch.nn.functional as F

def load_and_resize_image(image_path, image_size):
    # Load and normalize to [0, 1]
    image = Image.open(image_path)
    image = torch.FloatTensor(np.array(image)) / 255.0  # (H, W, C)
    image = image[..., :3]

    # Resize if necessary (TA style)
    H_old, W_old = image.shape[:2]
    H_new, W_new = image_size

    scale_h = H_new / H_old
    scale_w = W_new / W_old

    if abs(scale_h - scale_w) > 1e-3:
        raise ValueError("Non-isotropic scaling is not allowed.")

    if scale_h != 1.0:
        image = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        image = F.interpolate(image, size=(H_new, W_new), mode='bilinear')
        image = image.squeeze(0).permute(1, 2, 0)  # (H, W, C)

    return image

class NeRFSyntheticDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=(800, 800), transform=None, load_depth=False):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, split)
        self.depth_dir = os.path.join(root_dir, 'depth_float/' + split + '/raw-depth')
        self.transform = transform
        self.load_depth = load_depth
        self.image_size = image_size

        # Load metadata and cameras
        json_path = os.path.join(root_dir, f"transforms_{split}.json")
        self.meta = json.load(open(json_path, 'r'))
        self.cameras = load_nerfsynthetic_camera(json_path)
        self.cameras.image_size = torch.tensor([list(image_size)] * len(self.cameras.R), dtype=torch.float32)


        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))

        self.resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_and_resize_image(self.image_paths[idx], self.image_size)
        if self.transform:
            img = self.transform(img)        
        '''
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.resize(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img) #(C, H, W)
            img = img.permute(1, 2, 0)  # (H, W, C)
        '''

        # Construct 4x4 pose matrix from PerspectiveCameras
        pose = torch.eye(4)
        pose[:3, :3] = self.cameras.R[idx]
        pose[:3, 3] = self.cameras.T[idx]

        # Create 3x3 intrinsic matrix
        fx, fy = self.cameras.focal_length[idx]
        cx, cy = self.cameras.principal_point[idx]
        intrinsics = torch.tensor([
            [fx.item(), 0, cx.item()],
            [0, fy.item(), cy.item()],
            [0, 0, 1]
        ], dtype=torch.float32)

        sample = {
            'image': img,
            'pose': pose,
            'intrinsics': intrinsics,
            'camera': self.cameras[idx],
            'camera_idx': idx
        }

        # Optional depth loading
        if self.load_depth:
            base = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
            npy_path = os.path.join(self.depth_dir, base + '_depth.npy')
            npy_gz_path = npy_path + '.gz'
            depth = None
            if os.path.exists(npy_path):
                depth = np.load(npy_path)
            elif os.path.exists(npy_gz_path):
                import gzip
                with gzip.open(npy_gz_path, 'rb') as f:
                    depth = np.load(f)

            if depth is not None:
                depth = torch.from_numpy(depth).float()
                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth.squeeze(0)
                sample['depth'] = depth
            else:
                sample['depth'] = None

        return sample
