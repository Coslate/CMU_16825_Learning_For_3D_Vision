# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import json
import os


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data"
)

DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"


def load_nerfsynthetic_camera(json_path, num_cams=None):
    with open(json_path, "r") as f:
        meta = json.load(f)

    cam_angle_x = float(meta["camera_angle_x"])
    frames = meta["frames"]
    if num_cams is not None:
        frames = frames[:num_cams]
    n_cams = len(frames)

    # Original image size for NeRF synthetic
    H, W = 800, 800

    # Compute focal length in NDC space
    fx = fy = 0.5 * W / np.tan(0.5 * cam_angle_x)
    fx_ndc = 2 * fx / W
    fy_ndc = 2 * fy / H
    cx_ndc = 0.0  # centered
    cy_ndc = 0.0  # centered

    focal_length = torch.tensor([[fx_ndc, fy_ndc]], dtype=torch.float32).expand(n_cams, -1)
    principal_point = torch.tensor([[cx_ndc, cy_ndc]], dtype=torch.float32).expand(n_cams, -1)

    R_all, T_all = [], []
    flip = np.diag([-1, 1, -1]).astype(np.float32)  # OpenGL to OpenCV
    for f in frames:
        c2w = np.array(f["transform_matrix"], dtype=np.float32)  # (4, 4)

        # Step 1: Convert NeRF-style c2w to w2c
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]

        R_w2c = R_c2w.T
        T_w2c = -R_w2c @ T_c2w

        # Step 2: Apply OpenGL to OpenCV camera convention to w2c (R and T)
        R_final = (flip @ R_w2c).T
        T_final = flip @ T_w2c        

        R_all.append(torch.tensor(R_final, dtype=torch.float32))
        T_all.append(torch.tensor(T_final, dtype=torch.float32))

    R = torch.stack(R_all)
    T = torch.stack(T_all)

    cameras = PerspectiveCameras(
        R=R, T=T,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=[[H, W]] * n_cams,
        in_ndc=True,
        device="cpu"
    )

    return cameras

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]


def get_nerf_datasets(
    dataset_name: str,  # 'lego | fern'
    image_size: Tuple[int, int],
    few_views: int = 0,
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.
        autodownload: Auto-download the dataset files in case they are missing.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """

    # if dataset_name not in ALL_DATASETS:
    #     raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")

    train_data = torch.load(cameras_path)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))[..., :3]
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]

    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

    '''
    for cami in range(n_cameras):
        for k, v in train_data['cameras'].items():
            print(f"k = {k}, value = {v[cami][None]}")
    '''

    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    train_idx, val_idx, test_idx = train_data["split"]
    if few_views != 0:
        train_idx = train_idx[:few_views] 

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset


def download_data(
    dataset_names: Optional[List[str]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    url_root: str = DEFAULT_URL_ROOT,
) -> None:
    """
    Downloads the relevant dataset files.
    """

    if dataset_names is None:
        raise RuntimeError

    os.makedirs(data_root, exist_ok=True)

    for dataset_name in dataset_names:
        cameras_file = dataset_name + ".pth"
        images_file = cameras_file.replace(".pth", ".png")
        license_file = cameras_file.replace(".pth", "_license.txt")

        for fl in (cameras_file, images_file, license_file):
            local_fl = os.path.join(data_root, fl)
            remote_fl = os.path.join(url_root, fl)

            print(f"Downloading dataset {dataset_name} from {remote_fl} to {local_fl}.")

            r = requests.get(remote_fl)

            with open(local_fl, "wb") as f:
                f.write(r.content)


class RunningAverageMeter:
    """Simple running average estimator.
    Args:
      momentum (float): running average decay.
    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.avg = 0
        self.val = None

    def update(self, val):
        """Update running average given a new value.
        The new running average estimate is given as a weighted combination \
        of the previous estimate and the current value.
        Args:
          val (float): new value
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1.0 - self.momentum)
        self.val = val

class AverageMeter:
    """Running average estimator using arithmetic mean."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
        self.val = None

    def update(self, val):
        """Update running average with new value."""
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count