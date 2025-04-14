import json
import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
import os
from PIL import Image


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

    focal_length = torch.tensor([[fx_ndc, fy_ndc]]).expand(n_cams, -1)
    principal_point = torch.tensor([[cx_ndc, cy_ndc]]).expand(n_cams, -1)

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

        R_all.append(torch.tensor(R_final))
        T_all.append(torch.tensor(T_final))

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


def load_ta_train_cameras(pth_path, max_train=100):
    data = torch.load(pth_path, map_location="cpu")
    cams = data["cameras"]
    train_idx = data["split"][0][:max_train]  # first 100 training indices

    R = cams["R"][train_idx]
    T = cams["T"][train_idx]
    focal_length = cams["focal_length"][train_idx]
    principal_point = cams["principal_point"][train_idx]

    cameras = PerspectiveCameras(
        R=R, T=T,
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=True,
        device="cpu"
    )
    return cameras    


if __name__ == "__main__":
    nerf_json = "/data/patrick/nerf_synthetic/materials/transforms_train.json"
    ta_path = "/home/patrick/CMU_16825_Learning_for_3D_Vision/CMU_16825_Learning_For_3D_Vision/proj4/Q1/data/materials/materials.pth"

    ta_cameras = load_ta_train_cameras(ta_path)
    nerf_cameras = load_nerfsynthetic_camera(nerf_json, num_cams=100)

    print(f"Num TA train cameras    : {len(ta_cameras)}")
    print(f"Num NeRF train cameras  : {len(nerf_cameras)}\n")

    focal_diff_thresh = 1e-5
    principal_diff_thresh = 1e-5
    rotation_diff_thresh = 1e-5
    translation_diff_thresh = 1e-5
    for i in range(min(len(ta_cameras), len(nerf_cameras))):
        if not torch.allclose(
            ta_cameras.focal_length[i].to(torch.float64),
            nerf_cameras.focal_length[i].to(torch.float64),
            atol=focal_diff_thresh
        ):
            raise ValueError(f"[Camera {i}] ❌ Focal length mismatch")            

        if not torch.allclose(
            ta_cameras.principal_point[i].to(torch.float64),
            nerf_cameras.principal_point[i].to(torch.float64),
            atol=principal_diff_thresh
        ):
            raise ValueError(f"[Camera {i}] ❌ Principal point mismatch")

        if not torch.allclose(
            ta_cameras.R[i].to(torch.float64),
            nerf_cameras.R[i].to(torch.float64),
            atol=rotation_diff_thresh
        ):
            raise ValueError(f"[Camera {i}] ❌ Rotation matrix mismatch")

        if not torch.allclose(
            ta_cameras.T[i].to(torch.float64),
            nerf_cameras.T[i].to(torch.float64),
            atol=translation_diff_thresh
        ):
            raise ValueError(f"[Camera {i}] ❌ Translation vector mismatch")


        # If all is good, print normal info
        print(f"--- Camera {i} ---")
        print(f"[TA]   Focal       : {ta_cameras.focal_length[i].tolist()}")
        print(f"[Nerf] Focal       : {nerf_cameras.focal_length[i].tolist()}")
        print(f"[TA]   PrincipalPt : {ta_cameras.principal_point[i].tolist()}")
        print(f"[Nerf] PrincipalPt : {nerf_cameras.principal_point[i].tolist()}")
        print(f"[TA]   R           :\n{ta_cameras.R[i]}")
        print(f"[Nerf] R           :\n{nerf_cameras.R[i]}")
        print(f"[TA]   T           : {ta_cameras.T[i].tolist()}")
        print(f"[Nerf] T           : {nerf_cameras.T[i].tolist()}")
        print()        

    print("✅ All cameras matched within thresholds!")

