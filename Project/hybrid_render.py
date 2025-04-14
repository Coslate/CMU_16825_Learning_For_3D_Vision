import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import yaml
from omegaconf import OmegaConf

from model import Scene, Gaussians
from model_uncertainty import TinyUNet
from train_nerf.volume_rendering_main import create_model, get_rays_from_pixels
from train_uncertainty_map import extract_features, compute_ssim_error_map, ndc_to_screen_camera
from data_utils_harder_scene import get_nerf_datasets
import sys
sys.path.append(os.path.abspath("train_nerf"))

@torch.no_grad()
def visualize_hybrid_render(cfg, args):
    device = args.device
    image_size = tuple(args.image_size)

    # === Load datasets ===
    _, val_dataset, _ = get_nerf_datasets("materials", args.data_path, image_size=image_size)
    val_sample = val_dataset[args.val_index]
    gt_image = val_sample["image"].to(device)
    camera_ndc = val_sample["camera"].to(device)
    camera_screen = ndc_to_screen_camera(camera_ndc)

    # === Step 5.1: Render base image using GS ===
    gaussians = Gaussians(num_points=args.init_random_numpoints, device=device, isotropic=False)
    scene = Scene(gaussians)
    gs_ckpt = torch.load(args.gs_ckpt)
    scene.gaussians.load_state_dict(gs_ckpt['gaussians_state_dict'])
    gs_img, gs_depth, _ = scene.render(camera_screen, per_splat=-1, img_size=image_size, bg_colour=(0, 0, 0))

    # === Step 5.2: Identify fallback pixels via confidence mask ===
    unc_model = TinyUNet(in_channels=4, out_channels=1).to(device)
    unc_model.load_state_dict(torch.load(args.uncertainty_ckpt)['model_state_dict'])
    unc_model.eval()
    features = extract_features(scene, camera_screen, args, img_size=image_size)
    pred_uncertainty = unc_model(features.permute(2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()
    mask = pred_uncertainty > args.threshold

    # === Step 5.3: Re-render fallback rays via NeRF ===
    print(f"cfg.data.image_size = {cfg.data.image_size}")
    model, _, _, _, _ = create_model(cfg)
    model = model.to(device)
    model.eval()
    H, W = image_size
    xy = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, H),
        indexing='ij'), dim=-1).reshape(-1, 2).to(device)
    ray_bundle = get_rays_from_pixels(xy, image_size, camera_ndc)
    out = model(ray_bundle)
    nerf_img = out["feature_fine"].reshape(H, W, 3) if "feature_fine" in out else out["feature"].reshape(H, W, 3)

    # === Step 5.4: Compose Hybrid ===
    nerf_depth = out["depth_fine"].reshape(H, W) if "depth_fine" in out else out["depth"].reshape(H, W)
    hybrid = gs_img.clone().cpu()
    hybrid[mask] = nerf_img.cpu()[mask]

    # === Evaluate ===
    gt_np = gt_image.cpu().numpy()
    gs_np = gs_img.cpu().numpy()
    hybrid_np = hybrid.numpy()

    ssim_gs = structural_similarity(gt_np, gs_np, channel_axis=-1, data_range=1.0)
    psnr_gs = peak_signal_noise_ratio(gt_np, gs_np, data_range=1.0)

    ssim_hybrid = structural_similarity(gt_np, hybrid_np, channel_axis=-1, data_range=1.0)
    psnr_hybrid = peak_signal_noise_ratio(gt_np, hybrid_np, data_range=1.0)

    print(f"[Val {args.val_index}] SSIM - GS: {ssim_gs:.4f}, Hybrid: {ssim_hybrid:.4f}")
    print(f"[Val {args.val_index}] PSNR - GS: {psnr_gs:.2f}, Hybrid: {psnr_hybrid:.2f}")

    # === Save hybrid + RGB + depth renderings ===
    ssim_error = compute_ssim_error_map(gs_img, gt_image).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 14))
    axes[0, 0].imshow(gt_np); axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(gs_np); axes[0, 1].set_title("GS Prediction")
    axes[0, 2].imshow(nerf_img.cpu().numpy()); axes[0, 2].set_title("NeRF Prediction")
    axes[1, 0].imshow(ssim_error, cmap='magma'); axes[1, 0].set_title("SSIM Error Map")
    axes[1, 1].imshow(pred_uncertainty, cmap='magma'); axes[1, 1].set_title("Predicted Uncertainty")
    axes[1, 2].imshow(hybrid_np); axes[1, 2].set_title("Hybrid Render")
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()

    os.makedirs(args.out_path, exist_ok=True)
    base = f"val{args.val_index:03d}"
    plt.savefig(os.path.join(args.out_path, f"{base}_hybrid_visualization.png"), dpi=150)
    plt.imsave(os.path.join(args.out_path, f"{base}_hybrid_rgb.png"), hybrid_np)
    plt.imsave(os.path.join(args.out_path, f"{base}_gs_depth.png"), gs_depth.squeeze().cpu().numpy(), cmap='magma')
    plt.imsave(os.path.join(args.out_path, f"{base}_mask.png"), mask.astype(np.uint8) * 255, cmap='gray')
    plt.imsave(os.path.join(args.out_path, f"{base}_nerf_depth.png"), nerf_depth.cpu().numpy(), cmap='magma')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, default="./train_nerf/configs/nerf_materials_highres.yaml", help="YAML config file for NeRF.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Image height and width")
    parser.add_argument("--data_path", type=str, default="./data/materials")
    parser.add_argument("--gs_ckpt", type=str, required=True)
    parser.add_argument("--uncertainty_ckpt", type=str, required=True)
    parser.add_argument("--init_random_numpoints", type=int, default=15000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, default="./output_hybrid")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--val_index", type=int, default=0)
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    print(OmegaConf.to_yaml(cfg))

    visualize_hybrid_render(cfg, args)