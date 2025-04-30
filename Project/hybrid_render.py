import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import yaml
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm
import torch.nn.functional as F
import time

from model import Scene, Gaussians
from model_uncertainty import TinyUNet, UncertaintyMLP
from train_nerf.volume_rendering_main import create_model, render_images
from train_nerf.dataset import (
    trivial_collate,
)
from train_nerf.data_utils import (
    create_surround_cameras,
)
from train_nerf.ray_utils import (
    get_pixels_from_image,
    get_rays_from_pixels
)
from train_uncertainty_map import extract_features, compute_ssim_error_map, ndc_to_screen_camera, predict_uncertainty_map
from data_utils_harder_scene import get_nerf_datasets
import sys
sys.path.append(os.path.abspath("train_nerf"))

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def render_worker(queue, device_id, cameras_chunk, cfg):
    torch.cuda.set_device(device_id)
    model, _, _, _, _ = create_model(cfg)
    model = model.to(f"cuda:{device_id}").eval()
    with torch.no_grad():
        imgs = render_images(model, cameras_chunk, cfg.data.image_size, file_prefix=f'nerf_{device_id}')
    queue.put((device_id, imgs))    

def parallel_render_images(model, cfg, args, base, device_ids=[0]):
    full_cameras = create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0)
    n_devices = len(device_ids)
    chunk_size = len(full_cameras) // n_devices
    camera_chunks = [full_cameras[i*chunk_size:(i+1)*chunk_size] for i in range(n_devices)]
    if len(full_cameras) % n_devices != 0:
        camera_chunks[-1] += full_cameras[n_devices*chunk_size:]

    queue = mp.Queue()
    procs = []
    for dev_id, cam_chunk in zip(device_ids, camera_chunks):
        p = mp.Process(target=render_worker, args=(queue, dev_id, cam_chunk, cfg))
        p.start()
        procs.append(p)

    all_images = []
    for _ in procs:
        _, imgs = queue.get()
        all_images.extend(imgs)

    for p in procs:
        p.join()

    out_file_name = os.path.join(args.out_path, f"{base}_testnerf.gif")
    imageio.mimsave(f'{out_file_name}', [np.uint8(im * 255) for im in all_images], loop=0)    

def render_nerf(model, base, args, cfg):
    with torch.no_grad():
        test_images = render_images(
            model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
            cfg.data.image_size, file_prefix='nerf', return_np=False
        )

        test_images_interp = []
        for img in test_images:
            img_tensor = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            #upscaled = F.interpolate(img_tensor, scale_factor=4.0, mode='bicubic', align_corners=False)  # [1, 3, 512, 512]
            upscaled = F.interpolate(img_tensor, scale_factor=4.0, mode='bilinear', align_corners=False, antialias=True)
            img_up = (upscaled.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # [512, 512, 3]
            test_images_interp.append(img_up)
        test_images = [x for x in test_images_interp]
        #test_images = [(x.detach().cpu().numpy()*255).astype(np.uint8) for x in test_images]
        out_file_name = os.path.join(args.out_path, f"{base}_testnerf.gif")
        imageio.mimsave(f'{out_file_name}', [im for im in test_images], loop=0)

def render_gs(scene, base, args, cfg):
    with torch.no_grad():
        camera = [ndc_to_screen_camera(camera_item).cuda() for camera_item in create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0)]
        test_images = []
        for camera_i in tqdm(camera):
            pred_img, _, _ = scene.render(
                                camera_i,
                                per_splat=args.gaussians_per_splat,
                                img_size=cfg.data.image_size,
                                bg_colour=(0.0, 0.0, 0.0)
                            )
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            #pred_img = F.interpolate(pred_img, scale_factor=4.0, mode='nearest', align_corners=False)
            pred_img = F.interpolate(pred_img, scale_factor=4.0, mode='bilinear', align_corners=False, antialias=True)
            pred_npy = (pred_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            test_images.append(pred_npy)
            #test_images.append(pred_img)

        #test_images = [(x.detach().cpu().numpy()*255).astype(np.uint8) for x in test_images]
        # Saving renderings
        out_file_name = os.path.join(args.out_path, f"{base}_testgs.gif")
        imageio.mimsave(f'{out_file_name}', test_images, loop=0)


def render_hybrid(scene, model, unc_model, base, args, cfg):
    image_size = tuple(args.image_size)
    model.eval()
    unc_model.eval()

    with torch.no_grad():
        camera_ndc = create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0)
        test_images_nerf = render_images(
            model, camera_ndc,
            cfg.data.image_size, file_prefix='nerf', return_np=False
        )

        camera = [ndc_to_screen_camera(camera_item).cuda() for camera_item in camera_ndc]
        test_images = []
        for idx, camera_i in enumerate(tqdm(camera)):
            pred_img, _, _ = scene.render(
                                camera_i,
                                per_splat=args.gaussians_per_splat,
                                img_size=cfg.data.image_size,
                                bg_colour=(0.0, 0.0, 0.0)
                            )

            features, timing = extract_features(scene, camera_i, args, img_size=image_size, time_profiling=True)
            if args.use_tiny_unet:
                pred_uncertainty = predict_uncertainty_map(unc_model, features)
            else:
                pred_uncertainty = unc_model(features.view(-1, 4)).view(image_size[0], image_size[1]) #(H, W)

            unc_flat = pred_uncertainty.view(-1)

            # Get top-k indices and values
            k = int(args.threshold * unc_flat.numel())
            topk_vals, topk_idxs = torch.topk(unc_flat, k=k, largest=True, sorted=False)

            # Create mask directly from top-k indices
            flat_mask = torch.zeros_like(unc_flat, dtype=torch.bool)
            flat_mask[topk_idxs] = True
            mask = flat_mask.view(pred_uncertainty.shape)  # reshape to [H, W]

            # Calculate the threshold k value for mask
            # Flatten to compute percentile
            hybrid = pred_img.clone()
            hybrid[mask] = test_images_nerf[idx][mask]

            if idx == 0:
                replace_ratio = mask.sum().item() / mask.numel()
                print(f"Hybrid GS-NeRF replacement rate: {replace_ratio * 100:.2f}%")

            hybrid = hybrid.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            hybrid = F.interpolate(hybrid, scale_factor=4.0, mode='bilinear', align_corners=False, antialias=True)
            hybrid_npy = (hybrid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            test_images.append(hybrid_npy)

        #test_images = [(x.detach().cpu().numpy()*255).astype(np.uint8) for x in test_images]
        # Saving renderings
        out_file_name = os.path.join(args.out_path, f"{base}_testhybrid.gif")
        imageio.mimsave(f'{out_file_name}', test_images, loop=0)

@torch.no_grad()
def visualize_hybrid_all_render(cfg, args):
    device = args.device
    image_size = tuple(args.image_size)

    few_views = cfg.get("few_views", 0)
    _, val_dataset, test_dataset = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        data_root=cfg.data_path,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
        few_views=few_views
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # NeRF Model Loading
    model, _, _, _, _ = create_model(cfg)
    model = model.to(device).eval()

    # GS Model Loading
    gaussians = Gaussians(num_points=args.init_random_numpoints, init_type="random", device=args.device, isotropic=False)
    scene = Scene(gaussians)
    gs_ckpt = torch.load(args.gs_ckpt)
    scene.gaussians.load_state_dict(gs_ckpt['gaussians_state_dict'])

    # Uncertainty Prediction Model Loading
    if args.use_tiny_unet:
        unc_model = TinyUNet(in_channels=4, out_channels=1).to(device)
    else:
        unc_model = UncertaintyMLP(input_dim=4).to(device)

    unc_model.load_state_dict(torch.load(args.uncertainty_ckpt)['model_state_dict'])
    unc_model.eval()

    ssim_gs_total, ssim_nerf_total, ssim_hybrid_total = 0, 0, 0
    psnr_gs_total, psnr_nerf_total, psnr_hybrid_total = 0, 0, 0
    time_gs, time_nerf = [], []
    time_hybrid_extract_feature, time_hybrid_render = [], []
    time_alpha = []
    time_color = []
    time_footprint = []
    time_viewdir = []
    time_feat = []
    time_proj_gau = []
    time_gau_act = []
    time_depth_sorting = []
    time_depth_compute = []
    time_forward = []
    time_flat = []
    time_topk = []
    time_mask = []

    #for val_index, sample in enumerate(tqdm(val_dataloader)):
    for val_index, sample in enumerate(tqdm(test_dataloader)):
        # -------------------Rendering-----------------------#
        gt_image = sample[0]["image"].cuda()
        camera_ndc = sample[0]["camera"].cuda()
        camera_screen = ndc_to_screen_camera(camera_ndc).cuda()

        # GS render
        #start = time.time()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        gs_img, gs_depth, _ = scene.render(camera_screen, per_splat=-1, img_size=image_size, bg_colour=(0, 0, 0))
        end_event.record()

        torch.cuda.synchronize()  # Wait for render to complete
        elapsed_time_gs = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_gs.append(elapsed_time_gs)

        # NeRF render
        #start = time.time()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        H, W = image_size
        xy_grid = get_pixels_from_image(image_size, camera_ndc)
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera_ndc)
        out = model(ray_bundle)
        nerf_img = out["feature_fine"].reshape(H, W, 3) if "feature_fine" in out else out["feature"].reshape(H, W, 3)
        end_event.record()

        torch.cuda.synchronize()  # Wait for NeRF render to complete
        elapsed_time_nerf = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_nerf.append(elapsed_time_nerf)

        # Hybrid render
        #start_init = time.time()
        #start = time.time()
        start_event_init = torch.cuda.Event(enable_timing=True)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_event_init.record()
        features, timing = extract_features(scene, camera_screen, args, img_size=image_size, time_profiling=True)
        end_event.record()
        
        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_feat = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_feat.append(elapsed_time_feat)

        # model forward
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if args.use_tiny_unet:
            pred_uncertainty = predict_uncertainty_map(unc_model, features)
        else:
            pred_uncertainty = unc_model(features.view(-1, 4)).view(H, W) #(H, W)
        end_event.record()

        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_forward = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_forward.append(elapsed_time_forward)

        # uncertainty flat
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        unc_flat = pred_uncertainty.view(-1)
        end_event.record()

        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_flat = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_flat.append(elapsed_time_flat)

        # Get top-k indices and values
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        k = int(args.threshold * unc_flat.numel())
        topk_vals, topk_idxs = torch.topk(unc_flat, k=k, largest=True, sorted=False)
        end_event.record()

        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_topk = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_topk.append(elapsed_time_topk)

        # Create mask directly from top-k indices
        start_event = torch.cuda.Event(enable_timing=True)
        end_event_init = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        flat_mask = torch.zeros_like(unc_flat, dtype=torch.bool)
        flat_mask[topk_idxs] = True
        mask = flat_mask.view(pred_uncertainty.shape)  # reshape to [H, W]

        end_event.record()
        end_event_init.record()
        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_mask = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_mask.append(elapsed_time_mask)
        time_hybrid_extract_feature.append(start_event_init.elapsed_time(end_event_init) / 1000.0)
        time_alpha.append(timing['alpha_sum_time'])
        time_color.append(timing['color_var_time'])
        time_footprint.append(timing['footprint_time'])
        time_viewdir.append(timing['view_direction_time'])
        time_proj_gau.append(timing['proj_gau_time'])
        time_gau_act.append(timing['gau_act_time'])
        time_depth_sorting.append(timing['depth_sorting_time'])
        time_depth_compute.append(timing['depth_compute_time'])

        # Re-generate rays only on masked pixels for hybrid render
        hybrid = gs_img.clone().view(-1, 3)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        xy_all = get_pixels_from_image(image_size, camera_ndc)
        mask_indices = mask.view(-1).nonzero(as_tuple=False).squeeze(1).to(xy_all.device)
        xy_selected = xy_all[mask_indices]
        if xy_selected.numel() > 0:
            ray_bundle_masked = get_rays_from_pixels(xy_selected, image_size, camera_ndc)
            out_masked = model(ray_bundle_masked)
            nerf_selected = out_masked["feature_fine"] if "feature_fine" in out_masked else out_masked["feature"]
            hybrid[mask.view(-1)] = nerf_selected
        else:
            print(f"No high-uncertainty pixels selected - skipped NeRF fall back for this frame.")
        end_event.record()
        torch.cuda.synchronize()  # Wait for any previous GPU ops to finish
        elapsed_time_render = start_event.elapsed_time(end_event) / 1000.0  # milliseconds -> seconds
        time_hybrid_render.append(elapsed_time_render)

        # -------------------Evaluate-----------------------#
        gt_np = gt_image.detach().cpu().numpy()
        gs_np = gs_img.detach().cpu().numpy()
        nerf_np = nerf_img.detach().cpu().numpy()
        hybrid_np = hybrid.view(H, W, 3).detach().cpu().numpy()

        ssim_gs_total += structural_similarity(gt_np, gs_np, channel_axis=-1, data_range=1.0)
        ssim_nerf_total += structural_similarity(gt_np, nerf_np, channel_axis=-1, data_range=1.0)
        ssim_hybrid_total += structural_similarity(gt_np, hybrid_np, channel_axis=-1, data_range=1.0)

        psnr_gs_total += peak_signal_noise_ratio(gt_np, gs_np, data_range=1.0)
        psnr_nerf_total += peak_signal_noise_ratio(gt_np, nerf_np, data_range=1.0)
        psnr_hybrid_total += peak_signal_noise_ratio(gt_np, hybrid_np, data_range=1.0)

        if val_index in args.inspect_ids:
            # === Evaluate ===
            gs_depth_np = gs_depth.detach().cpu().numpy()
            nerf_img = nerf_img.detach()
            nerf_depth = out["depth_fine"].reshape(H, W) if "depth_fine" in out else out["depth"].reshape(H, W)
            nerf_vis = (nerf_img.cpu().numpy()*255).astype(np.uint8)
            nerf_depth_np = nerf_depth.detach().cpu().numpy()
            pred_uncertainty_vis = (pred_uncertainty.detach().cpu().numpy() * 255).astype(np.uint8)

            gt_vis = (gt_np*255).astype(np.uint8)
            gs_vis = (gs_np*255).astype(np.uint8)
            hybrid_vis = (hybrid_np*255).astype(np.uint8)
            gs_depth_vis = (gs_depth_np*255).astype(np.uint8).squeeze()
            nerf_depth_vis = (nerf_depth_np*255).astype(np.uint8).squeeze()

            ssim_gs = structural_similarity(gt_np, gs_np, channel_axis=-1, data_range=1.0)
            psnr_gs = peak_signal_noise_ratio(gt_np, gs_np, data_range=1.0)

            ssim_hybrid = structural_similarity(gt_np, hybrid_np, channel_axis=-1, data_range=1.0)
            psnr_hybrid = peak_signal_noise_ratio(gt_np, hybrid_np, data_range=1.0)

            print(f"[Val {val_index}] SSIM - GS: {ssim_gs:.4f}, Hybrid: {ssim_hybrid:.4f}")
            print(f"[Val {val_index}] PSNR - GS: {psnr_gs:.2f}, Hybrid: {psnr_hybrid:.2f}")

            # === Save hybrid + RGB + depth renderings ===
            ssim_error = compute_ssim_error_map(gs_img, gt_image)
            ssim_error_np = ssim_error.detach().cpu().numpy()
            ssim_error_vis = (ssim_error_np*255).astype(np.uint8)
            mask_vis = mask.cpu().numpy().astype(np.uint8) * 255


            # === Save each image individually ===
            base = f"val{val_index:03d}"
            os.makedirs((os.path.join(args.out_path, f"cam_view_val_index_{base}")), exist_ok=True)

            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_gt.png"), gt_vis)
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_gs_pred.png"), gs_vis)
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_nerf_pred.png"), nerf_vis)
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_ssim_error.png"), ssim_error_vis, cmap='magma')
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_pred_uncertainty.png"), pred_uncertainty_vis, cmap='magma')
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_mask_threshold.png"), mask_vis, cmap='gray')
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_hybrid_render.png"), hybrid_vis)
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_gs_depth.png"), gs_depth_vis, cmap='magma')
            plt.imsave(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_nerf_depth.png"), nerf_depth_vis, cmap='magma')

            # === Save debug comparison: Predicted Uncertainty vs Mask (optional) ===
            fig_debug, debug_axes = plt.subplots(1, 2, figsize=(8, 6))
            debug_axes[0].imshow(pred_uncertainty_vis, cmap='magma')
            debug_axes[0].set_title("Predicted Uncertainty", fontsize=14)
            debug_axes[0].axis('off')
            debug_axes[1].imshow(mask_vis, cmap='gray')
            debug_axes[1].set_title("Mask After Thresholding", fontsize=14)
            debug_axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_path, f"cam_view_val_index_{base}", f"{base}_uncertainty_vs_mask.png"), dpi=300)
            plt.close(fig_debug)            

            '''
            fig, axes = plt.subplots(2, 4, figsize=(16, 14))
            # ====== Top row (row 0) ======
            axes[0, 0].axis('off')  # Empty
            axes[0, 1].imshow(gt_vis)
            axes[0, 1].set_title("Ground Truth")
            axes[0, 2].imshow(gs_vis)
            axes[0, 2].set_title("GS Prediction")
            axes[0, 3].imshow(nerf_vis)
            axes[0, 3].set_title("NeRF Prediction")

            # ====== Bottom row (row 1) ======
            axes[1, 0].imshow(ssim_error_vis, cmap='magma')
            axes[1, 0].set_title("SSIM Error Map")
            axes[1, 1].imshow(pred_uncertainty_vis, cmap='magma')
            axes[1, 1].set_title("Predicted Uncertainty")
            axes[1, 2].imshow(mask_vis, cmap='gray')
            axes[1, 2].set_title("Mask After Thresholding")
            axes[1, 3].imshow(hybrid_vis)
            axes[1, 3].set_title("Hybrid Render")            

            for ax in axes.flat: ax.axis('off')
            plt.tight_layout()

            os.makedirs(args.out_path, exist_ok=True)
            base = f"val{val_index:03d}"
            plt.savefig(os.path.join(args.out_path, f"{base}_hybrid_visualization.png"), dpi=300)
            plt.imsave(os.path.join(args.out_path, f"{base}_hybrid_rgb.png"), hybrid_vis)
            plt.imsave(os.path.join(args.out_path, f"{base}_gs_depth.png"), gs_depth_vis, cmap='magma')
            plt.imsave(os.path.join(args.out_path, f"{base}_nerf_depth.png"), nerf_depth_vis, cmap='magma')
            plt.close()

            # === Optional Debug Plot: Predicted Uncertainty vs Mask ===
            fig_debug, debug_axes = plt.subplots(1, 2, figsize=(8, 6))
            debug_axes[0].imshow(pred_uncertainty_vis, cmap='magma')
            debug_axes[0].set_title("Predicted Uncertainty", fontsize=14)
            debug_axes[0].axis('off')
            debug_axes[1].imshow(mask_vis, cmap='gray')
            debug_axes[1].set_title("Mask After Thresholding", fontsize=14)
            debug_axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_path, f"{base}_uncertainty_vs_mask.png"), dpi=300)
            plt.close(fig_debug)    
            '''

    #num = len(val_dataloader)
    num = len(test_dataloader)
    print("=== Average Metrics ===")
    print(f"GS      - PSNR: {psnr_gs_total/num:.2f}, SSIM: {ssim_gs_total/num:.4f}, Time/frame: {np.mean(time_gs):.3f}s")
    print(f"NeRF    - PSNR: {psnr_nerf_total/num:.2f}, SSIM: {ssim_nerf_total/num:.4f}, Time/frame: {np.mean(time_nerf):.3f}s")
    print(f"Hybrid  - PSNR: {psnr_hybrid_total/num:.2f}, SSIM: {ssim_hybrid_total/num:.4f}, Time/frame-extract_feature: {np.mean(time_hybrid_extract_feature):.3f}s, Time/frame-render: {np.mean(time_hybrid_render):.3f}s")
    print(f"Breakdown for Time/frame-extract_feature:")
    print(f"Time/frame-feat: {np.mean(time_feat):.3f}s")
    print(f"|")
    print(f"--Time/frame-proj_gau: {np.mean(time_proj_gau):.3f}s")
    print(f"--Time/frame-gau_act: {np.mean(time_gau_act):.3f}s")
    print(f"--Time/frame-depth_sorting: {np.mean(time_depth_sorting):.3f}s")
    print(f"--Time/frame-depth_compute: {np.mean(time_depth_compute):.3f}s")
    print(f"--Time/frame-alpha_sum: {np.mean(time_alpha):.3f}s")
    print(f"--Time/frame-color_variance: {np.mean(time_color):.3f}s")
    print(f"--Time/frame-footprint_area: {np.mean(time_footprint):.3f}s")
    print(f"--Time/frame-view_direction: {np.mean(time_viewdir):.3f}s")
    print(f"Time/frame-forward: {np.mean(time_forward):.3f}s")
    print(f"Time/frame-flat: {np.mean(time_flat):.3f}s")
    print(f"Time/frame-topk: {np.mean(time_topk):.3f}s")
    print(f"Time/frame-mask: {np.mean(time_mask):.3f}s")
    print(f"")

    os.makedirs(args.out_path, exist_ok=True)
    base = f"val"
    render_nerf(model, base, args, cfg)
    render_gs(scene, base, args, cfg)
    render_hybrid(scene, model, unc_model, base, args, cfg)

@torch.no_grad()
def visualize_hybrid_render(cfg, args):
    device = args.device
    image_size = tuple(args.image_size)

    # === Load datasets ===
    few_views    = cfg.get("few_views", 0)
    _, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        data_root=cfg.data_path,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
        few_views = few_views
    )

    val_sample = val_dataset[args.val_index]
    gt_image = val_sample["image"].to(device)
    camera_ndc = val_sample["camera"].to(device)
    camera_screen = ndc_to_screen_camera(camera_ndc).cuda()

    # === Step 5.1: Render base image using GS ===
    gaussians = Gaussians(num_points=args.init_random_numpoints, init_type="random", device=args.device, isotropic=False)
    scene = Scene(gaussians)
    gs_ckpt = torch.load(args.gs_ckpt)
    scene.gaussians.load_state_dict(gs_ckpt['gaussians_state_dict'])
    gs_img, gs_depth, _ = scene.render(camera_screen, per_splat=-1, img_size=image_size, bg_colour=(0, 0, 0))

    # === Step 5.2: Identify fallback pixels via confidence mask ===
    unc_model = TinyUNet(in_channels=4, out_channels=1).to(device)
    unc_model.load_state_dict(torch.load(args.uncertainty_ckpt)['model_state_dict'])
    unc_model.eval()
    features = extract_features(scene, camera_screen, args, img_size=image_size)

    if args.use_tiny_unet:
        pred_uncertainty = predict_uncertainty_map(unc_model, features)
    else:
        pred_uncertainty = unc_model(features.view(-1, 4)).view(H, W) #(H, W)

    unc_flat = pred_uncertainty.view(-1)

    # Get top-k indices and values
    k = int(args.threshold * unc_flat.numel())
    topk_vals, topk_idxs = torch.topk(unc_flat, k=k, largest=True, sorted=False)

    # Create mask directly from top-k indices
    flat_mask = torch.zeros_like(unc_flat, dtype=torch.bool)
    flat_mask[topk_idxs] = True
    mask = flat_mask.view(pred_uncertainty.shape)  # reshape to [H, W]

    print(f"[DEBUG] Mask sum: {mask.sum().item()} / {mask.numel()} ({mask.sum().item()/mask.numel()*100}%) pixels above threshold")    

    # === Step 5.3: Re-render fallback rays via NeRF ===
    print(f"cfg.data.image_size = {cfg.data.image_size}")
    model, _, _, _, _ = create_model(cfg)
    model = model.to(device)
    model.eval()
    H, W = image_size
    xy_grid = get_pixels_from_image(image_size, camera_ndc)
    ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera_ndc)
    out = model(ray_bundle)
    nerf_img = out["feature_fine"].reshape(H, W, 3) if "feature_fine" in out else out["feature"].reshape(H, W, 3)
    nerf_img = nerf_img.detach()
    nerf_depth = out["depth_fine"].reshape(H, W) if "depth_fine" in out else out["depth"].reshape(H, W)
    nerf_vis = (nerf_img.cpu().numpy()*255).astype(np.uint8)

    # === Step 5.4: Compose Hybrid ===
    hybrid = gs_img.clone()
    hybrid[mask] = nerf_img[mask]
    #hybrid[mask] = gt_image.detach().cpu()[mask]

    # === Evaluate ===
    gt_np = gt_image.detach().cpu().numpy()
    gs_np = gs_img.detach().cpu().numpy()
    hybrid_np = hybrid.detach().cpu().numpy()
    gs_depth_np = gs_depth.detach().cpu().numpy()
    nerf_depth_np = nerf_depth.detach().cpu().numpy()

    gt_vis = (gt_np*255).astype(np.uint8)
    gs_vis = (gs_np*255).astype(np.uint8)
    hybrid_vis = (hybrid_np*255).astype(np.uint8)
    gs_depth_vis = (gs_depth_np*255).astype(np.uint8).squeeze()
    nerf_depth_vis = (nerf_depth_np*255).astype(np.uint8).squeeze()

    ssim_gs = structural_similarity(gt_np, gs_np, channel_axis=-1, data_range=1.0)
    psnr_gs = peak_signal_noise_ratio(gt_np, gs_np, data_range=1.0)

    ssim_hybrid = structural_similarity(gt_np, hybrid_np, channel_axis=-1, data_range=1.0)
    psnr_hybrid = peak_signal_noise_ratio(gt_np, hybrid_np, data_range=1.0)

    print(f"[Val {args.val_index}] SSIM - GS: {ssim_gs:.4f}, Hybrid: {ssim_hybrid:.4f}")
    print(f"[Val {args.val_index}] PSNR - GS: {psnr_gs:.2f}, Hybrid: {psnr_hybrid:.2f}")

    # === Save hybrid + RGB + depth renderings ===
    ssim_error = compute_ssim_error_map(gs_img, gt_image)
    ssim_error_np = ssim_error.detach().cpu().numpy()
    ssim_error_vis = (ssim_error_np*255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(14, 14))
    axes[0, 0].imshow(gt_vis); axes[0, 0].set_title("Ground Truth")
    axes[0, 1].imshow(gs_vis); axes[0, 1].set_title("GS Prediction")
    axes[0, 2].imshow(nerf_vis); axes[0, 2].set_title("NeRF Prediction")
    axes[1, 0].imshow(ssim_error_vis, cmap='magma'); axes[1, 0].set_title("SSIM Error Map")
    axes[1, 1].imshow(pred_uncertainty_vis, cmap='magma'); axes[1, 1].set_title("Predicted Uncertainty")
    axes[1, 2].imshow(hybrid_vis); axes[1, 2].set_title("Hybrid Render")
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()

    os.makedirs(args.out_path, exist_ok=True)
    base = f"val{args.val_index:03d}"
    plt.savefig(os.path.join(args.out_path, f"{base}_hybrid_visualization.png"), dpi=300)
    plt.imsave(os.path.join(args.out_path, f"{base}_hybrid_rgb.png"), hybrid_vis)
    plt.imsave(os.path.join(args.out_path, f"{base}_gs_depth.png"), gs_depth_vis, cmap='magma')
    plt.imsave(os.path.join(args.out_path, f"{base}_nerf_depth.png"), nerf_depth_vis, cmap='magma')
    plt.close()

    # === Optional Debug Plot: Predicted Uncertainty vs Mask ===
    mask_vis = mask.cpu().numpy().astype(np.uint8) * 255
    fig_debug, debug_axes = plt.subplots(1, 2, figsize=(8, 6))
    debug_axes[0].imshow(pred_uncertainty_vis, cmap='magma')
    debug_axes[0].set_title("Predicted Uncertainty", fontsize=14)
    debug_axes[0].axis('off')
    debug_axes[1].imshow(mask_vis, cmap='gray')
    debug_axes[1].set_title("Mask After Thresholding", fontsize=14)
    debug_axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path, f"{base}_uncertainty_vs_mask.png"), dpi=300)
    plt.close(fig_debug)    


    render_nerf(model, base, args, cfg)
    render_gs(scene, base, args, cfg)

    #cfg.data.image_size = (400, 400)
    #cfg.data.image_size = (256, 256)
    #cfg.sampler.n_pts_per_ray = 300
    #model, _, _, _, _ = create_model(cfg)
    #model = model.to(device)
    #model.eval()
    '''
    with torch.no_grad():
        test_images = render_images(
            model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
            cfg.data.image_size, file_prefix='nerf', return_np=False
        )

        test_images_interp = []
        for img in test_images:
            img_tensor = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            #upscaled = F.interpolate(img_tensor, scale_factor=4.0, mode='bicubic', align_corners=False)  # [1, 3, 512, 512]
            upscaled = F.interpolate(img_tensor, scale_factor=4.0, mode='bilinear', align_corners=False, antialias=True)
            img_up = (upscaled.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # [512, 512, 3]
            test_images_interp.append(img_up)
        test_images = [x for x in test_images_interp]
        #test_images = [(x.detach().cpu().numpy()*255).astype(np.uint8) for x in test_images]
        out_file_name = os.path.join(args.out_path, f"{base}_testnerf.gif")
        imageio.mimsave(f'{out_file_name}', [im for im in test_images], loop=0)

    with torch.no_grad():
        camera = [ndc_to_screen_camera(camera_item).cuda() for camera_item in create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0)]
        test_images = []
        for camera_i in tqdm(camera):
            pred_img, _, _ = scene.render(
                                camera_i,
                                per_splat=args.gaussians_per_splat,
                                img_size=cfg.data.image_size,
                                bg_colour=(0.0, 0.0, 0.0)
                            )
            pred_img = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            #pred_img = F.interpolate(pred_img, scale_factor=4.0, mode='nearest', align_corners=False)
            pred_img = F.interpolate(pred_img, scale_factor=4.0, mode='bilinear', align_corners=False, antialias=True)
            pred_npy = (pred_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            test_images.append(pred_npy)
            #test_images.append(pred_img)

        #test_images = [(x.detach().cpu().numpy()*255).astype(np.uint8) for x in test_images]
        # Saving renderings
        out_file_name = os.path.join(args.out_path, f"{base}_testgs.gif")
        imageio.mimsave(f'{out_file_name}', test_images, loop=0)
    '''
    # === Parallel Test Rendering ===
    #cfg.data.image_size = (400, 400)
    #parallel_render_images(model, cfg, args, base, device_ids=args.device_ids)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, default="./train_nerf/configs/nerf_materials_highres.yaml", help="YAML config file for NeRF.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Image height and width")
    parser.add_argument("--data_path", type=str, default="./data/materials")
    parser.add_argument("--gs_ckpt", type=str, required=True)
    parser.add_argument("--uncertainty_ckpt", type=str, required=True)
    parser.add_argument("--init_random_numpoints", type=int, default=15000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, default="./output_hybrid")
    parser.add_argument("--threshold", type=float, default=0.2, help='The top threshold*100 % pixels in uncertainty_map will be used to fallback for NeRF.')
    parser.add_argument("--val_index", type=int, default=0)
    parser.add_argument('--chunksize_colvar', default=1024, type=int, help="The chunk size of calculating color vairance.")
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="List of GPU device IDs to use")
    parser.add_argument("--inspect_ids", type=int, nargs='+', default=[0], help="List of validation index to inspect the rendering result. Use with --render_all")
    parser.add_argument("--render_all", action="store_true", help="Run all validation dataset and calculate mean PSNR/SSIM and renders --inspect_ids instances. Otherwise, only calculate PSNR/SSIM on --val_index instance and render the result.")
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument("--use_tiny_unet", action="store_true", help="Whether to use tiny unet or MLP. 1: tiny unet; 0: MLP")
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    #print(OmegaConf.to_yaml(cfg))
    print(f"cfg.sampler.type = {cfg.sampler.type}")
    print(f"cfg.sampler.use_fine_sampling = {cfg.sampler.use_fine_sampling}")
    print(f"cfg.sampler.n_coarse_pts_per_ray= {cfg.sampler.n_coarse_pts_per_ray}")
    print(f"cfg.sampler.n_fine_pts_per_ray = {cfg.sampler.n_fine_pts_per_ray}")

    if args.render_all:
        visualize_hybrid_all_render(cfg, args)
    else:
        visualize_hybrid_render(cfg, args)