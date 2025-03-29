import argparse
import os
import os.path as osp
import random
import time
import math

import imageio
import numpy as np
import torch
from nerf.config_parser import add_config_arguments
from nerf.network_grid import NeRFNetwork
from nerf.provider import NeRFDataset
from optimizer import Adan
from PIL import Image
from utils import prepare_embeddings, seed_everything
import torch.nn.functional as F
import lpips
from SDS import SDS
from torchvision import transforms

def get_view_dependent_embedding(prompt, azimuth, sds, embeddings):
    """
    Returns a smoothly blended view-dependent text embedding.
    
    Args:
        prompt (str): The original text prompt.
        azimuth (float or tensor): Azimuth angle in degrees [-180, 180].
        sds (SDS): SDS object for getting text embeddings.
        embeddings (dict): Dictionary containing precomputed embeddings for 'front', 'side', 'back', 'default'.
    
    Returns:
        Tensor: Blended text embedding.
    """
    # Ensure azimuth is a float and wrap to [0, 360)
    if isinstance(azimuth, torch.Tensor):
        az = azimuth.item()
    else:
        az = float(azimuth)
    az = (az + 360) % 360
    az_rad = math.radians(az)

    # Canonical view directions
    dir_names = ["front", "side", "back", "side"]
    dir_centers = [0, 90, 180, 270]  # degrees
    dir_centers_rad = [math.radians(d) for d in dir_centers]

    # Cosine-based weights
    weights = []
    for c in dir_centers_rad:
        delta = az_rad - c
        weight = torch.cos(torch.tensor(delta))
        weights.append(torch.clamp(weight, min=0.0))

    weights = torch.tensor(weights, dtype=torch.float32, device=embeddings["default"].device)
    if weights.sum() == 0:
        weights = torch.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    # Blend the embeddings
    view_embs = [embeddings[name] for name in dir_names]
    text_cond = sum(w * e for w, e in zip(weights, view_embs))
    return text_cond


def optimize_nerf_pixel_sds(
    sds,
    prompt,
    device="cpu",
    log_interval=20,
    args=None,
):
    model = NeRFNetwork(args).to(device)

    optimizer = Adan(
        model.parameters(),
        lr=5e-3,
        eps=1e-8,
        weight_decay=2e-5,
        max_grad_norm=5.0,
        foreach=False,
    )
    if args.loss_scaling:
        scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    # Dataset
    train_loader = NeRFDataset(
        args, device=device, type="train", H=args.h, W=args.w,
        size=args.dataset_size_train * args.batch_size,
    ).dataloader()
    test_loader = NeRFDataset(
        args, device=device, type="test", H=args.h, W=args.w,
        size=args.dataset_size_test,
    ).dataloader(batch_size=1)

    # Loss
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    transform = transforms.Compose([
        transforms.Resize((args.h, args.w)),
        transforms.ToTensor()
    ])    

    # Output paths
    os.makedirs(f"{args.output_dir}/images", exist_ok=True)
    os.makedirs(f"{args.output_dir}/videos", exist_ok=True)
    checkpoint_path = osp.join(sds.output_dir, f"nerf_checkpoint_pixelsds.pth")

    loss_dict = {}
    cached_sd_images = {}
    global_step = 0
    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
    target_rgb = None
    embeddings = prepare_embeddings(sds, prompt, neg_prompt="", view_dependent=True)
    for epoch in range(max_epoch):
        model.train()
        for data in train_loader:
            global_step += 1
            optimizer.zero_grad()
            exp_iter_ratio = (global_step - args.exp_start_iter) / (
                args.exp_end_iter - args.exp_start_iter
            )

            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            mvp = data["mvp"]
            H, W = data["H"], data["W"]
            B = rays_o.shape[0]

            # Set the shading and background color for rendering
            if exp_iter_ratio <= args.latent_iter_ratio:
                ambient_ratio = 1.0
                shading = "normal"
                bg_color = None

            else:
                # random shading
                ambient_ratio = (
                    args.min_ambient_ratio
                    + (1.0 - args.min_ambient_ratio) * random.random()
                )
                rand = random.random()
                if rand >= (1.0 - args.textureless_ratio):
                    shading = "textureless"
                else:
                    shading = "lambertian"

                # random background
                rand = random.random()
                if args.bg_radius > 0 and rand > 0.5:
                    bg_color = None  # use bg_net
                else:
                    bg_color = torch.rand(3).to(device)  # single color random bg

            # Forward pass to render NeRF model
            outputs = model.render(
                rays_o,
                rays_d,
                mvp,
                H,
                W,
                staged=False,
                perturb=True,
                bg_color=bg_color,
                ambient_ratio=ambient_ratio,
                shading=shading,
                binarize=False,
                max_ray_batch=args.max_ray_batch,
            )
            pred_rgb = outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

            # Resize to LPIPS input size (typically 64 or 128, but LPIPS supports 512)
            #target_rgb = data.get("target", torch.rand_like(pred_rgb))  # Dummy target if none provided
            if global_step == 1:
                view_prompts = {
                    "front": "front view",
                    "side": "side view",
                    "back": "back view",
                    "top": "top view",
                    "bottom": "bottom view"
                }
                with torch.no_grad():
                    for k, v in view_prompts.items():
                        prompt_view = f"{prompt}, {v}"
                        print(f"[INFO] Caching SD image for: '{prompt_view}'")
                        cached_sd_images[k] = sds.text_to_image(prompt=prompt_view, height=H, width=W, guidance_scale=args.guidance_scale)            

            azimuth = data["azimuth"]  # [-180, 180]
            assert azimuth.shape[0] == 1, "Batch size should be 1"
            if global_step >= args.latent_only_iters:
                with torch.no_grad():
                    elevation = data.get("elevation", torch.tensor([0.0])).item()            

                    if elevation > 60:
                        view_key = "top"
                    elif elevation < -20:
                        view_key = "bottom"
                    else:
                        az = azimuth.item()
                        assert -180 <= az <= 180, f"Azimuth {az} out of expected range [-180, 180]"

                        # Convert to [0, 360)
                        az = (az + 360) % 360
                        if (az >= 315 or az < 45):
                            view_key = "front"
                        elif 45 <= az < 135:
                            view_key = "side"
                        elif 135 <= az < 225:
                            view_key = "back"
                        else:
                            view_key = "side"  # reuse side for both left/right                        

                    target_rgb = cached_sd_images[view_key]

            if global_step < args.latent_only_iters:
                # === MODIFIED: use latent SDS loss first ===
                text_uncond = embeddings["uncond"]
                text_cond = get_view_dependent_embedding(prompt, azimuth, sds, embeddings)
                with torch.no_grad():
                    latents = sds.encode_imgs(F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False))
                loss = sds.sds_loss(latents, text_cond, text_uncond, guidance_scale=args.guidance_scale)
            else:
                # pixel-space LPIPS + L2 (your current code)
                pred_resized = F.interpolate(pred_rgb, size=(256, 256), mode="bilinear", align_corners=False)
                target_resized = F.interpolate(target_rgb, size=(256, 256), mode="bilinear", align_corners=False)
                loss_lpips = lpips_loss_fn(pred_resized, target_resized).mean()
                loss_l2 = F.mse_loss(pred_resized, target_resized)
                l2_weight = min(0.1 * (global_step / args.iters), 0.1)
                loss = loss_lpips if global_step < args.lpips_only_iters else loss_lpips + l2_weight * loss_l2

            '''
            loss_lpips = lpips_loss_fn(pred_resized, target_resized).mean()
            loss_l2 = F.mse_loss(pred_resized, target_resized)
            l2_weight = min(0.1 * (global_step / args.iters), 0.1)

            if global_step < args.lpips_only_iters:
                loss = loss_lpips
            else:
                loss = loss_lpips + l2_weight * loss_l2
            '''
            # regularizations
            if args.lambda_entropy > 0:
                alphas = outputs["weights"].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (
                    -alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
                ).mean()
                lambda_entropy = args.lambda_entropy * min(
                    1, 2 * global_step / args.iters
                )
                loss = loss + lambda_entropy * loss_entropy

            if args.lambda_orient > 0 and "loss_orient" in outputs:
                loss_orient = outputs["loss_orient"]
                loss = loss + args.lambda_orient * loss_orient

            # Backward pass
            if args.loss_scaling:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            # Log
            #print(f"Epoch {epoch}, global_step {global_step}, loss {loss.item()}")
            if global_step < args.latent_only_iters:
                print(f"Epoch {epoch}, [Step {global_step}] Latent SDS loss: {loss.item():.4f}")
            else:
                print(f"Epoch {epoch}, [Step {global_step}] LPIPS+L2 loss: {loss.item():.4f}")

            if global_step % 100 == 0:
                loss_dict[global_step] = loss.item()
                # save the nerf rendering as the logging output, instead of the decoded latent
                imgs = (
                    pred_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
                )  # torch to numpy, shape [1, 512, 512, 3]
                imgs = (imgs * 255).round()  # [0, 1] => [0, 255]
                rgb = Image.fromarray(imgs[0].astype("uint8"))
                output_path = (
                    f"{sds.output_dir}/images/rgb_epoch_{epoch}_iter_{global_step}.png"
                )
                rgb.save(output_path)

        # Save checkpoint
        if epoch % log_interval == 0 and global_step > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        if epoch % log_interval == 0 or epoch == max_epoch - 1:
            model.eval()
            all_preds = []
            all_preds_depth = []

            print(f"Epoch {epoch}, testing and save rgb and depth to video...")

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    rays_o = data["rays_o"]  # [B, N, 3]
                    rays_d = data["rays_d"]  # [B, N, 3]
                    mvp = data["mvp"]

                    B, N = rays_o.shape[:2]
                    H, W = data["H"], data["W"]

                    if bg_color is not None:
                        bg_color = bg_color.to(rays_o.device)

                    shading = data["shading"] if "shading" in data else "albedo"
                    ambient_ratio = (
                        data["ambient_ratio"] if "ambient_ratio" in data else 1.0
                    )
                    light_d = data["light_d"] if "light_d" in data else None

                    outputs = model.render(
                        rays_o,
                        rays_d,
                        mvp,
                        H,
                        W,
                        staged=True,
                        perturb=False,
                        light_d=light_d,
                        ambient_ratio=ambient_ratio,
                        shading=shading,
                        bg_color=bg_color,
                    )

                    preds = outputs["image"].reshape(B, H, W, 3)
                    preds_depth = outputs["depth"].reshape(B, H, W)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min() + 1e-6
                    )
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            # save the video
            if args.out_gif:
                imageio.mimsave(
                    os.path.join(sds.output_dir, "videos", f"rgb_ep_{epoch}.gif"),
                    all_preds,
                    fps=25,  # Typical frame rate for GIFs
                    loop=0
                )

                # Save Depth GIF
                imageio.mimsave(
                    os.path.join(sds.output_dir, "videos", f"depth_ep_{epoch}.gif"),
                    all_preds_depth,
                    fps=25,
                    loop=0
                )                

            else:
                imageio.mimwrite(
                    os.path.join(sds.output_dir, "videos", f"rgb_ep_{epoch}.mp4"),
                    all_preds,
                    fps=25,
                    quality=8,
                    macro_block_size=1,
                    format='ffmpeg',
                )
                imageio.mimwrite(
                    os.path.join(sds.output_dir, "videos", f"depth_ep_{epoch}.mp4"),
                    all_preds_depth,
                    fps=25,
                    quality=8,
                    macro_block_size=1,
                    format='ffmpeg',
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/pixel_sds")
    parser.add_argument("--loss_scaling", type=int, default=1)
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_orient', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--latent_iter_ratio', type=float, default=0, help="training iters that only use albedo shading")
    parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale for text-to-image generation")
    parser.add_argument("--out_gif", default=True, type=bool, help="Output gif instead of mp4")
    parser.add_argument("--inter_genv", default=100, type=int, help="Interval to gen new SD view target.")
    parser.add_argument("--lpips_only_iters", default=1000, type=int)
    parser.add_argument("--latent_only_iters", default=2000, type=int)

    parser = add_config_arguments(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir = osp.join(args.output_dir, args.prompt.replace(" ", "_"))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    sds = SDS(sd_version="2.1", device=device, output_dir=args.output_dir)
    optimize_nerf_pixel_sds(sds, prompt=args.prompt, device=device, args=args)
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
