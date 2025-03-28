import argparse
import os
import os.path as osp
import random
import time

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


def optimize_nerf_pixel_sds(
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

    # Output paths
    os.makedirs(f"{args.output_dir}/images", exist_ok=True)
    os.makedirs(f"{args.output_dir}/videos", exist_ok=True)

    global_step = 0
    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)

    for epoch in range(max_epoch):
        model.train()
        for data in train_loader:
            global_step += 1
            optimizer.zero_grad()

            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            mvp = data["mvp"]
            H, W = data["H"], data["W"]
            B = rays_o.shape[0]

            outputs = model.render(
                rays_o, rays_d, mvp, H, W, staged=False, perturb=True,
                bg_color=None, ambient_ratio=1.0, shading="lambertian",
                max_ray_batch=args.max_ray_batch,
            )
            pred_rgb = outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

            # Resize to LPIPS input size (typically 64 or 128, but LPIPS supports 512)
            target_rgb = data.get("target", torch.rand_like(pred_rgb))  # Dummy target if none provided
            pred_resized = F.interpolate(pred_rgb, size=(256, 256), mode="bilinear", align_corners=False)
            target_resized = F.interpolate(target_rgb, size=(256, 256), mode="bilinear", align_corners=False)

            loss = lpips_loss_fn(pred_resized, target_resized).mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
                img = (pred_rgb[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(f"{args.output_dir}/images/iter_{global_step}.png")

        # Save GIF
        if epoch % log_interval == 0 or epoch == max_epoch - 1:
            model.eval()
            frames = []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    rays_o = data["rays_o"]
                    rays_d = data["rays_d"]
                    mvp = data["mvp"]
                    H, W = data["H"], data["W"]
                    outputs = model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False)
                    img = outputs["image"].reshape(H, W, 3).detach().cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    frames.append(img)
            imageio.mimsave(
                os.path.join(args.output_dir, "videos", f"rgb_ep_{epoch}.gif"),
                frames,
                fps=25,
                loop=0
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a hamburger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/pixel_sds")
    parser = add_config_arguments(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir = osp.join(args.output_dir, args.prompt.replace(" ", "_"))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    optimize_nerf_pixel_sds(prompt=args.prompt, device=device, args=args)
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
