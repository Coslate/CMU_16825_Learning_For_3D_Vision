import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from data_utils import visualize_renders
from SDS import SDS
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from torch.utils.data import DataLoader
from utils import (
    get_cosine_schedule_with_warmup,
    get_mesh_renderer_soft,
    init_mesh,
    prepare_embeddings,
    render_360_views,
    seed_everything,
)
from pytorch3d.renderer import FoVPerspectiveCameras
import math
import torch.nn.functional as F
import imageio


def make_trainable(gaussians):
    gaussians.means.requires_grad = True
    gaussians.pre_act_scales.requires_grad = True
    gaussians.pre_act_opacities.requires_grad = True
    gaussians.colours.requires_grad = True
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad = True

def ndc_to_screen_camera(camera, img_size = (128, 128)):
    min_size = min(img_size[0], img_size[1])
    screen_focal = camera.fov * math.pi / 180  # This should be used to get focal length
    screen_focal = 0.5 * min_size / torch.tan(screen_focal / 2.0)

    if not isinstance(screen_focal, torch.Tensor):
        screen_focal = torch.tensor([[screen_focal, screen_focal]], dtype=torch.float32, device=camera.device)
    
    screen_principal = torch.tensor([[img_size[0] / 2, img_size[1] / 2]], dtype=torch.float32, device=camera.device)    

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
        device=camera.device
    )

def generate_random_cameras(num_views=64, dist=3.0, elev_range=(-30, 60), azim_range=(0, 360), device="cuda"):
    query_cameras = [] # optional
    num_cameras = num_views
    for _ in range(num_cameras):
        elev = torch.FloatTensor(1).uniform_(*elev_range)
        azim = torch.FloatTensor(1).uniform_(*azim_range)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cam = FoVPerspectiveCameras(R=R, T=T, device=device)
        query_cameras.append(cam)        
    return query_cameras

def render_360_gif(scene, output_path, radius=3.0, num_frames=60, res=128, device="cuda"):
    frames = []
    for i in range(num_frames):
        azim = 360.0 * i / num_frames
        elev = 0  # fixed elevation for slight top-down view
        R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)
        camera = PerspectiveCameras(R=R, T=T, in_ndc=False, device=device,
                                    focal_length=torch.tensor([[res, res]], device=device).float(),
                                    principal_point=torch.tensor([[res/2, res/2]], device=device).float(),
                                    image_size=((res, res),))

        img, _, _ = scene.render(camera, img_size=(res, res), per_splat=-1, bg_colour=(0.0, 0.0, 0.0))
        frame_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame_np)

    gif_path = os.path.join(output_path, "final_output_rendered_360.gif")
    imageio.mimwrite(gif_path, frames, fps=10, loop = 0)
    print(f"[*] Rendered 360° GIF saved to: {gif_path}")    

def run_sds_gaussian_optimization(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init SDS
    sds = SDS(sd_version="2.1", device=device, output_dir=args.output_dir)
    embeddings = prepare_embeddings(sds, args.prompt, view_dependent=False)

    # Init Gaussians and Scene
    gaussians = Gaussians(
        init_type="random",
        device=args.device,
        num_points=args.init_random_numpoints,
        isotropic=False
    )
    scene = Scene(gaussians)
    make_trainable(gaussians)

    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05},
        {'params': [gaussians.pre_act_scales], 'lr': 0.02},
        {'params': [gaussians.colours], 'lr': 0.005},
        {'params': [gaussians.means], 'lr': 0.0016},
        {'params': [gaussians.pre_act_quats], 'lr': 0.001} if not gaussians.is_isotropic else {}
    ], eps=1e-15)

    # Camera views
    cameras = generate_random_cameras(num_views=args.num_views, device=device)

    # Training loop
    viz_frames = []
    viz_depth_frames = []
    viz_gif_path = os.path.join(args.output_dir, "training_procedure.gif")
    viz_gif_depth_path = os.path.join(args.output_dir, "training_procedure_depth.gif")
    for itr in tqdm(range(args.num_itrs)):
        camera_ndc = cameras[np.random.randint(0, len(cameras))]
        camera = ndc_to_screen_camera(camera_ndc, img_size=(args.res, args.res))

        pred_img, pred_depth, _ = scene.render(
            camera,
            per_splat=args.gaussians_per_splat,
            img_size=(args.res, args.res),
            bg_colour=(0.0, 0.0, 0.0)
        )

        # SDS optimization
        image_tensor = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        pred_img_resized = F.interpolate(image_tensor, size=(512, 512), mode="bilinear", align_corners=False)
        latents = sds.encode_imgs(pred_img_resized)
        loss = sds.sds_loss(latents=latents,
                            text_embeddings=embeddings['default'],
                            text_embeddings_uncond=embeddings['uncond'],
                            guidance_scale=args.guidance_scale)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if itr % args.viz_freq == 0 or itr == args.num_itrs - 1:
            decoded = sds.decode_latents(latents.detach())
            img = Image.fromarray(decoded.astype("uint8"))
            img.save(os.path.join(args.output_dir, f"iter_{itr}.png"))
            viz_frames.append(img)

            # Save predicted depth
            depth_np = pred_depth.detach().cpu().numpy()
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_uint8 = np.squeeze(depth_uint8)  # remove dimensions like (1, 1, 1)
            depth_img = Image.fromarray(depth_uint8)
            depth_img.save(os.path.join(args.output_dir, f"depth_{itr}.png"))            
            viz_depth_frames.append(depth_img)

    print("[*] SDS-based optimization complete.")
    imageio.mimwrite(viz_gif_path, viz_frames, loop=0, duration=(1/10.0)*1000)
    imageio.mimwrite(viz_gif_depth_path, viz_depth_frames, loop=0, duration=(1/10.0)*1000)
    render_360_gif(scene, output_path=args.output_dir, radius=3.0, num_frames=60, res=args.res, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a red sports car")
    parser.add_argument("--output_dir", type=str, default="./output/q242_gaussian")
    parser.add_argument("--init_random_numpoints", type=int, default=10000)
    parser.add_argument("--num_views", type=int, default=64)
    parser.add_argument("--num_itrs", type=int, default=1000)
    parser.add_argument("--viz_freq", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=100.0)
    parser.add_argument("--gaussians_per_splat", type=int, default=-1)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()

    run_sds_gaussian_optimization(args)
