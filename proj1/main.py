import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np

from starter.dolly_zoom import dolly_zoom
import starter.utils

def setup_renderer(image_size=256):
    # Equal to: renderer = starter.utils.get_mesh_renderer(image_size=image_size, device=device)
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    shader = pytorch3d.renderer.HardPhongShader(device=device)
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )    

    return renderer

def render_360cow(meshes=None, image_size=256, output_path="./outputs",num_views=12, device=None):
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights)
    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"{output_path}/360_cow.gif", my_images, duration=duration, loop=0, palettesize=256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")    

    # Load data and prepare vertix, face, texture
    vertices, faces = starter.utils.load_cow_mesh(path=args.cow_path)
    vertices = vertices.unsqueeze(0)  # 1 x N_v x 3
    faces = faces.unsqueeze(0)  # 1 x N_f x 3
    texture_rgb = torch.ones_like(vertices) # 1 x N_v X 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb) # important

    # Construct renderer and lights
    renderer = starter.utils.get_mesh_renderer(image_size=args.image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Construct meshes
    meshes = pytorch3d.structures.Meshes(
        verts=vertices, # batched tensor or a list of tensors
        faces=faces,
        textures=textures,
    )
    meshes = meshes.to(device)  # Move mesh to GPU

    # Q1.1
    print(f"> Executing Q1.1...")
    render_360cow(meshes, output_path=args.output_path, num_views=36, device=device)
    print(f"> Done.")

    # Q1.2
    print(f"> Executing Q1.2...")
    dolly_zoom(image_size=args.image_size, num_frames=40, duration=1000//15, device=device, output_file=f"{args.output_path}/dolly_zoom.gif")
    print(f"> Done.")
