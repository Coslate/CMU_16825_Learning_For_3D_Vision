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

def render_surround(meshes=None, output_path="./outputs", num_views=12, lights=None, fps=15, device=None):
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
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="./outputs")
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
    render_surround(meshes, output_path=f"{args.output_path}/360_cow.gif", num_views=36, lights=lights, fps=15, device=device)
    print(f"> Done.")

    # Q1.2
    print(f"> Executing Q1.2...")
    dolly_zoom(image_size=args.image_size, num_frames=40, duration=1000//15, device=device, output_file=f"{args.output_path}/dolly_zoom.gif")
    print(f"> Done.")

    # Q2.1
    print(f"> Executing Q2.1...")
    tetra_vertices = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    tetra_faces    = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    tetra_vertices = tetra_vertices.unsqueeze(0)  # 1 x N_v x 3
    tetra_faces    = tetra_faces.unsqueeze(0)  # 1 x N_f x 3
    tetra_texture_rgb    = torch.ones_like(tetra_vertices) # 1 x N_v X 3
    tetra_texture_rgb    = tetra_texture_rgb * torch.tensor([0.7, 0.7, 1])
    tetra_textures       = pytorch3d.renderer.TexturesVertex(tetra_texture_rgb) # important

    tetra_meshes = pytorch3d.structures.Meshes(
        verts=tetra_vertices, # batched tensor or a list of tensors
        faces=tetra_faces,
        textures=tetra_textures,
    )
    tetra_meshes = tetra_meshes.to(device)  # Move mesh to GPU
    render_surround(tetra_meshes, output_path=f"{args.output_path}/tetrahedron.gif", num_views=36, lights=lights, fps=15, device=device)
    print(f"> Done.")

    # Q2.2
    print(f"> Executing Q2.2...")
    cube_vertices = torch.tensor([[-0.9, -0.9, -0.9], #bottom left back
                                  [ 0.9, -0.9, -0.9], #bottom right back
                                  [ 0.9,  0.9, -0.9], #top right back
                                  [-0.9,  0.9, -0.9], #top left back
                                  [-0.9, -0.9,  0.9], #bottom left back
                                  [ 0.9, -0.9,  0.9], #bottom right back
                                  [ 0.9,  0.9,  0.9], #top right back
                                  [-0.9,  0.9,  0.9]]) #top left back

    cube_faces    = torch.tensor([[0, 1, 2], 
                                  [0, 2, 3], #back face
                                  [4, 5, 6], 
                                  [4, 6, 7], #front face
                                  [0, 3, 7],
                                  [0, 7, 4], #left face
                                  [1, 2, 6],
                                  [1, 6, 5], #right face
                                  [0, 1, 5],
                                  [0, 5, 4], #bottom face
                                  [3, 2, 6],
                                  [3, 6, 7]])#top face

    cube_vertices = cube_vertices.unsqueeze(0)  # 1 x N_v x 3
    cube_faces    = cube_faces.unsqueeze(0)  # 1 x N_f x 3
    cube_texture_rgb    = torch.ones_like(cube_vertices) # 1 x N_v X 3
    cube_texture_rgb    = cube_texture_rgb * torch.tensor([0.7, 0.7, 1])
    cube_textures       = pytorch3d.renderer.TexturesVertex(cube_texture_rgb) # important

    cube_meshes = pytorch3d.structures.Meshes(
        verts=cube_vertices, # batched tensor or a list of tensors
        faces=cube_faces,
        textures=cube_textures,
    )
    cube_meshes = cube_meshes.to(device)  # Move mesh to GPU
    render_surround(cube_meshes, output_path=f"{args.output_path}/cube.gif", num_views=36, lights=lights, fps=12, device=device)
    print(f"> Done.")
