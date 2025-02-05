import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np
import mcubes
import time
import psutil
import os
from PIL import Image

from starter.dolly_zoom import dolly_zoom
from starter.camera_transforms import render_textured_cow
from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image
import starter.utils
from starter.utils import get_device

def sample_distribution(num_samples=5, distribution=None) -> []:
    '''
    distribution: (N, ) array of probability distribution from sample1 to sampleN
    sample_index: (M, ) array of sampled index follow the probability distribution
    '''
    sample_index = []
    for i in range(num_samples):
        u = torch.rand(1)
        for index, prob in enumerate(distribution):
            u -= prob
            if u <= 0:
                sample_index.append(index)
                break

    return sample_index

def sample_points_from_mesh(verts, faces, num_samples) -> torch.Tensor:
    """
    verts: (V, 3) tensor of mesh vertices.
    faces: (F, 3) tensor of vertex indices forming faces.
    sampled_points: (num_samples, 3) tensor of sampled points.
    """
    # Compute face vertices
    A = verts[faces[:, 0]]  # (F, 3)
    B = verts[faces[:, 1]]  # (F, 3)
    C = verts[faces[:, 2]]  # (F, 3)

    # Compute face areas (used for weighted sampling)
    AB = B - A  # Edge vector
    AC = C - A  # Edge vector
    face_areas = 0.5 * torch.norm(torch.linalg.cross(AB, AC), dim=1)  # (F,)
    
    # Normalize to create a probability distribution
    face_probs = face_areas / face_areas.sum()

    # Sample faces based on area-weighted probabilities
    face_indices = torch.multinomial(face_probs, num_samples, replacement=True)
    #face_indices = sample_distribution(num_samples, face_probs) #same as the above line

    # Retrieve selected faces
    A_samples = A[face_indices]
    B_samples = B[face_indices]
    C_samples = C[face_indices]

    # Sample random barycentric coordinates: To make sure the sampling inside a triangle is distributed evenly
    u = torch.rand(num_samples, 1)
    v = torch.rand(num_samples, 1)
    
    sqrt_u = torch.sqrt(u)
    lambda_1 = 1 - sqrt_u
    lambda_2 = sqrt_u * (1 - v)
    lambda_3 = v * sqrt_u

    # Compute final sampled points using barycentric interpolation
    sampled_points = lambda_1 * A_samples + lambda_2 * B_samples + lambda_3 * C_samples

    return sampled_points


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

def render_UFO_cow(dist=3, output_path="./output/custom_fun.gif", device=None, num_views=24, fps=15, renderer=None):
    #========================== UFO object===============================#
    obj_filename = "./data/ufo/UFO_Empty_OBJ/UFO_Empty.obj"
    texture_filename = "./data/ufo/textures/UFO_color.jpg"

    # Load the mesh
    ufo_verts, ufo_faces, ufo_aux = pytorch3d.io.load_obj(obj_filename, load_textures=False)
    ufo_verts = ufo_verts+torch.tensor([0, 3, 0])

    # Load the texture image
    texture_image = Image.open(texture_filename).convert("RGB")
    texture_image = np.array(texture_image) / 255.0  # Normalize to [0,1]

    # Convert texture to tensor
    texture_image = torch.tensor(texture_image, dtype=torch.float32).to(device)

    # Get UV coordinates and faces
    verts_uvs = ufo_aux.verts_uvs.to(device) if ufo_aux.verts_uvs is not None else None
    faces_uvs = ufo_faces.textures_idx.to(device) if ufo_faces.textures_idx is not None else None

    # Create a texture object
    ufo_textures = pytorch3d.renderer.Textures(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=[texture_image]).to(device)

    # Create a Meshes object
    #ufo_mesh = pytorch3d.structures.Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=textures).to(device)
    #========================== cow object===============================#
    obj_filename = "./data/cow.obj"
    texture_filename = "./data/cow_texture.png"

    # Load the mesh
    verts, faces, aux = pytorch3d.io.load_obj(obj_filename, load_textures=False)
    verts = verts+torch.tensor([0, -3, 0])

    # Load the texture image
    texture_image = Image.open(texture_filename).convert("RGB")
    texture_image = np.array(texture_image) / 255.0  # Normalize to [0,1]

    # Convert texture to tensor
    texture_image = torch.tensor(texture_image, dtype=torch.float32).to(device)

    # Get UV coordinates and faces
    verts_uvs = aux.verts_uvs.to(device) if aux.verts_uvs is not None else None
    faces_uvs = faces.textures_idx.to(device) if faces.textures_idx is not None else None

    # Create a texture object
    cow_textures = pytorch3d.renderer.Textures(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=[texture_image]).to(device)
    
    # Rendering setup
    rotation_angles = np.linspace(0, 2 * np.pi, num_views)  # 360-degree rotation
    y_positions = np.linspace(-3, 3, num_views)  # Smooth upward motion
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)


    # Store frames for GIF
    frames = []

    # Iterate through frames to create smooth motion and rotation
    for i in range(num_views):
        # Move cow along y-axis
        translated_verts = verts.clone().to(device)
        translated_verts[:, 1] += y_positions[i]  # Apply y-axis translation

        # Apply rotation to cow
        cow_rotation_matrix = pytorch3d.transforms.axis_angle_to_matrix(torch.tensor([0, rotation_angles[i], 0], dtype=torch.float32))
        translated_verts = torch.matmul(translated_verts, cow_rotation_matrix.to(device))

        # Create cow mesh
        cow_mesh = pytorch3d.structures.Meshes(verts=[translated_verts], faces=[faces.verts_idx.to(device)], textures=cow_textures)

        # Rotate UFO
        ufo_rotation_matrix = pytorch3d.transforms.axis_angle_to_matrix(torch.tensor([0, -rotation_angles[i], 0], dtype=torch.float32))
        ufo_verts_rotated = torch.matmul(ufo_verts.to(device), ufo_rotation_matrix.to(device))
        ufo_mesh_rotated = pytorch3d.structures.Meshes(verts=[ufo_verts_rotated], faces=[ufo_faces.verts_idx.to(device)], textures=ufo_textures)

        # Combine UFO and moving cow into one scene
        joint_mesh = pytorch3d.structures.join_meshes_as_scene([cow_mesh, ufo_mesh_rotated])

        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist,
            elev=0,
            azim=0,
        )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        image = renderer(joint_mesh, cameras=camera, lights=lights)
        frame = (image[0].cpu().numpy()[:, :, :3] * 255).astype('uint8')
        frames.append(frame)

    # Save as looping animated GIF
    imageio.mimsave(output_path, frames, fps=15, loop=0, palettesize=256)  # loop=0 makes it infinite    

def render_custom(image_size=256, num_samples=200, device=None, num_views=24, fps=12, dist=3, a=1, c=1, output_path="./outputs", point_cloud_renderer=None, lights=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(-2, 2, num_samples)  # Finite range for visibility    

    # Densely sample phi and theta on a grid
    cap_u, cap_v = torch.meshgrid(u, v)

    x = a*torch.cosh(cap_v)*torch.cos(cap_u)
    y = a*torch.cosh(cap_v)*torch.sin(cap_u)
    z = c*torch.sinh(cap_v)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    images = point_cloud_renderer(point_cloud.extend(num_views), cameras=many_cameras, lights=lights)
    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=256)

def render_torus(image_size=256, num_samples=200, cap_r=0, r=1, device=None, num_views=36, fps=12, dist=3, output_path="./outputs", point_cloud_renderer=None, lights=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """
    #print(f"==============================")
    #print(f"========profiling-begin=======")
    #print(f"==============================")
    process = psutil.Process(os.getpid())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    start_time = time.perf_counter()
    #print(f"==============================")
    #print(f"========profiling-end=========")
    #print(f"==============================")

    phi = torch.linspace(0, 2*np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (cap_r + r*torch.cos(Theta)) * torch.cos(Phi)
    y = (cap_r + r*torch.cos(Theta)) * torch.sin(Phi)
    z = r*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    images = point_cloud_renderer(torus_point_cloud.extend(num_views), cameras=many_cameras, lights=lights)

    #print(f"==============================")
    #print(f"========profiling-begin=======")
    #print(f"==============================")
    torch.cuda.synchronize()  # Ensure all GPU tasks finish before measuring
    end_time = time.perf_counter()
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_mem_used = gpu_mem_after - gpu_mem_before
        print(f"GPU Memory Used (During Execution): {gpu_mem_used:.2f} MB")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("=" * 50)
    #print(f"==============================")
    #print(f"========profiling-end=========")
    #print(f"==============================")
    
    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=256)

def render_custom_mesh(image_size=256, voxel_size=64, min_value=-10, max_value=10, a=1, b=1, c=1, device=None, dist=10, num_views=24, output_path="./output", fps=12):
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X**2 / a**2) - (Y**2 / b**2) - (Z**2 / c**2) - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    if vertices.numel() == 0:
        raise ValueError("No vertices were extracted! Check voxel grid calculations.")    

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = starter.utils.get_mesh_renderer(image_size=image_size, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=32)

def render_torus_mesh(image_size=256, voxel_size=64, min_value=-10, max_value=10, cap_r=3, r=1, device=None, dist=10, num_views=24, output_path="./output", fps=12):
    #print(f"==============================")
    #print(f"========profiling-begin=======")
    #print(f"==============================")
    process = psutil.Process(os.getpid())
    cap_r = torch.tensor(cap_r)  # Ensure it's a tensor
    r = torch.tensor(r)          # Ensure it's a tensor    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    start_time = time.perf_counter()
    #print(f"============================")
    #print(f"========profiling-end=======")
    #print(f"============================")

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = torch.pow(cap_r - torch.sqrt(torch.pow(X, 2) + torch.pow(Y, 2)), 2) + torch.pow(Z, 2) - torch.pow(r, 2)
    
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    if vertices.numel() == 0:
        raise ValueError("No vertices were extracted! Check voxel grid calculations.")    

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = starter.utils.get_mesh_renderer(image_size=image_size, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)

    #print(f"==============================")
    #print(f"========profiling-begin=======")
    #print(f"==============================")
    torch.cuda.synchronize()  # Ensure all GPU tasks finish before measuring
    end_time = time.perf_counter()
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_mem_used = gpu_mem_after - gpu_mem_before
        print(f"GPU Memory Used (During Execution): {gpu_mem_used:.2f} MB")

    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("=" * 50)
    #print(f"============================")
    #print(f"========profiling-end=======")
    #print(f"============================")

    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=32)

def render_point(point_clouds=None, output_path="./outputs", num_views=12, lights=None, fps=15, device=None, dist=3, elev=0, azim=0, point_cloud_renderer=None, R_relative=torch.tensor(np.eye(3)), T_relative=torch.tensor([0, 0, 0])):
    R_relative = torch.tensor(R_relative).float()

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R@R_relative,
        T=T,
        device=device
    )
    images = point_cloud_renderer(point_clouds.extend(num_views), cameras=many_cameras, lights=lights)
    my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=256)

def render_surround(meshes=None, output_path="./outputs", num_views=12, lights=None, fps=15, device=None, dist=3, renderer=None):
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
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

    # Ensure the output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created output directory: {args.output_path}")        

    point_cloud_renderer = starter.utils.get_points_renderer(device=device, radius=0.03)

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
    render_surround(meshes, output_path=f"{args.output_path}/360_cow.gif", num_views=36, lights=lights, fps=15, device=device, renderer=renderer)
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
    render_surround(tetra_meshes, output_path=f"{args.output_path}/tetrahedron.gif", num_views=36, lights=lights, fps=15, device=device, renderer=renderer)
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
    render_surround(cube_meshes, output_path=f"{args.output_path}/cube.gif", num_views=36, lights=lights, fps=12, device=device, renderer=renderer)
    print(f"> Done.")

    # Q3
    print(f"> Executing Q3...")
    color1 = torch.tensor([0, 0, 1])
    color2 = torch.tensor([1, 0, 0])
    texture_rgb = vertices.clone() # 1 x N_v x 3
    alpha = (texture_rgb[:, :, 2] - texture_rgb[:, :, 2].min())/(texture_rgb[:, :, 2].max() - texture_rgb[:, :, 2].min())
    for i in range(len(alpha[0])):
        color = alpha[0][i]*color2 + (1-alpha[0][i])*color1
        texture_rgb[0][i] = color
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb) # important

    # Construct renderer and lights
    renderer = starter.utils.get_mesh_renderer(image_size=args.image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # Construct meshes
    retex_meshes = pytorch3d.structures.Meshes(
        verts=vertices, # batched tensor or a list of tensors
        faces=faces,
        textures=textures,
    )
    retex_meshes = retex_meshes.to(device)  # Move mesh to GPU
    render_surround(retex_meshes, output_path=f"{args.output_path}/retex_cow.gif", num_views=36, lights=lights, fps=12, device=device, renderer=renderer)
    print(f"> Done.")

    # Q4
    print(f"> Executing Q4...")
    # Transform1
    print(f"> Transoform1")
    T_relative1 = np.array([0, 0, 0])
    R_relative1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T
    rend_result = render_textured_cow(cow_path='./data/cow_with_axis.obj', image_size=args.image_size, R_relative=R_relative1, T_relative=T_relative1, device=device)
    plt.imsave(f"{args.output_path}/trans1_cow.jpg", rend_result)
    print(f"> Done.")
    # Transform2
    print(f"> Transoform2")
    T_relative2 = np.array([0, 0, 2])
    R_relative2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    rend_result = render_textured_cow(cow_path='./data/cow_with_axis.obj', image_size=args.image_size, R_relative=R_relative2, T_relative=T_relative2, device=device)
    plt.imsave(f"{args.output_path}/trans2_cow.jpg", rend_result)
    print(f"> Done.")
    # Transform3
    print(f"> Transoform3")
    T_relative3 = np.array([0.35, -0.5, 0])
    theta = -5 #degree
    R_relative3 = np.array([[np.cos(np.pi/180*(theta)), 0, np.sin(np.pi/180*(theta))],
                            [0, 1, 0],
                            [-np.sin(np.pi/180*(theta)), 0, np.cos(np.pi/180*(theta))]]).T
    rend_result = render_textured_cow(cow_path='./data/cow_with_axis.obj', image_size=args.image_size, R_relative=R_relative3, T_relative=T_relative3, device=device)
    plt.imsave(f"{args.output_path}/trans3_cow.jpg", rend_result)
    print(f"> Done.")
    # Transform4
    print(f"> Transoform4")
    T_relative4 = np.array([-3, 0, 3])
    theta = -90 #degree
    R_relative4 = np.array([[np.cos(np.pi/180*(theta)), 0, np.sin(np.pi/180*(theta))],
                            [0, 1, 0],
                            [-np.sin(np.pi/180*(theta)), 0, np.cos(np.pi/180*(theta))]]).T
    rend_result = render_textured_cow(cow_path='./data/cow_with_axis.obj', image_size=args.image_size, R_relative=R_relative4, T_relative=T_relative4, device=device)
    plt.imsave(f"{args.output_path}/trans4_cow.jpg", rend_result)
    print(f"> Done.")

    # Q5
    # Q5.1
    print(f"> Q5.1")
    data = load_rgbd_data(path="data/rgbd_data.pkl")
    #for key, _ in data.items():
        #print(f"key = {key}")
        #print(f"type = {type(data[key])}")
    #keys: rgb1, mask1, depth1, rgb2, mask2, depth2, cameras1, cameras2

    pts1, color1 = unproject_depth_image(torch.tensor(data['rgb1']), torch.tensor(data['mask1']), torch.tensor(data['depth1']), data['cameras1'])
    pts2, color2 = unproject_depth_image(torch.tensor(data['rgb2']), torch.tensor(data['mask2']), torch.tensor(data['depth2']), data['cameras2'])

    pointcloud1 = pytorch3d.structures.Pointclouds(
        points=pts1.unsqueeze(0),
        features=color1.unsqueeze(0),
        ).to(device)
    
    pointcloud2 = pytorch3d.structures.Pointclouds(
        points=pts2.unsqueeze(0),
        features=color2.unsqueeze(0),
        ).to(device)

    pts3 = torch.cat((pts1, pts2), 0)
    color3 = torch.cat((color1, color2), 0)

    pointcloud3 = pytorch3d.structures.Pointclouds(
        points=pts3.unsqueeze(0),
        features=color3.unsqueeze(0),
        ).to(device)
    
    R_relative = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                           [np.sin(np.pi),  np.cos(np.pi), 0],
                           [0            ,  0            , 1]]).T

    render_point(point_clouds=pointcloud1, output_path=f"{args.output_path}/point_cloud1.gif", num_views=24, lights=None, fps=12, device=device, dist=6, elev=0, azim=0, point_cloud_renderer=point_cloud_renderer, R_relative=R_relative)
    render_point(point_clouds=pointcloud2, output_path=f"{args.output_path}/point_cloud2.gif", num_views=24, lights=None, fps=12, device=device, dist=6, elev=0, azim=0, point_cloud_renderer=point_cloud_renderer, R_relative=R_relative)
    render_point(point_clouds=pointcloud3, output_path=f"{args.output_path}/point_cloud3.gif", num_views=24, lights=None, fps=12, device=device, dist=6, elev=0, azim=0, point_cloud_renderer=point_cloud_renderer, R_relative=R_relative)
    print(f"> Done.")

    # Q5.2
    print(f"> Q5.2")
    render_torus(image_size=512, num_samples=600, cap_r=5, r=2, device=device, num_views=36, fps=12, dist=15, output_path=f"{args.output_path}/torus.gif", point_cloud_renderer=point_cloud_renderer, lights=None)
    render_custom(image_size=512, num_samples=200, device=device, fps=12, a=1, c=1, dist=15, output_path=f"{args.output_path}/custom_one_sheet_hyperboloid.gif", point_cloud_renderer=point_cloud_renderer, lights=None)
    print(f"> Done.")

    # Q5.3
    print(f"> Q5.3")
    render_torus_mesh(image_size=512, voxel_size=600, min_value=-10, max_value=10, cap_r=3, r=1.5, device=device, dist=10, num_views=36, output_path=f"{args.output_path}/torus_mesh.gif", fps=12)
    render_custom_mesh(image_size=512, voxel_size=64, min_value=-10, max_value=10, a=1, b=1, c=1, device=device, dist=25, num_views=36, output_path=f"{args.output_path}/custom_two_sheet_hyperboloid_mesh.gif", fps=12)
    print(f"> Done.")

    # Q6
    print(f"> Q6")
    ufo_renderer = starter.utils.get_mesh_renderer(image_size=512, device=device)
    render_UFO_cow(dist=10, output_path=f"{args.output_path}/custom_fun.gif", device=device, num_views=24, fps=15, renderer=ufo_renderer)
    print(f"> Done.")

    # Q7
    print(f"> Q7")
    verts = vertices.squeeze(0)  # 1 x N_v x 3
    faces = faces.squeeze(0)  # 1 x N_f x 3
    num_samples_list = [10, 100, 1000, 10000]
    num_views = 24
    fps = 15
    output_path = f"{args.output_path}/"
    dist = 3

    point_cloud_renderer_q7 = starter.utils.get_points_renderer(device=device, radius=0.015)
    mesh_renderer_q7 = starter.utils.get_mesh_renderer(image_size=512, device=device)
    render_surround(meshes, output_path=f"{args.output_path}/original_cow.gif", num_views=num_views, lights=lights, fps=fps, device=device, renderer=mesh_renderer_q7)

    for num_samples in num_samples_list:
        points = sample_points_from_mesh(verts, faces, num_samples)
        color = (points - points.min()) / (points.max() - points.min())

        point_cloud = pytorch3d.structures.Pointclouds(
            points=[points], features=[color],
        ).to(device)

        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=dist,
            elev=0,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device
        )
        images = point_cloud_renderer_q7(point_cloud.extend(num_views), cameras=many_cameras, lights=lights)
        my_images = [((image.cpu().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
        duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
        imageio.mimsave(output_path+f"cow_sampled{num_samples}.gif", my_images, duration=duration, loop=0, palettesize=256)
    print(f"> Done.")
    

