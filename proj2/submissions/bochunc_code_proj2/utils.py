import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import pytorch3d
import imageio
import numpy as np
import mcubes

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_pure_mesh(mesh_src=None, dist=3, num_views=24, device_tag=None, image_size=512, output_path = "./output", fps=15):
    if device_tag == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    

    vertices, faces = mesh_src.verts_packed(), mesh_src.faces_packed()
    if vertices.numel() == 0:
        raise ValueError("No vertices were extracted! Check voxel grid calculations.")    

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

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
    my_images = [((image.cpu().detach().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=32)

def render_point_cloud(src_points, image_size=512, device_tag='cuda', num_views=36, fps=12, dist=3, radius=0.03, output_path="./outputs"):
    src_points = src_points.squeeze(0) # BxPx3 -> Px3
    if device_tag == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    

    color = (src_points - src_points.min()) / (src_points.max() - src_points.min())
    ren_point_cloud = pytorch3d.structures.Pointclouds(
        points=[src_points], features=[color],
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
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    point_cloud_renderer = get_points_renderer(image_size=image_size, device=device, radius=radius)
    images = point_cloud_renderer(ren_point_cloud.extend(num_views), cameras=many_cameras, lights=lights)
    my_images = [((image.cpu().detach().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=256)

def render_voxel_mesh(voxels_src=None, is_logit=False, dist=3, num_views=24, device_tag=None, image_size=512, output_path = "./output", fps=15):
    if device_tag == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    

    if is_logit:
        voxels_probs = torch.sigmoid(voxels_src)
    else:
        voxels_probs = voxels_src

    meshes_obj = pytorch3d.ops.cubify(voxels_probs, thresh=0.5)
    vertices, faces = meshes_obj.verts_packed(), meshes_obj.faces_packed()
    if vertices.numel() == 0:
        raise ValueError("No vertices were extracted! Check voxel grid calculations.")    

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

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
    my_images = [((image.cpu().detach().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=32)

def render_occ_mesh(voxels_src=None, is_logit=False, dist=3, num_views=24, device_tag=None, image_size=512, output_path = "./output", fps=15):
    # assume input voxel_src.shape = (D, H, W)
    if device_tag == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    

    if is_logit:
        voxels_probs = torch.sigmoid(voxels_src)
    else:
        voxels_probs = voxels_src

    vertices, faces = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    if vertices.numel() == 0:
        print("No vertices were extracted! Check voxel grid calculations.")    
        return None

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

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
    my_images = [((image.cpu().detach().numpy()[:, :, :3] * 255).astype('uint8')) for image in images]
    duration = 1000 // fps  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output_path, my_images, duration=duration, loop=0, palettesize=32)

    

