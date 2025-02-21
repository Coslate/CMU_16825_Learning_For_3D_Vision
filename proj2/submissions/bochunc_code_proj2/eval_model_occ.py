import argparse
import time
import torch
import os
from model import SingleViewto3D
from model_occnet import ImageConditionedOccupancyNetwork
#from model_occnet2 import ImageConditionedOccupancyNetwork2
from model_occnet3 import ImageConditionedOccupancyNetwork2
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np
from utils import render_voxel_mesh
from utils import render_point_cloud
from utils import render_pure_mesh
from utils import render_occ_mesh
import torch.nn.functional as F
import losses
import imageio

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=None, type=int)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh', 'occ'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--vis_freq', default=1, type=int)
    parser.add_argument('--eval_chk_file', default=f'./outputs/checkpoint.pth', type=str)
    parser.add_argument('--output_eval_path', default=f'./outputs', type=str)
    parser.add_argument('--n_sample_pt', default=32*32*32, type=int)
    return parser    


def render_occupancy_grid(predictions, grid_size=32, threshold=0.5, filename="occupancy.gif"):
    """
    Renders a 3D occupancy grid as a GIF by slicing along the Z-axis.
    
    Args:
        predictions: Tensor of shape (B, grid_size, grid_size, grid_size), occupancy probabilities.
        grid_size: Integer, size of the voxel grid.
        threshold: Float, threshold for occupancy.
        filename: String, name of the output GIF file.
    """
    predictions = predictions.squeeze(0).cpu().numpy()  # Convert to numpy
    occupancy_grid = (predictions >= threshold).astype(np.uint8)  # Apply threshold

    images = []
    for z in range(grid_size):
        plt.figure(figsize=(4, 4))
        plt.imshow(occupancy_grid[:, :, z], cmap="gray", origin="lower")
        plt.title(f"Z-Slice {z}")
        plt.axis("off")

        # Save frame
        plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)
        images.append(imageio.imread("temp.png"))
        plt.close()

    # Save as GIF
    imageio.mimsave(filename, images, fps=10)
    print(f"Saved GIF to {filename}")

def sample_3d_points(voxel_grid, num_samples):
    B, Z, Y, X = voxel_grid.shape  # Get voxel grid dimensions

    # Sample normalized (0,1) points to ensure correct voxel indexing
    coords = np.random.uniform(0, 1, size=(B, num_samples, 3))
    coords[..., 0] *= (X - 1)  # Scale X
    coords[..., 1] *= (Y - 1)  # Scale Y
    coords[..., 2] *= (Z - 1)  # Scale Z

    # Convert to integer indices
    voxel_indices_x = coords[..., 0].astype(int)
    voxel_indices_y = coords[..., 1].astype(int)
    voxel_indices_z = coords[..., 2].astype(int)

    # Ensure indices stay within bounds
    voxel_indices_x = np.clip(voxel_indices_x, 0, X - 1)
    voxel_indices_y = np.clip(voxel_indices_y, 0, Y - 1)
    voxel_indices_z = np.clip(voxel_indices_z, 0, Z - 1)

    '''
    labels = np.zeros((B, num_samples))
    for b in range(B):
        for i in range(num_samples):
            x, y, z = voxel_indices_x[b, i], voxel_indices_y[b, i], voxel_indices_z[b, i]
            labels[b, i] = voxel_grid[b, z, y, x]  # Ensure correct indexing
    '''
    labels = voxel_grid[np.arange(B)[:, None], voxel_indices_z, voxel_indices_y, voxel_indices_x] #(B, num_samples)

    # Convert to PyTorch tensors
    coords = torch.tensor(coords, dtype=torch.float32).reshape(-1, 3)  # (B * num_samples, 3)
    occupancy_labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)  # (B * num_samples, 1)

    return coords, occupancy_labels    


def preprocess(feed_dict, args, grid_size=32):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']

    lin_space = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(lin_space, lin_space, lin_space, indexing='ij')  # Shape: (32, 32, 32)
    coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # Shape: (32768, 3)

    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), coords.to(args.device), mesh

    return images.to(args.device), coords.to(args.device), mesh

def preprocess_orig(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'{args.output_eval_path}/eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def compute_normals(vertices, faces):
    """
    Compute normals from vertices and faces using cross-product.

    Args:
        vertices: (N, 3) vertex positions.
        faces: (M, 3) triangle indices.

    Returns:
        normals: (N, 3) vertex normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute face normals
    normals = torch.cross(v1 - v0, v2 - v0)  # (M, 3)
    
    # Normalize normals
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # Compute per-vertex normals (average of face normals)
    vertex_normals = torch.zeros_like(vertices)
    for i in range(faces.shape[0]):
        vertex_normals[faces[i]] += normals[i]

    # Normalize final per-vertex normals
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
    
    return vertex_normals

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox" or args.type == 'occ':
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])

        if mesh_src.verts_packed().shape[0] == 0:
            return None

        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

def extract_mesh_from_occupancy(occupancy_grid, threshold=0.5):
    """
    Convert (32, 32, 32) occupancy predictions into 3D mesh using Marching Cubes (mcubes).

    Args:
        occupancy_grid: NumPy array of shape (32, 32, 32), predicted occupancy values.
        threshold: Isosurface threshold (default=0.5).

    Returns:
        verts: (N, 3) vertices
        faces: (M, 3) faces
    """
    verts, faces = mcubes.marching_cubes(occupancy_grid, isovalue=threshold)
    if verts.size == 0:
        print("[WARNING] marching_cubes() extracted zero vertices. Check occupancy values.")
        return None, None, None  # Return None to handle failure case

    verts = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)

    # Compute normals correctly for extracted vertices
    normals = compute_normals(verts, faces)

    return verts, faces, normals

def compute_iou(predictions, targets, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for occupancy predictions.
    
    Args:
        predictions: Tensor of shape (B * num_samples, 1), raw logits or probabilities.
        targets: Tensor of shape (B * num_samples, 1), ground-truth occupancy labels.
        threshold: Float, decision threshold for occupancy.
    
    Returns:
        iou: Tensor, scalar IoU value.
    """
    preds = (predictions >= threshold).float()
    intersection = (preds * targets).sum()
    union = (preds + targets).clamp(0, 1).sum()
    return intersection / union if union > 0 else torch.tensor(0.0)

def compute_chamfer_distance(pred_verts, gt_verts):
    """
    Compute Chamfer-L1 distance between predicted and ground-truth meshes.
    
    Args:
        pred_verts: (N, 3) predicted mesh vertices
        gt_verts: (M, 3) ground-truth mesh vertices
    
    Returns:
        Chamfer Distance (L1)
    """
    pred_verts = pred_verts.unsqueeze(0)  # Add batch dim
    gt_verts = gt_verts.unsqueeze(0)      # Add batch dim
    #return losses.chamfer_distance(pred_verts, gt_verts, norm=1)[0].item()
    return chamfer_distance(pred_verts, gt_verts, norm=1)[0].item()

def normal_consistency(pred_normals, gt_normals):
    """
    Compute Normal Consistency metric.
    
    Args:
        pred_normals: Tensor of shape (N, 3), predicted surface normals.
        gt_normals: Tensor of shape (N, 3), ground-truth surface normals.
    
    Returns:
        normal_consistency: Tensor, scalar value representing normal consistency.
    """
    similarity = F.cosine_similarity(pred_normals, gt_normals, dim=-1)
    return similarity.mean()


def evaluate_model(args):
    if not os.path.exists(args.output_eval_path):
        os.makedirs(args.output_eval_path)
        print(f"Created output directory: {args.output_eval_path}")

    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    #model = ImageConditionedOccupancyNetwork(args)
    model = ImageConditionedOccupancyNetwork2(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        #checkpoint = torch.load(f'checkpoint_{args.type}_smoooooth.pth')
        checkpoint = torch.load(f'{args.eval_chk_file}')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader) if args.max_iter is None else args.max_iter
    print(f"max_iter = {max_iter}")

    iou_scores = []
    chamfer_scores = []
    normal_consistency_scores = []

    for step in range(start_iter, max_iter):
        iter_start_time = time.time()
        read_start_time = time.time()

        feed_dict = next(eval_loader)

        #images_gt, (coords, gt_occupancy), mesh_gt = preprocess(feed_dict, args)
        images_gt, coords, mesh_gt = preprocess(feed_dict, args, grid_size=32)


        read_time = time.time() - read_start_time

        #images_gt.shape = torch.Size([32, 137, 137, 3])
        #mesh_gt.shape = torch.Size([32, 1, 32, 32, 32])
        #[batch_size, channels, depth, height, width]
        voxel_size = 32
        B = images_gt.shape[0]
        predictions = torch.sigmoid(model(images_gt, coords, args)) #probability, (B*n_sample_pt, 1)
        predictions = predictions.view(B, 1, voxel_size, voxel_size, voxel_size) #(B, 1, 32, 32, 32)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)
        if metrics is None:
            print(f"> [WARNING]: empty mesh found for evaluation step = {step}")
            continue

        '''
        # Render both
        # Get ground truth from dataset
        gt_voxels = feed_dict["voxels"].float().squeeze(1).squeeze(0).cpu().numpy() #B=1, C=1, (32, 32, 32)
        # Convert to voxel grid
        pred_voxels = predictions.reshape(voxel_size, voxel_size, voxel_size).detach().cpu().numpy() # B=1, (32, 32, 32)

        # Extract meshes using mcubes
        gt_verts, gt_faces, gt_normals = extract_mesh_from_occupancy(gt_voxels)
        if gt_verts is None:
            print(f"Step {step} has zero gt_verts. Continue...")
            continue
        pred_verts, pred_faces, pred_normals = extract_mesh_from_occupancy(pred_voxels)
        if pred_verts is None:
            print(f"Step {step} has zero pred_verts. Continue...")
            continue
        render_occupancy_grid(pred_voxels.unsqueeze(0), filename=f"{args.output_eval_path}/q3.1_step_{step}_optimized_occ.gif")
        render_occupancy_grid(gt_voxels.unsqueeze(0), filename=f"{args.output_eval_path}/q3.1_step_{step}_gt_occ.gif")
        '''

        if (step % args.vis_freq) == 0:
            if args.type == "occ":
                #if metrics['F1@0.050000'] > 90 or (step == 110 or step == 114  or step == 329):
                #if (step in [110, 108, 116, 273, 292, 329, 385, 608, 611, 646]):
                if metrics['F1@0.050000'] > 35 :
                #if True:
                    f1_05 = metrics['F1@0.050000'].item()
                    f1_05_str = f"{f1_05:.4f}"
                    out_img = images_gt[0, ..., :3].detach().cpu().numpy()

                    plt.imsave(f'{args.output_eval_path}/gt_image_step_{step}_{args.type}.png', out_img)
                    render_voxel_mesh(voxels_src=predictions.squeeze(1), is_logit=False, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_eval_path}/q3.1_step_{step}_f1_{f1_05}_optimized_occ.gif", fps=10)
                    render_voxel_mesh(voxels_src=feed_dict['voxels'].to(args.device).squeeze(1), is_logit=False, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_eval_path}/q3.1_step_{step}_f1_{f1_05}_groundtruth_vox.gif", fps=10)
                    render_pure_mesh(mesh_src=mesh_gt, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_eval_path}/q3.1_step_{step}_f1_{f1_05}_groundtruth_mesh.gif", fps=10)
      
        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
