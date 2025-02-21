import torch.nn.functional as F
import cv2
import argparse
import time
import torch
import os
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np
from utils import render_voxel_mesh
from utils import render_point_cloud
from utils import render_pure_mesh
from tqdm import tqdm, trange
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')  # Use non-GUI backend

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=None, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--vis_freq', default=1, type=int)
    parser.add_argument('--eval_chk_file', default=f'./outputs/checkpoint.pth', type=str)
    parser.add_argument('--output_eval_path', default=f'./outputs', type=str)
    parser.add_argument('--examine_step', default=None, type=int)
    return parser    

def preprocess(feed_dict, args):
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

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
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

def vis_saliency_map_vox(input_image, args, step, model):
    # Prevent OpenCV Qt plugin issues (for remote environments)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"    
    input_image.requires_grad = True  

    # Forward pass
    voxels_pred = model(input_image, args)
    voxels_pred = voxels_pred.permute(0,1,4,3,2) #(B, C, D, H, W)

    # Compute overall influence by summing all output points
    target = voxels_pred.sum()
    input_image.grad = None  # Clear previous gradients
    target.backward()

    # Compute absolute gradient values (importance of each pixel)
    saliency_map = input_image.grad.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()

    # Normalize saliency map for visualization
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Convert to saliencymap
    saliencymap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    saliencymap = cv2.cvtColor(saliencymap, cv2.COLOR_BGR2RGB)

    # Overlay saliencymap on input image
    input_img_np = input_image.squeeze(0).cpu().detach().numpy()
    overlayed_img = (0.5 * input_img_np + 0.5 * saliencymap / 255).clip(0, 1)
    resized_overlayed_img = cv2.resize(overlayed_img, (256, 256), interpolation=cv2.INTER_CUBIC)

    # ------------------ Save Images Instead of Displaying ------------------ #
    # Save the saliencymap overlay image
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_overlayed_img)
    plt.axis("off")
    plt.title("Input Influence on 3D Reconstruction")
    plt.savefig(f"{args.output_eval_path}/saliency_map_{args.type}_{step}.png", bbox_inches='tight')
    plt.close()

    # Save the 3D scatter plot of the predicted point cloud
    out_img = input_image[0, ..., :3].detach().cpu().numpy()
    plt.imsave(f'{args.output_eval_path}/gt_image_step_{step}_{args.type}.png', out_img)
    render_voxel_mesh(voxels_src=voxels_pred.squeeze(1), is_logit=False, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_eval_path}/{args.type}_step_{step}_optimized_vox.gif", fps=10)


def vis_saliency_map_mesh(input_image, args, step, model):
    # Prevent OpenCV Qt plugin issues (for remote environments)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"    
    input_image.requires_grad = True  

    # Forward pass
    mesh_pred = model(input_image, args)

    # Compute overall influence by summing all output points
    target = mesh_pred.verts_packed().sum()
    input_image.grad = None  # Clear previous gradients
    target.backward()

    # Compute absolute gradient values (importance of each pixel)
    saliency_map = input_image.grad.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()

    # Normalize saliency map for visualization
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Convert to saliencymap
    saliencymap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    saliencymap = cv2.cvtColor(saliencymap, cv2.COLOR_BGR2RGB)

    # Overlay saliencymap on input image
    input_img_np = input_image.squeeze(0).cpu().detach().numpy()
    overlayed_img = (0.5 * input_img_np + 0.5 * saliencymap / 255).clip(0, 1)
    resized_overlayed_img = cv2.resize(overlayed_img, (256, 256), interpolation=cv2.INTER_CUBIC)

    # ------------------ Save Images Instead of Displaying ------------------ #
    # Save the saliencymap overlay image
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_overlayed_img)
    plt.axis("off")
    plt.title("Input Influence on 3D Reconstruction")
    plt.savefig(f"{args.output_eval_path}/saliency_map_{args.type}_{step}.png", bbox_inches='tight')
    plt.close()

    # Save the 3D scatter plot of the predicted point cloud
    out_img = input_image[0, ..., :3].detach().cpu().numpy()
    plt.imsave(f'{args.output_eval_path}/gt_image_step_{step}_{args.type}.png', out_img)
    render_pure_mesh(mesh_src=mesh_pred, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_eval_path}/{args.type}_step_{step}_optimized_mesh.gif", fps=10)

def vis_saliency_map_point(input_image, args, step, model):
    # Prevent OpenCV Qt plugin issues (for remote environments)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"    
    input_image.requires_grad = True  

    # Forward pass
    predicted_pointcloud = model(input_image, args)  # Shape: (Batch, n_point, 3)

    # Compute overall influence by summing all output points
    target = predicted_pointcloud.sum()
    input_image.grad = None  # Clear previous gradients
    target.backward()

    # Compute absolute gradient values (importance of each pixel)
    saliency_map = input_image.grad.abs().sum(dim=-1).squeeze(0).cpu().detach().numpy()

    # Normalize saliency map for visualization
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Convert to saliencymap
    saliencymap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    saliencymap = cv2.cvtColor(saliencymap, cv2.COLOR_BGR2RGB)

    # Overlay saliencymap on input image
    input_img_np = input_image.squeeze(0).cpu().detach().numpy()
    overlayed_img = (0.5 * input_img_np + 0.5 * saliencymap / 255).clip(0, 1)
    resized_overlayed_img = cv2.resize(overlayed_img, (256, 256), interpolation=cv2.INTER_CUBIC)

    # ------------------ Save Images Instead of Displaying ------------------ #
    # Save the saliencymap overlay image
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_overlayed_img)
    plt.axis("off")
    plt.title("Input Influence on 3D Reconstruction")
    plt.savefig(f"{args.output_eval_path}/saliency_map_{args.type}_{step}.png", bbox_inches='tight')
    plt.close()

    # Save the 3D scatter plot of the predicted point cloud
    out_img = input_image[0, ..., :3].detach().cpu().numpy()
    plt.imsave(f'{args.output_eval_path}/gt_image_step_{step}_{args.type}.png', out_img)
    render_point_cloud(src_points=predicted_pointcloud.squeeze(0), image_size=256, device_tag=args.device, num_views=20, fps=10, dist=3, radius=0.01, output_path=f"{args.output_eval_path}/{args.type}_step_{step}_optimized_point.gif")


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device=='cuda' else "cpu")
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

    # Move model and input to GPU
    model = SingleViewto3D(args)
    model.to(device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        #checkpoint = torch.load(f'checkpoint_{args.type}_smoooooth.pth')
        checkpoint = torch.load(f'{args.eval_chk_file}')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader) if args.max_iter is None else args.max_iter
    print(f"max_iter = {max_iter}")

    if args.examine_step == None:
        larger_step = 330
    else:
        larger_step = args.examine_step if args.examine_step > 330 else 330

    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        if args.type == 'point':
            if step == args.examine_step or (step == 5 or step == 64 or step == 76):
                vis_saliency_map_point(images_gt, args, step, model)
        elif args.type == 'mesh':
            if step == args.examine_step or (step == 13 or step == 110 or step == 51):
                vis_saliency_map_mesh(images_gt, args, step, model)
        elif args.type == 'vox':
            if step == args.examine_step or (step == 110 or step == 108 or step == 329):
                vis_saliency_map_vox(images_gt, args, step, model)

        if step == larger_step:
            break
        #images_gt.shape = torch.Size([32, 137, 137, 3])
        #mesh_gt.shape = torch.Size([32, 1, 32, 32, 32])
        #[batch_size, channels, depth, height, width]
        #predictions = model(images_gt, args)

        '''
        if args.type == "vox":
            #predictions = torch.nn.Sigmoid()(predictions)
            predictions = predictions.permute(0,1,4,3,2)

        if args.type == "point":
            pointclouds_tgt = sample_points_from_meshes(mesh_gt, args.n_points)
        '''
      
        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f);" % (step, max_iter, total_time, read_time, iter_time))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)