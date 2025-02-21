import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from utils import render_voxel_mesh
from utils import render_point_cloud
from utils import render_pure_mesh
import dataset_location
import torch



def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--output_path', default='./outputs', type=str) 
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        #print(f'pointclouds_src.shape = {pointclouds_src.shape}')
        #print(f'pointclouds_tgt.shape = {pointclouds_tgt.shape}')
        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)
        #loss = losses.chamfer_loss_official(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')


def train_model(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created output directory: {args.output_path}")        
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']
        #print(f"voxel_coords.shape = {voxel_coords.shape}")
        #print(f"voxels_src.shape = {voxels_src.shape}")
        #print(f"voxels_tgt.shape = {voxels_tgt.shape}")
        '''
        voxel_coords.shape = torch.Size([1, 1398, 3])
        voxels_src.shape = torch.Size([1, 32, 32, 32])
        voxels_tgt.shape = torch.Size([1, 32, 32, 32])
        '''

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)
        render_voxel_mesh(voxels_src=voxels_src, is_logit=True, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_path}/q1.1_optimized.gif", fps=10)
        render_voxel_mesh(voxels_src=voxels_tgt, is_logit=False, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_path}/q1.1_groundtruth.gif", fps=10)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)
        render_point_cloud(src_points=pointclouds_src, image_size=256, device_tag=args.device, num_views=20, fps=10, dist=3, radius=0.03, output_path=f"{args.output_path}/q1.2_optimized.gif")
        render_point_cloud(src_points=pointclouds_tgt, image_size=256, device_tag=args.device, num_views=20, fps=10, dist=3, radius=0.03, output_path=f"{args.output_path}/q1.2_groundtruth.gif")
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        
        render_pure_mesh(mesh_src=mesh_src, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_path}/q1.3_optimized.gif", fps=10)
        render_pure_mesh(mesh_src=mesh_tgt, dist=3, num_views=20, device_tag=args.device, image_size=256, output_path = f"{args.output_path}/q1.3_groundtruth.gif", fps=10)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
