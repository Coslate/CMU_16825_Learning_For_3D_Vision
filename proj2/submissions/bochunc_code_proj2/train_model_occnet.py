import argparse
import time
import os

import dataset_location
import losses
import torch
import torch.nn as nn
from model import SingleViewto3D
from model_occnet import ImageConditionedOccupancyNetwork
#from model_occnet2 import ImageConditionedOccupancyNetwork2
from model_occnet3 import ImageConditionedOccupancyNetwork2
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
from tqdm import tqdm, trange
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--type', default='occ', choices=['vox', 'point', 'mesh', 'occ'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=1000, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', action='store_true')    
    parser.add_argument('--output_path', default='./outputs', type=str)           
    parser.add_argument('--early_stop_iter', default=None, type=int)           
    parser.add_argument('--lr_sch_on', default=1, type=int)
    parser.add_argument('--use_cas', default=0, type=int)
    parser.add_argument('--cas_warmup_steps', default=1000, type=int)
    parser.add_argument('--cas_min_lr', default=1e-5, type=float)
    parser.add_argument('--cas_final_lr', default=1e-4, type=float)
    parser.add_argument('--cas_T_0', default=2500, type=int)
    parser.add_argument('--cas_T_mult', default=1, type=int)
    parser.add_argument('--use_step_update', default=0, type=int)
    parser.add_argument('--show_lr_freq_step', default=100, type=int)
    parser.add_argument('--n_sample_pt', default=8192, type=int)
    return parser    

def analyze_occupancy_balance(occupancy_labels):
    """
    Analyze the distribution of occupied vs. non-occupied points.
    
    Args:
        occupancy_labels (torch.Tensor): Tensor of shape (B * num_samples, 1)
    """
    occupancy_labels = occupancy_labels.view(-1)  # Flatten to 1D
    num_occupied = (occupancy_labels > 0.5).sum().item()  # Count occupied points
    num_free = (occupancy_labels <= 0.5).sum().item()  # Count free space points
    total = num_occupied + num_free

    print(f"[occupancy_labels] Total Samples: {total}")
    print(f"[occupancy_labels] Occupied Voxels (1s): {num_occupied} ({num_occupied / total:.2%})")
    print(f"[occupancy_labels] Free Voxels (0s): {num_free} ({num_free / total:.2%})")

def sample_3d_points_balanced(voxel_grid, num_samples):
    B, Z, Y, X = voxel_grid.shape  # Get voxel grid dimensions

    coords_list = []
    labels_list = []

    for b in range(B):  # Iterate over each batch
        occupied_indices = np.array(np.where(voxel_grid[b] == 1)).T  # Shape: (N_occupied, 3)
        free_indices = np.array(np.where(voxel_grid[b] == 0)).T  # Shape: (N_free, 3)

        # Compute how many samples to take from each class
        num_occupied_samples = num_samples // 2
        num_free_samples = num_samples - num_occupied_samples  # Keep total samples balanced

        # Randomly sample occupied and free indices
        #occupied_sampled = occupied_indices[np.random.choice(len(occupied_indices), num_occupied_samples, replace=False)]

        #print(f"len(occupied_indices) = {len(occupied_indices)}")
        #print(f"num_samples//2 = {num_samples//2}")
        #print(f"num_occupied_samples = {num_occupied_samples}")
        #input()
        # Sample occupied points
        satisfied = 0
        occupied_sampled = occupied_indices
        if len(occupied_indices) >= num_occupied_samples:
            occupied_sampled = occupied_indices[np.random.choice(len(occupied_indices), num_occupied_samples, replace=False)]
            satisfied = 1
        else:
            missing_samples = num_occupied_samples - len(occupied_indices)
            augmented_samples = []
            augmented_batches = []

            while len(augmented_samples) < missing_samples:
                # Select a batch to take occupied samples from (not b)
                other_batches = list(range(B))  # All batch indices
                other_batches.remove(b)  # Remove current batch
                other_b = np.random.choice(other_batches)  # Select a different batch

                # Get occupied indices from this random batch
                augmented = np.array(np.where(voxel_grid[other_b] == 1)).T

                # Ensure augmented samples are still occupied
                valid_mask = voxel_grid[other_b, augmented[:, 0], augmented[:, 1], augmented[:, 2]] == 1

                # Append only valid augmented samples
                augmented_valid = augmented[valid_mask]
                augmented_samples.extend(augmented_valid)

                # Store which batch these samples came from
                augmented_batches.extend([other_b] * len(augmented_valid))

            # Convert list to array and trim to exact required size
            augmented_samples = np.array(augmented_samples)[:missing_samples]
            augmented_batches = np.array(augmented_batches)[:missing_samples]  # Trim batch indices

            # Merge with original occupied samples
            occupied_sampled = np.vstack((occupied_sampled, augmented_samples))


        free_sampled = free_indices[
            np.random.choice(len(free_indices), num_free_samples, replace=True)
        ]

        # Combine occupied and free samples
        sampled_indices = np.vstack((occupied_sampled, free_sampled))  # Shape: (num_samples, 3)

        # Normalize coordinates to (-1,1)
        coords = np.stack([
            (sampled_indices[:, 0] / (Z - 1)) * 2 - 1,  # Scale X
            (sampled_indices[:, 1] / (Y - 1)) * 2 - 1,  # Scale Y
            (sampled_indices[:, 2] / (X - 1)) * 2 - 1,  # Scale Z
        ], axis=-1)

        # Get corresponding occupancy labels
        #labels = voxel_grid[b, sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]
        
        # Get corresponding occupancy labels with correct batch indexing
        occupied_label_mask = np.ones(len(occupied_sampled), dtype=bool)  # True for occupied, False for free
        batch_indices = np.full(len(sampled_indices), b)  # Default all to current batch
        if satisfied == 0:
            batch_indices[:len(occupied_indices) + len(augmented_batches)] = np.concatenate(([b] * len(occupied_indices), augmented_batches))

        labels = voxel_grid[batch_indices, sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]        
        '''
        print(f"coords.shape = {coords.shape}")
        print(f"labels.shape = {labels.shape}")
        print(f"sampled_indices min: {sampled_indices.min(axis=0)}")
        print(f"sampled_indices max: {sampled_indices.max(axis=0)}")
        print(f"voxel_grid shape: {voxel_grid.shape}")        
        analyze_occupancy_balance(labels)
        input()
        '''

        coords_list.append(coords)
        labels_list.append(labels)

    # Convert to PyTorch tensors
    coords = torch.tensor(np.vstack(coords_list), dtype=torch.float32).reshape(-1, 3).to("cuda")  
    occupancy_labels = torch.tensor(np.concatenate(labels_list), dtype=torch.float32).reshape(-1, 1).to("cuda")  

    return coords, occupancy_labels    

'''
def sample_3d_points(voxel_grid, num_samples):
    B, Z, Y, X = voxel_grid.shape  # Get voxel grid dimensions

    # Sample points in (-1,1)^3 instead of (0,1)
    coords = np.random.uniform(-1, 1, size=(B, num_samples, 3))  # range (-1,1)

    # Convert continuous coords (-1,1) to voxel indices (0, Z-1), (0, Y-1), (0, X-1)
    voxel_indices_x = ((coords[..., 0] + 1) * (X - 1) / 2).astype(int)  # Scale X to (0, X-1)
    voxel_indices_y = ((coords[..., 1] + 1) * (Y - 1) / 2).astype(int)  # Scale Y to (0, Y-1)
    voxel_indices_z = ((coords[..., 2] + 1) * (Z - 1) / 2).astype(int)  # Scale Z to (0, Z-1)

    # Ensure indices stay within bounds
    voxel_indices_x = np.clip(voxel_indices_x, 0, X - 1)
    voxel_indices_y = np.clip(voxel_indices_y, 0, Y - 1)
    voxel_indices_z = np.clip(voxel_indices_z, 0, Z - 1)

    # Get occupancy labels from voxel grid
    labels = voxel_grid[np.arange(B)[:, None], voxel_indices_z, voxel_indices_y, voxel_indices_x]  # (B, num_samples)

    # Convert to PyTorch tensors
    coords = torch.tensor(coords, dtype=torch.float32).reshape(-1, 3).to(args.device)  # (B * num_samples, 3)
    occupancy_labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1).to(args.device)  # (B * num_samples, 1)

    return coords, occupancy_labels    
'''

def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    elif args.type == 'occ':
        # Sample 3D points from voxel grid for occupancy training
        voxels = feed_dict["voxels"].float().squeeze(1)  # Remove channel dim -> (B, Z, Y, X)
        coords, occupancy_labels = sample_3d_points_balanced(voxels, num_samples=args.n_sample_pt)
        ground_truth_3d = (coords.to(args.device), occupancy_labels.to(args.device))  # Return sampled points, not full grid

    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        if args.type == 'occ':
            return images.to(args.device), ground_truth_3d
        else:
            return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth, use_logit=False)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        #loss_reg = losses.chamfer_loss_test(sample_pred, sample_trg)
        #loss_reg = losses.chamfer_loss_test2(sample_pred, sample_trg)
        #loss_reg2 = losses.chamfer_loss_official(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        #print(f"Official Chamfer Loss: {loss_reg2.item()}")
        #print(f"Manual Chamfer Loss: {loss_reg.item()}")
        #diff = torch.abs(loss_reg2 - loss_reg)
        #print(f"Difference: {diff.item()}")
    elif args.type == 'occ':
        loss = losses.occ_loss(predictions, ground_truth, use_logit=True)
    return loss

class CustomScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr, max_lr, final_lr, T_0, T_mult):
        """
        Custom Learning Rate Scheduler.
        
        - Warmup (Linear): Increases from min_lr to max_lr over `warmup_steps`
        - Cosine Annealing: Decays from max_lr to mid-range (5e-4) until 8000 steps
        - Final Linear Decay: Reduces from 5e-4 to final_lr over last 2000 steps

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Starting learning rate
            max_lr: Peak learning rate
            final_lr: Final learning rate at the end of training
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.current_step = 0
        self.T_0 = T_0
        self.T_mult = T_mult

        # Cosine Annealing Phase (Mid-Phase: 5e-4 as transition point)
        #self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(8000 - warmup_steps), eta_min=5e-4)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warm-up
            progress = self.current_step / self.warmup_steps
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        elif self.current_step < 4000:
            # Cosine annealing decay
            self.cosine_scheduler.step()
            new_lr = self.cosine_scheduler.get_last_lr()[0]
        else:
            # Final decay to stabilize learning
            progress = (self.current_step - 4000) / (self.total_steps - 4000)
            new_lr = 3e-4 + (self.final_lr - 3e-4) * progress

        # Apply new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.2)  # Reduce gain to prevent extreme activations
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)                

def train_model(args):
    #torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created output directory: {args.output_path}")        

    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )
    print(f"args.load_feat = {args.load_feat}")

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    #model = ImageConditionedOccupancyNetwork(args)
    model = ImageConditionedOccupancyNetwork2(args)
    model.to(args.device)
    model.train()
    model.apply(init_weights)

    # ============ Optimizer ================#
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-5)  # to use with ViTs

    # ============ Cosine Annealing Scheduler ================#
    learning_rate = args.lr  # Max LR, 9e-4
    warmup_steps = args.cas_warmup_steps   # Steps for warm-up
    total_steps = args.early_stop_iter if args.early_stop_iter is not None else args.max_iter  # Total training steps
    min_lr = args.cas_min_lr         # Minimum LR after decay
    final_lr = args.cas_final_lr         # Final minimum LR for convergence
    T_0 = args.cas_T_0
    T_mult = args.cas_T_mult
    cas_scheduler = CustomScheduler(optimizer, warmup_steps, total_steps, min_lr, learning_rate, final_lr, T_0, T_mult)

    # ============ LR Scheduler  ============
    milestone_perc = [10, 20, 30, 50, 70, 85, 90]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(perc/100*args.max_iter/len(train_loader)) for perc in milestone_perc], gamma=0.5)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"{args.output_path}/checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #cas_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    print(f"len(train_loader) = {len(train_loader)}")
    epoch_cnt = int(start_iter/len(train_loader))

    if args.lr_sch_on == 1:
        if args.use_step_update == 1:
            if args.use_cas == 1:
                cas_scheduler.step(start_iter)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cas_scheduler.get_last_lr()[0]
            else:
                lr_scheduler.step(start_iter)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cas_scheduler.get_last_lr()[0]
        else:
            if args.use_cas == 1:
                cas_scheduler.step(epoch_cnt)
            else:
                lr_scheduler.step(epoch_cnt)

    scaler = torch.cuda.amp.GradScaler()

    for step in range(start_iter, args.max_iter):
        if args.early_stop_iter is not None and step > args.early_stop_iter:
            break

        iter_start_time = time.time()


        if args.lr_sch_on == 1:
            if args.use_step_update == 1:
                if args.use_cas == 1:
                    cas_scheduler.step(step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = cas_scheduler.get_last_lr()[0]
                else:
                    lr_scheduler.step(step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = cas_scheduler.get_last_lr()[0]

                if (step % args.show_lr_freq_step) == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Step {step} | Learning Rate: {current_lr:.6f}")

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)
            epoch_cnt += 1

            if args.use_step_update == 0:
                if args.lr_sch_on == 1:
                    if args.use_cas == 1:
                        cas_scheduler.step(epoch_cnt)
                    else:
                        lr_scheduler.step(epoch_cnt)
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch_cnt} | Learning Rate: {current_lr:.6f}")

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        #print(f"images_gt.shape = {images_gt.shape}")
        #print(f"ground_truth_3d.shape = {ground_truth_3d.shape}")
        #images_gt.shape = torch.Size([32, 137, 137, 3])
        #ground_truth_3d.shape = torch.Size([32, 1, 32, 32, 32])
        #[batch_size, channels, depth, height, width]

        # Handle different training types
        coords, occupancy_labels = ground_truth_3d  # (B * num_samples, 3), (B * num_samples, 1)
        if torch.isnan(occupancy_labels).any() or torch.isinf(occupancy_labels).any():
            print(f"[Step {step}] WARNING: NaN or Inf detected in occupancy labels!")
            occupancy_labels = torch.clamp(occupancy_labels, min=0.0, max=1.0)  # Clamp between 0 and 1        
            input()

        # Forward pass
        #print(f"coords.shape = {coords.shape}")
        #input()
        prediction_3d = model(images_gt, coords, args)  # Predict occupancy
        if step % 50 == 0:
            print(f"...[Step {step}] prediction_3d.max =  {prediction_3d.max()}") 
            print(f"...[Step {step}] prediction_3d.min =  {prediction_3d.min()}") 
            analyze_occupancy_balance(occupancy_labels)
        if torch.isnan(prediction_3d).any() or torch.isinf(prediction_3d).any():
            print(f"[Step {step}] WARNING: NaN or Inf detected in predictions!")
            prediction_3d = torch.clamp(prediction_3d, min=-10, max=10)  # Clamp as safety        
            input()
        loss = calculate_loss(prediction_3d, occupancy_labels, args)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"[Step {step}] WARNING: NaN or Inf detected in loss!")
            input()
        #print(f"prediction_3d.shape = {prediction_3d.shape}")
        #prediction3d.shape = torch.Size([32, 1, 32, 32, 32])
        #a = input()

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.08)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    #"scheduler_state_dict": cas_scheduler.state_dict(),
                },
                f"{args.output_path}/checkpoint_{args.type}.pth",
            )

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
