import argparse
import time
import os

import dataset_location
import losses
import torch
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
from tqdm import tqdm, trange
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


'''
def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser
'''

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
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
    parser.add_argument('--use_step_update', default=0, type=int)
    parser.add_argument('--show_lr_freq_step', default=100, type=int)
    return parser    


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
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
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
    return loss

class CustomScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr, max_lr, final_lr):
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

        # Cosine Annealing Phase (Mid-Phase: 5e-4 as transition point)
        #self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(8000 - warmup_steps), eta_min=5e-4)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=self.min_lr)

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warm-up
            progress = self.current_step / self.warmup_steps
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        elif self.current_step < 8000:
            # Cosine annealing decay
            self.cosine_scheduler.step()
            new_lr = self.cosine_scheduler.get_last_lr()[0]
        else:
            # Final decay to stabilize learning
            if step < 14000:
                progress = (self.current_step - 8000) / (self.total_steps - 8000)
                new_lr = 6e-4 + (self.final_lr - 6e-4) * progress
            else:
                progress = (self.current_step - 14000) / (self.total_steps - 14000)
                new_lr = 5e-4 + (self.final_lr - 5e-4) * progress  # Increase min LR in last phase

        # Apply new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_model(args):
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

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ Optimizer ================#
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs

    # ============ Cosine Annealing Scheduler ================#
    learning_rate = args.lr  # Max LR, 9e-4
    warmup_steps = args.cas_warmup_steps   # Steps for warm-up
    total_steps = args.early_stop_iter if args.early_stop_iter is not None else args.max_iter  # Total training steps
    min_lr = args.cas_min_lr         # Minimum LR after decay
    final_lr = args.cas_final_lr         # Final minimum LR for convergence
    cas_scheduler = CustomScheduler(optimizer, warmup_steps, total_steps, min_lr, learning_rate, final_lr)

    # ============ LR Scheduler  ============
    milestone_perc = [10, 20, 30, 50, 70, 85, 90]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(perc/100*args.max_iter/len(train_loader)) for perc in milestone_perc], gamma=0.5)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"{args.output_path}/checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)
        #print(f"prediction_3d.shape = {prediction_3d.shape}")
        #prediction3d.shape = torch.Size([32, 1, 32, 32, 32])
        #a = input()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
