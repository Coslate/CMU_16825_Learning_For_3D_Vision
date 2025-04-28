import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Scene, Gaussians
from data_utils_harder_scene import get_nerf_datasets, trivial_collate
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import argparse
from pytorch3d.renderer import PerspectiveCameras
from scheduler import CustomScheduler  # [UPDATED to use CustomScheduler]
from data_utils_harder_scene import AverageMeter
from model_uncertainty import *
from scipy.stats import spearmanr

def plot_metrics_vs_step(val_mse_steps, val_mse_losses, spearman_steps, spearman_scores, out_path):
    plt.figure(figsize=(6, 4))

    # Plot Spearman correlation with circle markers
    plt.plot(spearman_steps, spearman_scores, marker='o', linestyle='-', label="Spearman ρ (rank correlation)")

    # Plot MSE Loss with dashed line and triangle markers
    plt.plot(val_mse_steps, val_mse_losses, marker='^', linestyle='--', label="MSE")

    plt.xlabel("Global Step")
    plt.ylabel("Metric Value")
    plt.title("Validation MSE & Spearman Correlation Over Steps")
    plt.legend()
    plt.grid(True)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, "metrics_vs_step.png"), bbox_inches='tight')
    plt.close()

def predict_uncertainty_map(model, feat_tensor):
    feat_tensor = feat_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) → (1, C, H, W)
    uncertainty_map = model(feat_tensor).squeeze(0).squeeze(0)      # (1, 1, H, W) → (H, W)
    return uncertainty_map

def compute_ssim_error_map(pred_img, gt_img):
    pred_np = pred_img.detach().cpu().numpy()
    gt_np = gt_img.detach().cpu().numpy()
    ssim_val, ssim_map = structural_similarity(
        pred_np, gt_np, channel_axis=-1, data_range=1.0, full=True
    )
    ssim_map = np.dot(ssim_map, [0.299, 0.587, 0.114])  # (H, W)
    return 1.0 - torch.tensor(ssim_map, dtype=torch.float32).to(pred_img.device)


def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def extract_features(scene, camera, args, img_size=(128, 128), time_profiling=False):
    if time_profiling:
        timings = {}  # Dictionary to store extraction time for each feature

    #Depth sorting
    if time_profiling:
        start_depth_sorting = time.time()
    z_vals = scene.compute_depth_values(camera)
    if time_profiling:
        timings['depth_compute_time'] = time.time() - start_depth_sorting
        start_depth_sorting = time.time()
    idxs = scene.get_idxs_to_filter_and_sort(z_vals)
    if time_profiling:
        timings['depth_sorting_time'] = time.time() - start_depth_sorting

    #Select & activate Gaussians
    means_3D = scene.gaussians.means[idxs]
    quats = scene.gaussians.pre_act_quats[idxs]
    scales = scene.gaussians.pre_act_scales[idxs]
    opacities = scene.gaussians.pre_act_opacities[idxs]
    colours = scene.gaussians.colours[idxs]
    if time_profiling:
        start_gau_act = time.time()
    quats, scales, opacities = scene.gaussians.apply_activations(quats, scales, opacities)
    if time_profiling:
        timings['gau_act_time'] = time.time() - start_gau_act

    #Project to 2D views
    if time_profiling:
        start_proj_gau = time.time()
    view_dirs = scene.calculate_gaussian_directions(means_3D, camera) #(N, 3)
    cov_2D = scene.gaussians.compute_cov_2D(means_3D, quats, scales, camera, img_size)
    if time_profiling:
        timings['proj_gau_time'] = time.time() - start_proj_gau

    #Compute alpha(visibility) map
    if time_profiling:
        start_alpha = time.time()
    alphas = scene.compute_alphas(opacities, scene.gaussians.compute_means_2D(means_3D, camera), cov_2D, img_size) #(N, H, W)
    trans = scene.compute_transmittance(alphas) #(N, H, W)
    alpha_sum = torch.sum(alphas, dim=0) #(H, W)
    if time_profiling:
        timings['alpha_sum_time'] = time.time() - start_alpha

    #Compute color variance map
    if time_profiling:
        start_color_var = time.time()
    weights = (alphas * trans).unsqueeze(-1) #(N, H, W, 1)
    colours_exp = colours[:, None, None, :] #(N, 1, 1, 3)
    weighted_color = (weights * colours_exp).sum(0) #(H, W, 3)
    mean_color = weighted_color / (weights.sum(0) + 1e-6) #(H, W, 3)
    # [CHUNKED] compute color variance in chunks to avoid OOM
    chunk_size = args.chunksize_colvar
    color_var_accum = 0.0
    weight_accum = 0.0
    for i in range(0, colours_exp.shape[0], chunk_size):
        chunk = colours_exp[i:i+chunk_size]
        w_chunk = weights[i:i+chunk_size]
        diff_sq = (chunk - mean_color[None]) ** 2
        weighted_diff = w_chunk * diff_sq
        color_var_accum += weighted_diff.sum(0)
        weight_accum += w_chunk.sum(0)
    color_var = (color_var_accum / (weight_accum + 1e-6)).mean(-1)  # (H, W)    
    #color_var = ((weights * (colours_exp - mean_color[None])**2).sum(0) / (weights.sum(0) + 1e-6)).mean(-1) #(H, W)
    if time_profiling:
        timings['color_var_time'] = time.time() - start_color_var

    #Compute weighted footprint size map
    if time_profiling:
        start_footprint = time.time()
    det_cov = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 0, 1] * cov_2D[:, 1, 0] #(N,)
    det_map = torch.sum(alphas * det_cov[:, None, None], dim=0) / (alpha_sum + 1e-6)
    if time_profiling:
        timings['footprint_time'] = time.time() - start_footprint

    # Compute view direction feature
    if time_profiling:
        start_viewdir = time.time()
    view_angle_cos = view_dirs[:, 2]
    view_map = torch.sum(alphas * view_angle_cos[:, None, None], dim=0) / (alpha_sum + 1e-6)
    feat_map = torch.stack([alpha_sum, color_var, det_map, view_map], dim=-1) #(H, W, 4)
    if time_profiling:
        timings['view_direction_time'] = time.time() - start_viewdir
        return feat_map, timings
    else:
        return feat_map

def visualize_prediction(gt_img, pred_img, ssim_error, uncertainty_map, out_dir, step, args):
    gt = (gt_img.detach().cpu().numpy() * 255).astype(np.uint8)
    pred = (pred_img.detach().cpu().numpy() * 255).astype(np.uint8)
    ssim_error_vis = (ssim_error.detach().cpu().numpy() * 255).astype(np.uint8)
    uncertainty_vis = (uncertainty_map.detach().cpu().numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))  # Larger space for image and titles

    imgs = [gt, pred, ssim_error_vis, uncertainty_vis]
    titles = ["Ground Truth", "GS Prediction", "SSIM Error Map", "Predicted Uncertainty"]
    cmaps = [None, None, 'magma', 'magma']

    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=16, pad=10)  # pad ensures title visibility
        ax.axis("off")
        ax.set_xlim(0, gt.shape[1])
        ax.set_ylim(gt.shape[0], 0)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.01, hspace=0.1)

    os.makedirs(out_dir, exist_ok=True)
    filename = f"val_step_tunet_{step:05d}.png" if args.use_tiny_unet else f"val_step_mlp_{step:05d}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=150)  # [✔] Don't use bbox_inches
    plt.close()    

'''
def visualize_prediction(gt_img, pred_img, ssim_error, uncertainty_map, out_dir, step, args):
    gt = (gt_img.detach().cpu().numpy() * 255).astype(np.uint8)
    pred = (pred_img.detach().cpu().numpy() * 255).astype(np.uint8)
    ssim_error_vis = (ssim_error.detach().cpu().numpy() * 255).astype(np.uint8)
    uncertainty_vis = (uncertainty_map.detach().cpu().numpy() * 255).astype(np.uint8)

    # [MODIFIED] 2x2 grid instead of 1x4, larger images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(gt, interpolation='none', origin='upper')
    axes[0, 0].set_title("Ground Truth")
    
    axes[0, 1].imshow(pred, interpolation='none', origin='upper')
    axes[0, 1].set_title("GS Prediction")
    
    axes[1, 0].imshow(ssim_error_vis, cmap='magma', interpolation='none', origin='upper')
    axes[1, 0].set_title("SSIM Error Map")
    
    axes[1, 1].imshow(uncertainty_vis, cmap='magma', interpolation='none', origin='upper')
    axes[1, 1].set_title("Predicted Uncertainty")

    for ax in axes.flatten():
        ax.axis("off")

    # [ADDED] better spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=2.0)

    # [MODIFIED] distinguish filename by model
    suffix = "tunet" if args.use_tiny_unet else "mlp"
    filename = os.path.join(out_dir, f"val_step_{suffix}_{step:05d}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
'''

def load_gaussian_checkpoint(checkpoint_path, scene):
    checkpoint = torch.load(checkpoint_path)
    scene.gaussians.load_state_dict(checkpoint['gaussians_state_dict'])
    print(f"[+] Loaded Gaussian checkpoint from iteration {checkpoint['itr']}")
    return checkpoint['itr']

def save_uncertainty_checkpoint(path, model, optimizer, scheduler, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
        'epoch': epoch
    }, path)
    print(f"[*] MLP/Unet Checkpoint saved to {path}")    

def load_uncertainty_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load the .npz file
    loaded_data = np.load(os.path.join(args.out_path, f"{args.load_metric_file}"))
    train_losses = list(loaded_data["train_losses"])
    train_steps = list(loaded_data["train_steps"])
    val_losses = list(loaded_data["val_losses"])
    val_steps = list(loaded_data["val_steps"])
    val_mse_losses = list(loaded_data["val_mse_losses"])
    val_mse_steps = list(loaded_data["val_mse_steps"])
    spearman_scores = list(loaded_data["spearman_scores"])
    spearman_steps = list(loaded_data["spearman_steps"])

    print(f"[+] Loaded MLP/Unet checkpoint from {path}")
    return checkpoint.get('epoch', 0), train_losses, train_steps, val_losses, val_steps, val_mse_losses, val_mse_steps, spearman_scores, spearman_steps 

def cosine_warmup_scheduler(args, total_steps):
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1. + np.cos(np.pi * progress))
    return lr_lambda    

def train_uncertainty_predictor(args):
    device = torch.device(args.device)
    print("Using device:", device)
    print("Device name:", torch.cuda.get_device_name(device)) if 'cuda' in str(device) else 'CPU'

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials",
        data_root=args.data_path,
        image_size=[args.img_size, args.img_size],
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=trivial_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=trivial_collate)

    gaussians = Gaussians(num_points=args.init_random_numpoints, init_type="random", device=args.device, isotropic=False)
    scene = Scene(gaussians)
    if args.use_tiny_unet:
        model = TinyUNet(in_channels=4, out_channels=1).to(device)  # [MODIFIED] match class signature
    else:
        model = UncertaintyMLP(input_dim=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.num_epochs * len(train_loader)
    scheduler = CustomScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,#1e-4,
        max_lr=args.lr,
        final_lr=args.final_lr,#1e-6,
        T_0=total_steps,
        T_mult=1
    )

    os.makedirs(args.out_path, exist_ok=True)

    # Load GS checkpoint if provided
    if args.resume_gs and os.path.exists(args.load_gs_ckpt):
        load_gaussian_checkpoint(args.load_gs_ckpt, scene)

    start_epoch = 0  # [ADDED] default start epoch
    global_step = 0
    train_losses = []
    train_steps = []
    val_losses = []
    val_steps = []
    val_mse_losses = []
    val_mse_steps = []
    spearman_scores = []
    spearman_steps = []
    # Load MLP/Unet checkpoint if provided
    if args.resume_unc_model and os.path.exists(args.load_unc_model_ckpt):
        start_epoch, train_losses, train_steps, val_losses, val_steps, val_mse_losses, val_mse_steps, spearman_scores, spearman_steps = load_uncertainty_checkpoint(args.load_unc_model_ckpt, model, optimizer, scheduler)        
        global_step = start_epoch * len(train_loader)

    print(f"Start Epoch = {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):  # [MODIFIED] use start_epoch loaded from checkpoint
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        loss_meter = AverageMeter()
        avg_loss = []
        for step, data in enumerate(pbar):
            gt_img = data[0]["image"].to(device)
            camera = ndc_to_screen_camera(data[0]["camera"]).to(device)

            with torch.no_grad():
                pred_img, _, _ = scene.render(camera, per_splat=-1, img_size=(args.img_size, args.img_size), bg_colour=(0.0, 0.0, 0.0))
                ssim_error_map = compute_ssim_error_map(pred_img, gt_img)
                features = extract_features(scene, camera, args, img_size=(args.img_size, args.img_size))

            ssim_error_flat = ssim_error_map.view(-1)
            if args.use_tiny_unet:
                feat_map = features.permute(2, 0, 1).unsqueeze(0)  # (H, W, 4) → (1, 4, H, W)
                pred = model(feat_map).squeeze(0).squeeze(0)   # (1, 1, H, W) → (H, W)
            else:
                feat_flat = features.view(-1, 4)
                pred = model(feat_flat).squeeze()

            loss = torch.nn.functional.smooth_l1_loss(pred.view(-1), ssim_error_flat, beta=1.0)  # Huber loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(global_step)  # [UPDATED for CustomScheduler]

            #print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}")
            loss_meter.update(loss.item())
            pbar.set_description(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f} | Avg. Loss {loss_meter.avg:.4f}")
            avg_loss.append(loss.item())

            if global_step % args.val_step == 0:
                val_data = next(iter(val_loader))
                gt_val = val_data[0]["image"].to(device)
                cam_val = ndc_to_screen_camera(val_data[0]["camera"]).to(device)
                with torch.no_grad():
                    pred_val, _, _ = scene.render(cam_val, per_splat=-1, img_size=(args.img_size, args.img_size), bg_colour=(0.0, 0.0, 0.0))
                    ssim_error_val = compute_ssim_error_map(pred_val, gt_val)
                    feat_val = extract_features(scene, cam_val, args, img_size=(args.img_size, args.img_size))
                    if args.use_tiny_unet:
                        uncertainty_val = predict_uncertainty_map(model, feat_val)  # [ALREADY PRESENT]
                    else:
                        uncertainty_val = model(feat_val.view(-1, 4)).view(args.img_size, args.img_size)

                    # Loss calculation
                    val_mse_loss = torch.nn.functional.mse_loss(uncertainty_val.view(-1), ssim_error_val.view(-1)).item()
                    val_loss = torch.nn.functional.smooth_l1_loss(uncertainty_val.view(-1), ssim_error_val.view(-1), beta=1.0).item()  # Huber loss

                    # Spearman correlation (rho close to 1.0 = good)
                    uncertainty_flat = uncertainty_val.detach().cpu().numpy().flatten()
                    ssim_error_flat = ssim_error_val.detach().cpu().numpy().flatten()
                    rho, _ = spearmanr(uncertainty_flat, ssim_error_flat)
                    print(f"[Validation] Spearman rank correlation: {rho:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}")  # [ADDED] log learning rate

                spearman_scores.append(rho)
                spearman_steps.append(global_step)
                val_losses.append(val_loss)
                val_steps.append(global_step)
                val_mse_losses.append(val_mse_loss)
                val_mse_steps.append(global_step)
                visualize_prediction(gt_val, pred_val, ssim_error_val, uncertainty_val, args.out_path, global_step, args)

            if global_step % args.save_step == 0:
                if args.use_tiny_unet:
                    save_uncertainty_checkpoint(os.path.join(args.out_path, f"uncertainty_tunet.{global_step}.pth"), model, optimizer, scheduler, epoch)
                else:
                    save_uncertainty_checkpoint(os.path.join(args.out_path, f"uncertainty_mlp.{global_step}.pth"), model, optimizer, scheduler, epoch)
                np.savez(os.path.join(args.out_path, f"{args.load_metric_file}"),
                        train_losses=train_losses,
                        val_losses=val_losses,
                        val_steps=val_steps,
                        val_mse_losses=val_mse_losses,
                        val_mse_steps=val_mse_steps,
                        spearman_scores=spearman_scores,
                        spearman_steps=spearman_steps)

            global_step += 1
        train_losses.append(np.mean(avg_loss))
        train_steps.append(global_step)

    if args.use_tiny_unet:
        save_uncertainty_checkpoint(os.path.join(args.out_path, f"uncertainty_tunet.{global_step}.pth"), model, optimizer, scheduler, args.num_epochs-1)
    else:
        save_uncertainty_checkpoint(os.path.join(args.out_path, f"uncertainty_mlp.{global_step}.pth"), model, optimizer, scheduler, args.num_epochs-1)
    print("MLP/Unet training complete and saved")

    # === Save Train Loss & Validation Loss Plot ===
    global_steps = np.arange(1, args.num_epochs*len(train_loader) + 1)

    # Interpolate validation losses to match every epoch
    train_interp = np.interp(global_steps, train_steps, train_losses)  
    val_interp = np.interp(global_steps, val_steps, val_losses)  
    val_mse_interp = np.interp(global_steps, val_mse_steps, val_mse_losses)  
    spearman_interp = np.interp(global_steps, spearman_steps, spearman_scores)  

    # Training Loss
    plt.plot(train_steps, train_losses,
         label="Training Loss (Actual)", color="blue", linestyle='-', marker='o')
    plt.plot(global_steps, train_interp,
         label="Train Loss (Interpolated)", color="blue", linestyle='--', alpha=0.5)

    # Validation Loss (Actual & Interpolated)
    plt.plot(val_steps, val_losses,
         label="Validation Loss (Actual)", color="crimson", linestyle='-', marker='s')
    plt.plot(global_steps, val_interp,
         label="Validation Loss (Interpolated)", color="crimson", linestyle='--', alpha=0.5)

    '''
    # MSE Loss (Actual & Interpolated)
    plt.plot(val_mse_steps, val_mse_losses,
         label="Validation MSE Loss (Actual)", color="darkorange", linestyle='-', marker='^')
    plt.plot(global_steps, val_mse_interp,
         label="Validation MSE Loss (Interpolated)", color="darkorange", linestyle='--', alpha=0.5)

    # Spearman Score (Actual & Interpolated)
    plt.plot(spearman_steps, spearman_scores,
         label="Spearman Correlation (Actual)", color="forestgreen", linestyle='-', marker='x')
    plt.plot(global_steps, spearman_interp,
         label="Spearman Correlation (Interpolated)", color="forestgreen", linestyle='--', alpha=0.5, marker='.')
    '''
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss and Metrics vs Steps")
    plt.legend()
    plt.grid(True)

    # Save plot
    if args.use_tiny_unet:
        plt.savefig(os.path.join(args.out_path, "loss_curve_tunet.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(args.out_path, "loss_curve_mlp.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory    

    # Save metrics
    plot_metrics_vs_step(val_mse_steps, val_mse_losses, spearman_steps, spearman_scores, args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/materials", help="Path to NeRF synthetic dataset")
    parser.add_argument("--out_path", type=str, default="./output_uncertainty", help="Directory to save results")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for training and rendering")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--init_random_numpoints", type=int, default=10000, help="Initial number of Gaussians")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--val_step", type=int, default=100, help="Validation frequency (in steps)")
    parser.add_argument("--save_step", type=int, default=500, help="Saving checkpoint frequency (in steps)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-6)
    parser.add_argument("--resume_gs", action="store_true", help="Whether to load pretrained Gaussian checkpoint")
    parser.add_argument("--load_gs_ckpt", type=str, default="./checkpoint_gs.pth", help="Path to GS checkpoint")
    parser.add_argument("--resume_unc_model", action="store_true", help="Whether to resume MLP/Unet training from checkpoint")
    parser.add_argument("--load_unc_model_ckpt", type=str, default="./uncertainty_mlp.pth", help="Path to MLP/Unet checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Number of warmup steps for LR scheduler") #3%*num_epochs*100
    parser.add_argument('--load_metric_file', default='metrics_loss_data.npz', type=str)
    parser.add_argument('--chunksize_colvar', default=1024, type=int, help="The chunk size of calculating color vairance.")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="The weight decay of MLP/Unet parameters.")
    parser.add_argument("--use_tiny_unet", action="store_true", help="Whether to use tiny unet or MLP. 1: tiny unet; 0: MLP")
    args = parser.parse_args()

    train_uncertainty_predictor(args)


