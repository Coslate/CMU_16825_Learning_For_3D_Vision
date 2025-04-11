import os
import numpy as np
import torch
import argparse
from models import cls_model
from data_loader import get_data_loader
from utils import viz_seg, create_dir, viz_cls

def rotate_point_cloud_z(batch_pc, angle_deg):
    """Rotate the point cloud along the Z-axis by angle in degrees."""
    angle_rad = np.radians(angle_deg)
    cosval = np.cos(angle_rad)
    sinval = np.sin(angle_rad)
    rotation_matrix = torch.tensor([[cosval, -sinval, 0],
                                    [sinval,  cosval, 0],
                                    [0,       0,      1]], dtype=torch.float32, device=batch_pc.device)
    return torch.matmul(batch_pc, rotation_matrix)  # (B, N, 3) @ (3, 3) -> (B, N, 3)    

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_dir(args.output_dir)
    class_names = ["chair", "vase", "lamp"]

    # Load test data
    ind = np.random.choice(10000, args.num_points, replace=False)
    data = torch.from_numpy(np.load(args.test_data)[:, ind, :]).float().to(device)  # (B, num_points, 3)
    B = data.shape[0]

    # Load model
    model = cls_model(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    test_loader = get_data_loader(args=args, train=False)
    preds = []
    gts = []
    # Make predictions
    with torch.no_grad():
        #outputs = model(data)
        #preds = outputs.argmax(dim=1).cpu()
        for batch_points, batch_labels in test_loader:
            batch_points = batch_points.to(device)
            batch_points = rotate_point_cloud_z(batch_points, angle_deg=args.rotate_angle)
            batch_outputs = model(batch_points)             # (B, num_classes)
            batch_preds = batch_outputs.argmax(dim=1).cpu() # (B,)
            preds.append(batch_preds)
            gts.append(batch_labels.cpu())        

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(gts, dim=0)

    # Visualization color mapping
    colors = {
        0: torch.tensor([1.0, 0.0, 0.0]),  # chair → red
        1: torch.tensor([0.0, 1.0, 0.0]),  # vase  → green
        2: torch.tensor([0.0, 0.0, 1.0]),  # lamp  → blue
    }

    print("\n Examine:")
    for i, idx in enumerate([806, 333, 690, 406, 750, 650]):
        pred_cls = preds[idx].item()
        gt_cls = labels[idx].item()
        print(f"[{i}] GT: {class_names[gt_cls]} | Pred: {class_names[pred_cls]}")
        fake_labels = torch.full((args.num_points,), pred_cls)
        viz_cls(
            verts=rotate_point_cloud_z(data[idx:idx+1], angle_deg=args.rotate_angle).squeeze(0).cpu(),
            labels=fake_labels,
            path=os.path.join(args.output_dir, f"examine_{i}_{class_names[pred_cls]}_idx_{idx}.gif"),
            device=device,
            colors=colors,
            loop=0
        )

    # Visualize random correct predictions
    print("\n Random correct predictions:")
    correct_indices = (preds == labels).nonzero().squeeze().tolist()
    random_correct = np.random.choice(correct_indices, args.num_visualize, replace=False)

    for i, idx in enumerate(random_correct):
        pred_cls = preds[idx].item()
        gt_cls = labels[idx].item()
        print(f"[{i}] GT: {class_names[gt_cls]} | Pred: {class_names[pred_cls]}")
        fake_labels = torch.full((args.num_points,), pred_cls)
        viz_cls(
            verts=rotate_point_cloud_z(data[idx:idx+1], angle_deg=args.rotate_angle).squeeze(0).cpu(),
            labels=fake_labels,
            path=os.path.join(args.output_dir, f"correct_{i}_{class_names[pred_cls]}_idx_{idx}.gif"),
            device=device,
            colors=colors,
            loop=0
        )

    # Visualize one failure per class
    print("\n Failure cases:")
    failures = (preds != labels).nonzero().squeeze().tolist()
    seen_classes = set()
    for idx in failures:
        gt = labels[idx].item()
        pred = preds[idx].item()
        if gt not in seen_classes:
            print(f"- GT: {class_names[gt]} | Pred: {class_names[pred]} | Sample #{idx}")
            fake_labels = torch.full((args.num_points,), pred)
            viz_cls(
                verts=rotate_point_cloud_z(data[idx:idx+1], angle_deg=args.rotate_angle).squeeze(0).cpu(),
                labels=fake_labels,
                path=os.path.join(args.output_dir, f"fail_GT_{class_names[gt]}_PRED_{class_names[pred]}_idx_{idx}.gif"),
                device=device,
                colors=colors,
                loop=0
            )
            seen_classes.add(gt)

            # Example interpretations
            if gt == 0:
                print("   → Misclassified chair: could lack clear backrest or have vertical symmetry.")
            elif gt == 1:
                print("   → Misclassified vase: may resemble lamp base or cylindrical object.")
            elif gt == 2:
                print("   → Misclassified lamp: may appear too simple or smooth, like a vase.")

        if len(seen_classes) == args.num_classes:
            break

def create_parser():
    parser = argparse.ArgumentParser(description="Visualize classification predictions on test point clouds.")
    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy', help='Path to test data .npy file')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy', help='Path to test label .npy file')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/cls/best_model.pt', help='Trained model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./output/cls_vis', help='Directory to save output GIFs')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points per sample')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of object classes')
    parser.add_argument('--num_visualize', type=int, default=5, help='Number of correct predictions to visualize')
    parser.add_argument('--main_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--task', type=str, default='cls', help='Task name: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--rotate_angle', type=float, default=0.0, help='Angle in degrees to rotate input point clouds (Z-axis)')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)
