import os
import argparse
import numpy as np
import torch
from models import seg_model
from utils import viz_seg, create_dir
from data_loader import get_data_loader


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seg_class', type=int, default=6)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg_vis')
    parser.add_argument('--exp_name', type=str, default='seg_demo')
    parser.add_argument('--num_visualize', type=int, default=5)
    parser.add_argument('--label_names', nargs='+', default=['chair', 'table', 'lamp', 'vase', 'bed', 'sofa'])
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    create_dir(args.output_dir)

    # Load model
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    model_path = f'./checkpoints/seg/{args.load_checkpoint}.pt'
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load test data
    test_data = torch.from_numpy(np.load(args.test_data)).float()  # (B, N, 3)
    test_label = torch.from_numpy(np.load(args.test_label)).long()  # (B, N)
    B, N, _ = test_data.shape

    # Predict
    pred_all, acc_all = [], []
    with torch.no_grad():
        for i in range(B):
            pc = test_data[i:i+1].to(args.device)  # (1, N, 3)
            gt = test_label[i]  # (N,)
            pred = model(pc).argmax(dim=2).squeeze(0).cpu()  # (N,)
            acc = pred.eq(gt).sum().item() / N
            pred_all.append(pred)
            acc_all.append(acc)

    acc_all = np.array(acc_all)
    sorted_indices = np.argsort(acc_all)
    bad_ids = sorted_indices[:2]
    #good_ids = np.random.choice(sorted_indices[2:], args.num_visualize - 2, replace=False)
    good_ids = sorted_indices[-4:-1]
    selected_ids = list(bad_ids) + list(good_ids)

    print("\n Visualizing segmentation predictions...")
    for i, obj_idx in enumerate(selected_ids):
        pts = test_data[obj_idx]
        gt = test_label[obj_idx]
        pred = pred_all[obj_idx]
        acc = acc_all[obj_idx]

        if acc < 0.50:
            base_name = f"fail_{obj_idx}"
        elif acc > 0.90:
            base_name = f"correct_{obj_idx}"
        else:
            continue

        viz_seg(pts, gt, f"{args.output_dir}/{base_name}_gt.gif", args.device)
        viz_seg(pts, pred, f"{args.output_dir}/{base_name}_pred.gif", args.device)
        print(f"[{base_name}] accuracy: {acc:.2%}")

    print(f"\n Saved all visualizations to: {args.output_dir}")
