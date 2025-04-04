import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default='seg')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr1e-2_numepochs300_etamin1e-6_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin3e-6_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin5e-5_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin1e-4_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin8e-5_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin1e-5_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)
    #model_path = './checkpoints_lr9e-3_numepochs300_etamin5e-6_warmupsteps10/seg/{}.pt'.format(args.load_checkpoint)

    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    '''
    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
    '''

    test_dataloader = get_data_loader(args, train=False)
    # ------ TO DO: Make Prediction ------
    '''
    with torch.no_grad():
        outputs = model(test_data.to(args.device))  # (B, N, num_classes)
        pred_label = outputs.argmax(dim=2).cpu()    # (B, N)
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])

    '''
    # Evaluate as in train.py
    correct_point = 0
    num_point = 0
    all_preds = []
    all_labels = []
    all_data = []
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            outputs = model(point_clouds)  # (B, N, C)
            pred_labels = outputs.argmax(dim=2)  # (B, N)

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view(-1).size()[0]

            # Store for visualization
            all_preds.append(pred_labels.cpu())
            all_labels.append(labels.cpu())
            all_data.append(point_clouds.cpu())

    pred_label = torch.cat(all_preds, dim=0)
    test_label = torch.cat(all_labels, dim=0)        
    test_data  = torch.cat(all_data, dim=0)        
    test_accuracy = correct_point / num_point
    print ("total test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    # Reload just the i-th object from raw data
    '''
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
    acc_i = pred_label[args.i].eq(test_label[args.i]).sum().item() / test_label[args.i].numel()
    print(f"Test accuracy for object {args.i}: {acc_i}")
    '''

    if args.i >= len(all_preds):
        print(f"Invalid index: args.i={args.i}, but test set has only {len(all_preds)} samples")
        exit(1)

    viz_seg(test_data[args.i], test_label[args.i], f"{args.output_dir}/gt_{args.exp_name}.gif", args.device)
    viz_seg(test_data[args.i], pred_label[args.i], f"{args.output_dir}/pred_{args.exp_name}.gif", args.device)
    acc_i = all_preds[args.i].eq(all_labels[args.i]).sum().item() / all_labels[args.i].numel()
    print(f"test accuracy for object {args.i}: {acc_i}")    
