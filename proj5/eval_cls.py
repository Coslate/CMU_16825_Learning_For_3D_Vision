import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir
from data_loader import get_data_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--main_dir', type=str, default='./data/')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    print(f"model_path = {model_path}")
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    test_dataloader = get_data_loader(args=args, train=False)
    '''
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))
    '''

    preds = []
    gts = []
    # ------ TO DO: Make Prediction ------
    with torch.no_grad():
        #outputs = model(test_data.to(args.device))       # (B, num_classes)
        #pred_label = outputs.argmax(dim=1).cpu()         # (B,)    
        for batch_points, batch_labels in test_dataloader:
            batch_points = batch_points.to(args.device)
            batch_outputs = model(batch_points)                   # (B, num_classes)
            batch_preds = batch_outputs.argmax(dim=1).cpu()       # (B,)
            preds.append(batch_preds)
            gts.append(batch_labels.cpu())                        # (B,)

    pred_label = torch.cat(preds, dim=0)  # (total_samples,)
    test_label = torch.cat(gts, dim=0)    # (total_samples,)

    # Accuracy
    #test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    test_accuracy = pred_label.eq(test_label).sum().item() / test_label.size(0)
    print("Test Accuracy: {:.4f}".format(test_accuracy))