import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cls_model, seg_model
from models_dgcnn import DGCNN_cls, DGCNN_seg
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir
from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine_lambda(epoch):
    if epoch < args.warmup_steps:
        return epoch / args.warmup_steps  # Linear warmup
    else:
        progress = (epoch - args.warmup_steps) / max(1, args.num_epochs - args.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return args.eta_min / args.lr + (1 - args.eta_min / args.lr) * cosine_decay

def train(train_dataloader, model, opt, epoch, args, writer):
    
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)
        labels = labels.to(args.device).to(torch.long)

        # ------ TO DO: Forward Pass ------
        predictions = model(point_clouds) 

        if (args.task == "seg"):
            labels = labels.reshape([-1])
            predictions = predictions.reshape([-1, args.num_seg_class])
            
        # Compute Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('train_loss', loss.item(), step+i)

    return epoch_loss

def test(test_dataloader, model, epoch, args, writer):
    
    model.eval()

    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                outputs = model(point_clouds) #(B, C)
                pred_labels = outputs.argmax(dim=1) #(B,)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj
                
        
    # Evaluation in Segmentation Task
    else:
        correct_point = 0
        num_point = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():     
                outputs = model(point_clouds) #(B, N, C)
                pred_labels = outputs.argmax(dim=2) #(B, N)

            correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_point / num_point

    writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{0}'.format(args.task+"_"+args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        if args.use_dgcnn:
            #model = DGCNN_cls(num_classes=3).to(args.device)
            model = DGCNN_cls(num_classes=3)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs.")
                model = torch.nn.DataParallel(model)
            model = model.to(args.device)
        else:
            model = cls_model(num_classes=3).to(args.device)
    else:
        if args.use_dgcnn:
            #model = DGCNN_seg(num_seg_classes=args.num_seg_class).to(args.device)
            model = DGCNN_seg(num_seg_classes=args.num_seg_class)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs.")
                model = torch.nn.DataParallel(model)
            model = model.to(args.device)
        else:
            model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)    

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Scheduler: Cosine Annealing over total epochs
    scheduler = LambdaLR(opt, lr_lambda=warmup_cosine_lambda)
    
    # Load Checkpoint 
    start_epoch = 0
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint_file)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            if 'optimizer_state_dict' in state_dict:
                opt.load_state_dict(state_dict['optimizer_state_dict'])
            if 'scheduler_state_dict' in state_dict:
                scheduler.load_state_dict(state_dict['scheduler_state_dict'])  # <-- Add this            

            start_epoch = state_dict.get('epoch', 0) + 1  # resume from next epoch            
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Dataloader for Training & Testing
    train_dataloader = get_data_loader(args=args, train=True)
    test_dataloader = get_data_loader(args=args, train=False)

    print ("successfully loaded data")

    best_acc = args.best_acc if args.best_acc != 0.0 else -1
    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        current_lr = opt.param_groups[0]['lr']

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        
        # Test
        current_acc = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            current_lr = opt.param_groups[0]['lr']
            print(f"learning_rate = {current_lr}")
            save_checkpoint(epoch=epoch, model=model, args=args, opt=opt, scheduler=scheduler, best=False)

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, opt=opt, scheduler=scheduler, best=True)


    print ("======== training completes ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--load_checkpoint_file', type=str, default='./best_model.pt')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Min LR after cosine annealing')
    parser.add_argument('--best_acc', type=float, default=0.0, help='Best test accuracy during last run.')
    parser.add_argument("--use_dgcnn", action="store_true", help="Whether to use DGCNN architecture for cls and seg tasks.")

    

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

    main(args)