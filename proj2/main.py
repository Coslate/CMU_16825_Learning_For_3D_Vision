import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio
import numpy as np
import time
import os
import subprocess
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./outputs")
    args = parser.parse_args()

    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")    

    # Ensure the output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created output directory: {args.output_path}")        

    # Q1
    commands = [
        "python fit_data.py --type 'vox'",
        "python fit_data.py --type 'point'",
        "python fit_data.py --type 'mesh'"
    ]

    for index, cmd in enumerate(commands):
        print(f"> Q1.{index+1}")
        subprocess.run(cmd, shell=True)    
        print(f"> Done.")



    

