# This is the config for MoDL dataset. 

import argparse
import os

from datetime import datetime

import torch

this_dir = os.path.dirname(os.path.realpath(__file__))
default_logdir = os.path.join(
    this_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S")
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", 
    type = int, 
    default = 100, 
    help= "Training Epochs"
)
parser.add_argument(
    "--batch_size",
    type = int, 
    default = 1, 
    help= "Training batch size"
)

parser.add_argument(
    "--val_batch",
    type = int, 
    default = 1, 
    help= "Validation batch size"
)

parser.add_argument(
    "--sigma",
    type = float,
    default = 0.01,
    help = "Noise level"
)