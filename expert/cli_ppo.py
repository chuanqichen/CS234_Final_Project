import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--dirpath", default="ppo")              
    parser.add_argument("--filename", default="ppo")               
    parser.add_argument("--placement", default="fixed", help="fixed or random")               # fixed, random
    parser.add_argument("--operation", default="both", help="train, test, or both")               # train', 'test', or 'both'
    parser.add_argument("--start_subtask", default=2, type=int)               # start subtask 
    # Experiment
    parser.add_argument("--pi", default="200,50", help="arch for pi (separated with comma), such as 200, 50")  # policy architecture 
    parser.add_argument("--vf", default="200,50", help="arch for vf (separated with comma), such as 200,50")  # policy architecture  
    parser.add_argument("--env", default="stack2")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factortarget policy during critic update : 0.35
    parser.add_argument("--learning_rate", default=0.001, type=float)                # Learning Rate
    parser.add_argument("--clip_range", default=0.5, type=float)                # Clipping rate [0, 1]
    parser.add_argument("--clip_range_vf", default=0.5, type=float)             # Clipping rate [0, 1]   


    args = parser.parse_args()
    return args 

def save_confg(args):
    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)
    
    with open(os.path.join(args.dirpath + '/config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def read_confg(args):
    parser = argparse.argumentParser()
    args = parser.parse_args()
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
