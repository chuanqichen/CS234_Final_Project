import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--dirpath", default="td3")              
    parser.add_argument("--filename", default="td3")               
    parser.add_argument("--placement", default="fixed", help="fixed or random")               # fixed, random
    parser.add_argument("--operation", default="both", help="train, test, or both")               # train', 'test', or 'both'

    # Experiment
    parser.add_argument("--pi_arch", default="200,50", help="arch for pi (separated with comma), such as 200, 50")  # policy architecture 
    parser.add_argument("--qf_arch", default="200,50", help="arch for qf (separated with comma), such as 200,50")  # policy architecture     
    parser.add_argument("--policy", default="TD3")               # Policy name
    parser.add_argument("--env", default="stack2")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment

	# TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    args = parser.parse_args()
    return args 

def save_confg(args):
    with open(os.path.join(args.dirpath + 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def read_confg(args):
    parser = argparse.argumentParser()
    args = parser.parse_args()
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
