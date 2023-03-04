import os
import torch
import pandas as pd
from network_utils import build_mlp, np2torch
from environment import Environment


DATADIR = "trajectories"

env_generator = Environment()
env = env_generator.create_env()


action_dim = env.action_dim

obs_filepaths = [
    os.path.join(DATADIR, filename)\
        for filename in os.listdir(DATADIR) if "observations" in filename
]

first_df = pd.read_csv(obs_filepaths[0])
print(first_df.head())