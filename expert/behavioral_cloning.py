import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from network_utils import build_mlp, np2torch, device
from environment import Environment
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------- #
#                                     Input                                    #
# ---------------------------------------------------------------------------- #
# Can modify
DATADIR = "trajectories"
batch_size = 64

# ---------------------------------------------------------------------------- #
#                                  Environment                                 #
# ---------------------------------------------------------------------------- #

env_generator = Environment()
env = env_generator.create_env()
action_dim = env.action_dim
with open(os.path.join(DATADIR, "obs_dims.json"), "r") as f:
    obs_dict = json.load(f)
    obs_dims = sum([v for k, v in obs_dict.items()])
    train_dims = sum([v for k, v in obs_dict.items() if k in [
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel"
    ]])

with open(os.path.join(DATADIR, "img_dims.json"), "r") as f:
    img_dims = sum([v for k, v in json.load(f).items()])

# ---------------------------------------------------------------------------- #
#                                    Helper                                    #
# ---------------------------------------------------------------------------- #
def create_dataloader(obs, actions, batch_size: int):
    tensor_dataset = TensorDataset(np2torch(obs), np2torch(actions))
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #

network = build_mlp(
    input_size=obs_dims, output_size=action_dim, n_layers=5, size=obs_dims * 2
).to(device=device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.0000075)
criterion = torch.nn.MSELoss()

obs_filepaths = [
    os.path.join(DATADIR, filename)\
        for filename in os.listdir(DATADIR) if "observations" in filename
]

imgs_filepaths = [
    os.path.join(DATADIR, filename)\
        for filename in os.listdir(DATADIR) if "imgs" in filename
]

combined_filepaths = []
for filename in obs_filepaths:
    common_name = "_" + "_".join(filename.split("_")[-2:])
    img_filename = [img_file for img_file in imgs_filepaths if common_name in img_file][0]
    combined_filepaths.append((filename, img_filename))


epochs = 10
file_no = 0
total_file = len(obs_filepaths) * epochs
for i in range(epochs):
    for filepath_obs, filepath_imgs in combined_filepaths:
        print()
        file_no += 1
        df_obs = pd.read_csv(filepath_obs)
        df_imgs = pd.read_csv(filepath_imgs)
        obs = df_obs.iloc[:, 0:train_dims]
        imgs = df_imgs.iloc[:, 0:img_dims]
        
        obs_imgs = pd.concat([obs, imgs], axis=1).values.astype(np.float32)
        actions = df_obs.iloc[:, obs_dims:(obs_dims+action_dim)].values.astype(np.float32)
        
        dataloader = create_dataloader(obs_imgs, actions, batch_size=batch_size)
        n_dataloader = len(dataloader)
        del obs, imgs, obs_imgs, actions, df_obs, df_imgs
        break

        # TODO: To be modified with CNN + Dense, use MSE LOSS without tanh on output layer.
        # for batch, (X, y) in enumerate(dataloader):

        #     y_hat = network(X)
        #     loss = criterion(y_hat, y)
        #     print(
        #         f"File: {file_no}/{total_file}, Batch: {batch}/{n_dataloader}, Loss: {loss}",
        #         end="\r"
        #     )

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

    break


torch.save(network.state_dict(), "model.pt") # saving the model        