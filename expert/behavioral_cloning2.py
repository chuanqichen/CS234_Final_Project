import os
import sys
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from network_utils import NetworkBC, MultiLayerCNN, np2torch
from config import device, device_name
from environment import Environment
import matplotlib.pyplot as plt


# All observations and observations csv columns, for reference
# robot0_joint_pos_cos     0 through 6
# robot0_joint_pos_sin     7 through 13
# robot0_joint_vel        14 through 20
# robot0_eef_pos          21 through 23
# robot0_eef_quat         24 through 27
# robot0_gripper_qpos     28 through 29
# robot0_gripper_qvel     30 through 31
# agentview_image         (see images csv)
# cubeA_pos               32 through 34
# cubeA_quat              35 through 38
# cubeB_pos               39 through 41
# cubeB_quat              42 through 45
# gripper_to_cubeA        46 through 48
# gripper_to_cubeB        49 through 51
# cubeA_to_cubeB          52 through 54
# robot0_proprio-state    55 through 86
# object-state            87 through 109


# ---------------------------------------------------------------------------- #
#                                     Input                                    #
# ---------------------------------------------------------------------------- #
# Can modify
DATADIR = input("Enter dataset directory: ")
if not os.path.exists(DATADIR):
    print("Directory doesn't exists!")
    sys.exit(1)
batch_size = 64
TRAINING_MODE = input("Enter training mode (pick, place, all): ").lower()


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
        "gripper_to_cubeA",
    ]])

## with open(os.path.join(DATADIR, "img_dims.json"), "r") as f:
    ## img_dims = sum([v for k, v in json.load(f).items()])


# ---------------------------------------------------------------------------- #
#                                    Helper                                    #
# ---------------------------------------------------------------------------- #
def create_dataloader(obs, actions, batch_size: int):
    tensor_dataset = TensorDataset(np2torch(obs), np2torch(actions))
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ---------------------------------------------------------------------------- #
#                                    Network                                   #
# ---------------------------------------------------------------------------- #

## network = MultiLayerCNN(
    ## obs_input_size=train_dims,
    ## img_input_height=int(np.sqrt(img_dims / 3)),
    ## img_input_width=int(np.sqrt(img_dims / 3)),
    ## output_size=action_dim
## ).to(device=device)
network = NetworkBC(
    obs_input_size=train_dims,
    output_size=action_dim
).to(device=device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


# ---------------------------------------------------------------------------- #
#                                   Training                                   #
# ---------------------------------------------------------------------------- #

obs_filepaths = [
    os.path.join(DATADIR, filename)\
        for filename in os.listdir(DATADIR) if "observations" in filename
]

## imgs_filepaths = [
    ## os.path.join(DATADIR, filename)\
        ## for filename in os.listdir(DATADIR) if "imgs" in filename
## ]

combined_filepaths = obs_filepaths
## combined_filepaths = []
## for filename in obs_filepaths:
    ## common_name = "_" + "_".join(filename.split("_")[-2:])
    ## img_filename = [img_file for img_file in imgs_filepaths if common_name in img_file][0]
    ## combined_filepaths.append((filename, img_filename))

# Change the epochs here
epochs = 10
file_no = 0
total_file = len(obs_filepaths) * epochs
print("TOTAL_FILE:", total_file)
print("...Start Training...")
for i in range(epochs):
    ## for filepath_obs, filepath_imgs in combined_filepaths:
    for filepath_obs in combined_filepaths:
        print()
        file_no += 1
        try:
            df_obs = pd.read_csv(filepath_obs)
            ## df_imgs = pd.read_csv(filepath_imgs)
            if TRAINING_MODE == "pick":
                df_obs = df_obs[df_obs["subtask_id"].isin([2])]
                ## df_imgs = df_imgs[df_imgs["subtask_id"] <= 4]
            elif TRAINING_MODE == "place":
                df_obs = df_obs[df_obs["subtask_id"].isin([6])]
                ## df_imgs = df_imgs[df_imgs["subtask_id"] > 4]
        except Exception as e:
            print(f"EXCEPTION ({type(e)}):", e)
            continue
        obs = df_obs.iloc[:, 46:49]  # gripper_to_cubeA
        ## obs = df_obs.iloc[:, 0:train_dims]
        ## imgs = df_imgs.iloc[:, 0:img_dims]
        
        obs_imgs = obs.values.astype(np.float32)
        ## obs_imgs = pd.concat([obs, imgs], axis=1).values.astype(np.float32)
        actions = df_obs.iloc[:, obs_dims:(obs_dims+action_dim)].values.astype(np.float32)
        
        dataloader = create_dataloader(obs_imgs, actions, batch_size=batch_size)
        n_dataloader = len(dataloader)
        del obs, obs_imgs, actions, df_obs
        ## del obs, imgs, obs_imgs, actions, df_obs, df_imgs

        for batch, (X, y) in enumerate(dataloader):

            y_hat = network(X)
            loss = criterion(y_hat, y)
            print(
                f"File: {file_no}/{total_file}, Batch: {batch+1}/{n_dataloader}, Loss: {loss}",
                end="\r"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Saving the model
            if i % 50 == 0:
                # saving the mode
                if TRAINING_MODE == "pick":
                    torch.save(network.state_dict(), "model_pick.pt") 
                elif TRAINING_MODE == "place":
                    torch.save(network.state_dict(), "model_place.pt") 
                else:
                    torch.save(network.state_dict(), "model.pt")      














