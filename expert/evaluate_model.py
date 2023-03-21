import numpy as np
import torch
from environment import Environment
from network_utils import NetworkBC, NetworkBC2, MultiLayerCNN, np2torch
from config import device, device_name

np.random.seed(3001)

# Create environment instance
env_generator = Environment()
env = env_generator.create_env(fixed_placement=False)

# Load model
## network = MultiLayerCNN(
        ## obs_input_size=32,
        ## img_input_height=64,
        ## img_input_width=64,
        ## output_size=4
## ).to(device=device)
pick_network = NetworkBC(
    obs_input_size=5,
    output_size=env.action_dim
).to(device=device)
pick_network.load_state_dict(torch.load("model_pick.pt", 
        map_location=torch.device(device)))
place_network = NetworkBC2(
    obs_input_size=5,
    output_size=env.action_dim
).to(device=device)
place_network.load_state_dict(torch.load("model_place.pt",
        map_location=torch.device(device)))

# Run
episode_no = 0
n_success = 0
max_nsteps = 570
for i in range(3000):
    print(f"Episode {episode_no}, success = {n_success}/{episode_no}")
    # reset the environment
    obs = env.reset()
    episode_no += 1
    
    # Store trajectory
    actions = []
    observations = [obs]
    rewards = []
    max_persistence = 70
    task_persistence = -max_persistence
    # Iterate over steps
    for i in range(max_nsteps):
        print(f"{episode_no}.{i}; ", end="")
        if task_persistence <= 0:  # Pick
        ## if np.sign(obs["robot0_gripper_qpos"][0]) > 0:  # Pick
            print("Picking; ", end="")
            obs_flat = np.concatenate([
                    obs["robot0_gripper_qpos"],
                    obs["gripper_to_cubeA"],
            ]).astype(np.float32)
            action = pick_network(np2torch(obs_flat)).cpu().detach().numpy().flatten()
        else:
            print("Placing; ", end="")
            obs_flat = np.concatenate([
                    obs["robot0_gripper_qpos"],
                    obs["gripper_to_cubeB"],
            ]).astype(np.float32) 
            action = place_network(np2torch(obs_flat)).cpu().detach().numpy().flatten()
        print(f"grip_action={action[-1]:.4f} ", end="")
        task_persistence = np.clip(task_persistence + int(np.sign(action[-1])), -max_persistence, max_persistence)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        env.render()
        observations.append(obs)
        rewards.append(reward)
        print(f"reward={reward:.4f}", end="")
        print("",end="\r")
    if max(rewards) >= 1:
        n_success += 1
    print(f"----- Episode {episode_no}, success {n_success}/{episode_no}, reward={max(rewards)} -----")
