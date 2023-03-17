import numpy as np
import torch
from environment import Environment
from network_utils import NetworkBC, MultiLayerCNN, np2torch
from config import device, device_name

np.random.seed(1001)

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
network = NetworkBC(
    obs_input_size=3,
    output_size=env.action_dim
).to(device=device)
network.load_state_dict(torch.load("model_pick.pt", 
        map_location=torch.device(device)))

# Run
episode_no = 0
n_success = 0
max_nsteps = 570
for i in range(3):
    # reset the environment
    obs = env.reset()
    episode_no += 1
    
    # Store trajectory
    actions = []
    observations = [obs]
    rewards = []
    
    # Iterate over steps
    for i in range(max_nsteps):
        # Collect observations from previous step
        obs_flat = np.concatenate([
                obs["gripper_to_cubeA"],
        ])
        ## img_flat = obs["agentview_image"].flatten()
        obs_img = obs_flat.astype(np.float32)
        ## obs_img = np.array([
                ## np.concatenate([obs_flat, img_flat])]).astype(np.float32)
        # Act according to model
        action = network(np2torch(obs_img)).cpu().detach().numpy().flatten()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        env.render()
        observations.append(obs)
        rewards.append(reward)
        print(f"Episode {episode_no}, Step {i+1}/{max_nsteps}, Reward {reward}",
                end="\r")

    n_success += (max(rewards) >= 1.0)
    print(f"----- Episode {episode_no}, success {n_success}/{episode_no} -----")
