import numpy as np
import torch
from environment import Environment
from network_utils import MultiLayerCNN, np2torch, device

np.random.seed(1001)

# Create environment instance
env_generator = Environment()
env = env_generator.create_env(fixed_placement=True)

# Load model
network = MultiLayerCNN(
        obs_input_size=32,
        img_input_height=64,
        img_input_width=64,
        output_size=7
).to(device=device)
network.load_state_dict(torch.load("model_pick.pt", map_location=torch.device(device)))

# Run
episode_no = 0
n_success = 0
max_nsteps = 570
while True:
    # reset the environment
    obs = env.reset()
    episode_no += 1
    
    # Store trajectory
    actions = []
    observations = [obs]
    rewards = []
    
    # Iterate over steps
    for i in range(max_nsteps):
        print(f"Episode: {episode_no}, Step: {i+1}/{max_nsteps}", end="\r")
        # Collect observations from previous step
        obs_flat = np.concatenate([
                obs["robot0_joint_pos_cos"],
                obs["robot0_joint_pos_sin"],
                obs["robot0_joint_vel"],
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
                obs["robot0_gripper_qvel"]
        ])
        img_flat = obs["agentview_image"].flatten()
        obs_img = np.array([np.concatenate([obs_flat, img_flat])]).astype(np.float32)
        # Act according to model
        action = network(np2torch(obs_img)).cpu().detach().numpy().flatten()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        env.render()
        observations.append(obs)
        rewards.append(reward)

    n_success += (sum(rewards) >= 1.0)
    print(f"----- Episode {episode_no}, success {n_success}/{episode_no} -----")
