import os
import pandas as pd
import numpy as np
import robosuite.utils.transform_utils as T
import json
from environment import Environment


np.random.seed(2000)


# ---------------------------------------------------------------------------- #
#                                 Data Settings                                #
# ---------------------------------------------------------------------------- #

dirpath = input("Enter dirpath: ")
filepath = input("Enter filename: ")
save_every = int(input("Save every: "))
num_iterations = int(input("Number of episodes: "))
fixed_placement_input = input("Fixed placement y/n: ")
fixed_placement = "y" in fixed_placement_input or "Y" in fixed_placement_input

if not os.path.exists(dirpath):
    os.makedirs(dirpath)


# ---------------------------------------------------------------------------- #
#                                  Environment                                 #
# ---------------------------------------------------------------------------- #

# create environment instance
env_generator = Environment()
env = env_generator.create_env(
    fixed_placement=fixed_placement, use_camera_obs=True, use_object_obs=True
)



# ---------------------------------------------------------------------------- #
#                                      Run                                     #
# ---------------------------------------------------------------------------- #

df_obs = pd.DataFrame()
df_imgs = pd.DataFrame()
scene_no = 0
while True:
    # print(f"------ Scene {scene_no} ------", end="\r")
    scene_no += 1
    # reset the environment
    env.reset()
    
    # Target poses, absolute (x, y, z, gripper)
    home = np.array([0.0, 0.0, 1.1, -1])
    
    # Initial observation to get block poses
    obs, reward, done, info = env.step(np.zeros(4))
    robot0_eef_pos = obs["robot0_eef_pos"]
    cubeA_pos = obs["cubeA_pos"] 
    cubeB_pos = obs["cubeB_pos"]
    # Red cube (A)
    A_ungrasped = np.concatenate([
            cubeA_pos, 
            [-1]
    ])
    A_primed_ungrasped = A_ungrasped + [0, 0, .1, 0]
    A_grasped = np.concatenate([A_ungrasped[:3], [1]])
    A_primed_grasped = np.concatenate([A_primed_ungrasped[:3], [1]])
    # Green cube (B)
    B_ungrasped = np.concatenate([
            cubeB_pos + [0, 0, 0.03], 
            [-1]
    ])
    B_primed_ungrasped = B_ungrasped + [0, 0, .1, 0]
    B_grasped = np.concatenate([B_ungrasped[:3], [1]])
    B_primed_grasped = np.concatenate([B_primed_ungrasped[:3], [1]])
    waypoints = [
            home,                  # 0
            A_primed_ungrasped,    # 1
            A_ungrasped,           # 2
            A_grasped,             # 3
            A_primed_grasped,      # 4
            B_primed_grasped,      # 5
            B_grasped,             # 6
            B_ungrasped,           # 7
            B_primed_ungrasped,    # 8
            home                   # 9
    ]
    noise = np.array([1E-2, 1E-2, 1E-2, 0])
    # Subtasks
    # 0 = initial home
    # 1 = prime pick 
    # 2 = descend pick
    # 3 = grasp
    # 4 = ascent pick
    # 5 = prime place
    # 6 = descent place
    # 7 = release
    # 8 = ascend place
    # 9 = final home
    subtask = 0
    # Target durations, in number of steps
    durations = [75, 100, 50, 10, 50, 100, 50, 10, 50, 75]

    observations = []
    imgs = []
    imgs.append(obs["agentview_image"])
    obs.pop("agentview_image", "")
    observations.append(obs)
    actions = []
    rewards = []
    subtasks = []

    sum_durations = np.sum(durations)
    cumsum_durations = np.cumsum(durations)
    print()
    for i in range(sum_durations):
        print(f"Episode: {scene_no}, Iteration: {i+1}/{sum_durations}", end="\r")
        subtask = int(np.sum(i > cumsum_durations))
        action = waypoints[subtask] - np.concatenate([robot0_eef_pos, [0]])
        action += np.random.uniform(-noise, noise)  # Noise
        obs, reward, done, info = env.step(action)  # move towards waypoint
        ## if i % 1 == 0:
            ## env.render()  # render on display
        robot0_eef_pos = obs["robot0_eef_pos"]
        actions.append(action)
        rewards.append(reward)
        subtasks.append(subtask)
        if i == (sum_durations - 1):
            break
        imgs.append(obs["agentview_image"])
        obs.pop("agentview_image", "")
        observations.append(obs)

    # Remove unsuccessful trajectories
    if sum(rewards) < 1.0:
        print("\nTask unsuccessful, remove trajectory...", end="\r")
    else:
        sample_obs = observations[0]
        state_dims = {key: sample_obs[key].shape[0] for key in sample_obs.keys()}

        if not os.path.exists(os.path.join(dirpath, "obs_dims.json")):
            with open(os.path.join(dirpath, "obs_dims.json"), "w") as f:
                json.dump(state_dims, f)

        sample_imgs = imgs[0]
        state_img_dims = {
            "img": sample_imgs.shape[0] * sample_imgs.shape[1] * sample_imgs.shape[2]
        }
        if not os.path.exists(os.path.join(dirpath, "img_dims.json")):
            with open(os.path.join(dirpath, "img_dims.json"), "w") as f:
                json.dump(state_img_dims, f)


        observations_flatten = [
            np.concatenate([obs[k] for k in state_dims]) for obs in observations
        ]
        imgs_flatten = [img.flatten() for img in imgs]
        df_actions = pd.DataFrame(actions)
        df_actions.columns = [f"a_{i}" for i in range(df_actions.shape[1])]
        df_obs_trajectory = pd.DataFrame(observations_flatten)
        df_obs_trajectory = pd.concat([df_obs_trajectory, df_actions], axis=1)
        df_obs_trajectory["rewards"] = rewards
        df_obs_trajectory["trajectory_id"] = scene_no
        df_obs_trajectory["subtask_id"] = subtasks
        df_imgs_trajectory = pd.DataFrame(imgs_flatten)
        df_imgs_trajectory = pd.concat([df_imgs_trajectory, df_actions], axis=1)
        df_imgs_trajectory["rewards"] = rewards
        df_imgs_trajectory["trajectory_id"] = scene_no
        df_imgs_trajectory["subtask_id"] = subtasks

        df_obs = pd.concat([df_obs, df_obs_trajectory], axis=0)
        df_imgs = pd.concat([df_imgs, df_imgs_trajectory], axis=0)

    if scene_no % save_every == 0:
        print(f"\nSaving to csv")
        indexer = scene_no // save_every
        index_before = ((indexer - 1) * save_every) + 1
        index_after = indexer * save_every
        obs_filename = f"{filepath}_observations_{index_before}_{index_after}.csv"
        imgs_filename = f"{filepath}_imgs_{index_before}_{index_after}.csv"
        df_obs.to_csv(os.path.join(dirpath, obs_filename), index=False)
        df_imgs.to_csv(os.path.join(dirpath, imgs_filename), index=False)
        df_obs = pd.DataFrame()
        df_imgs = pd.DataFrame()

    if scene_no == num_iterations:
        if scene_no % save_every == 0:
            break
        print(f"\nSaving to csv")
        index_after = scene_no
        indexer = scene_no // save_every
        index_before = (indexer * save_every) + 1
        obs_filename = f"{filepath}_observations_{index_before}_{index_after}.csv"
        imgs_filename = f"{filepath}_imgs_{index_before}_{index_after}.csv"
        df_obs.to_csv(os.path.join(dirpath, obs_filename), index=False)
        df_imgs.to_csv(os.path.join(dirpath, imgs_filename), index=False)
        break
        
