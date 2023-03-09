import os
import pandas as pd
import numpy as np
import robosuite.utils.transform_utils as T
import json
from environment import Environment

# ---------------------------------------------------------------------------- #
#                                   Settings                                   #
# ---------------------------------------------------------------------------- #

def find_cube_rotation(cube_quat, home_quat):
    """ Finds the orientation of the cube most suitable for robot to grasp. """
    cube_quat = T.quat_multiply(cube_quat, [1,0,0,0])
    options = [
            T.quat2mat(cube_quat),
            np.array([[0,-1,0],[1,0,0],[0,0,1]])@T.quat2mat(cube_quat),
            np.array([[-1,0,0],[0,-1,0],[0,0,1]])@T.quat2mat(cube_quat),
            np.array([[0,1,0],[-1,0,0],[0,0,1]])@T.quat2mat(cube_quat),
    ]
    idx = np.argmin([np.linalg.norm(
            T.get_orientation_error(T.mat2quat(options[i]), home_quat)
        ) for i in range(len(options))]
    )
    return T.mat2quat(options[idx])


np.random.seed(1001)

# ---------------------------------------------------------------------------- #
#                                  Environment                                 #
# ---------------------------------------------------------------------------- #

# create environment instance
env_generator = Environment()
env = env_generator.create_env(fixed_placement=True)

# ---------------------------------------------------------------------------- #
#                                 Data Settings                                #
# ---------------------------------------------------------------------------- #

dirpath = input("Enter dirpath: ")
filepath = input("Enter filename: ")
save_every = int(input("Save every: "))
num_iterations = int(input("Number of episodes: "))

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

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
    
    # Target poses, absolute (x, y, z, rx, ry, rz, gripper)
    # (rx, ry, rz) is the axis of rotation with magnitude the angle of rotation)
    home_axisangle = np.array([np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0.])
    home_quat = T.axisangle2quat(home_axisangle)
    home = np.array([0.0, 0.0, 1.1, *home_axisangle, -1])
    
    # Initial observation to get block poses
    obs, reward, done, info = env.step(np.zeros(7))
    cubeA_pos = obs["cubeA_pos"] 
    cubeA_quat = obs["cubeA_quat"]
    cubeB_pos = obs["cubeB_pos"]
    cubeB_quat = obs["cubeB_quat"] 
    pick_quat = find_cube_rotation(cubeA_quat, home_quat)
    place_quat = find_cube_rotation(cubeB_quat, home_quat)
    # Red cube (A)
    A_ungrasped = np.concatenate([
            cubeA_pos, 
            T.quat2axisangle(pick_quat), 
            [-1]
    ])
    A_primed_ungrasped = A_ungrasped + [0, 0, .1, 0, 0, 0, 0]
    A_grasped = np.concatenate([A_ungrasped[:6], [1]])
    A_primed_grasped = np.concatenate([A_primed_ungrasped[:6], [1]])
    # Green cube (B)
    B_ungrasped = np.concatenate([
            cubeB_pos + [0, 0, 0.03], 
            T.quat2axisangle(place_quat), 
            [-1]
    ])
    B_primed_ungrasped = B_ungrasped + [0, 0, .1, 0, 0, 0, 0]
    B_grasped = np.concatenate([B_ungrasped[:6], [1]])
    B_primed_grasped = np.concatenate([B_primed_ungrasped[:6], [1]])
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
    # Permissible random delta variation for each waypoint, [hi, low]
    home_variation = [
            np.array([-0.003, -0.003, -0.003, -0.010, -0.010, -0.010, 0]),
            np.array([ 0.003,  0.003,  0.003,  0.010,  0.010,  0.010, 0])
    ]
    prime_variation = [  # Primed height above cube
            np.array([-0.003, -0.003, -0.001, -0.003, -0.003, -0.003, 0]),
            np.array([ 0.003,  0.003,  0.003,  0.003,  0.003,  0.003, 0])
    ]
    engage_variation = [  # Descended down onto cube
            np.array([-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, 0]),  
            np.array([ 0.001,  0.001,  0.001,  0.001,  0.001,  0.001, 0])
    ]
    variations = [
        home_variation,    # 0
        prime_variation,   # 1
        engage_variation,  # 2
        engage_variation,  # 3
        prime_variation,   # 4
        prime_variation,   # 5
        engage_variation,  # 6
        engage_variation,  # 7
        prime_variation,   # 8
        home_variation     # 9
    ]
    
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
        action = waypoints[subtask]
        obs, reward, done, info = env.step(action)  # move towards waypoint
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
        
