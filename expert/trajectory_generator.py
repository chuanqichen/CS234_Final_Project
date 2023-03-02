import os
import pandas as pd
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler

controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = False  # Use absolute position
controller_config["kp"] = 15  
controller_config["damping_ratio"] = 2 
controller_config["uncouple_pos_ori"] = False  

# create environment instance
env = suite.make(
    env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Panda" and "Jaco"
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    render_camera="frontview",
    has_offscreen_renderer=True,
    control_freq=20,
    horizon=200,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=True,
    camera_heights=64,
    camera_widths=64,
    placement_initializer=UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.35,0.35],
        y_range=[-0.35,0.35],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01
    )
)

df_obs = pd.DataFrame()
df_imgs = pd.DataFrame()
dirpath = input("Enter dirpath: ")
filepath = input("Enter filename: ")
save_every = int(input("Save every: "))
num_iterations = int(input("Number of episodes: "))

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

scene_no = 0
while True:
    # print(f"------ Scene {scene_no} ------", end="\r")
    scene_no += 1
    # reset the environment
    obs = env.reset()
    
    # Initial observation to get block poses
    obs, reward, done, info = env.step(np.zeros(7))
    cubeA_pos = obs["cubeA_pos"] ##np.array([0.02236725, 0.07769993, 0.8197210])
    cubeA_quat = obs["cubeA_quat"] ##np.array([0, 0, 7.27025e-01, 6.866109e-01])
    cubeB_pos = obs["cubeB_pos"] ##np.array([0.0676399, -0.0796457, 0.82472101])
    cubeB_quat = obs["cubeB_quat"] ##np.array([0., 0., 9.95409e-01, 9.5708e-02])

    # Target poses, absolute (x, y, z, rx, ry, rz, gripper)
    # (rx, ry, rz) is the axis of rotation with magnitude the angle of rotation)
    home_axisangle = np.array([np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0.])
    home_quat = T.axisangle2quat(home_axisangle)
    home = np.array([0.0, 0.0, 1.1, *home_axisangle, -1])

    # Red cube (A)
    A_ungrasped = np.concatenate([
            cubeA_pos, 
            T.quat2axisangle(T.quat_multiply(cubeA_quat, [1,0,0,0])), 
            [-1]
    ])
    A_primed_ungrasped = A_ungrasped + [0, 0, .1, 0, 0, 0, 0]
    A_grasped = np.concatenate([A_ungrasped[:6], [1]])
    A_primed_grasped = np.concatenate([A_primed_ungrasped[:6], [1]])
    # Green cube (B)
    B_ungrasped = np.concatenate([
            cubeB_pos + [0, 0, 0.03], 
            T.quat2axisangle(T.quat_multiply(cubeB_quat, [1,0,0,0])), 
            [-1]
    ])
    B_primed_ungrasped = B_ungrasped + [0, 0, .1, 0, 0, 0, 0]
    B_grasped = np.concatenate([B_ungrasped[:6], [1]])
    B_primed_grasped = np.concatenate([B_primed_ungrasped[:6], [1]])
    waypoints = [
            home, 
            A_primed_ungrasped,
            A_ungrasped,
            A_grasped,
            A_primed_grasped,
            B_primed_grasped,
            B_grasped,
            B_ungrasped,
            B_primed_ungrasped,
            home
    ]

    # Target durations, in number of steps
    durations = [75, 105, 50, 20, 50, 105, 50, 20, 50, 75]

    observations = []
    imgs = []
    imgs.append(obs["agentview_image"])
    obs.pop("agentview_image", "")
    observations.append(obs)
    actions = []
    rewards = []

    sum_durations = np.sum(durations)
    cumsum_durations = np.cumsum(durations)
    print()
    for i in range(sum_durations):
        print(f"Episode: {scene_no}, Iteration: {i+1}/{sum_durations}", end="\r")
        action = waypoints[int(np.sum(i > cumsum_durations))]
        obs, reward, done, info = env.step(action)  # move towards waypoint
        actions.append(action)
        rewards.append(reward)
        if i == (sum_durations - 1):
            break
        imgs.append(obs["agentview_image"])
        obs.pop("agentview_image", "")
        observations.append(obs)


    sample_obs = observations[0]
    state_dims = {key: sample_obs[key].shape for key in sample_obs.keys()}

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
    df_imgs_trajectory = pd.DataFrame(imgs_flatten)
    df_imgs_trajectory = pd.concat([df_imgs_trajectory, df_actions], axis=1)
    df_imgs_trajectory["rewards"] = rewards
    df_imgs_trajectory["trajectory_id"] = scene_no

    df_obs = pd.concat([df_obs, df_obs_trajectory], axis=0)
    df_imgs = pd.concat([df_imgs, df_imgs_trajectory], axis=0)

    if scene_no % save_every == 0:
        print(f"Saving to csv")
        indexer = scene_no // save_every
        index_before = ((indexer - 1) * save_every) + 1
        index_after = indexer * save_every
        obs_filename = f"{filepath}_observations_{index_before}_{index_after}.csv"
        imgs_filename = f"{filepath}_imgs_{index_before}_{index_after}.csv"
        df_obs.to_csv(os.path.join(dirpath, obs_filename))
        df_imgs.to_csv(os.path.join(dirpath, imgs_filename))

    if scene_no == num_iterations:
        if scene_no % save_every == 0:
            break
        index_after = scene_no
        indexer = scene_no // save_every
        index_before = (indexer * save_every) + 1
        obs_filename = f"{filepath}_observations_{index_before}_{index_after}.csv"
        imgs_filename = f"{filepath}_imgs_{index_before}_{index_after}.csv"
        df_obs.to_csv(os.path.join(dirpath, obs_filename))
        df_imgs.to_csv(os.path.join(dirpath, imgs_filename))
        break
        
