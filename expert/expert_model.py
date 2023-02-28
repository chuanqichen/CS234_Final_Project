import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T

np.random.seed(10)  # 5 is good

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
    ## use_object_obs=True,
    ## use_camera_obs=True,
    ## camera_heights=84,
    ## camera_widths=84
)
scene_no = 0
while True:
    print(f"------ Scene {scene_no} ------")
    scene_no += 1
    # reset the environment
    env.reset()
    
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

    for i in range(np.sum(durations)):
        action = waypoints[int(np.sum(i > np.cumsum(durations)))]
        obs, reward, done, info = env.step(action)  # move towards waypoint
        env.render()  # render on display
        print("step:", i, " reward:", reward)
        ## print("cubeA_pos = np.array(", obs["cubeA_pos"])
        ## print("cubeA_quat = np.array(", obs["cubeA_quat"])
        ## print("cubeB_pos = np.array(", obs["cubeB_pos"])
        ## print("cubeB_quat = np.array(", obs["cubeB_quat"])





