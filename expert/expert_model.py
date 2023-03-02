import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler


## from pyquaternion import Quaternion
## # Distance thresholds to move to next state
## HORIZ_THRESH = 0.05
## VERT_THRESH = 0.02
## ANG_THRESH = 0.06
## def close_enough(eef_pos, eef_quat, waypoint):
    ## """ Returns if eef is close enough to waypoint to transition to next task.
    ## Args:
       ## eef_pos (np.array shape(3,)): End effector position
       ## eef_quat (np.array shape(4,)): End effector quaternion (x,y,z,w)
       ## waypoint [x, y, z, ax, ay, az, gripper]: OSC controller target waypoint
    ## """
    ## horiz_close = np.linalg.norm(eef_pos[0:2] - waypoint[0:2]) < HORIZ_THRESH
    ## vert_close = np.abs(eef_pos[2] - waypoint[2]) < VERT_THRESH
    ## ang_close = ang_dist(ee_quat, T.axisangle2quat(waypoint[3:6])) < ANG_TRESH


## def ang_dist(q1, q2):
    ## """ Gives the angular distance between two quaternions (x,y,z,w). """
    ## Q1 = Quaternion(q1[3], q1[0], q1[1], q1[2])  # Convert to w,x,y,z form
    ## Q2 = Quaternion(q2[3], q2[0], q2[1], q2[2])  # Convert to w,x,y,z form
    ## return Quaternion.distance(Q1, Q2)

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
scene_no = 0
while True:
    print(f"------ Scene {scene_no} ------")
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

    for i in range(np.sum(durations)):
        subtask = int(np.sum(i > np.cumsum(durations)))
        action = waypoints[subtask]
        action += np.random.uniform(*variations[subtask])
        obs, reward, done, info = env.step(action)  # move towards waypoint
        ## if i % 1 == 0:
            ## env.render()  # render on display
        print("step:", i, "subtask:", subtask, "reward:", reward)
        ## print("cubeA_pos = np.array(", obs["cubeA_pos"])
        ## print("cubeA_quat = np.array(", obs["cubeA_quat"])
        ## print("cubeB_pos = np.array(", obs["cubeB_pos"])
        ## print("cubeB_quat = np.array(", obs["cubeB_quat"])





