import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = False  # Use absolute position

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

action_dim = 6  # 6-dof cartesian pose
gripper_dim = env.robots[0].gripper.dof  # 1


# reset the environment
env.reset()

# Target poses, absolute (x, y, z, rx, ry, rz, gripper)
# (rx, ry, rz are the axis of rotation whose magnitude is the angle of rotation)
actions = [
    np.array([0.0, 0.0, 1.1, 0., np.pi, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 0.8, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.1, 0.1, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.0, 0.0, 1.1, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
    np.array([0.2, 0.25, 0.9, -np.pi*np.sqrt(2)/2, np.pi*np.sqrt(2)/2, 0., 0.]),
]

# Target durations, in number of steps
durations = [75, 50] + (len(actions)-2) * [25]

for i in range(np.sum(durations)-1):
    print("step", i)
    action = actions[int(np.sum(i // np.cumsum(durations)))]
    obs, reward, done, info = env.step(action)  # take action 
    print("reward:", reward)
    env.render()  # render on display





