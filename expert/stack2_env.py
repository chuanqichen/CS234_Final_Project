
from robosuite.environments.base import register_env
from stack2 import Stack2, placement_initializer2
import robosuite as suite
import numpy as np

register_env(Stack2)

# create environment instance
env = suite.make(
    env_name="Stack2", # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,  # can set to false for training
    #controller_configs=controller_config,
    render_camera="frontview",
    has_offscreen_renderer=True,
    use_object_obs=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    placement_initializer=placement_initializer2
)

# reset the environment
env.reset()

for i in range(100):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
