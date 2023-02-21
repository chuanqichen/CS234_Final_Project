import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
    use_camera_obs=True,         # use camera observations
    use_object_obs=False,        # no object feature when training on pixels
    reward_shaping=True          # (optional) using a shaping reward
)

# create an environment for learning on pixels
'''
env2 = suite.make(
    "SawyerLift",
    has_renderer=False,          # no on-screen renderer
    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
    ignore_done=True,            # (optional) never terminates episode
    use_camera_obs=True,         # use camera observations
    camera_height=84,            # set camera height
    camera_width=84,             # set camera width
    camera_name='agentview',     # use "agentview" camera
    use_object_obs=False,        # no object feature when training on pixels
    reward_shaping=True          # (optional) using a shaping reward
)
'''

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
