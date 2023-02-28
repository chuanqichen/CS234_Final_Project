import numpy as np
import robosuite as suite

# create environment instance

class Environment:
    def __init__(self, render=False):
        self.env = suite.make(
            env_name="Stack", # try with other tasks like "Stack" and "Door"
            robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=render,  # can set to false for training
            render_camera="frontview",
            has_offscreen_renderer=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=84,
            camera_widths=84
        )