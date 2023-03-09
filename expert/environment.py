"""Script for Environment"""
from stack2 import Stack2, placement_initializer2
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

class Environment:
    def __init__(self):
        self.controller_config = load_controller_config(default_controller="OSC_POSE")
        self.controller_config["control_delta"] = False  # Use absolute position
        self.controller_config["kp"] = 15  
        self.controller_config["damping_ratio"] = 2 
        self.controller_config["uncouple_pos_ori"] = False

        self.placement_sampler = UniformRandomSampler(
                name="ObjectSampler",
                x_range=[-0.35,0.35],
                y_range=[-0.35,0.35],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array((0, 0, 0.8)),
                z_offset=0.01
        )

    def create_env(self):
        # create environment instance
        env = suite.make(
            env_name="Stack2", # try with other tasks like "Stack" and "Door"
            robots="Sawyer",  # try with other robots like "Panda" and "Jaco"
            gripper_types="default",
            controller_configs=self.controller_config,
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
            placement_initializer=self.placement_sampler
        )
        return env
