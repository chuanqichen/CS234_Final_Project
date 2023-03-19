"""Script for Environment"""
from stack2 import Stack2, placement_initializer2
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class Environment:
    def __init__(self):
        self.controller_config = load_controller_config(
                default_controller="OSC_POSITION")
        self.controller_config["control_delta"] = True  # Use relative position
        self.controller_config["kp"] = 1500
        self.controller_config["damping_ratio"] = 1

        self.placement_sampler = UniformRandomSampler(
                name="ObjectSampler",
                x_range=[-0.15,0.15],
                y_range=[-0.15,0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array((0, 0, 0.8)),
                z_offset=0.01
        )

    def create_env(self, fixed_placement=False, use_object_obs=True,
            use_camera_obs=True, ignore_done=True):
        # create environment instance
        env = suite.make(
            env_name="Stack2", # try with other tasks like "Stack" and "Door"
            robots="Sawyer",  # try with other robots like "Panda" and "Jaco"
            gripper_types="default",
            controller_configs=self.controller_config,
            has_renderer=True,
            render_camera="frontview",
            has_offscreen_renderer=use_camera_obs,
            control_freq=20,
            horizon=600,
            ignore_done=ignore_done,
            use_object_obs=use_object_obs,
            use_camera_obs=use_camera_obs,
            camera_heights=64,
            camera_widths=64,
            reward_shaping=True,
            placement_initializer=self.placement_sampler if not fixed_placement else None
            #placement_initializer=None  # fixed bricks location, read from bricks.json if this is None
        )
        return env

    def make_sb_env(fixed_placement=True,
                use_object_obs=True, use_camera_obs=True, ignore_done=False, train=False):
        # Create environment instance
        env_generator = Environment()
        env = env_generator.create_env(fixed_placement,
                use_object_obs, use_camera_obs, ignore_done)
        if not use_camera_obs:
            wrapped_env = CustomWrapperWithoutImage(env)
        else:
            wrapped_env = CustomWrapper(env)
        #wrapped_env = Monitor(wrapped_env)
                ## # Needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda : wrapped_env])
                # Needed for all environments (e.g. used for mulit-processing)
        wrapped_env = VecNormalize(wrapped_env)
                # Needed for improving training when using MuJoCo envs?
        wrapped_env.training = train
        return wrapped_env, env


class CustomWrapper(GymWrapper):
    def __init__(self, env, keys=None):
        super().__init__(env, keys)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observations flattened into a 1d array
        """
        obs_vector = np.concatenate([
            v for k, v in obs_dict.items() if k in [
                "robot0_joint_pos_cos",
                "robot0_joint_pos_sin",
                "robot0_joint_vel",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel"
            ]
        ])
        obs_img = obs_dict["agentview_image"]
        obs_vector = np.concatenate([obs_vector, obs_img.flatten()])
        return obs_vector

class CustomWrapperWithoutImage(GymWrapper):
    def __init__(self, env, keys=None):
        super().__init__(env, keys)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observations flattened into a 1d array
        """
        obs_vector = np.concatenate([v for k, v in obs_dict.items()])
        '''
        To modify obs_vector to only includes our choice of observations:
        obs_vector = np.concatenate([
            v for k, v in obs_dict.items() if k in [
                "obs_1",
                "obs_2",
                ...
            ]
        ])
        '''
        return obs_vector
    

if __name__ == "__main__":
    env_generator = Environment()
    env = env_generator.create_env()
    env = CustomWrapper(env)
    obs = env.reset()
    print(obs.shape)
    print(type(obs))