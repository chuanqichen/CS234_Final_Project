import os
import gym
import numpy as np

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from network_utils import MultiLayerCNNFeaturesExtractor
from config import device, device_name

# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

from environment import Environment, CustomWrapper

np.random.seed(90)

operation = input("Operation ('train', 'test', or 'both'): ")
dirpath = input("Enter dirpath: ")
filename = input("Enter filename: ") 

from OpenGL import error as gl_error
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=gl_error.Error)

#
#  TRAINING
#
if operation == 'train' or operation == 'both':
    # Create environment instance
    env_generator = Environment()
    env = env_generator.create_env(fixed_placement=False, use_object_obs=True,
                                   use_camera_obs=True, ignore_done=False)
    obs = env.reset()
    obs_vector = np.concatenate([
        v for k, v in obs.items() if k in [
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel"
        ]
    ])
    obs_input_size = len(obs_vector)
    obs_img = obs["agentview_image"]
    obs_img_height = obs_img.shape[0]
    obs_img_width = obs_img.shape[1]
    action_dim = env.action_dim


    wrapped_env = CustomWrapper(env)
    wrapped_env = DummyVecEnv([lambda : wrapped_env])
    wrapped_env = VecNormalize(wrapped_env)

    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        wrapped_env,
        verbose=1,
        learning_rate=0.0001,
        gamma=0.98,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128],
                vf=[256, 128],
            ),
            features_extractor_class=MultiLayerCNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                obs_input_size=obs_input_size,
                img_input_width=obs_img_width,
                img_input_height=obs_img_height,
                features_dim=256
            )
        ),
	device=device_name
    )
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(1E5), progress_bar=True, log_interval=10)
    # Save the agent
    model.save(os.path.join(dirpath, filename))
    del model  # delete trained model to demonstrate loading

#
#  TESTING
#
if operation == 'test' or operation == 'both':
    # Create environment instance
    test_env_generator = Environment()
    test_env = test_env_generator.create_env(fixed_placement=False,
            use_object_obs=True, use_camera_obs=True, ignore_done=False)
    wrapped_test_env = CustomWrapper(test_env)
    ## wrapped_env = Monitor(wrapped_env)
            ## # Needed for extracting eprewmean and eplenmean
    wrapped_test_env = DummyVecEnv([lambda : wrapped_test_env])
            # Needed for all environments (e.g. used for mulit-processing)
    wrapped_test_env = VecNormalize(wrapped_test_env)
            # Needed for improving training when using MuJoCo envs?
    wrapped_test_env.training = False

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    ## model = PPO.load(os.path.join(dirpath, filename), env=wrapped_test_env)
    model = PPO.load(os.path.join(dirpath, filename), env=wrapped_test_env, device=device_name)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    ## mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
            ## n_eval_episodes=10)

    # Run trained agent
    obs = wrapped_test_env.reset()
    for i in range(10000):
        print(f"Step {i}", end="\r")
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = wrapped_test_env.step(action)
        wrapped_test_env.render()
        if True in dones:
            obs = wrapped_test_env.reset()
    wrapped_test_env.close()

