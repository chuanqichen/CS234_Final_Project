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

# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

from environment import Environment
from robosuite.wrappers import GymWrapper

np.random.seed(9)

operation = input("Operation ('train', 'test', or 'both'): ")
dirpath = input("Enter dirpath: ")
filename = input("Enter filename: ") 


#
#  TRAINING
#
if operation == 'train' or operation == 'both':
    # Create environment instance
    env_generator = Environment()
    env = env_generator.create_env(fixed_placement=False, use_object_obs=True,
            use_camera_obs=False, ignore_done=False)
    wrapped_env = GymWrapper(env)
    ## wrapped_env = Monitor(wrapped_env)
            ## # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])
            # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)
            # Needed for improving training when using MuJoCo envs?

    # Instantiate the agent
    ## model = PPO("MlpPolicy", wrapped_env, verbose=1)   
    model = TD3("MlpPolicy", wrapped_env, learning_rate=0.0001, verbose=1, 
            buffer_size=2048, learning_starts=100, gamma=0.99)   
            #TODO CUSTOMIZE MODEL ARCHITECTURE 
            #TODO Prevent from using block observations?
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
            use_object_obs=True, use_camera_obs=False, ignore_done=False)
    wrapped_test_env = GymWrapper(test_env)
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
    model = TD3.load(os.path.join(dirpath, filename), env=wrapped_test_env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    ## mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
            ## n_eval_episodes=10)

    # Run trained agent
    obs = wrapped_test_env.reset()
    for i in range(100):
        print(f"Step {i}", end="\r")
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = wrapped_test_env.step(action)
        wrapped_test_env.render()
        if True in dones:
            obs = wrapped_test_env.reset()
    wrapped_test_env.close()


