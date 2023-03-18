import os
os.environ['DISPLAY'] = ':0.0'
import gym
import numpy as np

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement, StopTrainingOnMaxEpisodes
from network_utils import MultiLayerCNNFeaturesExtractor
from config import device, device_name
from environment import Environment, CustomWrapper
from OpenGL import error as gl_error
import warnings
from motion import goto_subtask

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=gl_error.Error)

np.random.seed(9)

operation = input("Operation ('train', 'test', or 'both'): ")
dirpath = input("Enter dirpath: ")
filename = input("Enter filename: ")

#
#  TRAINING
#
if operation == 'train' or operation == 'both':
    # Create environment instance
    pi_arch = input("Enter arch for pi (separated with comma): ")
    pi_arch = [int(k) for k in pi_arch.split(",")]
    qf_arch = input("Enter arch for qf (separated with comma): ")
    qf_arch = [int(k) for k in qf_arch.split(",")]
    timesteps = int(input("Enter timesteps: "))
    train_env, env = Environment.make_sb_env(fixed_placement=True,
                use_object_obs=True, use_camera_obs=False, ignore_done=False, train=True)
    print(f"\nUsing pi:{pi_arch}")
    print(f"Using qf:{qf_arch}")
    print(f"Timesteps: {timesteps}")
    # Instantiate the env and the agent for the stable baseline3
    #goto_subtask(env, target_subtask=2, train=False)

    model = TD3(
        "MlpPolicy",
        train_env,
        verbose=1,
        buffer_size=4096,
        learning_rate=0.001,
        learning_starts=100,
        gamma=0.98,
        policy_kwargs=dict(
            net_arch=dict(
                pi=pi_arch,
                qf=qf_arch,
            )
        ),
        device=device_name,
        tensorboard_log="./logs/"
    )
    # Train the agent and display a progress bar
    # Save a checkpoint every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(dirpath,"logs"),
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Stops training when the model reaches the maximum number of episodes
    #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=5, verbose=1)
    eval_env, _ = Environment.make_sb_env(fixed_placement=True,
                use_object_obs=True, use_camera_obs=True, ignore_done=False, train=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(dirpath, "best_model"), callback_after_eval=stop_train_callback,
                             log_path=os.path.join(dirpath, "best_model"), eval_freq=3000,
                             deterministic=True, render=False)

    class CustomCallback(BaseCallback):
        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass
        
        def _on_training_start(self) -> None:
            pass

        def _on_rollout_start(self) -> None:
            goto_subtask(env, target_subtask=2, train=True)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            pass

        def _on_training_end(self) -> None:
            pass

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, CustomCallback()])
 
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        log_interval=10,
        tb_log_name="td3_run",
        reset_num_timesteps=False
    )
    # Save the agent
    model.save(os.path.join(dirpath, filename))
    del model  # delete trained model to demonstrate loading

#
#  TESTING
#
if operation == 'test' or operation == 'both':
    wrapped_test_env, env =   Environment.make_sb_env(fixed_placement=True,
                use_object_obs=True, use_camera_obs=True, ignore_done=False, train=False)
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    ## model = PPO.load(os.path.join(dirpath, filename), env=wrapped_test_env)
    model = TD3.load(os.path.join(dirpath, filename), env=wrapped_test_env, device=device_name)
    
    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    ## mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
            ## n_eval_episodes=10)

    # Run trained agent
    obs = wrapped_test_env.reset()
    goto_subtask(env, target_subtask=2, train=False)
    for i in range(10000):
        print(f"Step {i}", end="\r")
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = wrapped_test_env.step(action)
        wrapped_test_env.render()
        if True in dones:
            obs = wrapped_test_env.reset()
            goto_subtask(env, target_subtask=2, train=False)

    wrapped_test_env.close()

