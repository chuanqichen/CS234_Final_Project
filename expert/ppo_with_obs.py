import os
os.environ['DISPLAY'] = ':0.0'
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnNoModelImprovement, StopTrainingOnMaxEpisodes
from network_utils import MultiLayerCNNFeaturesExtractor, SimpleFeaturesExtractor
from config import device, device_name, linear_schedule
from environment import Environment, CustomWrapper
from OpenGL import error as gl_error
import warnings
from motion import goto_subtask
from cli_ppo import parse_args, save_confg
#from pink import PinkNoiseDist, PinkActionNoise

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=gl_error.Error)

np.random.seed(9)

args = parse_args()
save_confg(args)
print(args)

operation = args.operation
dirpath = args.dirpath
filename = args.filename
fixed_placement = args.placement=="fixed"

def train(args):
    #
    #  TRAINING
    #
    if args.operation  == 'train' or args.operation == 'both':
        # Create environment instance
        pi_arch = args.pi
        pi_arch = [int(k) for k in pi_arch.split(",")]
        vf_arch = args.vf
        vf_arch = [int(k) for k in vf_arch.split(",")]
        timesteps = int(args.max_timesteps)
        train_env, env = Environment.make_sb_env(
            controller=args.controller,
            fixed_placement=fixed_placement,
            use_object_obs=True,
            use_camera_obs=False,
            ignore_done=False,
            train=True,
            has_render=False
        )
        print(f"\nUsing pi:{pi_arch}")
        print(f"Using qf:{vf_arch}")
        print(f"Timesteps: {timesteps}")
        max_action = float(train_env.action_space.high[0])
        seq_len = timesteps
        action_dim = train_env.action_space.shape[-1]
        kwargs = {
                # PPO
                #"action_noise": PinkActionNoise(args.action_noise, seq_len, action_dim),
                "batch_size": int(args.batch_size),
                "clip_range":float(args.clip_range),
                "clip_range_vf":float(args.clip_range_vf),
                "gamma": float(args.discount)
        }

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=linear_schedule(float(args.learning_rate)),
            use_sde=True,
            policy_kwargs=dict(
                net_arch=dict(
                    #pi=pi_arch,
                    #qf=qf_arch,
                    pi=[],
                    vf=[],                    
                ),
                features_extractor_class=SimpleFeaturesExtractor,
                features_extractor_kwargs=dict(
                    obs_input_size=8,
                    output_size=env.action_dim)
            ),
            device=device_name,
            tensorboard_log=os.path.join(dirpath, "./logs/"),
            **kwargs
        )
        # Train the agent and display a progress bar
        # Save a checkpoint every 5000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=os.path.join(dirpath,"checkpoint"),
            name_prefix="ppo_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Stops training when the model reaches the maximum number of episodes
        #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=5, verbose=1)
        eval_env, _ = Environment.make_sb_env(fixed_placement=fixed_placement,
                    use_object_obs=True, use_camera_obs=False, ignore_done=False, train=False, has_render=False)
        eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(dirpath, "best_model"), callback_after_eval=stop_train_callback,
                                log_path=os.path.join(dirpath, "best_model"), eval_freq=3000,
                                deterministic=True, render=False)

        class CustomCallback(BaseCallback):
            def _on_training_start(self) -> None:
                pass
            
            def _on_training_start(self) -> None:
                pass

            def _on_rollout_start(self) -> None:
                goto_subtask(env, start_subtask=args.start_subtask, train=True)

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
            tb_log_name="task_" + str(args.start_subtask) + "_"+   args.placement +"_pi_"+  args.pi +"_vf_"+  args.vf,
            reset_num_timesteps=False
        )
        # Save the agent
        model.save(os.path.join(dirpath, filename))
        del model  # delete trained model to demonstrate loading

def test(args):
    #
    #  TESTING
    #
    if args.operation == 'test' or args.operation == 'both':
        wrapped_test_env, env =   Environment.make_sb_env(fixed_placement=fixed_placement,
                    use_object_obs=True, use_camera_obs=False, ignore_done=False, train=False, has_render=True)
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
        goto_subtask(env, start_subtask=args.start_subtask, train=False)
        for i in range(10000):
            print(f"Step {i}", end="\r")
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = wrapped_test_env.step(action)
            wrapped_test_env.render()
            if True in dones:
                obs = wrapped_test_env.reset()
                goto_subtask(env, start_subtask=args.start_subtask, train=False)

        wrapped_test_env.close()


if __name__ == '__main__':
    train(args)
    test(args)