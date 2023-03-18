import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler
from stack2 import Stack2, placement_initializer2
from environment import Environment, CustomWrapper

def goto_subtask(env, target_subtask=2, train=True, log=False):
    scene_no = 0
    n_success = 0
    for i in range(1):
        print(f"Scene {scene_no}, success = {n_success}/{scene_no}")
        scene_no += 1
        # reset the environment
        env.reset()
        
        # Target poses, absolute (x, y, z, gripper)
        home = np.array([0.0, 0.0, 1.1, -1])
        
        # Initial observation to get block poses
        obs, reward, done, info = env.step(np.zeros(4))
        robot0_eef_pos = obs["robot0_eef_pos"]
        cubeA_pos = obs["cubeA_pos"] 
        cubeB_pos = obs["cubeB_pos"]
        # Red cube (A)
        A_ungrasped = np.concatenate([
                cubeA_pos + [0, 0, -0.005], 
                [-1]
        ])
        A_primed_ungrasped = A_ungrasped + [0, 0, .1, 0]
        A_grasped = np.concatenate([A_ungrasped[:3], [1]])
        A_primed_grasped = np.concatenate([A_primed_ungrasped[:3], [1]])
        # Green cube (B)
        B_ungrasped = np.concatenate([
                cubeB_pos + [0, 0, 0.04], 
                [-1]
        ])
        B_primed_ungrasped = B_ungrasped + [0, 0, .1, 0]
        B_grasped = np.concatenate([B_ungrasped[:3], [1]])
        B_primed_grasped = np.concatenate([B_primed_ungrasped[:3], [1]])
        waypoints = [
                home,                  # 0
                A_primed_ungrasped,    # 1
                A_ungrasped,           # 2
                A_grasped,             # 3
                A_primed_grasped,      # 4
                B_primed_grasped,      # 5
                B_grasped,             # 6
                B_ungrasped,           # 7
                B_primed_ungrasped,    # 8
                home                   # 9
        ]
        # Subtasks
        # 0 = initial home
        # 1 = prime pick 
        # 2 = descend pick
        # 3 = grasp
        # 4 = ascent pick
        # 5 = prime place
        # 6 = descent place
        # 7 = release
        # 8 = ascend place
        # 9 = final home
        subtask = 0
        # Target durations, in number of steps
        durations = [75, 100, 50, 10, 50, 100, 50, 10, 50, 75]
        reward = 0
        for i in range(np.sum(durations)):
            if subtask > target_subtask: 
                break
            subtask = int(np.sum(i > np.cumsum(durations)))
            action = waypoints[subtask] - np.concatenate([robot0_eef_pos, [0]])
            obs, reward, done, info = env.step(action)  # move towards waypoint
            if i % 1 == 0 and not train:
                env.render()  # render on display
            robot0_eef_pos = obs["robot0_eef_pos"]
            if log:
                print("step:", i, "subtask:", subtask, "reward:", reward, end="\r")
        if reward >= 1:
            n_success += 1
        if log:
            print()

if __name__ == '__main__':
    train_env, env = Environment.make_env(train=True)
    while True:
        goto_subtask(env, target_subtask=2, train=False, log=True)
 