from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda

from robosuite.models.grippers import gripper_factory


world = MujocoWorldBase()
mujoco_robot = Panda()


gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)