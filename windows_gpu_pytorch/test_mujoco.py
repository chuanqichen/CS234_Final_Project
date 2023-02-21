import mujoco
from mujoco import viewer
import os

<<<<<<< Updated upstream
with open(os.path.join(os.path.join("model", "humanoid"), "humanoid.xml"), 'r') as f:
=======
<<<<<<< HEAD
with open('model/humanoid/humanoid.xml', 'r') as f:
=======
with open(os.path.join(os.path.join("model", "humanoid"), "humanoid.xml"), 'r') as f:
>>>>>>> 41e55e73f5ccaebc82286c87f6c7faac640cb801
>>>>>>> Stashed changes
    XML = f.read()

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)

viewer.launch(model, data)
