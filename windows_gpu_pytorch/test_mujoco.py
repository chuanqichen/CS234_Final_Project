import mujoco
from mujoco import viewer

with open('humanoid\\humanoid.xml', 'r') as f:
    XML = f.read()

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)

viewer.launch(model, data)