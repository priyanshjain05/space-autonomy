"""M0: is the simulator installed and working?

Drops a cube onto a floor, checks the physics, renders one image.
    python scripts/m0_smoke.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="cube" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0.45 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
while data.time < 1.0:
    mujoco.mj_step(model, data)

z = data.qpos[2]
print(f"MuJoCo {mujoco.__version__}: cube dropped from 1.0 m, resting at z = {z:.3f} m (expected ~0.1)")
assert abs(z - 0.1) < 0.01, "physics looks wrong"

renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data)
out = Path("results/m0")
out.mkdir(parents=True, exist_ok=True)
plt.imsave(out / "smoke.png", renderer.render())
print(f"Rendered frame -> {out / 'smoke.png'}")
