"""检查 SurfaceGripper API"""
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})

from isaacsim.robot.manipulators.grippers import SurfaceGripper
import inspect

print('=== SurfaceGripper signature ===')
print(inspect.signature(SurfaceGripper.__init__))
print()
print('=== SurfaceGripper docstring ===')
print(SurfaceGripper.__init__.__doc__)

app.close()
