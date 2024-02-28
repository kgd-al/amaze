import pprint

from amaze.simu import *

print("="*80)
maze = Maze.from_string("M16_10x10_U")
robot = Robot.BuildData.from_string("DD")
simulation = Simulation(maze, robot)

print(f"Maze stats: {pprint.pformat(maze.stats())}")

print("="*80)
controller = load("examples/agents/unicursive_tabular.zip")
print(f"Agent infos: {pprint.pformat(controller.infos)}")

simulation.run(controller)

print("="*80)
print("Target reached:", simulation.success())
print("Raw reward:", simulation.cumulative_reward())
print("Normalized reward:", simulation.normalized_reward())
print(f"Details: {pprint.pformat(simulation.infos())}")
