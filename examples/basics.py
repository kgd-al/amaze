import pprint
from amaze.simu import *

print("="*80)
maze = Maze.generate_from_string("M16_10x10_U")
print(f"Maze stats: {pprint.pformat(maze.stats())}")

robot = Robot.BuildData(
    InputType.DISCRETE,
    OutputType.DISCRETE
)

simulation = Simulation(maze, robot)

print("="*80)
controller = load("examples/agents/unicursive_tabular.zip")
print(f"Agent infos: {pprint.pformat(controller.infos)}")

action = controller(simulation.observations)
while not simulation.done():
    simulation.step(action)
    action = controller(simulation.observations)

print("="*80)
print("Simulation done.")
print("Raw reward:", simulation.cumulative_reward())
print("Normalized reward:", simulation.normalized_reward())
print(f"Details: {pprint.pformat(simulation.infos())}")
