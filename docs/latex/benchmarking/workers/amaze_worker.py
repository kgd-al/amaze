import math
import itertools
import pprint

from amaze import Maze, Robot, Simulation
from amaze.simu.controllers import RandomController

from common import Progress, STEPS


def _evaluate(maze, robot):
    #help(Simulation)
    simulation = Simulation(Maze.generate(maze), robot)
    simulation.reset()

    controller = RandomController(robot)
    for _ in range(STEPS):
        action = controller(simulation.observations)
        simulation.step(action)

        if simulation.done():
            simulation.reset()

    return True


def process(df):
    visions = [11, 15, 21, 37]
    sizes = [5, 10, 20, 50]
    p_signs = [0, .1, .5, 1]
    amaze_fields = [
        *itertools.product("D", visions[:1], sizes, p_signs, p_signs),
        *itertools.product("HC", visions, sizes, p_signs, p_signs[:1])
    ]

    with Progress(df, "AMaze") as progress:
        progress.add_task(len(amaze_fields))

        for robot, vision, size, p_lure, p_trap in amaze_fields:
            mbd = Maze.BuildData.from_string(f"M4_{size}x{size}_C1_l{p_lure}_L.25_t{p_trap}_T.5")
            rbd = Robot.BuildData.from_string(f"{robot}{vision}")
            name = f"{mbd.to_string()}-{rbd.to_string()}"
            progress.evaluate("AMaze", name, _evaluate, label=robot, maze=mbd, robot=rbd)

        progress.close()

    return progress.errors
