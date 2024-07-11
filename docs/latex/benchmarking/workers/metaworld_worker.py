import random
import metaworld
from common import evaluate, Progress, STEPS

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


def process(df, detailed):
    return
    amaze_fields = ["DHC"]
    if detailed:
        amaze_fields.extend([[11, 15, 21, 37], [5, 10, 20, 50], [0, .1, .5, 1]])
    else:
        amaze_fields.extend([[11, 15, 21], [5, 10, 20], [0, .5, 1]])

    with Progress("AMaze") as progress:
        task = progress.add_task(math.prod(len(f) for f in amaze_fields))

        for robot, vision, size, p_lure in itertools.product(*amaze_fields):
            namespace = "AMaze"
            if detailed:
                namespace += f"-{robot}"
            mbd = Maze.BuildData.from_string(f"M4_{size}x{size}_C1_l{p_lure}_L.25")
            rbd = Robot.BuildData.from_string(f"{robot}{vision}")
            name = f"{mbd.to_string()}-{rbd.to_string()}"
            evaluate(df, "AMaze", namespace, name, _evaluate, maze=mbd, robot=rbd)
            progress.update(task, name)

        progress.update(task, None)
