from random import Random
from typing import Tuple

from amaze.simu.controllers.base import BaseController
from amaze.simu.env.maze import Maze


class RandomController(BaseController):
    def __init__(self):
        self.rng = Random()
        self.stack = []
        self.visited = set()
        self.last_pos = None

    def __call__(self, pos, _) -> Tuple[float, float]:
        ci, cj = pos = pos.aligned()

        if self.last_pos and self.last_pos == pos:
            self.stack.pop()  # Cancel last move

        neighbors = []
        for d, (di, dj) in Maze._offsets.items():
            i, j = ci + di, cj + dj
            if ((i, j), d) not in self.visited:
                neighbors.append((d, (i, j), (di, dj)))

        self.last_pos = pos
        if len(neighbors) > 0:
            d_chosen, chosen, delta = self.rng.choice(neighbors)
            self.visited.add((pos, Maze._inverse_dir[d_chosen]))
            self.visited.add((chosen, d_chosen))
            self.stack.append((-delta[0], -delta[1]))

            return delta
        else:
            return self.stack.pop()
