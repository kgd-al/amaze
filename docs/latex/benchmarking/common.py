import os
import math
import time

from rich.progress import Progress as RichProgress

STEPS = 1000
REPLICATES = 10


def line(c='='):
    print(c*os.get_terminal_size()[0])


def evaluate(df, library, family, name, fn, **kwargs):
    if name not in df.index or math.isnan(df.loc[name]["Time"]):
        try:
            start = time.time_ns()
            for _ in range(REPLICATES):
                fn(**kwargs)
            t = (time.time_ns() - start) / (REPLICATES * 10**9)

            if not math.isnan(t):
                df.loc[name] = [library, family, t]

        except Exception as e:
            print(f"[ERROR] Processing {name}:\n{e}")


class Progress(RichProgress):
    def __init__(self, base_name):
        super().__init__(expand=True)
        self.base_name = base_name
        self.description = f"[green][{self.base_name:^9}] Timing"
        self.total = None

    def add_task(self, count):
        self.total = count
        return super().add_task(description=self.description, total=count)

    def update(self, task, subtask):
        if subtask is not None:
            super().update(task, advance=1, description=f"{self.description}: {subtask}",
                           refresh=True)
        else:
            super().update(
                task,
                description=f"[blue][{self.base_name:^9}] Evaluated {self.total} environments",
                advance=0, refresh=True)

