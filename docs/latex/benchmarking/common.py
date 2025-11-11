import os
import math
import time
import traceback

from rich.progress import Progress as RichProgress

STEPS = 1000
REPLICATES = 10


def line(c='='):
    print(c*os.get_terminal_size()[0])


class Progress(RichProgress):
    def __init__(self, df, library):
        super().__init__(expand=True)
        self.df, self.library = df, library
        self.description = f"[green][{self.library:^9}] Timing"
        self.environments = None
        self.steps = None
        self.errors = None
        self.task = None

    def add_task(self, count):
        self.environments = count
        self.steps = count * REPLICATES
        self.errors = 0
        self.task = super().add_task(description=self.description, total=self.steps)
        return self.task

    def close(self):
        if self.errors > 0:
            total = self.environments - self.errors
            color = "red"
        else:
            total = self.environments
            color = "blue"

        super().update(
            self.task,
            description=f"[{color}][{self.library:^9}] Evaluated {total} environments",
            advance=0, refresh=True)

        if self.errors > 0:
            super().console.print(f"[{color}] {self.errors} threw errors")

    def evaluate(self, family, name, fn, label=None, **kwargs):
        previously_completed = super().tasks[self.task].completed
        if name not in self.df.index or math.isnan(self.df.loc[name]["Time"]):
            try:
                start = time.time_ns()
                for i in range(REPLICATES):
                    if not fn(**kwargs):
                        start = float("NaN")

                    super().update(self.task, advance=1,
                                   description=f"{self.description}:"
                                               f" {name} [{i}/{REPLICATES}]",
                                   refresh=True)

                t = (time.time_ns() - start) / (REPLICATES * 10**9)

                if not math.isnan(t):
                    self.df.loc[name] = [self.library, family, label, t]

            except Exception as e:
                print(traceback.format_exc())
                print(f"[ERROR] Processing {name}:\n{e}")
                self.errors += 1

        super().update(self.task, completed=previously_completed+REPLICATES,
                       description=f"{self.description}: {name}",
                       refresh=True)

