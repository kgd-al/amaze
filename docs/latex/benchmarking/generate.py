#!/bin/env python3

import sys
import importlib
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from procgen.gym_registration import make_env as make_procgen_env, ENV_NAMES as PROCGEN_ENV_NAMES

import amaze
#from amaze.extensions.sb3.maze_env import MazeEnv

from common import line


pd.options.display.float_format = '{:.3f}'.format

detailed = (len(sys.argv) > 1)
fname = "table" + ("-detailed" if detailed else "")

import workers.amaze_worker, workers.gymnasium_worker
workers_list = [workers.amaze_worker, workers.gymnasium_worker]
if detailed:
    import workers.procgen_worker
    workers_list.extend([workers.procgen_worker])


# ================================================================================================
# == Record reading/creation
# ================================================================================================

line()
folder=Path(__file__).parent
datafile=folder.joinpath(f"{fname}.csv")
print("Datafile:", datafile)

try:
    df = pd.read_csv(datafile, index_col="Name")

    #print(f"\033[31mTruncating {datafile} from {len(df)}", end=' ')
    #df = df[df.Library != "Procgen"]
    #print(f"to {len(df)}\033[0m")

except FileNotFoundError:
    df = pd.DataFrame(columns=["Library", "Family", "Time"])
    df.index.name = "Name"

# ================================================================================================
# == Process workers
# ================================================================================================

errors = 0
for worker in workers_list:
    line()
    #print(worker, worker.process)
    try:
        worker.process(df, detailed)
    except Exception as e:
        print("Failed:", e)
        raise e
        errors += 1

# ================================================================================================
# == Write up
# ================================================================================================

df.to_csv(datafile)
print(df)
exit(42)

if errors > 0:
    exit(1)
