#!/bin/env python3

import sys
import importlib
import pandas as pd

from pathlib import Path

from common import line


pd.options.display.float_format = '{:.3f}'.format

worker = sys.argv[1]
detailed = (len(sys.argv) > 2)

# print(sys.argv)
# print(worker)
# print(detailed)

# ================================================================================================
# == Record reading/creation
# ================================================================================================

#line()
folder = Path(__file__).parent
datafile = folder.joinpath(f"table.csv")
#print("Datafile:", datafile)

columns = ["Library", "Family", "Class", "Time"]
try:
    df = pd.read_csv(datafile, index_col="Name")

    #print(f"\033[31mTruncating {datafile} from {len(df)}", end=' ')
    #df = df[df.Library != "Procgen"]
    #print(f"to {len(df)}\033[0m")

except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.index.name = "Name"

# ================================================================================================
# == Process worker
# ================================================================================================

errors = 0
line("-")
#print(worker, worker.process)
try:
    file = Path(__file__).parent.joinpath("workers").joinpath(f"{worker}_worker.py")
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)
    errors += module.process(df)
except Exception as e:
    print("Failed:", e)
    errors += 1
    raise e

# ================================================================================================
# == Write up
# ================================================================================================

# line("-")
# print("==", "Dataframe:")

df.sort_values(by=["Library", "Family", "Name"], inplace=True)
df.to_csv(datafile)
# print(df)

if errors > 0:
    print("Exiting unhappily after", errors, "errors")
    exit(1)
