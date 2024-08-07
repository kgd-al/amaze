#!/bin/env python3

import sys
import importlib
import pandas as pd

from pathlib import Path

from common import line


pd.options.display.float_format = '{:.3f}'.format

worker = sys.argv[1]
detailed = (len(sys.argv) > 2)
fname = "table" + ("-detailed" if detailed else "")

print(sys.argv)
print(worker)
print(detailed)
print(fname)

# ================================================================================================
# == Record reading/creation
# ================================================================================================

#line()
folder=Path(__file__).parent
datafile=folder.joinpath(f"{fname}.csv")
#print("Datafile:", datafile)

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
line("-")
#print(worker, worker.process)
try:
    file = Path(__file__).parent.joinpath("workers").joinpath(f"{worker}_worker.py")
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)
    module.process(df, detailed)
except Exception as e:
    print("Failed:", e)
    #raise e
    errors += 1

# ================================================================================================
# == Write up
# ================================================================================================

line("-")
print("==", "Dataframe:")

df.to_csv(datafile)
print(df)
#exit(42)

if errors > 0:
    exit(1)
