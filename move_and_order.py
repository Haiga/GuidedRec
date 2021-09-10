import os
import pandas as pd
import numpy as np
import sys

indexes = []
metrics = []

new_name = sys.argv[1]

for i in os.listdir("Output/"):
    if i.isdigit():
        os.rename(f"Output/{i}/", f"Output/{new_name}/{i}/")

    if os.path.isfile(f"Output/{i}"):
        os.rename(f"Output/{i}", f"Output/{new_name}/{i}")

