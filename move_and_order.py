import os
import pandas as pd
import numpy as np
import sys

indexes = []
metrics = []

new_name = sys.argv[1]
os.makedirs(f"Output/{new_name}/")
for i in os.listdir("Output/"):
    if i.isdigit():
        if not (990000 <= int(i) <= 999000):
            os.rename(f"Output/{i}/", f"Output/{new_name}/{i}/")

    if os.path.isfile(f"Output/{i}"):
        os.rename(f"Output/{i}", f"Output/{new_name}/{i}")
