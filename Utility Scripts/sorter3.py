import numpy as np
import pdb
import shutil
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)

converters = {
    "No Finding": "NORMAL",
}

data = np.loadtxt(f"{ROOT}/Data_Entry_2017.csv", dtype=str, delimiter=",", skiprows=1)
for line in data:
    src = f"{ROOT}/images/{line[0]}"
    findings = line[1].split("|")
    for finding in findings:
        dest = f"{DATA_ROOT}/{finding.upper() if finding not in converters.keys() else converters[finding]}/{line[0]}"
        shutil.copy(src, dest)
