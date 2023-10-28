import shutil
import numpy as np
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)
meta = np.loadtxt(f'{ROOT}/train.txt',str,delimiter=" ")
for line in meta:
    if(line[2] == 'positive'):
        src = f'{ROOT}/train/{line[1]}'
        dest = f'{DATA_ROOT}/COVID_19/{line[1]}'
        if os.path.exists(src):
            shutil.move(src, dest)
    else:
        src = f'{ROOT}/train/{line[1]}'
        dest = f'{DATA_ROOT}/TRASH/{line[1]}'
        if os.path.exists(src):
            shutil.move(src, dest)