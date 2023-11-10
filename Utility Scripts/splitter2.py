import shutil
import numpy as np
from glob import glob
import os
import pdb
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
folders = [
    "ATELECTASIS",
    "CARDIOMEGALY",
    "CONSOLIDATION",
    "COVID_19",
    "EDEMA",
    "EFFUSION",
    "EMPHYSEMA",
    "FIBROSIS",
    "HERNIA",
    "INFILTRATION",
    "MASS",
    "NODULE",
    "NORMAL",
    "PLEURAL_THICKENING",
    "PNEUMONIA",
    "PNEUMOTHORAX",
    "TUBERCULOSIS",
]
categories = ["TRAIN", "TEST", "VALIDATION"]
percentages = [1, 0.4, 0.1]

for folder in folders:
    for category in categories:
        os.makedirs(os.path.dirname(f"{ROOT}/{folder}/{category}/"), exist_ok=True)
    for image in tqdm(
        glob("*.png", root_dir=f"{ROOT}/{folder}/"),
        desc=f"Splitting images in folder {folder}",
        unit="img",
    ):
        src = f"{ROOT}/{folder}/{image}"
        category = categories[0]
        random_num = np.random.rand()
        if random_num < percentages[2]:
            category = categories[2]
        elif random_num < percentages[1]:
            category = categories[1]
        dest = f"{ROOT}/{folder}/{category}/{image}"
        shutil.move(src, dest)
