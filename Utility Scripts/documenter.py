from glob import glob
import os
import pdb
from tqdm import tqdm
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
folders = ['ATELECTASIS', 'CARDIOMEGALY', 'CONSOLIDATION', 'COVID_19', 'EDEMA', 'EFFUSION', 'EMPHYSEMA', 'FIBROSIS',
           'HERNIA', 'INFILTRATION', 'MASS', 'NODULE', 'NORMAL', 'PLEURAL_THICKENING', 'PNEUMONIA', 'PNEUMOTHORAX', 'TUBERCULOSIS']
categories = ['TRAIN', 'TEST', 'VALIDATION']

with open(f'{ROOT}/image_docs.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["image", "anomaly", "split category"]
    writer.writerow(field)
    for folder in folders:
        for category in categories:
            for image in tqdm(glob("*", root_dir=f'{ROOT}/{folder}/{category}'), desc=f'Documenting files in {ROOT}/{folder}/{category}'):
                writer.writerow([image, folder, category])
