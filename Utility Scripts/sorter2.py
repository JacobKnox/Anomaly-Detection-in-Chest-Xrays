import shutil
import numpy as np
from glob import glob
import os
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(ROOT)

options = {"atelectasis": "",
           "cardiomegaly": "",
           "consolidation": "",
           "covid-19": "COVID_19",
           "edema": "",
           "effusion": "",
           "emphysema": "",
           "fibrosis": "",
           "hernia": "",
           "infiltration": "",
           "mass": "",
           "nodule": "",
           "normal": "",
           "pleural thickening": "PLEURAL_THICKENING",
           "pneumonia": "",
           "pneumothorax": "",
           "tuberculosis": "",
           "tb": "TUBERCULOSIS"}

for reading, image in zip(glob('*',root_dir=f'{ROOT}/ClinicalReadings/'), glob('*',root_dir=f'{ROOT}/CXR_png/')):
    if reading.split('.')[0] != image.split('.')[0]:
        print(reading)
        print(image)
        break
    with open(f'{ROOT}/ClinicalReadings/{reading}', 'r', newline='\n') as file:
        reading_data = file.read().lower()
    for key, value in options.items():
        if key in reading_data:
            src = f'{ROOT}/CXR_png/{image}'
            dest = f'{DATA_ROOT}/{key.upper() if value == "" else value}/{image}'
            shutil.copy(src, dest)
    shutil.move(f'{ROOT}/CXR_png/{image}', f'{DATA_ROOT}/TRASH/{image}')
    shutil.move(f'{ROOT}/ClinicalReadings/{reading}', f'{DATA_ROOT}/TRASH/{reading}')