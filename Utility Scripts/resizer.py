import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pdb

DATA_DIR = "C:\\Users\\epicd\\Documents\\Data"
INFO_DIR = "C:\\Users\\epicd\\Documents\\GitHub\\Anomaly-Detection-in-Chest-Xrays\\image_docs.csv"


def main():
    data_info = np.loadtxt(
        INFO_DIR,
        dtype=str,
        delimiter=",",
        skiprows=1,
    )
    for row in tqdm(data_info[126000:], desc="Resizing images", unit="image"):
        file_loc = f"{DATA_DIR}\\{row[1]}\\{row[2]}\\{row[0]}"
        try:
            im = Image.open(file_loc)
        except UnidentifiedImageError:
            print(f"Couldn't load image {file_loc}.")
            continue
        out = im.resize((256, 256))
        try:
            out.save(file_loc)
        except OSError:
            print(f"Failed to resize and save image: {file_loc}.")
            continue


if __name__ == "__main__":
    main()
