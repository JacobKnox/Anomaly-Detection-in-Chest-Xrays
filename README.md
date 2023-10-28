# Anomaly-Detection-in-Chest-Xrays
MAT 4999 - Senior Seminar project attempting to use machine/deep learning concepts to detect anomalies in chest x-rays.

## Data
### Sources
Images were collected from a variety of datasets found on Kaggle (listed below).
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
- https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
- https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets
- https://www.kaggle.com/datasets/nabeelsajid917/covid-19-x-ray-10000-images
- https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
- https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- https://www.kaggle.com/datasets/nih-chest-xrays/data
- https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
### Preprocessing
The images were sorted into their proper anomaly categories either by hand (for datasets that had clear, easy organization) or by sorter scripts (found in the Utility Scripts folder). From there, a splitter (found in the Utility Scripts folder) was used to split the data in train, test, and validation sets for each anomaly. During the splitting process, only images of PNG format were kept. Finally, for ease of use, the image_docs.csv file was created using the documenter script in Utility Scripts.