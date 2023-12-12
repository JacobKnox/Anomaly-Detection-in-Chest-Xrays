# Anomaly-Detection-in-Chest-Xrays
MAT 4999 - Senior Seminar project attempting to use machine/deep learning concepts to detect anomalies in chest x-rays. A comprehensive write-up of this project can be found in the document titled "Application of Machine Learning to Anomaly Detection in Chest X-Rays."

## Data
Due to the large size of the compiled dataset and difficulties I had with moving the dataset a few times, it was not feasible to upload it to this repository.
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
The images were sorted into their proper anomaly categories either by hand (for datasets that had clear, easy organization) or by sorter scripts (found in the Utility Scripts folder). From there, a splitter (found in the Utility Scripts folder) was used to split the data in train, test, and validation sets for each anomaly in roughly 60%, 30%, 10% split respectively. During the splitting process, only images of PNG format were kept. Finally, for ease of use, the image_docs.csv file was created using the documenter script and all images were ultimately resized to 256 x 256 pixels using the resizer script (both found in the Utility Scripts folder).

## Convolutional Neural Network
The model created for this project can be found in the fcn.py file with a currently trained version of the model, my_model.keras, and an old version of the trained model under the Models directory of this repository.