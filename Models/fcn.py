from typing import Any
from keras import layers, Model
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pdb


class FCN:
    def __init__(self, num_classes: int = 17, dropout_rate: float = 0.2, filter: int = 32):
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.filter = filter
        self.model = self._make_model()

    def predict(self, input):
        return self.model.predict(input)

    def fit(self, input, labels):
        self.model.fit(input, labels)

    def summary(self) -> None:
        print(self.model.summary())
        print(f'Total number of layers: {len(self.model.layers)}')

    def _make_model(self) -> Model:
        input = layers.Input(shape=(None, None, 3))

        group_one = self._group(input, self.filter, 1)
        group_two = self._group(group_one, self.filter * 2, 1)
        group_three = self._group(group_two, self.filter * 4, 2)
        group_four = self._group(group_three, self.filter * 8, 2)
        group_five = self._group(group_four, self.filter * 16, 2)
        fully_connected_one = self._group(group_five, self.filter * 2, 1, 1)

        predictions = self._group(
            fully_connected_one, self.num_classes, 1, 1, True, 'softmax')

        return Model(inputs=input, outputs=predictions)

    def _group(self, previous_layer: Any | list | None, filter: int, stride: int, kernel: int = 3, max_pool: bool = False, activation: str = "relu") -> Any | None:
        layer = layers.Conv2D(
            filters=filter, kernel_size=kernel, strides=stride)(previous_layer)
        layer = layers.Dropout(self.dropout_rate)(layer)
        layer = layers.BatchNormalization()(layer)
        if max_pool:
            layer = layers.GlobalMaxPooling2D()(layer)
        layer = layers.Activation(activation)(layer)
        return layer


if __name__ == "__main__":
    data_dir = "C:\\Users\\epicd\\Documents\\Data"
    FCN()
    data_info = np.loadtxt(
        "C:\\Users\\epicd\\Documents\\GitHub\\Anomaly-Detection-in-Chest-Xrays\\image_docs.csv", dtype=str, delimiter=',', skiprows=1)
    train_data_info = data_info[np.where(data_info[:, 2] == "TRAIN")]
    train_labels = train_data_info[:, 1]
    train_data = []
    if os.path.exists(".\\Data\\train.npy"):
        train_data = np.load(".\\Data\\train.npy")
    else:
        for row in tqdm(train_data_info, desc="Loading training images", unit="image"):
            image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
            train_data.append(np.asarray(image))
        np.save(".\\Data\\train.npy", train_data)
    test_data_info = data_info[np.where(data_info[:, 2] == "TEST")]
    test_labels = test_data_info[:, 1]
    test_data = []
    if os.path.exists(".\\Data\\test.npy"):
        train_data = np.load(".\\Data\\test.npy")
    else:
        for row in tqdm(test_data_info, desc="Loading testing images", unit="image"):
            image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
            test_data.append(np.asarray(image))
        np.save(".\\Data\\test.npy", test_data)
    val_data_info = data_info[np.where(data_info[:, 2] == "VALIDATION")]
    val_labels = val_data_info[:, 1]
    val_data = []
    if os.path.exists(".\\Data\\val.npy"):
        train_data = np.load(".\\Data\\val.npy")
    else:
        for row in tqdm(val_data_info, desc="Loading validation images", unit="image"):
            image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
            val_data.append(np.asarray(image))
        np.save(".\\Data\\val.npy", val_data)
    pdb.set_trace()
