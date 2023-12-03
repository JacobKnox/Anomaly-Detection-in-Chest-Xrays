from typing import Any
from keras import layers, Model
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pdb
from time import time


class FCN:
    def __init__(
        self, num_classes: int = 17, dropout_rate: float = 0.2, filter: int = 32
    ):
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.filter = filter
        self.model = self._make_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.FalseNegatives(),
            ],
        )

    def predict(self, input):
        return self.model.predict(input)

    def fit(self, inputs, labels, vals, val_labels):
        self.model.fit(
            inputs,
            labels,
            batch_size=100,
            validation_data=(vals, val_labels),
            verbose="2",
        )

    def summary(self) -> None:
        print(self.model.summary())
        print(f"Total number of layers: {len(self.model.layers)}")

    def _make_model(self) -> Model:
        input = layers.Input(shape=(256, 256, 1))

        group_one = self._group(input, self.filter)
        group_two = self._group(group_one, self.filter * 2)
        group_three = self._group(group_two, self.filter * 4)
        group_four = self._group(group_three, self.filter * 8)
        group_five = self._group(group_four, self.filter * 16)
        fully_connected_one = self._group(group_five, self.filter * 2)

        # predictions = self._group(
        #     fully_connected_one, self.num_classes, 1, 1, True, "softmax"
        # )

        predictions = self._group(
            fully_connected_one, 1, max_pool=True, activation="softmax"
        )

        return Model(inputs=input, outputs=predictions)

    def _group(
        self,
        previous_layer: Any | list | None,
        filter: int,
        stride: tuple = (2, 2),
        kernel: tuple = (2, 2),
        max_pool: bool = False,
        activation: str = "relu",
    ) -> Any | None:
        layer = layers.Conv2D(filters=filter, kernel_size=kernel, strides=stride)(
            previous_layer
        )
        layer = layers.Dropout(self.dropout_rate)(layer)
        layer = layers.BatchNormalization()(layer)
        if max_pool:
            layer = layers.GlobalMaxPooling2D()(layer)
        layer = layers.Activation(activation)(layer)
        return layer


if __name__ == "__main__":
    data_dir = "C:\\Users\\epicd\\Documents\\Data"
    if os.path.exists(f"{os.getcwd()}/my_model.keras"):
        my_model = tf.keras.models.load_model(f"{os.getcwd()}/my_model.keras")
    else:
        my_model = FCN()
        my_model.summary()
        pdb.set_trace()
        data_info = np.loadtxt(
            "C:\\Users\\epicd\\Documents\\GitHub\\Anomaly-Detection-in-Chest-Xrays\\image_docs.csv",
            dtype=str,
            delimiter=",",
            skiprows=1,
        )
        train_data, train_labels = ([], [])
        if os.path.exists("./Data/train_data.npy"):
            print("Loading training images...")
            start = time()
            train_data = np.load("./Data/train_data.npy")
            print(f"Done! Took {time() - start} seconds.")
            start = time()
            print("Loading training labels...")
            train_labels = np.load("./Data/train_labels.npy")
            print(f"Done! Took {time() - start} seconds.")
        else:
            train_data_info = data_info[np.where(data_info[:, 2] == "TRAIN")]
            _, train_labels = np.unique(train_data_info[:, 1], return_inverse=True)
            train_labels += 1
            delete_indices = []
            i = -1
            for row in tqdm(
                train_data_info, desc="Loading training images", unit="image"
            ):
                i += 1
                image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
                image = np.asarray(image)
                if len(image.shape) > 2:
                    delete_indices.append(i)
                    continue
                train_data.append(image)
            train_labels = np.delete(train_labels, delete_indices)
            train_data = np.asarray(train_data)
            np.save("./Data/train_data.npy", train_data)
            np.save("./Data/train_labels.npy", train_labels)
        val_data, val_labels = ([], [])
        if os.path.exists("./Data/val_data.npy"):
            print("Loading validation images...")
            start = time()
            val_data = np.load("./Data/val_data.npy")
            print(f"Done! Took {time() - start} seconds.")
            start = time()
            print("Loading validation labels...")
            val_labels = np.load("./Data/val_labels.npy")
            print(f"Done! Took {time() - start} seconds.")
        else:
            val_data_info = data_info[np.where(data_info[:, 2] == "VALIDATION")]
            _, val_labels = np.unique(val_data_info[:, 1], return_inverse=True)
            val_labels += 1
            delete_indices = []
            i = -1
            for row in tqdm(
                val_data_info, desc="Loading validation images", unit="image"
            ):
                i += 1
                image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
                image = np.asarray(image)
                if len(image.shape) > 2:
                    delete_indices.append(i)
                    continue
                val_data.append(image)
            np.delete(val_labels, delete_indices)
            val_data = np.asarray(val_data)
            np.save("./Data/val_data.npy", val_data)
            np.save("./Data/val_labels.npy", val_labels)
        my_model.fit(train_data, train_labels, val_data, val_labels)
        my_model.save(f"{os.getcwd()}/my_model.keras")
        train_data, train_labels = (None, None)
        del train_data, train_labels
        val_data, val_labels = (None, None)
        del val_data, val_labels
    test_data, test_labels = ([], [])
    if os.path.exists("./Data/test_data.npy"):
        print("Loading testing images...")
        start = time()
        test_data = np.load("./Data/test_data.npy")
        print(f"Done! Took {time() - start} seconds.")
        start = time()
        print("Loading testing labels...")
        test_labels = np.load("./Data/test_labels.npy")
        print(f"Done! Took {time() - start} seconds.")
    else:
        test_data_info = data_info[np.where(data_info[:, 2] == "TEST")]
        _, test_labels = np.unique(test_data_info[:, 1], return_inverse=True)
        test_labels += 1
        delete_indices = []
        i = -1
        for row in tqdm(test_data_info, desc="Loading testing images", unit="image"):
            i += 1
            image = Image.open(f"{data_dir}\\{row[1]}\\{row[2]}\\{row[0]}")
            image = np.asarray(image)
            if len(image.shape) > 2:
                delete_indices.append(i)
                continue
            test_data.append(image)
        np.delete(test_labels, delete_indices)
        test_data = np.asarray(test_data)
        np.save("./Data/test_data.npy", test_data)
        np.save("./Data/test_labels.npy", test_labels)
    prediction = my_model.predict(test_data)
    print(f"Accuracy: {sum(prediction == test_labels)/len(test_labels)}")
    test_data, test_labels = (None, None)
    del test_data, test_labels
    pdb.set_trace()
