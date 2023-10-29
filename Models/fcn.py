from typing import Any
from keras import layers, Model


class FCN:
    def __init__(self, num_classes: int = 17, dropout_rate: float = 0.2, filter: int = 32):
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.filter = filter
        self.model = self._make_model()

    def predict(self):
        pass

    def fit(self):
        pass

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
    FCN()
