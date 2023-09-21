import numpy as np

import keras_core as keras

from keras_core import Model
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers


@keras.saving.register_keras_serializable()
class MyModel(Model):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.dense3 = layers.Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def get_config(self):
        config = super().get_config()
        config["hidden_dim"] = self.hidden_dim
        config["output_dim"] = self.output_dim
        return config


model = MyModel(hidden_dim=256, output_dim=16)

x = np.random.random((50000, 128))
y = np.random.random((50000, 16))
batch_size = 32
epochs = 6

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()],
)
history = model.fit(
    x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
)

print("History:")
print(history.history)

model.summary()

# Test saving model

model.save("basic_model.keras")
loaded_model = keras.models.load_model("basic_model.keras")
loaded_model.predict(x)