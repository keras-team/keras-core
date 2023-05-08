import numpy as np

from keras_core import layers
from keras_core import models


def get_test_data(
    train_samples, test_samples, input_shape, num_classes, random_seed=None
):
    """Generates test data to train a model on.

    Args:
        train_samples: Integer, how many training samples to generate.
        test_samples: Integer, how many test samples to generate.
        input_shape: Tuple of integers, shape of the inputs.
        num_classes: Integer, number of classes for the data and targets.
        random_seed: Integer, random seed used by Numpy to generate data.

    Returns:
        A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    num_sample = train_samples + test_samples
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    y = np.random.randint(0, num_classes, size=(num_sample,))
    x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
    for i in range(num_sample):
        x[i] = templates[y[i]] + np.random.normal(
            loc=0, scale=1.0, size=input_shape
        )
    return (
        (x[:train_samples], y[:train_samples]),
        (x[train_samples:], y[train_samples:]),
    )


def get_small_sequential_mlp(num_hidden, num_classes, input_dim=None):
    model = models.Sequential()
    model.add(layers.Dense(num_hidden, activation="relu"))
    activation = "sigmoid" if num_classes == 1 else "softmax"
    model.add(layers.Dense(num_classes, activation=activation))
    return model
