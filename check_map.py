import numpy as np

from keras_core.backend.jax.image import map_coordinates as jax_map_coordinates
from keras_core.backend.tensorflow.image import (
    map_coordinates as tf_map_coordinates,
)
from keras_core.backend.torch.image import (
    map_coordinates as torch_map_coordinates,
)

data = np.arange(12).reshape((4, 3))
coordinates = np.array([[0.5, 2], [0.5, 1]])

# print(jax_map_coordinates(data, coordinates, 1))
print(tf_map_coordinates(data, coordinates, 1))
# print(torch_map_coordinates(data, coordinates, 1))
