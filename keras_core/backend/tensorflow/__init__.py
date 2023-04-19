from tensorflow.experimental import numpy as tfnp

from keras_core.backend.tensorflow import numpy
from keras_core.backend.tensorflow import random
from keras_core.backend.tensorflow.basic import *
from keras_core.backend.tensorflow.variable import Variable

DYNAMIC_SHAPES_OK = True


def execute(op_name, *args, **kwargs):
    if hasattr(tfnp, op_name):
        op = getattr(tfnp, op_name)
        return op(*args, **kwargs)
    raise AttributeError(
        f"The TensorFlow backend does not support op '{op_name}'"
    )
