from jax import numpy as jnp

from keras_core.backend.jax import numpy
from keras_core.backend.jax import random
from keras_core.backend.jax.basic import *
from keras_core.backend.jax.variable import *

DYNAMIC_SHAPES_OK = False  # Dynamic shapes NG


### NumPy op delegation
def execute(op_name, *args, **kwargs):
    if hasattr(jnp, op_name):
        op = getattr(jnp, op_name)
        return op(*args, **kwargs)
    raise AttributeError(f"The JAX backend does not support op '{op_name}'")
