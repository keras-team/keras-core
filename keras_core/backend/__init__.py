import json
import os

from keras_core.backend.common import standardize_dtype
from keras_core.backend.common import standardize_shape
from keras_core.backend.config import epsilon
from keras_core.backend.config import floatx
from keras_core.backend.config import image_data_format
from keras_core.backend.config import set_epsilon
from keras_core.backend.config import set_floatx
from keras_core.backend.config import set_image_data_format
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.backend.keras_tensor import any_symbolic_tensors
from keras_core.backend.keras_tensor import is_keras_tensor
from keras_core.backend.stateless_scope import StatelessScope
from keras_core.utils.io_utils import print_msg

# Import backend functions.
if backend() == "tensorflow":
    print_msg("Using TensorFlow backend")
    from keras_core.backend.tensorflow import *
elif backend() == "jax":
    print_msg("Using JAX backend.")
    from keras_core.backend.jax import *
else:
    raise ValueError(f"Unable to import backend : {backend()}")
