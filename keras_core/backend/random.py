from keras_core.backend.config import backend

if backend() == "jax":
    from keras_core.backend.jax.random import *
else:
    from keras_core.backend.tensorflow.random import *
