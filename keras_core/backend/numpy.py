from keras_core.backend.config import backend

if backend() == "jax":
    from keras_core.backend.jax.numpy import *
else:
    from keras_core.backend.tensorflow.numpy import *
