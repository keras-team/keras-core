import tensorflow as tf

from keras_core.backend.common import standardize_dtype
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.utils.naming import auto_name


def convert_to_tensor(x, dtype=None):
    dtype = standardize_dtype(dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


def is_tensor(x):
    return tf.is_tensor(x)


def shape(x):
    return tf.shape(x)


def cast(x, dtype):
    dtype = standardize_dtype(dtype)
    return tf.cast(x, dtype=dtype)


def cond(pred, true_fun, false_fun):
    return tf.cond(pred, true_fn=true_fun, false_fun=false_fun)


def name_scope(name):
    return tf.name_scope(name)


def compute_output_spec(fn, *args, **kwargs):
    graph_name = auto_name("scratch_graph")
    with tf.__internal__.FuncGraph(graph_name).as_default():

        def convert_keras_tensor_to_tf(x):
            if isinstance(x, KerasTensor):
                return tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
            return x

        args, kwargs = tf.nest.map_structure(
            convert_keras_tensor_to_tf, (args, kwargs)
        )
        tf_out = fn(*args, **kwargs)

        def convert_tf_to_keras_tensor(x):
            if tf.is_tensor(x):
                return KerasTensor(x.shape, x.dtype)
            return x

        return tf.nest.map_structure(convert_tf_to_keras_tensor, tf_out)
