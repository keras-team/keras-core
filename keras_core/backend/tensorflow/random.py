import tensorflow as tf

from keras_core.backend.common import standardize_dtype
from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed


def tf_draw_seed(seed):
    if isinstance(seed, SeedGenerator):
        return seed.next()
    else:
        return draw_seed(seed)


def make_default_seed():
    rng = tf.random.Generator.from_seed(42)
    seed = tf.cast(rng.make_seeds(2)[0], tf.int32)
    return rng, seed


def make_initial_seed(seed):
    if isinstance(seed, (tuple, list, tf.Tensor)):
        raise ValueError(
            f"Initial seed should be a scalar value. Received seed={seed}."
        )
    if seed < 0:
        raise ValueError(
            f"Seed should be a non-negative number. Received seed={seed}."
        )
    rng = tf.random.Generator.from_seed(seed)
    seed = tf.cast(rng.make_seeds(2)[0], tf.int32)
    return rng, seed


def get_next_state(seed):
    return tf.random.split(seed, 2)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_uniform(
        shape=shape,
        minval=tf.cast(minval, dtype),
        maxval=tf.cast(maxval, dtype),
        dtype=dtype,
        seed=seed,
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    seed = tf_draw_seed(seed)
    output = tf.random.stateless_categorical(logits, num_samples, seed=seed)
    return tf.cast(output, dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    intemediate_dtype = dtype
    if standardize_dtype(dtype) not in ["int32", "int64"]:
        intemediate_dtype = "int64"
    seed = tf_draw_seed(seed)
    output = tf.random.stateless_uniform(
        shape=shape,
        minval=minval,
        maxval=maxval,
        dtype=intemediate_dtype,
        seed=seed,
    )
    return tf.cast(output, dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = tf_draw_seed(seed)
    return tf.random.stateless_truncated_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return tf.shape(inputs)

    concrete_inputs_shape = tf.shape(inputs)
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = tf_draw_seed(seed)
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    return tf.nn.experimental.stateless_dropout(
        inputs,
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )
