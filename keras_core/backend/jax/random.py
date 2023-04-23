import jax

from keras_core.backend.common.random import SeedGenerator
from keras_core.backend.common.random import draw_seed
from keras_core.backend.common.random import make_default_seed
from keras_core.backend.config import floatx


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Produce random number based on the normal distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, default to 0. Mean of the random values to generate.
        stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`).
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    sample = jax.random.normal(seed, shape=shape, dtype=dtype)
    return sample * stddev + mean


def uniform(shape, minval=0.0, maxval=None, dtype=None, seed=None):
    """Produce random number based on the uniform distribution.

    Args:
        shape: The shape of the random values to generate.
        minval: Floats, default to 0. Lower bound of the range of
            random values to generate (inclusive).
        minval: Floats, default to None. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    return jax.random.uniform(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Produce random number based on the truncated normal distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, default to 0. Mean of the random values to generate.
        stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    sample = jax.random.truncated_normal(
        seed, shape=shape, lower=-2.0, upper=2.0, dtype=dtype
    )
    return sample * stddev + mean


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    keep_prob = 1.0 - rate
    if noise_shape is None:
        noise_shape = inputs.shape
    mask = jax.random.bernoulli(seed, p=keep_prob, shape=noise_shape)
    mask = jax.numpy.broadcast_to(mask, inputs.shape)
    return jax.lax.select(
        mask, inputs / keep_prob, jax.numpy.zeros_like(inputs)
    )
