import jax

import random as python_random

from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed


def jax_draw_seed(seed):
    if isinstance(seed, SeedGenerator):
        return seed.next()
    else:
        return draw_seed(seed)


def make_default_seed():
    random_seed = python_random.randint(1, int(1e9))
    return None, jax.random.PRNGKey(seed=random_seed)


def make_initial_seed(seed):
    if isinstance(seed, (tuple, list, jax.Array)):
        raise ValueError(
            f"Initial seed should be a scalar value. Received seed={seed}."
        )
    if seed < 0:
        raise ValueError(
            f"Seed should be a non-negative number. Received seed={seed}."
        )
    return None, jax.random.PRNGKey(seed=seed)


def get_next_seed_state(seed):
    # JAX complains about the PRNG dtype if we don't cast it
    # TODO: Figure out the underlying problem
    return jax.random.split(jax.numpy.array(seed, jax.numpy.uint32), 2)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    sample = jax.random.normal(seed, shape=shape, dtype=dtype)
    return sample * stddev + mean


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    return jax.random.uniform(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    seed = jax_draw_seed(seed)
    output_shape = list(logits.shape)
    output_shape[1] = num_samples
    output_shape = tuple(output_shape)
    output = jax.random.categorical(
        seed, logits[..., None], shape=output_shape, axis=1
    )
    return output.astype(dtype)


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = jax_draw_seed(seed)
    return jax.random.randint(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = jax_draw_seed(seed)
    sample = jax.random.truncated_normal(
        seed, shape=shape, lower=-2.0, upper=2.0, dtype=dtype
    )
    return sample * stddev + mean


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = jax_draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to jax.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = jax.random.bernoulli(seed, p=keep_prob, shape=noise_shape)
    mask = jax.numpy.broadcast_to(mask, inputs.shape)
    return jax.lax.select(
        mask, inputs / keep_prob, jax.numpy.zeros_like(inputs)
    )
