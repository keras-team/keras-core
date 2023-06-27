import numpy as np

from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = np.prod(shape)
    random_numbers = np.empty(0)

    # loop until we have enough valid numbers to fill our desired shape
    while random_numbers.shape[0] < flat_shape:
        # Generate a batch of random numbers from a normal distribution
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)

        # Filter the numbers to keep only those within the specified bounds
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]

        # Append the valid numbers to the result array
        random_numbers = np.append(random_numbers, valid)

    # Truncate the result array to the desired size and reshape it
    return random_numbers[:flat_shape].astype(dtype).reshape(shape)


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)

    dropout_mask = rng.uniform(size=noise_shape) > rate

    dropout_inputs = inputs * dropout_mask / (1 - rate)

    if noise_shape:
        noise = rng.normal(size=noise_shape)
        dropout_inputs = dropout_inputs + noise

    return dropout_inputs.astype(inputs.dtype)
