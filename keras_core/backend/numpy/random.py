import numpy as np

from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, defaults to 0. Mean of the random values to generate.
        stddev: Floats, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras_core.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras_core.backend.set_floatx(float_dtype)`).
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
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Draw samples from a uniform distribution.

    Args:
        shape: The shape of the random values to generate.
        minval: Floats, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Floats, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras_core.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
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
    rng = np.random.default_rng(seed)
    return rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw samples from a truncated normal distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, defaults to 0. Mean of the random values to generate.
        stddev: Floats, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras_core.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras_core.backend.set_floatx(float_dtype)`)
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
    rng = np.random.default_rng(seed)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    # Initialize an empty array to store the random numbers
    random_numbers = np.empty(shape)

    # Generate random numbers using rejection sampling
    count = 0
    while count < shape[0]:
        # Generate a batch of random numbers from a normal distribution
        batch = rng.normal(loc=mean, scale=stddev, size=shape)

        # Filter the numbers to keep only those within the specified bounds
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]

        # Store the valid numbers in the result array
        num_valid = valid.shape[0]
        random_numbers[count : count + num_valid] = valid

        # Update the count of random numbers generated
        count += num_valid

    # Truncate the result array to the desired size
    return random_numbers[: shape[0]].astpye(dtype)


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed)

    dropout_mask = rng.uniform(size=noise_shape) > rate

    dropout_inputs = inputs * dropout_mask / (1 - rate)

    if noise_shape:
        noise = rng.normal(size=noise_shape)
        dropout_inputs = dropout_inputs + noise

    return dropout_inputs.astype(inputs.dtype)
