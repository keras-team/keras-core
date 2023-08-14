import random as python_random

import torch
import torch.nn.functional as tnn

from keras_core.backend.config import floatx
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import get_device
from keras_core.backend.torch.core import to_torch_dtype
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed


def torch_seed_generator(seed):
    if isinstance(seed, torch.Generator):
        return seed
    elif isinstance(seed, SeedGenerator):
        return seed.next()
    else:
        return draw_seed(seed)


def make_default_seed():
    device = get_device()
    if device == "meta":
        # Generator is not supported by the meta device.
        return None

    torch_seed_gen = torch.Generator(device=get_device())
    random_seed = python_random.randint(1, int(1e9))
    torch_seed_gen.manual_seed(random_seed)
    return torch_seed_gen, torch_seed_gen.get_state()


def make_initial_seed(seed):
    if isinstance(seed, (tuple, list, torch.Tensor)):
        raise ValueError(
            f"Initial seed should be a scalar value. Received seed={seed}."
        )
    if seed < 0:
        raise ValueError(
            f"Seed should be a non-negative number. Received seed={seed}."
        )
    device = get_device()
    if device == "meta":
        # Generator is not supported by the meta device.
        return None
    torch_seed_gen = torch.Generator(device=get_device())
    torch_seed_gen.manual_seed(seed)
    return torch_seed_gen, torch_seed_gen.get_state()


def get_next_seed_state(rng):
    return rng, rng.get_state()


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.normal(
            mean, stddev, size=shape, dtype=dtype, device=get_device()
        )
    generator = torch_seed_generator(seed)
    return torch.normal(
        mean,
        stddev,
        size=shape,
        generator=generator,
        dtype=dtype,
        device=get_device(),
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    logits = convert_to_tensor(logits)
    dtype = to_torch_dtype(dtype)
    probs = torch.softmax(logits, dim=-1)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.multinomial(
            probs,
            num_samples,
            replacement=True,
        ).type(dtype)
    generator = torch_seed_generator(seed)
    return torch.multinomial(
        probs,
        num_samples,
        replacement=True,
        generator=generator,
    ).type(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    requested_shape = shape
    if len(requested_shape) == 0:
        shape = (1,)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        rand_tensor = torch.rand(size=shape, dtype=dtype, device=get_device())
    else:
        generator = torch_seed_generator(seed)
        rand_tensor = torch.rand(
            size=shape, generator=generator, dtype=dtype, device=get_device()
        )

    output = (maxval - minval) * rand_tensor + minval

    if len(requested_shape) == 0:
        return output[0]
    return output


def randint(shape, minval, maxval, dtype="int32", seed=None):
    dtype = to_torch_dtype(dtype)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.randint(
            low=minval,
            high=maxval,
            size=shape,
            dtype=dtype,
            device=get_device(),
        )
    generator = torch_seed_generator(seed)
    return torch.randint(
        low=minval,
        high=maxval,
        size=shape,
        generator=generator,
        dtype=dtype,
        device=get_device(),
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    # Take a larger standard normal dist, discard values outside 2 * stddev
    # Offset by mean and stddev
    x = normal(tuple(shape) + (4,), mean=0, stddev=1, dtype=dtype, seed=seed)
    valid = (x > -2) & (x < 2)
    indexes = valid.max(-1, keepdim=True)[1]
    trunc_x = torch.empty(shape, device=get_device())
    trunc_x.data.copy_(x.gather(-1, indexes).squeeze(-1))
    trunc_x.data.mul_(stddev).add_(mean)
    return trunc_x


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
    if (
        seed is not None
        and not (isinstance(seed, SeedGenerator) and seed._initial_seed is None)
        or noise_shape is not None
    ):
        keep_prob = 1.0 - rate
        noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
        keep_prob_matrix = torch.full(
            noise_shape, keep_prob, device=get_device()
        )
        generator = torch_seed_generator(seed)

        # Do not use generator during symbolic execution.
        if get_device() == "meta":
            mask = torch.bernoulli(keep_prob_matrix)
        else:
            mask = torch.bernoulli(keep_prob_matrix, generator=generator)

        mask = mask.bool()
        mask = torch.broadcast_to(mask, inputs.shape)
        return torch.where(
            mask,
            inputs / keep_prob,
            torch.zeros_like(inputs, dtype=inputs.dtype),
        )
    # Fast path, unseeded (since torch doesn't support seeding dropout!!!!)
    # Using the above implementation is possible, but much slower.
    return torch.nn.functional.dropout(
        inputs, p=rate, training=True, inplace=False
    )
