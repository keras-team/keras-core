import jax
from jax import numpy as jnp
from tensorflow import nest

from keras_core.backend.common import standardize_dtype
from keras_core.backend.jax.variable import convert_to_tensor
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.backend.stateless_scope import StatelessScope


def is_tensor(x):
    if isinstance(x, jnp.ndarray):
        return True
    return False


def shape(x):
    # This will work as long as we disallow
    # dynamic shapes in JAX.
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fun, false_fun):
    return jax.lax.cond(pred, true_fn=true_fun, false_fun=false_fun)


def name_scope(name):
    return jax.named_scope(name)


def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def convert_keras_tensor_to_jax(x):
            if isinstance(x, KerasTensor):
                return jax.ShapeDtypeStruct(x.shape, dtype=x.dtype)
            return x

        built_in_types = (type(None), int, float, str, bool, complex, bytes)
        static_args = []
        maybe_symbolic_args = []
        for arg in args:
            if isinstance(arg, built_in_types):
                static_args.append(arg)
            else:
                maybe_symbolic_args.append(arg)
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for (
            k,
            arg,
        ) in kwargs.items():
            if isinstance(arg, built_in_types):
                static_kwargs[k] = arg
            else:
                maybe_symbolic_kwargs[k] = arg

        def wrapped_fn(*args, **kwargs):
            return fn(*args, *static_args, **kwargs, **static_kwargs)

        maybe_symbolic_args, maybe_symbolic_kwargs = nest.map_structure(
            convert_keras_tensor_to_jax,
            (maybe_symbolic_args, maybe_symbolic_kwargs),
        )
        _, jax_out = jax.make_jaxpr(wrapped_fn, return_shape=True)(
            *maybe_symbolic_args, **maybe_symbolic_kwargs
        )

        def convert_jax_spec_to_keras_tensor(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                return KerasTensor(x.shape, x.dtype)
            return x

        return nest.map_structure(convert_jax_spec_to_keras_tensor, jax_out)
