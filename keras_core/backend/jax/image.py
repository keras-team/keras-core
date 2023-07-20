import functools

import jax
import jax.numpy as jnp

from keras_core.backend.jax.core import convert_to_tensor

RESIZE_METHODS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def resize(
    image, size, method="bilinear", antialias=False, data_format="channels_last"
):
    if method not in RESIZE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{RESIZE_METHODS}. Received: method={method}"
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    if len(image.shape) == 4:
        if data_format == "channels_last":
            size = (image.shape[0],) + size + (image.shape[-1],)
        else:
            size = (image.shape[0], image.shape[1]) + size
    elif len(image.shape) == 3:
        if data_format == "channels_last":
            size = size + (image.shape[-1],)
        else:
            size = (image.shape[0],) + size
    else:
        raise ValueError(
            "Invalid input rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    return jax.image.resize(image, size, method=method, antialias=antialias)


AFFINE_TRANSFORM_METHODS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = (
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
)


def affine_transform(
    image,
    transform,
    method="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if method not in AFFINE_TRANSFORM_METHODS.keys():
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{set(AFFINE_TRANSFORM_METHODS.keys())}. Received: method={method}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: method={fill_mode}"
        )

    transform = convert_to_tensor(transform)

    if len(image.shape) not in (3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    # unbatched case
    need_squeeze = False
    if len(image.shape) == 3:
        image = jnp.expand_dims(image, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = jnp.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        image = jnp.transpose(image, (0, 2, 3, 1))

    batch_size = image.shape[0]

    # get indices
    meshgrid = jnp.meshgrid(
        *[jnp.arange(size) for size in image.shape[1:]], indexing="ij"
    )
    indices = jnp.concatenate(
        [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = jnp.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0]
    a2 = transform[:, 2]
    b1 = transform[:, 4]
    b2 = transform[:, 5]
    transform = transform.at[:, 0].set(b1)
    transform = transform.at[:, 2].set(b2)
    transform = transform.at[:, 4].set(a0)
    transform = transform.at[:, 5].set(a2)

    # deal with transform
    transform = jnp.pad(
        transform, pad_width=[[0, 0], [0, 1]], constant_values=1
    )
    transform = jnp.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2]
    offset = jnp.pad(offset, pad_width=[[0, 0], [0, 1]])
    transform = transform.at[:, 0:2, 2].set(0)

    # transform the indices
    coordinates = jnp.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = jnp.moveaxis(coordinates, source=-1, destination=1)
    coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

    # apply affine transformation
    _map_coordinates = functools.partial(
        jax.scipy.ndimage.map_coordinates,
        order=AFFINE_TRANSFORM_METHODS[method],
        mode=fill_mode,
        cval=fill_value,
    )
    affined = jax.vmap(_map_coordinates)(image, coordinates)

    if data_format == "channels_first":
        affined = jnp.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = jnp.squeeze(affined, axis=0)
    return affined
