import functools

import jax
import jax.numpy as jnp

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


AFFINE_METHODS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_FILL_MODES = (
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
)


def _affine_single_image(
    image,
    transform,
    order,
    mode,
    cval,
):
    meshgrid = jnp.meshgrid(
        *[jnp.arange(size) for size in image.shape], indexing="ij"
    )
    indices = jnp.concatenate(
        [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    new_transform = jnp.array(
        [
            [transform[4], transform[1], 0],
            [transform[3], transform[0], 0],
            [0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    offset = jnp.array([transform[5], transform[2], 0], dtype=jnp.float32)
    coordinates = indices @ new_transform
    coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)
    coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))
    affined = jax.scipy.ndimage.map_coordinates(
        image, coordinates, order=order, mode=mode, cval=cval
    )
    return affined


def affine(
    image,
    transform,
    method="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if method not in AFFINE_METHODS.keys():
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{AFFINE_METHODS.keys()}. Received: method={method}"
        )
    if fill_mode not in AFFINE_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_FILL_MODES}. Received: method={fill_mode}"
        )
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

    method = AFFINE_METHODS[method]

    # unbatched case
    need_squeeze = False
    if len(image.shape) == 3:
        image = jnp.expand_dims(image, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = jnp.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        image = jnp.transpose(image, (0, 2, 3, 1))

    _affine_single_image_impl = functools.partial(
        _affine_single_image, order=method, mode=fill_mode, cval=fill_value
    )
    affined = jax.vmap(_affine_single_image_impl)(image, transform)

    if data_format == "channels_first":
        affined = jnp.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = jnp.squeeze(affined, axis=0)
    return affined
