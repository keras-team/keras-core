import math
import tensorflow as tf

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
    if data_format == "channels_first":
        if len(image.shape) == 4:
            image = tf.transpose(image, (0, 2, 3, 1))
        elif len(image.shape) == 3:
            image = tf.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    # TODO: https://github.com/keras-team/keras-core/issues/294
    # XLA not compatible with UpSampling2D `tf.image.resize`
    # Workaround - Use `tf.repeat` for `nearest` interpolation
    if method == "nearest":
        resized = _resize_repeat_interpolation(image, size)
    else:
        resized = tf.image.resize(image, size, method=method, antialias=antialias)
        if data_format == "channels_first":
            if len(image.shape) == 4:
                resized = tf.transpose(resized, (0, 3, 1, 2))
            elif len(image.shape) == 3:
                resized = tf.transpose(resized, (2, 0, 1))
    return resized

def _resize_repeat_interpolation(image, size, data_format="channel_last"):
    """Resize via `tf.repeat` operation."""
    im_shape = image.shape
    channel_ax = len(im_shape) - 1
    batch_ax = None
    if len(image.shape) == 4:
        batch_ax = 0
        if data_format == "channel_first":
            new_im_shape = [im_shape[0:2], *size]
            channel_ax = 1
        else:
            new_im_shape = [im_shape[0], *size, im_shape[-1]]
    elif len(image.shape) == 3:
        if data_format == "channel_first":
            new_im_shape = [im_shape[0], *size]
            channel_ax = 0
        else:
            new_im_shape = [*size, im_shape[-1]]
    axes = range(len(im_shape))
    st_indexes = []
    for ax, new_sz, old_sz in zip(axes, new_im_shape, im_shape):
        if ax == channel_ax or ax == batch_ax:
            st_indexes.append(0)
            continue
        ratio = math.ceil(new_sz / old_sz)
        image = tf.repeat(image, ratio, axis=ax)
        # Track start indexes to center crop
        mid = image.shape[ax] // 2
        st_indexes.append(mid - new_sz // 2)

    return tf.slice(image, st_indexes, new_im_shape)