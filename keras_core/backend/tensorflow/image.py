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
        resized = _resize_repeat_interpolation(image, size, data_format)
    else:
        resized = tf.image.resize(
            image, size, method=method, antialias=antialias
        )
        if data_format == "channels_first":
            if len(image.shape) == 4:
                resized = tf.transpose(resized, (0, 3, 1, 2))
            elif len(image.shape) == 3:
                resized = tf.transpose(resized, (2, 0, 1))
    return resized


def _resize_repeat_interpolation(image, size, data_format="channel_last"):
    """Resize via `tf.repeat` operation."""
    im_shape = image.shape
    # Track batch and channel dimension as they are not modified by resize
    channel_ax = len(im_shape) - 1
    batch_ax = None
    if len(im_shape) == 4:
        batch_ax = 0
        if data_format == "channel_first":
            channel_ax = 1
    elif len(image.shape) == 3:
        if data_format == "channel_first":
            channel_ax = 0
    # Iterate through axes and resize the image on the axis dimension
    # by repeating nearest element value.
    axes = range(len(im_shape))
    sz_idx = 0
    for ax, old_sz in zip(axes, im_shape):
        if ax == channel_ax or ax == batch_ax:
            continue
        new_sz = size[sz_idx]
        sz_idx += 1
        repeat_ratio = new_sz // old_sz
        if new_sz % old_sz == 0:
            image = tf.repeat(image, repeat_ratio, axis=ax)
        else:
            # When `new_sz` is not a multiple of `old_sz`, repeat particular
            # indices of image along the `old_sz` dim to match the `new_sz`.
            input_indices = tf.ones(old_sz, dtype=tf.int32) * repeat_ratio
            reminder = new_sz - (repeat_ratio * old_sz)
            # scatter the remainder repeat indices
            upd_indices = tf.reshape(
                tf.cast(
                    (tf.range(old_sz) * new_sz / old_sz)[:reminder], tf.int32
                ),
                (reminder, 1),
            )
            updates = tf.ones(len(upd_indices), dtype=tf.int32)
            repeats = tf.tensor_scatter_nd_add(
                input_indices, upd_indices, updates
            )
            image = tf.repeat(image, repeats, axis=ax)
    return image
