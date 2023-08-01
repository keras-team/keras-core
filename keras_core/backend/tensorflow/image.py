from keras_core.utils.module_utils import tensorflow as tf
import numpy as np

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    data_format="channels_last",
):
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
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

    resized = tf.image.resize(
        image, size, method=interpolation, antialias=antialias
    )
    if data_format == "channels_first":
        if len(image.shape) == 4:
            resized = tf.transpose(resized, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            resized = tf.transpose(resized, (2, 0, 1))
    return resized


def decode_image(img, channels, expand_animations=False):
    return tf.image.decode_image(
        img, channels=channels, expand_animations=expand_animations
    )

def read_file(path):
    return tf.io.read_file(path)

AFFINE_TRANSFORM_INTERPOLATIONS = (
    "nearest",
    "bilinear",
)
AFFINE_TRANSFORM_FILL_MODES = (
    "constant",
    "nearest",
    "wrap",
    # "mirror", not supported by TF
    "reflect",
)

def smart_resize(x, size, interpolation="bilinear"):
    """Resize images to a target size without aspect ratio distortion.

    Warning: `tf.keras_core.backend.image.smart_resize` is not recommended for
    new code. Prefer `tf.keras.layers.Resizing`, which provides the same
    functionality as a preprocessing layer and adds `tf.RaggedTensor` support.
    See the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers)
    for an overview of preprocessing layers.

    TensorFlow image datasets typically yield images that have each a different
    size. However, these images need to be batched before they can be
    processed by Keras layers. To be batched, images need to share the same
    height and width.

    You could simply do:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: tf.image.resize(img, size))
    ```

    However, if you do this, you distort the aspect ratio of your images, since
    in general they do not all have the same aspect ratio as `size`. This is
    fine in many cases, but not always (e.g. for GANs this can be a problem).

    Note that passing the argument `preserve_aspect_ratio=True` to `resize`
    will preserve the aspect ratio, but at the cost of no longer respecting the
    provided target size. Because `tf.image.resize` doesn't crop images,
    your output images will still have different sizes.

    This calls for:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: smart_resize(img, size))
    ```

    Your output images will actually be `(200, 200)`, and will not be distorted.
    Instead, the parts of the image that do not fit within the target size
    get cropped out.

    The resizing process is:

    1. Take the largest centered crop of the image that has the same aspect
    ratio as the target size. For instance, if `size=(200, 200)` and the input
    image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
    the width.
    2. Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.

    Args:
      x: Input image or batch of images (as a tensor or NumPy array). Must be in
        format `(height, width, channels)` or `(batch_size, height, width,
        channels)`.
      size: Tuple of `(height, width)` integer. Target size.
      interpolation: String, interpolation to use for resizing. Defaults to
        `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`, `area`,
        `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.

    Returns:
      Array with shape `(size[0], size[1], channels)`. If the input image was a
      NumPy array, the output is a NumPy array, and if it was a TF tensor,
      the output is a TF tensor.
    """
    if len(size) != 2:
        raise ValueError(
            f"Expected `size` to be a tuple of 2 integers, but got: {size}."
        )
    img = tf.convert_to_tensor(x)
    if img.shape.rank is not None:
        if img.shape.rank < 3 or img.shape.rank > 4:
            raise ValueError(
                "Expected an image array with shape `(height, width, "
                "channels)`, or `(batch_size, height, width, channels)`, but "
                f"got input with incorrect rank, of shape {img.shape}."
            )
    shape = tf.shape(img)
    height, width = shape[-3], shape[-2]
    target_height, target_width = size
    if img.shape.rank is not None:
        static_num_channels = img.shape[-1]
    else:
        static_num_channels = None

    crop_height = tf.cast(
        tf.cast(width * target_height, "float32") / target_width, "int32"
    )
    crop_width = tf.cast(
        tf.cast(height * target_width, "float32") / target_height, "int32"
    )

    # Set back to input height / width if crop_height / crop_width is not
    # smaller.
    crop_height = tf.minimum(height, crop_height)
    crop_width = tf.minimum(width, crop_width)

    crop_box_hstart = tf.cast(
        tf.cast(height - crop_height, "float32") / 2, "int32"
    )
    crop_box_wstart = tf.cast(
        tf.cast(width - crop_width, "float32") / 2, "int32"
    )

    if img.shape.rank == 4:
        crop_box_start = tf.stack([0, crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([-1, crop_height, crop_width, -1])
    else:
        crop_box_start = tf.stack([crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([crop_height, crop_width, -1])

    img = tf.slice(img, crop_box_start, crop_box_size)
    img = tf.image.resize(images=img, size=size, method=interpolation)
    # Apparent bug in resize_images_v2 may cause shape to be lost
    if img.shape.rank is not None:
        if img.shape.rank == 4:
            img.set_shape((None, None, None, static_num_channels))
        if img.shape.rank == 3:
            img.set_shape((None, None, static_num_channels))
    if isinstance(x, np.ndarray):
        return img.numpy()
    return img

def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
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
    # unbatched case
    need_squeeze = False
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = tf.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        image = tf.transpose(image, (0, 2, 3, 1))

    affined = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.cast(transform, dtype=tf.float32),
        output_shape=tf.shape(image)[1:-1],
        fill_value=fill_value,
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
    )

    if data_format == "channels_first":
        affined = tf.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = tf.squeeze(affined, axis=0)
    return affined