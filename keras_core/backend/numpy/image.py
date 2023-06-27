import numpy as np
from PIL import Image

RESIZE_METHODS = {
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "bicubic": Image.BICUBIC,
}

UNSUPPORTED_METHODS = (
    "lanczos3",
    "lanczos5",
)


def resize_single_image(image, size, method):
    # Check the dtype
    # Covert numpy to PIL
    # Resize the image
    # Convert PIL to numpy
    pass


def resize(
    image,
    size,
    method="bilinear",
    antialias=False,
    data_format="channels_last",
):
    if method in UNSUPPORTED_METHODS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the NumPy backend. "
            f"Received: method={method}."
        )
    if method not in RESIZE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{RESIZE_METHODS}. Received: method={method}"
        )
    if method in ["bilinear", "bicubic"] and not antialias:
        raise ValueError(
            "Anti-aliasing must be enabled when using bilinear or "
            "bicubic interpolation. Received: antialias={antialias}."
        )
    if method == "nearest" and antialias:
        raise ValueError(
            "Anti-aliasing is not supported with nearest neighbor "
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    if data_format == "channels_first":
        if len(image.shape) == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    # PIL does not support batching of images. We need to loop over the batch
    # of images and resize them one by one.
    if len(image.shape) == 4:
        resized_images = [resize_single_image(x) for x in image]
        resized = np.stack(resized_images, axis=0)
    else:
        resized = resize_single_image(image)

    if data_format == "channels_first":
        if len(image.shape) == 4:
            resized = np.transpose(resized, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            resized = np.transpose(resized, (2, 0, 1))

    return resized
