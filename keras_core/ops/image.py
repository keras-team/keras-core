from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation


class Resize(Operation):
    def __init__(
        self,
        size,
        method="bilinear",
        antialias=False,
        data_format="channels_last",
    ):
        super().__init__()
        self.size = tuple(size)
        self.method = method
        self.antialias = antialias
        self.data_format = data_format

    def call(self, image):
        return backend.image.resize(
            image,
            self.size,
            method=self.method,
            antialias=self.antialias,
            data_format=self.data_format,
        )

    def compute_output_spec(self, image):
        if len(image.shape) == 3:
            return KerasTensor(
                self.size + (image.shape[-1],), dtype=image.dtype
            )
        elif len(image.shape) == 4:
            if self.data_format == "channels_last":
                return KerasTensor(
                    (image.shape[0],) + self.size + (image.shape[-1],),
                    dtype=image.dtype,
                )
            else:
                return KerasTensor(
                    (image.shape[0], image.shape[1]) + self.size,
                    dtype=image.dtype,
                )
        raise ValueError(
            "Invalid input rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )


@keras_core_export("keras_core.ops.image.resize")
def resize(
    image, size, method="bilinear", antialias=False, data_format="channels_last"
):
    
    """
    Resizes the image to the specified size.

    Args:
        image (Tensor): An input tensor representing an image or a batch of images.
        size (Tuple[int, int]): A tuple of two integers, (height, width), representing the new size of the image.
        method (str, optional): An interpolation method. One of "bilinear", "nearest", "bicubic", "area", or "lanczos3". Default is "bilinear".
        antialias (bool, optional): Whether to use an anti-aliasing filter when downsampling an image. Default is False.
        data_format (str, optional): A string, one of "channels_last" (default) or "channels_first".

    Returns:
        Tensor: The resized image(s).

    Example:
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> resize(img, (2, 2))
        array([[1., 3.],
               [7., 9.]])
    """
    
    if any_symbolic_tensors((image,)):
        return Resize(
            size, method=method, antialias=antialias, data_format=data_format
        ).symbolic_call(image)
    return backend.image.resize(
        image, size, method=method, antialias=antialias, data_format=data_format
    )
