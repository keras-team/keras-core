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
    # TODO: add docstring
    if any_symbolic_tensors((image,)):
        return Resize(
            size, method=method, antialias=antialias, data_format=data_format
        ).symbolic_call(image)
    return backend.image.resize(
        image, size, method=method, antialias=antialias, data_format=data_format
    )


class Affine(Operation):
    def __init__(
        self,
        method="bilinear",
        fill_mode="constant",
        fill_value=0,
        data_format="channels_last",
    ):
        super().__init__()
        self.method = method
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = data_format

    def call(self, image, transform):
        return backend.image.affine(
            image,
            transform,
            method=self.method,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

    def compute_output_spec(self, image, transform):
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
        return KerasTensor(image.shape, dtype=image.dtype)


@keras_core_export("keras_core.ops.image.affine")
def affine(
    image,
    transform,
    method="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    # TODO: add docstring
    if any_symbolic_tensors((image, transform)):
        return Affine(
            method=method,
            fill_mode=fill_mode,
            fill_value=fill_value,
            data_format=data_format,
        ).symbolic_call(image, transform)
    return backend.image.affine(
        image,
        transform,
        method=method,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format,
    )
