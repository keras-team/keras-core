import torch
from torch.nn import functional as F

from keras_core.backend.torch.core import convert_to_tensor

RESIZE_METHODS = {}  # populated after torchvision import

UNSUPPORTED_METHODS = (
    "lanczos3",
    "lanczos5",
)


def resize(
    image, size, method="bilinear", antialias=False, data_format="channels_last"
):
    try:
        import torchvision
        from torchvision.transforms import InterpolationMode as im

        RESIZE_METHODS.update(
            {
                "bilinear": im.BILINEAR,
                "nearest": im.NEAREST_EXACT,
                "bicubic": im.BICUBIC,
            }
        )
    except:
        raise ImportError(
            "The torchvision package is necessary to use `resize` with the "
            "torch backend. Please install torchvision."
        )
    if method in UNSUPPORTED_METHODS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the PyTorch backend. "
            f"Received: method={method}."
        )
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
    image = convert_to_tensor(image)
    if data_format == "channels_last":
        if image.ndim == 4:
            image = image.permute((0, 3, 1, 2))
        elif image.ndim == 3:
            image = image.permute((2, 0, 1))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    resized = torchvision.transforms.functional.resize(
        img=image,
        size=size,
        interpolation=RESIZE_METHODS[method],
        antialias=antialias,
    )
    if data_format == "channels_last":
        if len(image.shape) == 4:
            resized = resized.permute((0, 2, 3, 1))
        elif len(image.shape) == 3:
            resized = resized.permute((1, 2, 0))
    return resized


AFFINE_METHODS = (
    "nearest",
    "bilinear",
)
AFFINE_FILL_MODES = {
    "constant": "zeros",
    "nearest": "border",
    # "wrap",  not supported by torch
    # "mirror",  not supported by torch
    "reflect": "reflection",
}


def _apply_grid_transform(
    img,
    grid,
    method="bilinear",
    fill_mode="zeros",
    fill_value=None,
):
    """
    Modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_geometry.py
    """  # noqa: E501

    # We are using context knowledge that grid should have float dtype
    fp = img.dtype == grid.dtype
    float_img = img if fp else img.to(grid.dtype)

    shape = float_img.shape
    # Append a dummy mask for customized fill colors, should be faster than
    # grid_sample() twice
    if fill_value is not None:
        mask = torch.ones(
            (shape[0], 1, shape[2], shape[3]),
            dtype=float_img.dtype,
            device=float_img.device,
        )
        float_img = torch.cat((float_img, mask), dim=1)

    float_img = F.grid_sample(
        float_img,
        grid,
        mode=method,
        padding_mode=fill_mode,
        align_corners=False,
    )
    # Fill with required color
    if fill_value is not None:
        float_img, mask = torch.tensor_split(float_img, indices=(-1,), dim=-3)
        mask = mask.expand_as(float_img)
        fill_list = (
            fill_value
            if isinstance(fill_value, (tuple, list))
            else [float(fill_value)]
        )
        fill_img = torch.tensor(
            fill_list, dtype=float_img.dtype, device=float_img.device
        ).view(1, -1, 1, 1)
        if method == "nearest":
            bool_mask = mask < 0.5
            float_img[bool_mask] = fill_img.expand_as(float_img)[bool_mask]
        else:  # 'bilinear'
            # The following is mathematically equivalent to:
            # img * mask + (1.0 - mask) * fill =
            # img * mask - fill * mask + fill =
            # mask * (img - fill) + fill
            float_img = float_img.sub_(fill_img).mul_(mask).add_(fill_img)

    img = float_img.round_().to(img.dtype) if not fp else float_img
    return img


def affine(
    image,
    transform,
    method="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if method not in AFFINE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{AFFINE_METHODS}. Received: method={method}"
        )
    if fill_mode not in AFFINE_FILL_MODES.keys():
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_FILL_MODES.keys()}. Received: method={fill_mode}"
        )

    image = convert_to_tensor(image)
    transform = convert_to_tensor(transform)

    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    if transform.ndim not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    if fill_mode != "constant":
        fill_value = None
    fill_mode = AFFINE_FILL_MODES[fill_mode]

    # unbatched case
    need_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(dim=0)
        need_squeeze = True
    if transform.ndim == 1:
        transform = transform.unsqueeze(dim=0)

    if data_format == "channels_last":
        image = image.permute((0, 3, 1, 2))

    # deal with transform
    h, w = image.shape[1], image.shape[2]
    theta = torch.zeros((image.shape[0], 2, 3)).to(transform)
    theta[:, 0, 0] = transform[:, 0]
    theta[:, 0, 1] = transform[:, 1] * h / w
    theta[:, 0, 2] = (
        transform[:, 2] * 2 / w + theta[:, 0, 0] + theta[:, 0, 1] - 1
    )
    theta[:, 1, 0] = transform[:, 3] * w / h
    theta[:, 1, 1] = transform[:, 4]
    theta[:, 1, 2] = (
        transform[:, 5] * 2 / h + theta[:, 1, 0] + theta[:, 1, 1] - 1
    )

    grid = F.affine_grid(theta, image.shape)
    affined = _apply_grid_transform(image, grid, method, fill_mode, fill_value)

    if data_format == "channels_last":
        affined = affined.permute((0, 2, 3, 1))
    if need_squeeze:
        affined = affined.squeeze(dim=0)
    return affined
