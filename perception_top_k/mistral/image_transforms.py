"""Image preprocessing for the Mistral two-encoder contrastive model.

The model is variable-resolution (2D RoPE, no learned position embeddings), so it
accepts square *or* rectangular inputs as long as height and width are multiples of
``patch_size * spatial_merge_size`` (= 14 * 2 = 28).

Two transforms, matching the training/eval pipeline:

- ``build_squash_transform`` — resize directly to ``size x size`` (aspect ratio is
  distorted, no crop). This is what the bundled ``preprocessor_config.json`` /
  ``CLIPImageProcessor`` implements. Recommended for low / fixed resolution (e.g. 224, 336).
- ``build_keep_ratio_transform`` — resize the longest side to ``size`` keeping aspect
  ratio, rounding both sides down to a multiple of 28 so the ViT tiles exactly. Output is
  rectangular and variable-size. Recommended for high resolution. Because outputs vary in
  size, encode images one at a time (or batch only same-size images).

Normalization is OpenAI-CLIP statistics (``centering="default"`` in the training code).

Training used cv2 ``INTER_AREA`` resize; set ``use_cv2_resize=True`` to match it exactly,
otherwise torchvision bicubic is used (a close approximation).
"""

from collections.abc import Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
GRID_UNIT = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28: H and W must be multiples of this


def _cv2_resize(img: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    import cv2  # local import so cv2 is only required when use_cv2_resize=True

    np_img = img.permute(1, 2, 0).numpy()
    np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()


class _SquashResize(torch.nn.Module):
    def __init__(self, size: int, use_cv2_resize: bool) -> None:
        super().__init__()
        self.size = size
        self.use_cv2_resize = use_cv2_resize

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.use_cv2_resize:
            return _cv2_resize(img, self.size, self.size)
        return TF.resize(
            img, [self.size, self.size], interpolation=InterpolationMode.BICUBIC, antialias=True
        )


class _KeepRatioResize(torch.nn.Module):
    def __init__(self, max_size: int, use_cv2_resize: bool) -> None:
        super().__init__()
        self.max_size = max_size
        self.use_cv2_resize = use_cv2_resize

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        scale = self.max_size / max(h, w)
        new_h = max(GRID_UNIT, (int(h * scale) // GRID_UNIT) * GRID_UNIT)
        new_w = max(GRID_UNIT, (int(w * scale) // GRID_UNIT) * GRID_UNIT)
        if self.use_cv2_resize:
            return _cv2_resize(img, new_h, new_w)
        return TF.resize(
            img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True
        )


def build_squash_transform(size: int = 336, use_cv2_resize: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    """Resize to ``size x size`` (aspect ratio distorted) then normalize. Returns (3, size, size)."""
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            _SquashResize(size, use_cv2_resize),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def build_keep_ratio_transform(
    size: int = 672, use_cv2_resize: bool = False
) -> Callable[[Image.Image], torch.Tensor]:
    """Resize longest side to ``size`` keeping aspect ratio (rounded to multiples of 28), then normalize.

    Output is rectangular and varies per image, so encode one image at a time.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            _KeepRatioResize(size, use_cv2_resize),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
