from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def load_npz_images(path: str | os.PathLike) -> np.ndarray:
    data = np.load(path)
    key = "images" if "images" in data else data.files[0]
    images = data[key].astype(np.float32)
    if images.ndim == 3:
        images = images[..., None]
    if images.max() > 1.5:
        images = images / 255.0
    return images


def save_pgm(path: str | os.PathLike, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    header = f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(img.tobytes())



def save_png(path: str | os.PathLike, image: np.ndarray) -> None:
    pil = importlib.util.find_spec("PIL")
    if pil is None:
        return
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., 0]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)

def load_image_any(path: str | os.PathLike) -> np.ndarray:
    """Load an image using whatever optional decoder is installed.

    The base devcontainer has JAX/NumPy/SciPy but no JPEG decoder. This function
    keeps preprocessing flexible: install Pillow, imageio, OpenCV, or matplotlib
    and the same CLI will work.
    """
    path = str(path)
    pil = importlib.util.find_spec("PIL")
    if pil is not None:
        from PIL import Image

        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    imageio = importlib.util.find_spec("imageio")
    if imageio is not None:
        import imageio.v3 as iio

        img = iio.imread(path)
        return _to_gray_float(img)

    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec is not None:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img.astype(np.float32) / 255.0

    mpl = importlib.util.find_spec("matplotlib")
    if mpl is not None:
        import matplotlib.image as mpimg

        return _to_gray_float(mpimg.imread(path))

    raise RuntimeError(
        "No JPEG decoder is installed. Install one lightweight decoder, e.g. "
        "`python -m pip install pillow`, then rerun preprocessing."
    )


def resize_area(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a grayscale image, preferring Pillow's high-quality resampler."""
    pil = importlib.util.find_spec("PIL")
    if pil is not None:
        from PIL import Image

        arr = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
        resample = getattr(Image.Resampling, "LANCZOS", Image.BICUBIC)
        resized = Image.fromarray(arr, mode="L").resize((size, size), resample=resample)
        return np.asarray(resized, dtype=np.float32) / 255.0

    h, w = image.shape
    y_edges = np.linspace(0, h, size + 1).astype(np.int64)
    x_edges = np.linspace(0, w, size + 1).astype(np.int64)
    out = np.empty((size, size), dtype=np.float32)
    for yi in range(size):
        y0, y1 = y_edges[yi], max(y_edges[yi + 1], y_edges[yi] + 1)
        for xi in range(size):
            x0, x1 = x_edges[xi], max(x_edges[xi + 1], x_edges[xi] + 1)
            out[yi, xi] = image[y0:y1, x0:x1].mean()
    return out


def build_dataset(paths: Iterable[str], size: int) -> np.ndarray:
    images = []
    for path in paths:
        images.append(resize_area(load_image_any(path), size))
    return np.stack(images, axis=0)[..., None].astype(np.float32)


def sorted_images(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr

