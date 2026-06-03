from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class LearnedWaveletConfig:
    image_size: int = 256
    channels: int = 1
    levels: int = 2
    kernel_size: int = 3
    keep_detail_fraction: float = 0.35
    learning_rate: float = 5.0e-4
    edge_weight: float = 0.20
    edge_energy_weight: float = 0.10
    filter_regularization: float = 2.0e-3
    coefficient_l1: float = 5.0e-5


def validate_config(cfg: LearnedWaveletConfig) -> None:
    if cfg.channels != 1:
        raise ValueError("The learned lifting codec currently expects grayscale images.")
    if cfg.kernel_size % 2 != 1:
        raise ValueError("--kernel-size must be odd.")
    if cfg.image_size % (2**cfg.levels) != 0:
        raise ValueError("--image-size must be divisible by 2**levels.")
    if not 0.0 < cfg.keep_detail_fraction <= 1.0:
        raise ValueError("--keep-detail-fraction must be in (0, 1].")


def init_params(cfg: LearnedWaveletConfig) -> dict:
    """Initialize a 2D lifting transform at a Haar-like local average."""
    validate_config(cfg)
    center = cfg.kernel_size // 2
    shape = (cfg.levels, 3, cfg.kernel_size, cfg.kernel_size)
    predict = jnp.zeros(shape, dtype=jnp.float32)
    update = jnp.zeros(shape, dtype=jnp.float32)
    predict = predict.at[:, :, center, center].set(1.0)
    update = update.at[:, :, center, center].set(0.25)
    return {"predict": predict, "update": update}


def analysis(params: dict, images: Array, cfg: LearnedWaveletConfig):
    """Map pixels to one coarse image and three detail bands at every scale."""
    coarse = images
    details = []
    for level in range(cfg.levels):
        even_even = coarse[:, 0::2, 0::2, :]
        even_odd = coarse[:, 0::2, 1::2, :]
        odd_even = coarse[:, 1::2, 0::2, :]
        odd_odd = coarse[:, 1::2, 1::2, :]
        sources = (even_odd, odd_even, odd_odd)
        level_details = tuple(
            source - _conv2d(even_even, params["predict"][level, band])
            for band, source in enumerate(sources)
        )
        coarse = even_even + sum(
            _conv2d(detail, params["update"][level, band])
            for band, detail in enumerate(level_details)
        )
        details.append(level_details)
    return coarse, tuple(details)


def synthesis(params: dict, coarse: Array, details, cfg: LearnedWaveletConfig) -> Array:
    """Reverse `analysis` exactly for any learned lifting filters."""
    restored = coarse
    for level in reversed(range(cfg.levels)):
        level_details = details[level]
        even_even = restored - sum(
            _conv2d(detail, params["update"][level, band])
            for band, detail in enumerate(level_details)
        )
        even_odd, odd_even, odd_odd = tuple(
            detail + _conv2d(even_even, params["predict"][level, band])
            for band, detail in enumerate(level_details)
        )
        restored = jnp.zeros(
            (even_even.shape[0], even_even.shape[1] * 2, even_even.shape[2] * 2, even_even.shape[3]),
            dtype=even_even.dtype,
        )
        restored = restored.at[:, 0::2, 0::2, :].set(even_even)
        restored = restored.at[:, 0::2, 1::2, :].set(even_odd)
        restored = restored.at[:, 1::2, 0::2, :].set(odd_even)
        restored = restored.at[:, 1::2, 1::2, :].set(odd_odd)
    return restored


def compress(params: dict, images: Array, cfg: LearnedWaveletConfig) -> dict:
    """Encode, retain a fixed detail budget, and decode an image batch."""
    coarse, details = analysis(params, images, cfg)
    sparse_details, indices, values = retain_details(details, cfg.keep_detail_fraction)
    reconstruction = synthesis(params, coarse, sparse_details, cfg)
    return {
        "coarse": coarse,
        "details": details,
        "sparse_details": sparse_details,
        "detail_indices": indices,
        "detail_values": values,
        "reconstruction": reconstruction,
    }


def retain_details(details, fraction: float):
    """Keep the strongest continuous detail coefficients for every image."""
    flat = flatten_details(details)
    keep = kept_detail_count(flat.shape[1], fraction)
    _, indices = jax.lax.top_k(jnp.abs(flat), keep)
    values = jnp.take_along_axis(flat, indices, axis=1)
    rows = jnp.arange(flat.shape[0])[:, None]
    sparse = jnp.zeros_like(flat).at[rows, indices].set(values)
    return unflatten_details(sparse, detail_shapes_from_values(details)), indices, values


def decode_compact(params: dict, coarse: Array, detail_indices: Array, detail_values: Array, cfg: LearnedWaveletConfig) -> Array:
    """Decode the compact coefficient archive written by the training CLI."""
    total = detail_coefficient_count(cfg.image_size, cfg.channels, cfg.levels)
    rows = jnp.arange(coarse.shape[0])[:, None]
    flat = jnp.zeros((coarse.shape[0], total), dtype=coarse.dtype)
    flat = flat.at[rows, detail_indices].set(detail_values)
    shapes = detail_shapes(cfg.image_size, cfg.channels, cfg.levels)
    return synthesis(params, coarse, unflatten_details(flat, shapes), cfg)


def flatten_details(details) -> Array:
    return jnp.concatenate([band.reshape((band.shape[0], -1)) for level in details for band in level], axis=1)


def unflatten_details(flat: Array, shapes):
    offset = 0
    result = []
    for level_shapes in shapes:
        level = []
        for shape in level_shapes:
            count = 1
            for size in shape:
                count *= size
            level.append(flat[:, offset : offset + count].reshape((flat.shape[0], *shape)))
            offset += count
        result.append(tuple(level))
    return tuple(result)


def detail_shapes_from_values(details):
    return tuple(tuple(band.shape[1:] for band in level) for level in details)


def detail_shapes(image_size: int, channels: int, levels: int):
    shapes = []
    size = image_size
    for _ in range(levels):
        size //= 2
        shapes.append(tuple((size, size, channels) for _ in range(3)))
    return tuple(shapes)


def detail_coefficient_count(image_size: int, channels: int, levels: int) -> int:
    return sum(size[0] * size[1] * size[2] for level in detail_shapes(image_size, channels, levels) for size in level)


def coarse_coefficient_count(image_size: int, channels: int, levels: int) -> int:
    side = image_size // (2**levels)
    return side * side * channels


def kept_detail_count(total: int, fraction: float) -> int:
    return max(1, min(total, int(round(total * fraction))))


def pack_coefficients(coarse: Array, details) -> Array:
    """Arrange multiscale coefficients as a familiar 2D wavelet image."""
    packed = coarse
    for level in reversed(range(len(details))):
        horizontal, vertical, diagonal = details[level]
        packed = jnp.concatenate(
            [jnp.concatenate([packed, horizontal], axis=2), jnp.concatenate([vertical, diagonal], axis=2)],
            axis=1,
        )
    return packed


def filter_regularization(params: dict, reference: dict) -> Array:
    return jnp.mean((params["predict"] - reference["predict"]) ** 2) + jnp.mean((params["update"] - reference["update"]) ** 2)


def save_checkpoint(path: str | Path, params: dict, cfg: LearnedWaveletConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": jax.device_get(params), "config": cfg.__dict__, "format": "learned_2d_lifting_wavelet_v1"}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "learned_2d_lifting_wavelet_v1":
        raise ValueError("Not a learned 2D lifting-wavelet checkpoint.")
    return jax.tree.map(jnp.asarray, payload["params"]), LearnedWaveletConfig(**payload["config"])


def _conv2d(images: Array, kernel: Array) -> Array:
    return jax.lax.conv_general_dilated(
        images,
        kernel[:, :, None, None],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
