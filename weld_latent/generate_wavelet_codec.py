from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .io import load_npz_images, save_pgm, save_png
from .learned_wavelet import detail_coefficient_count, detail_shapes, load_checkpoint, synthesis, unflatten_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe new-image synthesis by remixing a learned 2D wavelet latent archive.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out", default="runs/learned_wavelet_generated")
    parser.add_argument("--dataset", default=None, help="Optional source dataset for donor images and novelty metrics.")
    parser.add_argument("--metric", default=None, help="Optional Grassmannian wavelet metric archive from fit_wavelet_metric.")
    parser.add_argument("--prior", default=None, help="Optional adaptive KDE archive from fit_wavelet_prior.")
    parser.add_argument("--metric-neighbors", choices=["grassmann", "embedding"], default="embedding")
    parser.add_argument("--strategy", choices=["auto", "metric-density", "local-interpolate", "crossover"], default="auto")
    parser.add_argument("--detail-interpolation", choices=["slerp", "linear"], default="slerp")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--interpolation-steps", type=int, default=9)
    parser.add_argument("--coarse-mix", choices=["hard", "soft"], default="hard")
    parser.add_argument("--detail-mix", choices=["hard", "soft"], default="soft")
    parser.add_argument("--detail-granularity", choices=["level", "band"], default="level")
    parser.add_argument("--neighbor-pool", type=int, default=16, help="Choose detail donors from this many metric neighbors; use 0 for global remixing.")
    parser.add_argument("--inverse-neighbors", type=int, default=2, help="Wavelet latents used to decode each sampled metric coordinate.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    if args.interpolation_steps < 2:
        raise ValueError("--interpolation-steps must be at least 2.")
    if args.neighbor_pool < 0:
        raise ValueError("--neighbor-pool must be non-negative.")
    if args.inverse_neighbors < 2:
        raise ValueError("--inverse-neighbors must be at least 2.")
    strategy = "metric-density" if args.strategy == "auto" and args.prior else "local-interpolate" if args.strategy == "auto" else args.strategy
    if strategy == "metric-density" and (not args.metric or not args.prior):
        raise ValueError("--strategy metric-density requires both --metric and --prior.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params, cfg = load_checkpoint(args.checkpoint)
    archive = load_latent_archive(args.codes)
    coarse = archive["coarse"]
    indices = archive["indices"]
    values = archive["values"]
    total_details = detail_coefficient_count(cfg.image_size, cfg.channels, cfg.levels)
    dense_details = expand_sparse_details(indices, values, total_details)
    metric = load_metric_archive(args.metric, coarse.shape[0]) if args.metric else None
    prior = load_prior_archive(args.prior, metric) if args.prior else None
    metric_neighbors = metric[f"{args.metric_neighbors}_neighbor_indices"] if metric else None
    rng = np.random.default_rng(args.seed)
    sampled_coordinates = None
    density_anchors = None

    if strategy == "metric-density":
        generated, donors, mixing, sampled_coordinates, generated_coordinates, density_anchors = generate_metric_density(
            params, cfg, coarse, dense_details, indices.shape[1], args.samples, args.inverse_neighbors, rng, metric, prior
        )
    elif strategy == "local-interpolate":
        generated, donors, mixing, generated_coordinates = generate_local_interpolations(
            params, cfg, coarse, dense_details, indices.shape[1], args.samples, args.neighbor_pool, args.detail_interpolation, rng, metric, metric_neighbors
        )
    else:
        generated, donors = generate_crossovers(
            params, cfg, coarse, dense_details, indices.shape[1], args.samples, args.coarse_mix, args.detail_mix,
            args.detail_granularity, args.neighbor_pool, rng, metric_neighbors
        )
        mixing = None
        generated_coordinates = None
    interpolation, interpolation_sources = generate_interpolation(params, cfg, coarse, dense_details, args.interpolation_steps, rng, metric_neighbors, args.neighbor_pool)
    save_images(out, "generated", generated)
    save_images(out, "interpolation", interpolation)
    save_montage(out / "generated_montage.png", generated)
    save_montage(out / "interpolation_montage.png", interpolation)

    metrics = {
        "strategy": strategy,
        "metric": args.metric,
        "prior": args.prior,
        "metric_neighbors": args.metric_neighbors,
        "detail_interpolation": args.detail_interpolation,
        "coarse_mix": args.coarse_mix,
        "detail_mix": args.detail_mix,
        "detail_granularity": args.detail_granularity,
        "neighbor_pool": int(args.neighbor_pool),
        "inverse_neighbors": int(args.inverse_neighbors),
        "note": "Metric-density samples an adaptive KDE in the Grassmann/MDS embedding and decodes through compatible local wavelet latents. Crossover remains a stress diagnostic.",
        "samples": int(generated.shape[0]),
        "interpolation_steps": int(interpolation.shape[0]),
        "interpolation_source_indices": [int(value) for value in interpolation_sources],
        "donor_indices": donors.tolist(),
        "mixing_weights": mixing.tolist() if mixing is not None else None,
        "density_anchor_indices": density_anchors.tolist() if density_anchors is not None else None,
        "generated_mean": float(np.mean(generated)),
        "generated_std": float(np.std(generated)),
        "generated_edge_energy": float(edge_energy(generated)),
        "generated_pairwise_mse": float(mean_pairwise_mse(generated)),
        "generated_image_sanity": image_sanity(generated),
    }
    if args.dataset:
        source = load_npz_images(args.dataset)
        if source.shape[0] != coarse.shape[0] or source.shape[1:] != generated.shape[1:]:
            raise ValueError(f"Dataset shape {source.shape} is incompatible with archive and generated images {generated.shape}.")
        nearest_indices, nearest_mse = nearest_sources(generated, source)
        metrics.update(
            {
                "source_mean": float(np.mean(source)),
                "source_std": float(np.std(source)),
                "source_edge_energy": float(edge_energy(source)),
                "source_image_sanity": image_sanity(source),
                "generated_over_source_edge_energy": float(edge_energy(generated) / (edge_energy(source) + 1.0e-12)),
                "nearest_source_indices": nearest_indices.tolist(),
                "nearest_source_mse_downsampled": nearest_mse.tolist(),
                "mean_nearest_source_mse_downsampled": float(np.mean(nearest_mse)),
            }
        )
        save_images(out, "nearest_source", source[nearest_indices])
        save_images(out, "coarse_donor", source[donors[:, 0]])
        save_montage(out / "nearest_source_montage.png", source[nearest_indices])
        save_montage(out / "coarse_donor_montage.png", source[donors[:, 0]])
    if generated_coordinates is not None:
        np.savetxt(out / "generated_metric_coordinates.csv", generated_coordinates, delimiter=",")
    if sampled_coordinates is not None:
        np.savetxt(out / "sampled_metric_coordinates.csv", sampled_coordinates, delimiter=",")
        metrics["metric_coordinate_reconstruction_mse"] = float(np.mean((sampled_coordinates - generated_coordinates) ** 2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote wavelet generation probe to {out}")


def load_latent_archive(path: str | Path) -> dict:
    archive = np.load(path)
    coarse = archive["coarse"].astype(np.float32)
    if "detail_index_deltas" in archive:
        indices = np.cumsum(archive["detail_index_deltas"].astype(np.uint32), axis=1)
    else:
        indices = archive["detail_indices"].astype(np.uint32)
    values = archive["detail_values"].astype(np.float32)
    if not (coarse.shape[0] == indices.shape[0] == values.shape[0]):
        raise ValueError("Compact latent arrays must have the same image count.")
    return {"coarse": coarse, "indices": indices, "values": values}


def load_metric_archive(path: str | Path, images: int) -> dict:
    archive = np.load(path)
    embedding = archive["embedding"].astype(np.float32)
    neighbors = archive["neighbor_indices"].astype(np.int32)
    grassmann_neighbors = archive["grassmann_neighbor_indices"].astype(np.int32)
    if embedding.shape[0] != images or neighbors.shape[0] != images:
        raise ValueError(f"Metric archive has {embedding.shape[0]} images; latent archive has {images}.")
    return {
        "embedding": embedding,
        "embedding_neighbor_indices": neighbors,
        "grassmann_neighbor_indices": grassmann_neighbors,
    }


def load_prior_archive(path: str | Path, metric: dict | None) -> dict:
    if metric is None:
        raise ValueError("A metric archive is required when loading a metric-space prior.")
    archive = np.load(path)
    embedding = archive["embedding"].astype(np.float32)
    if embedding.shape != metric["embedding"].shape or not np.allclose(embedding, metric["embedding"], atol=1.0e-5):
        raise ValueError("Prior and metric embeddings do not match.")
    weights = archive["component_weights"].astype(np.float64)
    weights /= weights.sum()
    return {
        "embedding": embedding,
        "local_cholesky": archive["local_cholesky"].astype(np.float32),
        "component_weights": weights,
        "bandwidth_scale": float(archive["bandwidth_scale"]),
    }


def expand_sparse_details(indices: np.ndarray, values: np.ndarray, total: int) -> np.ndarray:
    flat = np.zeros((indices.shape[0], total), dtype=np.float32)
    np.put_along_axis(flat, indices, values, axis=1)
    return flat


def generate_metric_density(params, cfg, coarse: np.ndarray, details: np.ndarray, keep: int, samples: int, inverse_neighbors: int, rng, metric: dict, prior: dict):
    """Sample the fitted metric-space KDE and decode through nearby wavelet latents."""
    anchors = rng.choice(prior["embedding"].shape[0], size=samples, p=prior["component_weights"])
    noise = rng.normal(size=(samples, prior["embedding"].shape[1])).astype(np.float32)
    offsets = np.einsum("nij,nj->ni", prior["local_cholesky"][anchors], noise, optimize=True)
    coordinates = prior["embedding"][anchors] + prior["bandwidth_scale"] * offsets
    donors, weights = local_coordinate_weights(coordinates, metric["embedding"], inverse_neighbors)
    generated_coarse = np.sum(weights[:, :, None, None, None] * coarse[donors], axis=1)
    generated_details = spherical_barycenter(details[donors], weights)
    generated_details = retain_strongest(generated_details, keep)
    decoded = np.clip(decode_dense(params, cfg, generated_coarse, generated_details), 0.0, 1.0)
    reconstructed = np.sum(weights[:, :, None] * metric["embedding"][donors], axis=1)
    return decoded, donors, weights, coordinates, reconstructed, anchors.astype(np.int32)


def local_coordinate_weights(coordinates: np.ndarray, embedding: np.ndarray, neighbors: int):
    """Map a sampled metric coordinate to a small compatible wavelet neighborhood."""
    count = min(neighbors, embedding.shape[0])
    distances = np.sum((coordinates[:, None, :] - embedding[None, :, :]) ** 2, axis=2)
    donors = np.argsort(distances, axis=1)[:, :count]
    selected = np.take_along_axis(distances, donors, axis=1)
    bandwidth = np.maximum(selected[:, -1:], 1.0e-12)
    scores = -0.5 * selected / bandwidth
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return donors.astype(np.int32), weights.astype(np.float32)


def spherical_barycenter(details: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Blend directions on the detail sphere while retaining local texture magnitude."""
    if details.shape[1] == 2:
        return spherical_interpolate(details[:, 0], details[:, 1], weights[:, 1])
    norms = np.linalg.norm(details, axis=2, keepdims=True) + 1.0e-12
    direction = np.sum(weights[:, :, None] * details / norms, axis=1)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1.0e-12
    radius = np.sum(weights * norms[:, :, 0], axis=1, keepdims=True)
    return (radius * direction).astype(np.float32)


def generate_local_interpolations(params, cfg, coarse: np.ndarray, details: np.ndarray, keep: int, samples: int, neighbor_pool: int, detail_interpolation: str, rng, metric=None, metric_neighbors=None):
    donors = select_donors(coarse, samples, 2, neighbor_pool, rng, metric_neighbors)
    mixing = rng.uniform(0.25, 0.75, size=samples).astype(np.float32)
    coarse_weight = mixing[:, None, None, None]
    detail_weight = mixing[:, None]
    generated_coarse = (1.0 - coarse_weight) * coarse[donors[:, 0]] + coarse_weight * coarse[donors[:, 1]]
    if detail_interpolation == "slerp":
        generated_details = spherical_interpolate(details[donors[:, 0]], details[donors[:, 1]], mixing)
    else:
        generated_details = (1.0 - detail_weight) * details[donors[:, 0]] + detail_weight * details[donors[:, 1]]
    generated_details = retain_strongest(generated_details, keep)
    decoded = np.clip(decode_dense(params, cfg, generated_coarse, generated_details), 0.0, 1.0)
    coordinates = None
    if metric:
        coordinates = (1.0 - mixing[:, None]) * metric["embedding"][donors[:, 0]] + mixing[:, None] * metric["embedding"][donors[:, 1]]
    return decoded, donors, mixing, coordinates


def spherical_interpolate(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Move between detail vectors along a norm-preserving spherical arc."""
    left_norm = np.linalg.norm(left, axis=1, keepdims=True) + 1.0e-12
    right_norm = np.linalg.norm(right, axis=1, keepdims=True) + 1.0e-12
    left_unit = left / left_norm
    right_unit = right / right_norm
    cosine = np.clip(np.sum(left_unit * right_unit, axis=1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    weight = alpha[:, None]
    linear = (1.0 - weight) * left_unit + weight * right_unit
    spherical = (np.sin((1.0 - weight) * angle) * left_unit + np.sin(weight * angle) * right_unit) / (sine + 1.0e-12)
    direction = np.where(np.abs(sine) < 1.0e-5, linear, spherical)
    radius = (1.0 - weight) * left_norm + weight * right_norm
    return (radius * direction).astype(np.float32)


def generate_crossovers(params, cfg, coarse: np.ndarray, details: np.ndarray, keep: int, samples: int, coarse_mix: str, detail_mix: str, detail_granularity: str, neighbor_pool: int, rng, metric_neighbors=None):
    shapes = detail_shapes(cfg.image_size, cfg.channels, cfg.levels)
    groups = cfg.levels if detail_granularity == "level" else cfg.levels * 3
    donors = select_donors(coarse, samples, 2 + 2 * groups, neighbor_pool, rng, metric_neighbors)
    if coarse_mix == "hard":
        generated_coarse = coarse[donors[:, 0]]
    else:
        coarse_weight = rng.uniform(0.25, 0.75, size=(samples, 1, 1, 1)).astype(np.float32)
        generated_coarse = coarse_weight * coarse[donors[:, 0]] + (1.0 - coarse_weight) * coarse[donors[:, 1]]
    generated_details = np.zeros((samples, details.shape[1]), dtype=np.float32)
    group_weights = rng.uniform(0.15, 0.85, size=(samples, groups)).astype(np.float32)
    offset = 0
    for level, level_shapes in enumerate(shapes):
        for band, shape in enumerate(level_shapes):
            count = int(np.prod(shape))
            group = level if detail_granularity == "level" else level * 3 + band
            left = details[donors[:, 2 + 2 * group], offset : offset + count]
            if detail_mix == "hard":
                generated_details[:, offset : offset + count] = left
            else:
                weight = group_weights[:, group : group + 1]
                right = details[donors[:, 3 + 2 * group], offset : offset + count]
                generated_details[:, offset : offset + count] = weight * left + (1.0 - weight) * right
            offset += count
    generated_details = retain_strongest(generated_details, keep)
    decoded = decode_dense(params, cfg, generated_coarse, generated_details)
    return np.clip(decoded, 0.0, 1.0), donors


def select_donors(coarse: np.ndarray, samples: int, donors_per_sample: int, neighbor_pool: int, rng, metric_neighbors=None) -> np.ndarray:
    anchors = rng.integers(0, coarse.shape[0], size=samples, dtype=np.int32)
    donors = np.empty((samples, donors_per_sample), dtype=np.int32)
    donors[:, 0] = anchors
    if neighbor_pool == 0:
        donors[:, 1:] = rng.integers(0, coarse.shape[0], size=(samples, donors_per_sample - 1), dtype=np.int32)
        return donors
    flattened = None if metric_neighbors is not None else coarse[:, ::2, ::2, :].reshape((coarse.shape[0], -1)).astype(np.float64)
    for row, anchor in enumerate(anchors):
        if metric_neighbors is not None:
            count = min(neighbor_pool, metric_neighbors.shape[1])
            neighbors = metric_neighbors[anchor, :count]
        else:
            distances = np.mean((flattened - flattened[anchor]) ** 2, axis=1)
            count = min(neighbor_pool, coarse.shape[0] - 1)
            neighbors = np.argsort(distances)[1 : count + 1]
        donors[row, 1:] = rng.choice(neighbors, size=donors_per_sample - 1, replace=True)
    return donors


def generate_interpolation(params, cfg, coarse: np.ndarray, details: np.ndarray, steps: int, rng, metric_neighbors=None, neighbor_pool: int = 32):
    if metric_neighbors is None or neighbor_pool == 0:
        sources = rng.choice(coarse.shape[0], size=2, replace=False)
    else:
        anchor = int(rng.integers(0, coarse.shape[0]))
        count = min(neighbor_pool, metric_neighbors.shape[1])
        sources = np.array([anchor, rng.choice(metric_neighbors[anchor, :count])], dtype=np.int32)
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    coarse_alpha = alpha[:, None, None, None]
    detail_alpha = alpha[:, None]
    mixed_coarse = (1.0 - coarse_alpha) * coarse[sources[0]] + coarse_alpha * coarse[sources[1]]
    mixed_details = (1.0 - detail_alpha) * details[sources[0]] + detail_alpha * details[sources[1]]
    return np.clip(decode_dense(params, cfg, mixed_coarse, mixed_details), 0.0, 1.0), sources


def retain_strongest(details: np.ndarray, keep: int) -> np.ndarray:
    if keep >= details.shape[1]:
        return details
    indices = np.argpartition(np.abs(details), -keep, axis=1)[:, -keep:]
    sparse = np.zeros_like(details)
    np.put_along_axis(sparse, indices, np.take_along_axis(details, indices, axis=1), axis=1)
    return sparse


def decode_dense(params, cfg, coarse: np.ndarray, details: np.ndarray) -> np.ndarray:
    @jax.jit
    def decode(batch_coarse, batch_details):
        return synthesis(params, batch_coarse, unflatten_details(batch_details, detail_shapes(cfg.image_size, cfg.channels, cfg.levels)), cfg)

    return np.asarray(decode(jnp.asarray(coarse), jnp.asarray(details)))


def nearest_sources(generated: np.ndarray, source: np.ndarray):
    generated_flat = generated[:, ::4, ::4, :].reshape((generated.shape[0], -1)).astype(np.float64)
    source_flat = source[:, ::4, ::4, :].reshape((source.shape[0], -1)).astype(np.float64)
    distances = (
        np.mean(generated_flat**2, axis=1)[:, None]
        + np.mean(source_flat**2, axis=1)[None, :]
        - 2.0 * (generated_flat @ source_flat.T) / generated_flat.shape[1]
    )
    nearest = np.argmin(distances, axis=1)
    return nearest.astype(np.int32), distances[np.arange(generated.shape[0]), nearest]


def edge_energy(images: np.ndarray) -> float:
    return float(0.5 * (np.mean(np.diff(images, axis=1) ** 2) + np.mean(np.diff(images, axis=2) ** 2)))


def image_sanity(images: np.ndarray) -> dict:
    horizontal = np.diff(images, axis=2)
    vertical = np.diff(images, axis=1)
    horizontal_energy = float(np.mean(horizontal**2))
    vertical_energy = float(np.mean(vertical**2))
    phase_means = [float(np.mean(images[:, y::2, x::2, :])) for y in range(2) for x in range(2)]
    return {
        "near_zero_fraction": float(np.mean(images <= 1.0 / 255.0)),
        "near_one_fraction": float(np.mean(images >= 254.0 / 255.0)),
        "horizontal_edge_energy": horizontal_energy,
        "vertical_edge_energy": vertical_energy,
        "directional_edge_ratio": horizontal_energy / (vertical_energy + 1.0e-12),
        "phase_means": phase_means,
        "phase_mean_range": max(phase_means) - min(phase_means),
    }


def mean_pairwise_mse(images: np.ndarray) -> float:
    if images.shape[0] < 2:
        return 0.0
    flat = images.reshape((images.shape[0], -1)).astype(np.float64)
    distances = np.mean((flat[:, None, :] - flat[None, :, :]) ** 2, axis=2)
    upper = distances[np.triu_indices(images.shape[0], k=1)]
    return float(np.mean(upper))


def save_images(out: Path, stem: str, images: np.ndarray) -> None:
    for index, image in enumerate(images):
        save_pgm(out / f"{stem}_{index:02d}.pgm", image)
        save_png(out / f"{stem}_{index:02d}.png", image)


def save_montage(path: Path, images: np.ndarray, columns: int = 4) -> None:
    count = min(images.shape[0], 16)
    rows = (count + columns - 1) // columns
    height, width = images.shape[1:3]
    montage = np.zeros((rows * height, columns * width, 1), dtype=np.float32)
    for index in range(count):
        row, column = divmod(index, columns)
        montage[row * height : (row + 1) * height, column * width : (column + 1) * width] = images[index]
    save_png(path, montage)


if __name__ == "__main__":
    main()
