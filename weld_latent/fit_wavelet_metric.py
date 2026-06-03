from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .io import load_npz_images
from .learned_wavelet import analysis, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an ordered latent metric that approximates Grassmannian geodesics between learned wavelet feature subspaces.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/weld_256_patches.npz")
    parser.add_argument("--out", default="runs/learned_wavelet_metric")
    parser.add_argument("--feature-grid", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--subspace-rank", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.patch_size < 1:
        raise ValueError("--patch-size must be positive.")
    if args.subspace_rank < 1 or args.embedding_dim < 1 or args.neighbors < 1:
        raise ValueError("Rank, embedding dimension, and neighbors must be positive.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params, codec_cfg = load_checkpoint(args.checkpoint)
    images = load_npz_images(args.dataset)
    maps = extract_wavelet_feature_maps(params, codec_cfg, images, args.batch_size)
    maps = resize_feature_maps(maps, args.feature_grid)
    channel_mean = maps.mean(axis=(0, 1, 2), keepdims=True)
    channel_scale = maps.std(axis=(0, 1, 2), keepdims=True) + 1.0e-6
    normalized = (maps - channel_mean) / channel_scale
    subspaces, singular_values = fit_feature_subspaces(normalized, args.patch_size, args.subspace_rank)
    grassmann = grassmann_distance_matrix(subspaces)
    embedding, eigenvalues = classical_mds(grassmann, args.embedding_dim)
    embedded = euclidean_distance_matrix(embedding)
    neighbors = min(args.neighbors, images.shape[0] - 1)
    metric_neighbors = np.argsort(embedded, axis=1)[:, 1 : neighbors + 1].astype(np.int32)
    grassmann_neighbors = np.argsort(grassmann, axis=1)[:, 1 : neighbors + 1].astype(np.int32)
    metrics = metric_quality(grassmann, embedded, metric_neighbors, grassmann_neighbors)
    metrics.update(
        {
            "images": int(images.shape[0]),
            "feature_maps": int(maps.shape[-1]),
            "feature_grid": int(args.feature_grid),
            "patch_size": int(args.patch_size),
            "subspace_rank": int(subspaces.shape[-1]),
            "embedding_dim": int(embedding.shape[1]),
            "neighbors": int(neighbors),
            "positive_mds_eigenvalues": int(np.count_nonzero(eigenvalues > 0.0)),
        }
    )
    np.savez_compressed(
        out / "metric.npz",
        embedding=embedding.astype(np.float32),
        grassmann_distances=grassmann.astype(np.float32),
        embedded_distances=embedded.astype(np.float32),
        neighbor_indices=metric_neighbors,
        grassmann_neighbor_indices=grassmann_neighbors,
        subspaces=subspaces.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        feature_channel_mean=channel_mean.reshape(-1).astype(np.float32),
        feature_channel_scale=channel_scale.reshape(-1).astype(np.float32),
        source_indices=np.arange(images.shape[0], dtype=np.int32),
    )
    np.savetxt(out / "embedding.csv", embedding, delimiter=",")
    np.savetxt(out / "neighbor_indices.csv", metric_neighbors, fmt="%d", delimiter=",")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved Grassmannian wavelet metric to {out}")


def extract_wavelet_feature_maps(params, cfg, images: np.ndarray, batch_size: int) -> np.ndarray:
    """Return aligned coarse, signed-detail, and detail-magnitude maps."""
    @jax.jit
    def analyze(batch):
        coarse, details = analysis(params, batch, cfg)
        return coarse, details

    outputs = []
    for start in range(0, images.shape[0], batch_size):
        coarse, details = jax.tree.map(lambda value: np.asarray(value), analyze(jnp.asarray(images[start : start + batch_size])))
        target = coarse.shape[1]
        signed = [coarse]
        for level in details:
            for band in level:
                signed.append(resize_feature_maps(band, target))
        signed = np.concatenate(signed, axis=-1)
        outputs.append(np.concatenate([signed, np.abs(signed[..., 1:])], axis=-1))
    return np.concatenate(outputs)


def resize_feature_maps(maps: np.ndarray, size: int) -> np.ndarray:
    if maps.shape[1] == size and maps.shape[2] == size:
        return maps
    if maps.shape[1] % size != 0 or maps.shape[2] % size != 0:
        raise ValueError(f"Cannot block-average feature maps shaped {maps.shape} to {size}x{size}.")
    fy, fx = maps.shape[1] // size, maps.shape[2] // size
    return maps.reshape((maps.shape[0], size, fy, size, fx, maps.shape[3])).mean(axis=(2, 4))


def fit_feature_subspaces(maps: np.ndarray, patch_size: int, rank: int):
    """Represent every image by the dominant span of its local wavelet patches."""
    ambient = maps.shape[-1] * patch_size * patch_size
    used_rank = min(rank, ambient)
    windows = np.lib.stride_tricks.sliding_window_view(maps, (patch_size, patch_size), axis=(1, 2))
    features = windows.reshape((maps.shape[0], -1, ambient)).astype(np.float64)
    features -= features.mean(axis=1, keepdims=True)
    covariance = np.einsum("npa,npb->nab", features, features, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[:, -used_rank:][:, ::-1]
    eigenvectors = eigenvectors[:, :, -used_rank:][:, :, ::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    return eigenvectors, singular_values


def grassmann_distance_matrix(subspaces: np.ndarray) -> np.ndarray:
    """Compute geodesic distance from principal angles on Gr(rank, ambient)."""
    count = subspaces.shape[0]
    distances = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            cosines = np.linalg.svd(subspaces[left].T @ subspaces[right], compute_uv=False)
            angles = np.arccos(np.clip(cosines, 0.0, 1.0))
            distance = float(np.sqrt(np.sum(angles**2)))
            distances[left, right] = distance
            distances[right, left] = distance
    return distances


def classical_mds(distances: np.ndarray, dimensions: int):
    """Embed geodesic distances in an ordered Euclidean latent approximation."""
    squared = distances**2
    centered = squared - squared.mean(axis=0, keepdims=True) - squared.mean(axis=1, keepdims=True) + squared.mean()
    gram = -0.5 * centered
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = values > 1.0e-10
    kept = min(dimensions, int(np.count_nonzero(positive)))
    if kept == 0:
        raise ValueError("Grassmannian distance matrix did not produce a positive MDS coordinate.")
    embedding = vectors[:, :kept] * np.sqrt(values[:kept])[None, :]
    return embedding, values


def euclidean_distance_matrix(embedding: np.ndarray) -> np.ndarray:
    squared = np.sum(embedding**2, axis=1)
    return np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2.0 * embedding @ embedding.T, 0.0))


def metric_quality(grassmann: np.ndarray, embedded: np.ndarray, metric_neighbors: np.ndarray, grassmann_neighbors: np.ndarray) -> dict:
    upper = np.triu_indices(grassmann.shape[0], k=1)
    target = grassmann[upper]
    approximation = embedded[upper]
    correlation = float(np.corrcoef(target, approximation)[0, 1])
    stress = float(np.sqrt(np.sum((approximation - target) ** 2) / np.sum(target**2)))
    overlap = []
    width = min(16, metric_neighbors.shape[1])
    for row in range(grassmann.shape[0]):
        overlap.append(len(set(metric_neighbors[row, :width]).intersection(grassmann_neighbors[row, :width])) / width)
    return {"distance_correlation": correlation, "normalized_stress": stress, "neighbor_overlap_at_16": float(np.mean(overlap))}


if __name__ == "__main__":
    main()
