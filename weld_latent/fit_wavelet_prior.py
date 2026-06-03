from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an adaptive KDE prior in the Grassmannian wavelet metric embedding.")
    parser.add_argument("--metric", required=True, help="Metric archive from fit_wavelet_metric.")
    parser.add_argument("--out", default="runs/learned_wavelet_prior")
    parser.add_argument("--neighbors", type=int, default=16, help="Local metric neighbors used to fit each KDE covariance.")
    parser.add_argument("--bandwidth-scale", type=float, default=0.55, help="Scale applied to the fitted local covariance while sampling.")
    parser.add_argument("--ridge-fraction", type=float, default=0.05, help="Isotropic regularization relative to each neighborhood radius.")
    args = parser.parse_args()
    if args.neighbors < 2:
        raise ValueError("--neighbors must be at least 2.")
    if args.bandwidth_scale <= 0.0 or args.ridge_fraction <= 0.0:
        raise ValueError("--bandwidth-scale and --ridge-fraction must be positive.")

    metric = np.load(args.metric)
    embedding = metric["embedding"].astype(np.float64)
    source_indices = metric["source_indices"].astype(np.int32) if "source_indices" in metric else np.arange(embedding.shape[0], dtype=np.int32)
    local_neighbors, local_covariances, local_cholesky, radii = fit_local_kde(embedding, args.neighbors, args.ridge_fraction)
    metrics = {
        "components": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "neighbors": int(local_neighbors.shape[1]),
        "bandwidth_scale": float(args.bandwidth_scale),
        "ridge_fraction": float(args.ridge_fraction),
        "mean_neighbor_radius": float(np.mean(radii)),
        "median_neighbor_radius": float(np.median(radii)),
        "minimum_neighbor_radius": float(np.min(radii)),
        "maximum_neighbor_radius": float(np.max(radii)),
        "note": "Adaptive Gaussian KDE in the saved Grassmann/MDS embedding. Decode sampled coordinates through compatible local wavelet latents.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "prior.npz",
        embedding=embedding.astype(np.float32),
        local_neighbor_indices=local_neighbors,
        local_covariances=local_covariances.astype(np.float32),
        local_cholesky=local_cholesky.astype(np.float32),
        component_weights=np.full(embedding.shape[0], 1.0 / embedding.shape[0], dtype=np.float32),
        bandwidth_scale=np.array(args.bandwidth_scale, dtype=np.float32),
        source_indices=source_indices,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved Grassmannian metric KDE prior to {out}")


def fit_local_kde(embedding: np.ndarray, neighbors: int, ridge_fraction: float):
    """Fit one regularized local covariance around each metric-space source point."""
    distances = euclidean_distance_matrix(embedding)
    count = min(neighbors, embedding.shape[0] - 1)
    local_neighbors = np.argsort(distances, axis=1)[:, 1 : count + 1].astype(np.int32)
    deltas = embedding[local_neighbors] - embedding[:, None, :]
    radii = np.sqrt(np.max(np.sum(deltas**2, axis=2), axis=1))
    covariance = np.einsum("nki,nkj->nij", deltas, deltas, optimize=True) / count
    ridge = np.maximum(radii * ridge_fraction, 1.0e-6) ** 2
    covariance += ridge[:, None, None] * np.eye(embedding.shape[1], dtype=np.float64)[None, :, :]
    cholesky = np.linalg.cholesky(covariance)
    return local_neighbors, covariance, cholesky, radii


def euclidean_distance_matrix(embedding: np.ndarray) -> np.ndarray:
    squared = np.sum(embedding**2, axis=1)
    return np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2.0 * embedding @ embedding.T, 0.0))


if __name__ == "__main__":
    main()
