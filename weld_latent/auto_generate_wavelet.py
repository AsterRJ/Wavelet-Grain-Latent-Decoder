from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .generate_wavelet_codec import (
    edge_energy,
    expand_sparse_details,
    generate_metric_density,
    image_sanity,
    load_latent_archive,
    load_metric_archive,
    load_prior_archive,
    mean_pairwise_mse,
    nearest_sources,
    save_montage,
)
from .io import load_npz_images, save_pgm, save_png
from .learned_wavelet import detail_coefficient_count, load_checkpoint


RAW_DEFAULTS = {
    "checkpoint": "runs/raw_learned_wavelet_codec/checkpoint.pkl",
    "codes": "runs/raw_learned_wavelet_codec/latent_codes.npz",
    "metric": "runs/raw_learned_wavelet_metric/metric.npz",
    "prior": "runs/raw_learned_wavelet_prior_wide/prior.npz",
    "dataset": "data/raw_weld_256.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically generate images from a learned wavelet decoder and metric-space prior.")
    parser.add_argument("--checkpoint", default=RAW_DEFAULTS["checkpoint"])
    parser.add_argument("--codes", default=RAW_DEFAULTS["codes"])
    parser.add_argument("--metric", default=RAW_DEFAULTS["metric"])
    parser.add_argument("--prior", default=RAW_DEFAULTS["prior"])
    parser.add_argument("--dataset", default=RAW_DEFAULTS["dataset"], help="Optional dataset used only for nearest-source novelty metrics; use '' to disable.")
    parser.add_argument("--out", default="runs/raw_learned_wavelet_auto")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inverse-neighbors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="generated")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.inverse_neighbors < 2:
        raise ValueError("--inverse-neighbors must be at least 2.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params, cfg = load_checkpoint(args.checkpoint)
    archive = load_latent_archive(args.codes)
    coarse = archive["coarse"]
    indices = archive["indices"]
    values = archive["values"]
    total_details = detail_coefficient_count(cfg.image_size, cfg.channels, cfg.levels)
    dense_details = expand_sparse_details(indices, values, total_details)
    metric = load_metric_archive(args.metric, coarse.shape[0])
    prior = load_prior_archive(args.prior, metric)
    rng = np.random.default_rng(args.seed)

    generated_batches = []
    donor_batches = []
    weight_batches = []
    sampled_coordinate_batches = []
    generated_coordinate_batches = []
    anchor_batches = []
    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        generated, donors, weights, sampled_coordinates, generated_coordinates, anchors = generate_metric_density(
            params,
            cfg,
            coarse,
            dense_details,
            indices.shape[1],
            count,
            args.inverse_neighbors,
            rng,
            metric,
            prior,
        )
        save_numbered_images(out, args.prefix, generated, start)
        generated_batches.append(generated)
        donor_batches.append(donors)
        weight_batches.append(weights)
        sampled_coordinate_batches.append(sampled_coordinates)
        generated_coordinate_batches.append(generated_coordinates)
        anchor_batches.append(anchors)
        print(f"generated {start + count}/{args.samples}")

    generated = np.concatenate(generated_batches, axis=0)
    donors = np.concatenate(donor_batches, axis=0)
    weights = np.concatenate(weight_batches, axis=0)
    sampled_coordinates = np.concatenate(sampled_coordinate_batches, axis=0)
    generated_coordinates = np.concatenate(generated_coordinate_batches, axis=0)
    anchors = np.concatenate(anchor_batches, axis=0)

    save_montage(out / f"{args.prefix}_montage.png", generated)
    np.savetxt(out / "sampled_metric_coordinates.csv", sampled_coordinates, delimiter=",")
    np.savetxt(out / "generated_metric_coordinates.csv", generated_coordinates, delimiter=",")

    manifest = {
        "checkpoint": args.checkpoint,
        "codes": args.codes,
        "metric": args.metric,
        "prior": args.prior,
        "dataset": args.dataset or None,
        "out": str(out),
        "samples": int(args.samples),
        "batch_size": int(args.batch_size),
        "inverse_neighbors": int(args.inverse_neighbors),
        "seed": int(args.seed),
        "prefix": args.prefix,
        "image_shape": list(generated.shape[1:]),
        "generated_mean": float(np.mean(generated)),
        "generated_std": float(np.std(generated)),
        "generated_edge_energy": float(edge_energy(generated)),
        "generated_pairwise_mse": float(mean_pairwise_mse(generated)) if args.samples <= 256 else None,
        "generated_image_sanity": image_sanity(generated),
        "metric_coordinate_reconstruction_mse": float(np.mean((sampled_coordinates - generated_coordinates) ** 2)),
        "density_anchor_indices": anchors.tolist(),
        "donor_indices": donors.tolist(),
        "mixing_weights": weights.tolist(),
        "files": {
            "montage": f"{args.prefix}_montage.png",
            "sampled_metric_coordinates": "sampled_metric_coordinates.csv",
            "generated_metric_coordinates": "generated_metric_coordinates.csv",
            "images": [f"{args.prefix}_{index:05d}.png" for index in range(args.samples)],
        },
    }

    if args.dataset:
        source = load_npz_images(args.dataset)
        if source.shape[0] != coarse.shape[0] or source.shape[1:] != generated.shape[1:]:
            raise ValueError(f"Dataset shape {source.shape} is incompatible with generated images {generated.shape}.")
        nearest_indices, nearest_mse = nearest_sources(generated, source)
        nearest = source[nearest_indices]
        save_numbered_images(out, "nearest_source", nearest, 0)
        save_montage(out / "nearest_source_montage.png", nearest)
        manifest.update(
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
        manifest["files"]["nearest_source_montage"] = "nearest_source_montage.png"

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in summary_keys(manifest)}, indent=2))
    print(f"wrote automatic wavelet generations to {out}")


def save_numbered_images(out: Path, stem: str, images: np.ndarray, offset: int) -> None:
    for index, image in enumerate(images, start=offset):
        save_pgm(out / f"{stem}_{index:05d}.pgm", image)
        save_png(out / f"{stem}_{index:05d}.png", image)


def summary_keys(manifest: dict) -> list[str]:
    keys = [
        "samples",
        "seed",
        "generated_mean",
        "generated_std",
        "generated_edge_energy",
        "generated_pairwise_mse",
        "generated_image_sanity",
        "metric_coordinate_reconstruction_mse",
    ]
    if "mean_nearest_source_mse_downsampled" in manifest:
        keys.extend(["generated_over_source_edge_energy", "mean_nearest_source_mse_downsampled"])
    return keys


if __name__ == "__main__":
    main()
