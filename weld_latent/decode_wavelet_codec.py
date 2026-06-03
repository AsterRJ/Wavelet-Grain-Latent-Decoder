from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .io import load_npz_images, save_pgm, save_png
from .learned_wavelet import decode_compact, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode compact continuous coefficients from a learned 2D lifting-wavelet codec.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out", default="runs/learned_wavelet_decoded")
    parser.add_argument("--dataset", default=None, help="Optional source dataset used only to report reconstruction MSE.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params, cfg = load_checkpoint(args.checkpoint)
    archive = np.load(args.codes)
    coarse = archive["coarse"]
    if "detail_index_deltas" in archive:
        indices = np.cumsum(archive["detail_index_deltas"].astype(np.uint32), axis=1)
    else:
        indices = archive["detail_indices"]
    values = archive["detail_values"]
    if not (coarse.shape[0] == indices.shape[0] == values.shape[0]):
        raise ValueError("Compact latent arrays must have the same image count.")

    @jax.jit
    def decode_batch(batch_coarse, batch_indices, batch_values):
        return decode_compact(params, batch_coarse, batch_indices, batch_values, cfg)

    reconstructions = []
    for start in range(0, coarse.shape[0], args.batch_size):
        reconstruction = decode_batch(
            jnp.asarray(coarse[start : start + args.batch_size]),
            jnp.asarray(indices[start : start + args.batch_size]),
            jnp.asarray(values[start : start + args.batch_size]),
        )
        reconstructions.append(np.asarray(reconstruction))
    reconstructions = np.concatenate(reconstructions)
    for index, image in enumerate(reconstructions[:32]):
        save_pgm(out / f"recon_{index:02d}.pgm", image)
        save_png(out / f"recon_{index:02d}.png", image)

    metrics = {"decoded_images": int(reconstructions.shape[0])}
    if args.dataset:
        source = load_npz_images(args.dataset)
        if source.shape != reconstructions.shape:
            raise ValueError(f"Dataset shape {source.shape} does not match decoded shape {reconstructions.shape}.")
        mse = float(np.mean((source - reconstructions) ** 2))
        metrics["mse"] = mse
        metrics["psnr_db"] = float(-10.0 * np.log10(mse + 1.0e-12))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote decoded images to {out}")


if __name__ == "__main__":
    main()
