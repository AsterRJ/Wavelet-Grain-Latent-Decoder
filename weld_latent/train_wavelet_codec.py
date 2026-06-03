from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .io import load_npz_images, save_pgm, save_png
from .learned_wavelet import (
    LearnedWaveletConfig,
    analysis,
    coarse_coefficient_count,
    compress,
    detail_coefficient_count,
    filter_regularization,
    flatten_details,
    init_params,
    kept_detail_count,
    pack_coefficients,
    save_checkpoint,
    synthesis,
    validate_config,
)
from .optim import adam_update, init_adam


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an interpretable continuous 2D lifting-wavelet codec for weld images.")
    parser.add_argument("--dataset", default="data/weld_256_patches.npz")
    parser.add_argument("--out", default="runs/learned_wavelet_codec")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--keep-detail-fraction", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--edge-weight", type=float, default=0.20)
    parser.add_argument("--edge-energy-weight", type=float, default=0.10)
    parser.add_argument("--filter-regularization", type=float, default=2.0e-3)
    parser.add_argument("--coefficient-l1", type=float, default=5.0e-5)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--selection-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.selection_interval < 1:
        raise ValueError("--selection-interval must be positive.")

    cfg = LearnedWaveletConfig(
        image_size=args.image_size,
        levels=args.levels,
        kernel_size=args.kernel_size,
        keep_detail_fraction=args.keep_detail_fraction,
        learning_rate=args.lr,
        edge_weight=args.edge_weight,
        edge_energy_weight=args.edge_energy_weight,
        filter_regularization=args.filter_regularization,
        coefficient_l1=args.coefficient_l1,
    )
    validate_config(cfg)
    images = load_npz_images(args.dataset)
    if images.shape[1:] != (cfg.image_size, cfg.image_size, cfg.channels):
        raise ValueError(f"Dataset has shape {images.shape}; expected (*, {cfg.image_size}, {cfg.image_size}, {cfg.channels}).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(images.shape[0])
    train_count = max(1, min(images.shape[0] - 1, int(round(images.shape[0] * args.train_fraction)))) if images.shape[0] > 1 else 1
    train = images[order[:train_count]]
    validation = images[order[train_count:]]

    params = init_params(cfg)
    reference = init_params(cfg)
    baseline = evaluate_dataset(params, images, cfg, args.eval_batch_size, collect_codes=False)
    baseline_validation = evaluate_dataset(params, validation, cfg, args.eval_batch_size, collect_codes=False) if validation.shape[0] else None
    opt = init_adam(params)
    train_step, evaluate_loss = make_train_step(cfg, reference)
    selection = validation if validation.shape[0] else train
    initial_selection = evaluate_loss(params, jnp.asarray(selection))
    best_loss = float(initial_selection["loss"])
    best_params = jax.device_get(params)
    best_step = 0
    for step in range(1, args.steps + 1):
        indices = rng.choice(train.shape[0], size=args.batch_size, replace=train.shape[0] < args.batch_size)
        params, opt, metrics = train_step(params, opt, jnp.asarray(train[indices]), args.lr)
        if step == 1 or step % args.selection_interval == 0 or step == args.steps:
            selection_metrics = evaluate_loss(params, jnp.asarray(selection))
            selection_loss = float(selection_metrics["loss"])
            if selection_loss < best_loss:
                best_loss = selection_loss
                best_params = jax.device_get(params)
                best_step = step
            print(
                f"step={step:05d} loss={float(metrics['loss']):.7f} "
                f"recon={float(metrics['recon']):.7f} edge={float(metrics['edge']):.7f} "
                f"edge_energy={float(metrics['edge_energy']):.7f} selection={selection_loss:.7f}"
            )

    params = jax.tree.map(jnp.asarray, best_params)
    full = evaluate_dataset(params, images, cfg, args.eval_batch_size, collect_codes=True)
    held_out = evaluate_dataset(params, validation, cfg, args.eval_batch_size, collect_codes=False) if validation.shape[0] else None
    total_details = detail_coefficient_count(cfg.image_size, cfg.channels, cfg.levels)
    kept_details = kept_detail_count(total_details, cfg.keep_detail_fraction)
    coarse_count = coarse_coefficient_count(cfg.image_size, cfg.channels, cfg.levels)
    summary = {
        "config": asdict(cfg),
        "dataset": args.dataset,
        "images": int(images.shape[0]),
        "training_images": int(train.shape[0]),
        "validation_images": int(validation.shape[0]),
        "pixel_values_per_image": int(cfg.image_size * cfg.image_size * cfg.channels),
        "coarse_coefficients": int(coarse_count),
        "retained_detail_coefficients": int(kept_details),
        "latent_values_per_image": int(coarse_count + kept_details),
        "compression_ratio": float(cfg.image_size * cfg.image_size * cfg.channels / (coarse_count + kept_details)),
        "selected_step": int(best_step),
        "selection_loss": float(best_loss),
        "haar_like_baseline": baseline["metrics"],
        "haar_like_baseline_validation": baseline_validation["metrics"] if baseline_validation else None,
        "trained": full["metrics"],
        "validation": held_out["metrics"] if held_out else None,
    }
    save_checkpoint(out / "checkpoint.pkl", params, cfg)
    np.savez_compressed(out / "filters.npz", predict=np.asarray(params["predict"]), update=np.asarray(params["update"]))
    latent_indices, latent_values = compact_storage(full["detail_indices"], full["detail_values"], total_details)
    np.savez_compressed(
        out / "latent_codes.npz",
        coarse=full["coarse"],
        detail_index_deltas=delta_encode_indices(latent_indices),
        detail_values=latent_values,
        source_indices=np.arange(images.shape[0], dtype=np.int32),
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_visuals(out, full["visuals"], np.asarray(params["predict"]), np.asarray(params["update"]))
    print(json.dumps(summary, indent=2))
    print(f"saved learned wavelet codec to {out}")


def make_train_step(cfg: LearnedWaveletConfig, reference: dict):
    @jax.jit
    def train_step(params, opt, batch, lr):
        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params, batch)
        params, opt = adam_update(params, grads, opt, lr)
        return params, opt, metrics

    def objective(params, batch):
        encoded = compress(params, batch, cfg)
        reconstruction = encoded["reconstruction"]
        recon = jnp.mean((reconstruction - batch) ** 2)
        edge = gradient_mse(reconstruction, batch)
        edge_energy = relative_edge_energy_error(reconstruction, batch)
        coefficient_l1 = jnp.mean(jnp.abs(flatten_details(encoded["details"])))
        regularization = filter_regularization(params, reference)
        loss = recon + cfg.edge_weight * edge + cfg.edge_energy_weight * edge_energy + cfg.filter_regularization * regularization + cfg.coefficient_l1 * coefficient_l1
        return loss, {
            "loss": loss,
            "recon": recon,
            "edge": edge,
            "edge_energy": edge_energy,
            "coefficient_l1": coefficient_l1,
            "filter_regularization": regularization,
        }

    @jax.jit
    def evaluate_loss(params, batch):
        _, metrics = objective(params, batch)
        return metrics

    return train_step, evaluate_loss


def evaluate_dataset(params, images: np.ndarray, cfg: LearnedWaveletConfig, batch_size: int, collect_codes: bool):
    if images.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty image set.")

    @jax.jit
    def evaluate_batch(batch):
        encoded = compress(params, batch, cfg)
        coarse, details = analysis(params, batch, cfg)
        exact = synthesis(params, coarse, details, cfg)
        packed = pack_coefficients(encoded["coarse"], encoded["sparse_details"])
        return {
            "reconstruction": encoded["reconstruction"],
            "exact": exact,
            "coarse": encoded["coarse"],
            "detail_indices": encoded["detail_indices"],
            "detail_values": encoded["detail_values"],
            "packed": packed,
        }

    pixel_error = 0.0
    pixel_count = 0
    gradient_error = 0.0
    gradient_count = 0
    input_edge_energy = 0.0
    recon_edge_energy = 0.0
    exact_max_abs = 0.0
    coarse_codes = []
    detail_indices = []
    detail_values = []
    visual_inputs = []
    visual_recons = []
    visual_coefficients = []
    for start in range(0, images.shape[0], batch_size):
        host_batch = np.asarray(images[start : start + batch_size])
        result = jax.tree.map(lambda value: np.asarray(value), evaluate_batch(jnp.asarray(host_batch)))
        reconstruction = result["reconstruction"]
        error = reconstruction - host_batch
        pixel_error += float(np.sum(error**2))
        pixel_count += error.size
        for axis in (1, 2):
            input_gradient = np.diff(host_batch, axis=axis)
            recon_gradient = np.diff(reconstruction, axis=axis)
            gradient_error += float(np.sum((recon_gradient - input_gradient) ** 2))
            gradient_count += input_gradient.size
            input_edge_energy += float(np.sum(input_gradient**2))
            recon_edge_energy += float(np.sum(recon_gradient**2))
        exact_max_abs = max(exact_max_abs, float(np.max(np.abs(result["exact"] - host_batch))))
        if collect_codes:
            coarse_codes.append(result["coarse"])
            detail_indices.append(result["detail_indices"])
            detail_values.append(result["detail_values"])
        remaining = 16 - len(visual_inputs)
        if remaining > 0:
            visual_inputs.extend(host_batch[:remaining])
            visual_recons.extend(reconstruction[:remaining])
            visual_coefficients.extend(result["packed"][:remaining])

    mse = pixel_error / pixel_count
    metrics = {
        "mse": mse,
        "psnr_db": float(-10.0 * np.log10(mse + 1.0e-12)),
        "gradient_mse": gradient_error / gradient_count,
        "edge_energy_ratio": recon_edge_energy / (input_edge_energy + 1.0e-12),
        "exact_inverse_max_abs": exact_max_abs,
    }
    return {
        "metrics": metrics,
        "coarse": np.concatenate(coarse_codes) if coarse_codes else None,
        "detail_indices": np.concatenate(detail_indices) if detail_indices else None,
        "detail_values": np.concatenate(detail_values) if detail_values else None,
        "visuals": {"input": visual_inputs, "reconstruction": visual_recons, "coefficients": visual_coefficients},
    }


def compact_storage(indices: np.ndarray, values: np.ndarray, total_details: int):
    """Sort sparse positions and store them in the smallest safe integer type."""
    order = np.argsort(indices, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    dtype = np.uint16 if total_details <= np.iinfo(np.uint16).max + 1 else np.uint32
    return indices.astype(dtype), values


def delta_encode_indices(indices: np.ndarray) -> np.ndarray:
    """Delta-code sorted sparse positions so NPZ compression can remove overhead."""
    leading_zero = np.zeros((indices.shape[0], 1), dtype=indices.dtype)
    return np.diff(indices, axis=1, prepend=leading_zero).astype(indices.dtype)


def gradient_mse(left, right):
    return 0.5 * (
        jnp.mean((jnp.diff(left, axis=1) - jnp.diff(right, axis=1)) ** 2)
        + jnp.mean((jnp.diff(left, axis=2) - jnp.diff(right, axis=2)) ** 2)
    )


def relative_edge_energy_error(left, right):
    left_energy = gradient_energy_per_image(left)
    right_energy = gradient_energy_per_image(right)
    return jnp.mean(((left_energy - right_energy) / (right_energy + 1.0e-12)) ** 2)


def gradient_energy_per_image(images):
    vertical = jnp.mean(jnp.diff(images, axis=1) ** 2, axis=(1, 2, 3))
    horizontal = jnp.mean(jnp.diff(images, axis=2) ** 2, axis=(1, 2, 3))
    return 0.5 * (vertical + horizontal)


def save_visuals(out: Path, visuals: dict, predict: np.ndarray, update: np.ndarray) -> None:
    for index, (source, reconstruction, coefficients) in enumerate(zip(visuals["input"], visuals["reconstruction"], visuals["coefficients"])):
        save_pgm(out / f"input_{index:02d}.pgm", source)
        save_png(out / f"input_{index:02d}.png", source)
        save_pgm(out / f"recon_{index:02d}.pgm", reconstruction)
        save_png(out / f"recon_{index:02d}.png", reconstruction)
        coefficient_image = normalize_signed(coefficients)
        save_pgm(out / f"coefficients_{index:02d}.pgm", coefficient_image)
        save_png(out / f"coefficients_{index:02d}.png", coefficient_image)
    for level in range(predict.shape[0]):
        for band in range(predict.shape[1]):
            save_png(out / f"predict_l{level + 1}_b{band + 1}.png", normalize_signed(predict[level, band]))
            save_png(out / f"update_l{level + 1}_b{band + 1}.png", normalize_signed(update[level, band]))


def normalize_signed(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    scale = float(np.percentile(np.abs(arr), 99.0)) + 1.0e-12
    return np.clip(0.5 + 0.5 * arr / scale, 0.0, 1.0)


if __name__ == "__main__":
    main()
