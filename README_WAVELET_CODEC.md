# Learned 2D Wavelet Latent Codec

This is the primary reconstruction path for the weld images. It intentionally
returns to a small, inspectable objective: learn a continuous 2D wavelet-style
latent representation and decode it back into images without a generative
neural decoder.

## Representation

`weld_latent.learned_wavelet` implements an exactly invertible multiscale 2D
lifting transform. At each scale, an image is split into its four interleaved
pixel phases. Three learned local `n x n` convolutional predictors convert the
non-reference phases into detail coefficients. Three learned update filters
form the next low-frequency image. Decoding reverses those same lifting steps
exactly.

The latent representation contains:

- The complete final low-frequency image.
- The strongest continuous detail coefficients across every scale.
- The coefficient indices needed to put those details back in the wavelet grid.

The default keeps `35%` of detail coefficients across two scales. For
`256 x 256` patches this stores `25,600` continuous values instead of `65,536`
pixels, a `2.56x` value-count compression ratio. Increase
`--keep-detail-fraction` when visual fidelity matters more than compression.

The learned loss includes pixel reconstruction, gradient reconstruction,
per-image edge-energy preservation, filter regularization, and a small detail
sparsity term. Filters are selected by held-out loss during training. If
learning does not improve the validation set, the Haar-like initial transform
is retained.

## Train

Run the main patch workflow:

```bash
python -m weld_latent.train_wavelet_codec \
  --dataset data/weld_256_patches.npz \
  --image-size 256 \
  --out runs/learned_wavelet_codec \
  --steps 1200
```

A shorter first pass is useful before a long run:

```bash
python -m weld_latent.train_wavelet_codec \
  --dataset data/weld_256_patches.npz \
  --image-size 256 \
  --out runs/learned_wavelet_codec_first_pass \
  --steps 100
```

For a higher-fidelity archive, retain more detail coefficients:

```bash
python -m weld_latent.train_wavelet_codec \
  --dataset data/weld_256_patches.npz \
  --image-size 256 \
  --out runs/learned_wavelet_codec_hifi \
  --keep-detail-fraction 0.60 \
  --steps 1200
```

## Decode

Decode the compact latent archive without loading the source images:

```bash
python -m weld_latent.decode_wavelet_codec \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --codes runs/learned_wavelet_codec/latent_codes.npz \
  --out runs/learned_wavelet_decoded
```

Add the source dataset only when you want reconstruction metrics:

```bash
python -m weld_latent.decode_wavelet_codec \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --codes runs/learned_wavelet_codec/latent_codes.npz \
  --dataset data/weld_256_patches.npz \
  --out runs/learned_wavelet_decoded
```

## Outputs

```text
runs/learned_wavelet_codec/checkpoint.pkl
runs/learned_wavelet_codec/latent_codes.npz
runs/learned_wavelet_codec/filters.npz
runs/learned_wavelet_codec/metrics.json
runs/learned_wavelet_codec/input_00.png
runs/learned_wavelet_codec/recon_00.png
runs/learned_wavelet_codec/coefficients_00.png
```

`latent_codes.npz` is the actual compact encoding. It contains `coarse`,
delta-coded sparse positions in `detail_index_deltas`, and `detail_values`.
`filters.npz` exposes the learned
predictor and update kernels for inspection.

## Verified First Pass

A `100`-step CPU verification run on all `520` prepared patches selected the
learned filters at step `100`. On the held-out patches:

```text
Haar-like baseline MSE:       0.004968
Learned transform MSE:        0.004290
Learned transform PSNR:       23.676 dB
Learned edge-energy ratio:    0.999445
Exact full inverse max error: 3.58e-7
```

The next modeling layer should operate on these continuous wavelet
coefficients only after the reconstruction tradeoff is visually acceptable.

## Baseline Crossover Probe

Before fitting the ordered metric, stress-test whether the learned coefficient
space can produce useful new images through interpretable multiscale remixing:

```bash
python -m weld_latent.generate_wavelet_codec \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --codes runs/learned_wavelet_codec/latent_codes.npz \
  --dataset data/weld_256_patches.npz \
  --out runs/learned_wavelet_generated_probe \
  --strategy crossover \
  --samples 16
```

The probe writes two kinds of output:

- `interpolation_*.png` checks that the continuous latent representation moves
  smoothly between two encoded patches. This is a continuity test, not proof of
  generation.
- `generated_*.png` keeps one donor's coarse geometry, remixes detail bands
  from multiple encoded patches, reapplies the learned detail budget, and
  decodes the result. This tests whether the wavelet representation supports
  structural recombination before introducing a learned latent distribution.

The balanced default preserves one coarse geometry while blending each
scale's complete detail triplet from nearby coarse-latent neighbors:

```text
--coarse-mix hard --detail-mix soft --detail-granularity level
```

Keeping the three orientation bands together respects the internal lifting
relationship.
The default `--neighbor-pool 16` avoids combining incompatible geometry and
texture.
Use `--neighbor-pool 0` for an intentionally unconstrained remix,
`--coarse-mix soft` to blend larger structures, `--detail-granularity band` to
stress-test orientation-band compatibility, or `--detail-mix hard` as a
deliberately harsher swap.

Inspect these contact sheets together:

```text
runs/learned_wavelet_generated_probe/generated_montage.png
runs/learned_wavelet_generated_probe/nearest_source_montage.png
runs/learned_wavelet_generated_probe/coarse_donor_montage.png
runs/learned_wavelet_generated_probe/interpolation_montage.png
```

`metrics.json` reports nearest-source distances and edge-energy preservation.
The crossover is intentionally a diagnostic, not a claim that the codec already
contains a trained generative prior. If these remixes preserve plausible weld
geometry, the next step is to fit a distribution over the wavelet latents.

## Grassmannian Metric Ordering

Fit a reusable ordered latent metric from the original patches:

```bash
python -m weld_latent.fit_wavelet_metric \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --dataset data/weld_256_patches.npz \
  --out runs/learned_wavelet_metric
```

Each image is represented by a low-rank subspace of local multiscale wavelet
features. Pairwise distances are Grassmannian geodesics computed from principal
angles. Classical MDS then produces a compact Euclidean coordinate system whose
distances approximate those geodesics.

Generate locally within that ordered space:

```bash
python -m weld_latent.generate_wavelet_codec \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --codes runs/learned_wavelet_codec/latent_codes.npz \
  --metric runs/learned_wavelet_metric/metric.npz \
  --dataset data/weld_256_patches.npz \
  --out runs/learned_wavelet_generated_metric \
  --strategy local-interpolate
```

The default local generator selects donors from the compact MDS embedding,
interpolates coarse geometry linearly, and interpolates detail coefficients
spherically. This preserves texture norm while remaining inside a
metric-compatible neighborhood. Use `--metric-neighbors grassmann` to select
donors directly from the exact principal-angle neighbor table.

## Verified Generation Probe Result

The multiscale crossover probe is implemented and useful as a diagnostic, but
it is not yet a production generator. A verified `16`-sample run found that
unconstrained orientation-band crossover introduces excess saturation and
interleaved phase imbalance. Keeping each scale's three detail bands together
reduces phase imbalance, but manual crossover still trades between overly
smoothed soft mixes and over-saturated hard swaps.

The codec itself remains sound: ordinary compact reconstructions preserve edge
energy and have low phase imbalance. The next generative step should fit a
learned conditional distribution over compatible coarse and detail latents
rather than continue hand-remixing coefficients.

The earlier crossover stress-test output remains useful for comparison:

```text
runs/learned_wavelet_generated_probe_level/generated_montage.png
runs/learned_wavelet_generated_probe_level/nearest_source_montage.png
runs/learned_wavelet_generated_probe_level/metrics.json
```

## Verified Grassmannian Ordering Result

The practical one-time metric fit uses a `16 x 16` feature grid, `2 x 2` local
wavelet windows, rank-`8` subspaces, and an `8`-dimensional MDS embedding. On the
`520` prepared patches it achieved:

```text
Grassmann/MDS distance correlation: 0.9659
Normalized embedding stress:        0.2668
Neighbor overlap at 16:              0.4076
```

With metric-local spherical interpolation, a verified `16`-sample generation
probe improved substantially over manual level crossover:

```text
                                Manual crossover   Metric-local slerp
Nearest-source MSE:             0.0654             0.0268
Edge-energy ratio:              0.7671             0.9014
Near-white fraction:            0.0804             0.0315
Interleaved phase-mean range:   0.00252            0.00140
```

Inspect the preferred metric-local output here:

```text
runs/learned_wavelet_generated_metric_default/generated_montage.png
runs/learned_wavelet_generated_metric_default/nearest_source_montage.png
runs/learned_wavelet_generated_metric_default/interpolation_montage.png
runs/learned_wavelet_generated_metric_default/generated_metric_coordinates.csv
```

This ordered local generator is retained as a deterministic geometry probe.
For fresh sampling, fit the metric-space prior below.

## Learned Metric-Space Prior

Fit the lightweight adaptive KDE prior once after fitting the Grassmannian
metric:

```bash
python -m weld_latent.fit_wavelet_prior \
  --metric runs/learned_wavelet_metric/metric.npz \
  --out runs/learned_wavelet_prior
```

Generate fresh samples from the learned density:

```bash
python -m weld_latent.generate_wavelet_codec \
  --checkpoint runs/learned_wavelet_codec/checkpoint.pkl \
  --codes runs/learned_wavelet_codec/latent_codes.npz \
  --metric runs/learned_wavelet_metric/metric.npz \
  --prior runs/learned_wavelet_prior/prior.npz \
  --dataset data/weld_256_patches.npz \
  --out runs/learned_wavelet_generated_density_default \
  --samples 16
```

When `--prior` is supplied, the default `--strategy auto` samples an adaptive
Gaussian KDE in the saved MDS coordinates. Each sampled point is decoded
through its two closest compatible wavelet latents using linear coarse-space
interpolation and spherical detail interpolation. This keeps the inverse map
local and preserves sharp texture. Use `--inverse-neighbors 3` or `4` only when
additional variation is worth a softer result.

## Verified Density-Sampler Result

The packaged prior uses `16` local metric neighbors, bandwidth scale `0.55`,
and a two-neighbor inverse map. With the same `16`-sample seed used above:

```text
                                Manual crossover   Metric-local slerp   KDE density sampler
Nearest-source MSE:             0.0654             0.0268               0.0329
Edge-energy ratio:              0.7671             0.9014               0.9189
Near-white fraction:            0.0804             0.0315               0.0294
Interleaved phase-mean range:   0.00252            0.00140              0.00181
```

Inspect the final density-sampled output here:

```text
runs/learned_wavelet_generated_density_default/generated_montage.png
runs/learned_wavelet_generated_density_default/nearest_source_montage.png
runs/learned_wavelet_generated_density_default/interpolation_montage.png
runs/learned_wavelet_generated_density_default/sampled_metric_coordinates.csv
runs/learned_wavelet_generated_density_default/generated_metric_coordinates.csv
```

## Raw Color Image Run

The raw JPG frames in `/workspaces/autoencoder_weld/raw` are RGB `2560 x 1920`
images. They can be converted through the existing preprocessing path into the
same model format used above: greyscale float32 tensors shaped `(N, 256, 256, 1)`.
The loader uses Pillow luminance conversion and Lanczos downscaling.

```bash
python prepare_dataset.py \
  --input 'raw/*.jpg' \
  --out data/raw_weld_256.npz \
  --size 256
```

A verified conversion produced `113` images and an input montage at:

```text
data/raw_weld_256_montage.png
```

Train and sample the raw-image codec with:

```bash
JAX_PLATFORMS=cpu python -m weld_latent.train_wavelet_codec \
  --dataset data/raw_weld_256.npz \
  --image-size 256 \
  --out runs/raw_learned_wavelet_codec \
  --steps 600 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --selection-interval 25 \
  --seed 7

JAX_PLATFORMS=cpu python -m weld_latent.fit_wavelet_metric \
  --checkpoint runs/raw_learned_wavelet_codec/checkpoint.pkl \
  --dataset data/raw_weld_256.npz \
  --out runs/raw_learned_wavelet_metric \
  --batch-size 16

python -m weld_latent.fit_wavelet_prior \
  --metric runs/raw_learned_wavelet_metric/metric.npz \
  --out runs/raw_learned_wavelet_prior

JAX_PLATFORMS=cpu python -m weld_latent.generate_wavelet_codec \
  --checkpoint runs/raw_learned_wavelet_codec/checkpoint.pkl \
  --codes runs/raw_learned_wavelet_codec/latent_codes.npz \
  --metric runs/raw_learned_wavelet_metric/metric.npz \
  --prior runs/raw_learned_wavelet_prior/prior.npz \
  --dataset data/raw_weld_256.npz \
  --out runs/raw_learned_wavelet_generated_density \
  --samples 16 \
  --seed 4
```

For a slightly more varied raw-frame sample, fit the wider prior:

```bash
python -m weld_latent.fit_wavelet_prior \
  --metric runs/raw_learned_wavelet_metric/metric.npz \
  --out runs/raw_learned_wavelet_prior_wide \
  --bandwidth-scale 0.85

JAX_PLATFORMS=cpu python -m weld_latent.generate_wavelet_codec \
  --checkpoint runs/raw_learned_wavelet_codec/checkpoint.pkl \
  --codes runs/raw_learned_wavelet_codec/latent_codes.npz \
  --metric runs/raw_learned_wavelet_metric/metric.npz \
  --prior runs/raw_learned_wavelet_prior_wide/prior.npz \
  --dataset data/raw_weld_256.npz \
  --out runs/raw_learned_wavelet_generated_density_wide \
  --samples 16 \
  --seed 4
```

Verified raw-run metrics:

```text
Codec PSNR:                       38.35 dB
Codec edge-energy ratio:          0.9952
Grassmann/MDS distance corr.:     0.9342
Neighbor overlap at 16:           0.6787

                                Default KDE   Wider KDE
Nearest-source MSE:             0.00195       0.00208
Edge-energy ratio:              0.9845        0.9884
Near-white fraction:            0.0000        0.0000
Interleaved phase-mean range:   0.00152       0.00157
```

The raw generation artifacts are summarized in:

```text
runs/raw_learned_wavelet_comparison.json
runs/raw_learned_wavelet_generated_density/generated_montage.png
runs/raw_learned_wavelet_generated_density_wide/generated_montage.png
```

### Automatic Decoder Sampling

Once the raw codec, metric, and prior artifacts exist, fresh images can be
produced directly from the decoder with the raw defaults:

```bash
JAX_PLATFORMS=cpu python -m weld_latent.auto_generate_wavelet \
  --out runs/raw_learned_wavelet_auto \
  --samples 32 \
  --batch-size 16 \
  --seed 12
```

By default this uses:

```text
runs/raw_learned_wavelet_codec/checkpoint.pkl
runs/raw_learned_wavelet_codec/latent_codes.npz
runs/raw_learned_wavelet_metric/metric.npz
runs/raw_learned_wavelet_prior_wide/prior.npz
data/raw_weld_256.npz
```

The command writes sequential PNG/PGM images, a montage, metric-space
coordinates, nearest-source comparisons, and a `manifest.json`:

```text
runs/raw_learned_wavelet_auto/generated_00000.png
runs/raw_learned_wavelet_auto/generated_montage.png
runs/raw_learned_wavelet_auto/nearest_source_montage.png
runs/raw_learned_wavelet_auto/manifest.json
```

Use `--dataset ''` to skip nearest-source comparison when generating a larger
unattended batch.
