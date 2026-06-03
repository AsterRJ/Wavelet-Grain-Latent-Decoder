from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from weld_latent.io import build_dataset, sorted_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize weld JPGs into an NPZ tensor for JAX training.")
    parser.add_argument("--input", default="data/data/*.jpg")
    parser.add_argument("--out", default="data/weld_128.npz")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    paths = sorted_images(args.input)
    if not paths:
        raise FileNotFoundError(f"No images matched {args.input!r}")
    images = build_dataset(paths, args.size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=images, paths=np.array(paths))
    print(f"wrote {args.out}: images={images.shape}")


if __name__ == "__main__":
    main()

