from __future__ import annotations

import jax
import jax.numpy as jnp


def init_adam(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(params, grads, opt, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = opt["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g * g), opt["v"], grads)
    mhat = jax.tree.map(lambda x: x / (1 - b1**t), m)
    vhat = jax.tree.map(lambda x: x / (1 - b2**t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}
