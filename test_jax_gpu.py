import jax
import jax.numpy as jnp

print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

x = jnp.ones((2048, 2048))
y = x @ x
print("Result:", y[0, 0])
