import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self):
        self.gamma = nnx.Param(jnp.ones((1,)))
        self.beta = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x / jnp.sqrt(rms + 1e-6)
        x = x * self.gamma + self.beta
        return x


class Embed(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        model_dtyp: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.norm = RMSNorm()
        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=model_dim,
            dtype=model_dtyp,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        x = self.embed.attend(x)
        return x


class Linear(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(
            in_features=din,
            out_features=dout,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x)
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.linear1 = Linear(din, dout * 4, jnp.float32, rngs)
        self.linear2 = Linear(dout * 4, dout, jnp.float32, rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class RoPE(nnx.Module):
    def __init__(
        self,
        T: int,
        model_dim: int,
    ):
        assert model_dim % 2 == 0, "Model dimension must be even for RoPE."

        freq = jnp.arange(1, T + 1, dtype=jnp.float32)[:, None]

        position = jnp.arange(model_dim // 2, dtype=jnp.float32)[:, None]
        position = position.repeat(2, axis=1).reshape(1, -1)
        log_theta_base = jnp.log(10000.0)
        theta = jnp.exp(-1 * position / model_dim * log_theta_base)

        self.cos = jnp.cos(freq * theta)
        self.sin = jnp.sin(freq * theta)

    def __call__(
        self,
        x: jax.Array,
        seq_start: int,
    ) -> jax.Array:
        B, T, C = x.shape
        dtype = x.dtype
        x = x.astype(jnp.float32)

        cos_rope = x * self.cos[seq_start : seq_start + T, :]

        x = x.reshape(B, T, C // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1] * -1
        x = jnp.stack([x2, x1], axis=-1)
        x = x.reshape((B, T, C))

        sin_rope = x * self.sin[seq_start : seq_start + T, :]

        x = cos_rope + sin_rope
        x = x.astype(dtype)
        return x
