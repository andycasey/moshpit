"""
Sky background model evaluation.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float

from .utils import poly_features_2d


@jit
def eval_sky(
    sky_coeffs: Float[Array, "6"],
    positions: Float[Array, "n_pix 2"]
) -> Float[Array, "n_pix"]:
    """Evaluate sky background at pixel positions.

    Args:
        sky_coeffs: Polynomial coefficients [1, y, y², x, xy, x²]
        positions: Pixel coordinates (n_pix, 2) with (x, y) ordering

    Returns:
        Sky values at each pixel position
    """
    features = vmap(poly_features_2d)(positions)  # (n_pix, 6)
    return features @ sky_coeffs
