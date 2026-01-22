"""
Forward model evaluation.
"""

import jax.numpy as jnp
from jax import jit, lax
from jaxtyping import Array, Float

from .models import ModelState
from .sky import eval_sky
from .psf import eval_psf_at_location


@jit
def eval_star_contribution(
    pixel_coords: Float[Array, "n_pix 2"],
    star_pos: Float[Array, "2"],
    amplitude: float,
    psf_sigmas: Float[Array, "n_basis"],
    psf_ref_weights: Float[Array, "n_basis"],
    psf_deviation_coeffs: Float[Array, "n_basis 5"],
    psf_coord_scale: float,
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix"]:
    """Evaluate contribution from a single star.

    Args:
        pixel_coords: Pixel coordinates (n_pix, 2)
        star_pos: Star position (x, y)
        amplitude: Star amplitude/flux
        psf_sigmas: PSF Gaussian widths
        psf_ref_weights: PSF reference weights
        psf_deviation_coeffs: PSF deviation coefficients
        psf_coord_scale: PSF coordinate scale
        field_center: Field center for PSF deviations

    Returns:
        Star's contribution to each pixel
    """
    dx = pixel_coords - star_pos
    psf_vals = eval_psf_at_location(
        psf_sigmas, psf_ref_weights, psf_deviation_coeffs, psf_coord_scale,
        star_pos, dx, field_center
    )
    return amplitude * psf_vals


@jit
def eval_model(
    pixel_coords: Float[Array, "n_pix 2"],
    state: ModelState,
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix"]:
    """Evaluate full forward model (sky + all stars).

    Args:
        pixel_coords: Pixel coordinates (n_pix, 2)
        state: Model state containing all parameters
        field_center: Field center for PSF deviations

    Returns:
        Model prediction at each pixel
    """
    # Sky background
    model = eval_sky(state.sky_coeffs, pixel_coords)

    # Add stars using lax.scan for JIT compatibility
    def add_star(carry, star_data):
        pos, amp = star_data
        contrib = eval_star_contribution(
            pixel_coords, pos, amp,
            state.psf_sigmas, state.psf_ref_weights,
            state.psf_deviation_coeffs, state.psf_coord_scale,
            field_center
        )
        return carry + contrib, None

    star_data = (state.positions, state.amplitudes)
    model, _ = lax.scan(add_star, model, star_data)

    return model
