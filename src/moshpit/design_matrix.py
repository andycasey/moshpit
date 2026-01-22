"""
Design matrix construction for linear least squares fitting.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float

from .utils import poly_features_2d
from .psf import eval_psf_at_location, eval_psf_derivative


@jit
def build_design_matrix_amplitudes(
    pixel_coords: Float[Array, "n_pix 2"],
    positions: Float[Array, "n_stars 2"],
    psf_sigmas: Float[Array, "n_basis"],
    psf_ref_weights: Float[Array, "n_basis"],
    psf_deviation_coeffs: Float[Array, "n_basis 5"],
    psf_coord_scale: float,
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix n_stars"]:
    """Build design matrix for amplitude parameters.

    Each column corresponds to one star and contains the PSF evaluated at that
    star's position for all pixels.

    Args:
        pixel_coords: Pixel coordinates (n_pix, 2)
        positions: Star positions (n_stars, 2)
        psf_sigmas: PSF Gaussian widths
        psf_ref_weights: PSF reference weights
        psf_deviation_coeffs: PSF deviation coefficients
        psf_coord_scale: PSF coordinate scale
        field_center: Field center for PSF deviations

    Returns:
        Design matrix (n_pix, n_stars)
    """
    def compute_column(star_pos):
        dx = pixel_coords - star_pos
        return eval_psf_at_location(
            psf_sigmas, psf_ref_weights, psf_deviation_coeffs, psf_coord_scale,
            star_pos, dx, field_center
        )

    # vmap over stars
    M = vmap(compute_column)(positions)  # (n_stars, n_pix)
    return M.T  # (n_pix, n_stars)


@jit
def build_design_matrix_positions(
    pixel_coords: Float[Array, "n_pix 2"],
    positions: Float[Array, "n_stars 2"],
    amplitudes: Float[Array, "n_stars"],
    psf_sigmas: Float[Array, "n_basis"],
    psf_ref_weights: Float[Array, "n_basis"],
    psf_deviation_coeffs: Float[Array, "n_basis 5"],
    psf_coord_scale: float,
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix 2*n_stars"]:
    """Build design matrix for position parameters.

    Each star contributes 2 columns (x and y derivatives) to the design matrix.

    Args:
        pixel_coords: Pixel coordinates (n_pix, 2)
        positions: Star positions (n_stars, 2)
        amplitudes: Star amplitudes (n_stars,)
        psf_sigmas: PSF Gaussian widths
        psf_ref_weights: PSF reference weights
        psf_deviation_coeffs: PSF deviation coefficients
        psf_coord_scale: PSF coordinate scale
        field_center: Field center for PSF deviations

    Returns:
        Design matrix (n_pix, 2*n_stars)
    """
    def compute_columns(star_pos, amp):
        dx = pixel_coords - star_pos
        deriv = eval_psf_derivative(
            psf_sigmas, psf_ref_weights, psf_deviation_coeffs, psf_coord_scale,
            star_pos, dx, field_center
        )  # (n_pix, 2)
        return amp * deriv  # (n_pix, 2)

    # vmap over stars
    G_per_star = vmap(compute_columns)(positions, amplitudes)  # (n_stars, n_pix, 2)

    # Reshape to (n_pix, 2*n_stars)
    n_stars = positions.shape[0]
    n_pix = pixel_coords.shape[0]
    return G_per_star.transpose(1, 0, 2).reshape(n_pix, 2 * n_stars)


@jit
def build_design_matrix_sky(
    pixel_coords: Float[Array, "n_pix 2"]
) -> Float[Array, "n_pix 6"]:
    """Build design matrix for sky parameters.

    Each column corresponds to one polynomial term.

    Args:
        pixel_coords: Pixel coordinates (n_pix, 2)

    Returns:
        Design matrix (n_pix, 6) with polynomial features
    """
    return vmap(poly_features_2d)(pixel_coords)
