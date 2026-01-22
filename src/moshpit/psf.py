"""
PSF model evaluation and derivatives.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float

from .utils import poly_features_2d_no_constant


@jit
def eval_single_gaussian(
    dx: Float[Array, "n_pix 2"],
    sigma: float
) -> Float[Array, "n_pix"]:
    """Evaluate normalized 2D Gaussian.

    Args:
        dx: Position offsets from Gaussian center (n_pix, 2)
        sigma: Gaussian width

    Returns:
        Gaussian values at each offset
    """
    r2 = jnp.sum(dx**2, axis=-1)
    return jnp.exp(-0.5 * r2 / sigma**2) / (2 * jnp.pi * sigma**2)


@jit
def eval_psf_at_location(
    psf_sigmas: Float[Array, "n_basis"],
    psf_ref_weights: Float[Array, "n_basis"],
    psf_deviation_coeffs: Float[Array, "n_basis 5"],
    psf_coord_scale: float,
    field_pos: Float[Array, "2"],
    dx: Float[Array, "n_pix 2"],
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix"]:
    """Evaluate spatially-varying PSF at a field position.

    The PSF is a weighted sum of Gaussian basis functions where the weights
    vary spatially according to a polynomial model.

    Args:
        psf_sigmas: Gaussian widths for each basis function
        psf_ref_weights: Reference weights at field center
        psf_deviation_coeffs: Polynomial coefficients for weight deviations
        psf_coord_scale: Scale factor for normalizing field coordinates
        field_pos: Position in field where PSF is evaluated
        dx: Pixel offsets from field_pos (n_pix, 2)
        field_center: Center of field for computing deviations

    Returns:
        PSF values at each pixel offset
    """
    # Normalized position relative to field center
    rel_pos = (field_pos - field_center) / psf_coord_scale

    # Deviation features (no constant term)
    dev_features = poly_features_2d_no_constant(rel_pos)  # (5,)

    # Compute deviations for each basis function
    deviations = psf_deviation_coeffs @ dev_features  # (n_basis,)

    # Total weights = reference + deviations
    basis_weights = psf_ref_weights + deviations  # (n_basis,)

    # Evaluate each Gaussian basis using vmap
    def eval_basis(sigma):
        return eval_single_gaussian(dx, sigma)

    gaussians = vmap(eval_basis)(psf_sigmas)  # (n_basis, n_pix)

    # Weighted sum over basis functions
    return jnp.sum(basis_weights[:, None] * gaussians, axis=0)


@jit
def eval_psf_derivative(
    psf_sigmas: Float[Array, "n_basis"],
    psf_ref_weights: Float[Array, "n_basis"],
    psf_deviation_coeffs: Float[Array, "n_basis 5"],
    psf_coord_scale: float,
    field_pos: Float[Array, "2"],
    dx: Float[Array, "n_pix 2"],
    field_center: Float[Array, "2"]
) -> Float[Array, "n_pix 2"]:
    """Evaluate PSF gradient with respect to star position.

    Args:
        psf_sigmas: Gaussian widths for each basis function
        psf_ref_weights: Reference weights at field center
        psf_deviation_coeffs: Polynomial coefficients for weight deviations
        psf_coord_scale: Scale factor for normalizing field coordinates
        field_pos: Position in field where PSF is evaluated
        dx: Pixel offsets from field_pos (n_pix, 2)
        field_center: Center of field for computing deviations

    Returns:
        PSF gradient at each pixel offset (n_pix, 2)
    """
    rel_pos = (field_pos - field_center) / psf_coord_scale
    dev_features = poly_features_2d_no_constant(rel_pos)
    deviations = psf_deviation_coeffs @ dev_features
    basis_weights = psf_ref_weights + deviations

    r2 = jnp.sum(dx**2, axis=-1)

    def eval_basis_deriv(sigma):
        gaussian = jnp.exp(-0.5 * r2 / sigma**2) / (2 * jnp.pi * sigma**2)
        # Derivative: -dx/sigma^2 * gaussian, then negate for d/d(star_pos)
        deriv = dx / sigma**2 * gaussian[:, None]  # (n_pix, 2)
        return deriv

    gaussian_derivs = vmap(eval_basis_deriv)(psf_sigmas)  # (n_basis, n_pix, 2)

    return jnp.sum(basis_weights[:, None, None] * gaussian_derivs, axis=0)
