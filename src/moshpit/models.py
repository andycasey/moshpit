"""
Data structures for PSF photometry model.
"""

from typing import NamedTuple
from jaxtyping import Array, Float


class PSFBasis(NamedTuple):
    """PSF with spatially varying weights.

    Attributes:
        sigmas: Gaussian widths for each basis function
        ref_weights: Reference weights at field center
        deviation_coeffs: Polynomial coefficients for spatial variation (excludes constant term)
        coord_scale: Scale factor for normalizing field coordinates
    """
    sigmas: Float[Array, "n_basis"]
    ref_weights: Float[Array, "n_basis"]
    deviation_coeffs: Float[Array, "n_basis 5"]  # Fixed size for order=2 (JIT compatible)
    coord_scale: float = 32.0


class SkyModel(NamedTuple):
    """Sky background as 2D polynomial.

    Attributes:
        coeffs: Polynomial coefficients [1, y, y², x, xy, x²] for order=2
    """
    coeffs: Float[Array, "6"]  # Fixed size for order=2 (JIT compatible)


class StarParams(NamedTuple):
    """Star parameters.

    Attributes:
        positions: Star positions (x, y) in pixels
        amplitudes: Star amplitudes/fluxes per band
    """
    positions: Float[Array, "n_stars 2"]
    amplitudes: Float[Array, "n_bands n_stars"]


class ModelState(NamedTuple):
    """Full model state for single band (JIT compatible).

    This flattened structure allows JIT compilation by avoiding lists.

    Attributes:
        positions: Star positions
        amplitudes: Star amplitudes for this band
        psf_sigmas: PSF basis Gaussian widths
        psf_ref_weights: PSF reference weights
        psf_deviation_coeffs: PSF deviation coefficients
        psf_coord_scale: PSF coordinate scale
        sky_coeffs: Sky polynomial coefficients
    """
    positions: Float[Array, "n_stars 2"]
    amplitudes: Float[Array, "n_stars"]
    psf_sigmas: Float[Array, "n_basis"]
    psf_ref_weights: Float[Array, "n_basis"]
    psf_deviation_coeffs: Float[Array, "n_basis 5"]
    psf_coord_scale: float
    sky_coeffs: Float[Array, "6"]


# Fixed polynomial order for JIT compatibility
POLY_ORDER = 2
N_POLY_COEFFS = 6  # (2+1)*(2+2)//2 = 6
N_DEVIATION_COEFFS = 5  # N_POLY_COEFFS - 1
