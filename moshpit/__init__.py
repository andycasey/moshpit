"""
Moshpit: JIT-compiled PSF photometry for JAX.

A package for fitting point spread functions (PSFs) to astronomical images
using JAX for high-performance automatic differentiation and GPU acceleration.
"""

# Data structures
from .models import (
    PSFBasis,
    SkyModel,
    StarParams,
    ModelState,
    POLY_ORDER,
    N_POLY_COEFFS,
    N_DEVIATION_COEFFS
)

# Utilities
from .utils import (
    poly_features_2d,
    poly_features_2d_no_constant,
    solve_weighted_lstsq,
    poisson_log_likelihood
)

# Sky model
from .sky import eval_sky

# PSF model
from .psf import (
    eval_single_gaussian,
    eval_psf_at_location,
    eval_psf_derivative
)

# Forward model
from .forward import (
    eval_star_contribution,
    eval_model
)

# Design matrices
from .design_matrix import (
    build_design_matrix_amplitudes,
    build_design_matrix_positions,
    build_design_matrix_sky
)

# Fitting
from .fitting import (
    irls_step,
    fit_model_jit,
    irls_step_joint,
    fit_model_jit_joint,
    irls_step_joint_linesearch,
    fit_model_jit_joint_linesearch
)

# Visualization
from .visualization import plot_results

__version__ = "0.1.0"

__all__ = [
    # Data structures
    "PSFBasis",
    "SkyModel",
    "StarParams",
    "ModelState",
    "POLY_ORDER",
    "N_POLY_COEFFS",
    "N_DEVIATION_COEFFS",
    # Utilities
    "poly_features_2d",
    "poly_features_2d_no_constant",
    "solve_weighted_lstsq",
    "poisson_log_likelihood",
    # Sky
    "eval_sky",
    # PSF
    "eval_single_gaussian",
    "eval_psf_at_location",
    "eval_psf_derivative",
    # Forward model
    "eval_star_contribution",
    "eval_model",
    # Design matrices
    "build_design_matrix_amplitudes",
    "build_design_matrix_positions",
    "build_design_matrix_sky",
    # Fitting
    "irls_step",
    "fit_model_jit",
    "irls_step_joint",
    "fit_model_jit_joint",
    # Visualization
    "plot_results",
]
