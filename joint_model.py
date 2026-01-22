"""
JIT-compatible PSF photometry model.

Key changes from model.py:
1. Use jax.lax.fori_loop instead of Python for loops
2. Use static poly_order (hardcoded or passed as static_argnums)
3. Use vmap for vectorization over stars
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from typing import NamedTuple
from jaxtyping import Array, Float


# Fixed polynomial order for JIT compatibility
POLY_ORDER = 2
N_POLY_COEFFS = 6  # (2+1)*(2+2)//2 = 6
N_DEVIATION_COEFFS = 5  # N_POLY_COEFFS - 1


class PSFBasis(NamedTuple):
    """PSF with spatially varying weights."""
    sigmas: Float[Array, "n_basis"]
    ref_weights: Float[Array, "n_basis"]
    deviation_coeffs: Float[Array, "n_basis 5"]  # Fixed size for JIT
    coord_scale: float = 32.0


class SkyModel(NamedTuple):
    """Sky as 2D polynomial (order 2)."""
    coeffs: Float[Array, "6"]  # Fixed size for order=2


class StarParams(NamedTuple):
    """Star parameters."""
    positions: Float[Array, "n_stars 2"]
    amplitudes: Float[Array, "n_bands n_stars"]


class ModelState(NamedTuple):
    """Full model state for single band (for JIT compatibility)."""
    positions: Float[Array, "n_stars 2"]
    amplitudes: Float[Array, "n_stars"]
    psf_sigmas: Float[Array, "n_basis"]
    psf_ref_weights: Float[Array, "n_basis"]
    psf_deviation_coeffs: Float[Array, "n_basis 5"]
    psf_coord_scale: float
    sky_coeffs: Float[Array, "6"]


@jit
def poly_features_2d(x: Float[Array, "2"]) -> Float[Array, "6"]:
    """2D polynomial features for order=2: [1, y, y², x, xy, x²]"""
    return jnp.array([
        1.0,
        x[1],
        x[1]**2,
        x[0],
        x[0] * x[1],
        x[0]**2
    ])


@jit
def poly_features_2d_no_constant(x: Float[Array, "2"]) -> Float[Array, "5"]:
    """2D polynomial features without constant: [y, y², x, xy, x²]"""
    return jnp.array([
        x[1],
        x[1]**2,
        x[0],
        x[0] * x[1],
        x[0]**2
    ])


@jit
def eval_sky(sky_coeffs: Float[Array, "6"], positions: Float[Array, "n_pix 2"]) -> Float[Array, "n_pix"]:
    """Evaluate sky at pixel positions."""
    features = vmap(poly_features_2d)(positions)  # (n_pix, 6)
    return features @ sky_coeffs


@jit
def eval_single_gaussian(
    dx: Float[Array, "n_pix 2"],
    sigma: float
) -> Float[Array, "n_pix"]:
    """Evaluate normalized Gaussian."""
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
    """Evaluate PSF at field position for pixel offsets dx."""
    # Normalized position relative to field center
    rel_pos = (field_pos - field_center) / psf_coord_scale

    # Deviation features (no constant)
    dev_features = poly_features_2d_no_constant(rel_pos)  # (5,)

    # Deviations for each basis: (n_basis,)
    deviations = psf_deviation_coeffs @ dev_features

    # Total weights
    basis_weights = psf_ref_weights + deviations  # (n_basis,)

    # Evaluate each Gaussian basis using vmap over basis functions
    def eval_basis(sigma):
        return eval_single_gaussian(dx, sigma)

    gaussians = vmap(eval_basis)(psf_sigmas)  # (n_basis, n_pix)

    # Weighted sum
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
    """Evaluate PSF gradient w.r.t. star position."""
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
    """Evaluate contribution from a single star."""
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
    """Evaluate full model (sky + all stars)."""
    # Sky
    model = eval_sky(state.sky_coeffs, pixel_coords)

    # Stars - use vmap over stars
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
    """Build design matrix for amplitudes."""
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
    """Build design matrix for position updates."""
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
def build_design_matrix_sky(pixel_coords: Float[Array, "n_pix 2"]) -> Float[Array, "n_pix 6"]:
    """Build sky design matrix."""
    return vmap(poly_features_2d)(pixel_coords)


@jit
def solve_weighted_lstsq(
    A: Float[Array, "n m"],
    b: Float[Array, "n"],
    weights: Float[Array, "n"],
    regularization: float = 1e-6
) -> Float[Array, "m"]:
    """Solve weighted least squares."""
    WA = weights[:, None] * A
    AtwA = A.T @ WA + regularization * jnp.eye(A.shape[1])
    Atwb = A.T @ (weights * b)
    return jnp.linalg.solve(AtwA, Atwb)


@jit
def poisson_log_likelihood(data: Float[Array, "n_pix"], model: Float[Array, "n_pix"]) -> float:
    """Compute Poisson log-likelihood."""
    model_safe = jnp.maximum(model, 1e-10)
    return jnp.sum(data * jnp.log(model_safe) - model_safe)


@partial(jit, static_argnums=(5, 6, 7))
def irls_step(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    state: ModelState,
    field_center: Float[Array, "2"],
    floor: float = 1.0,
    fit_sky: bool = True,
    fit_psf: bool = False,
    damping: float = 0.5
) -> ModelState:
    """One IRLS iteration for single band.

    Uses Fisher scoring: solve for parameter UPDATES, not full parameters.
    For Poisson: (X^T W X) delta = X^T W (d - mu) / mu * mu = X^T (d - mu)
    which simplifies to: (X^T W X) delta = X^T W (d - mu) with W = 1/mu
    """
    regularization_sky = 1e-6 #1e-3
    regularization_amplitudes = 1e-6 #1e-3
    regularization_positions = 1e-6 #1e-2

    # Current model and weights
    model = eval_model(pixel_coords, state, field_center)
    model = jnp.maximum(model, floor)
    weights = 1.0 / model

    # Residual: data - current model
    residual = data - model

    # Design matrices
    S = build_design_matrix_sky(pixel_coords)
    M = build_design_matrix_amplitudes(
        pixel_coords, state.positions,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    )

    # Step 1: Update sky (solve for delta_sky)
    def update_sky(_):
        delta_sky = solve_weighted_lstsq(S, residual, weights, regularization_sky)
        return state.sky_coeffs + damping * delta_sky

    sky_coeffs = lax.cond(fit_sky, update_sky, lambda _: state.sky_coeffs, None)

    # Recompute model and residual with new sky
    sky_model = S @ sky_coeffs
    source_model = M @ state.amplitudes
    model = sky_model + source_model
    model = jnp.maximum(model, floor)
    weights = 1.0 / model
    residual = data - model

    # Step 2: Update amplitudes (solve for delta_A)
    delta_A = solve_weighted_lstsq(M, residual, weights, regularization_amplitudes)
    amplitudes = state.amplitudes + damping * delta_A
    amplitudes = jnp.maximum(amplitudes, 1.0)

    # Recompute model and residual with new amplitudes
    source_model = M @ amplitudes
    model = sky_model + source_model
    model = jnp.maximum(model, floor)
    weights = 1.0 / model
    residual = data - model

    # Step 3: Update positions (solve for delta_pos)
    G = build_design_matrix_positions(
        pixel_coords, state.positions, amplitudes,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    )

    n_stars = state.positions.shape[0]
    delta_pos = solve_weighted_lstsq(G, residual, weights, regularization_positions)
    delta_pos = jnp.clip(delta_pos, -2.0, 2.0)
    positions = state.positions + 0.3 * delta_pos.reshape(n_stars, 2)

    return ModelState(
        positions=positions,
        amplitudes=amplitudes,
        psf_sigmas=state.psf_sigmas,
        psf_ref_weights=state.psf_ref_weights,
        psf_deviation_coeffs=state.psf_deviation_coeffs,
        psf_coord_scale=state.psf_coord_scale,
        sky_coeffs=sky_coeffs
    )

@partial(jit, static_argnums=(4, 5, 6))
def fit_model_jit(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    initial_state: ModelState,
    field_center: Float[Array, "2"],
    n_iter: int = 30,
    fit_sky: bool = True,
    track_likelihood: bool = False
):
    """Fit model using JIT-compiled IRLS.

    Args:
        track_likelihood: If True, return (final_state, log_likes). Otherwise just final_state.
    """

    # Always use lax.scan with log-likelihood array in carry
    # Pre-allocate array for log-likelihoods (n_iter + 1 for initial + each iteration)
    log_likes_array = jnp.zeros(n_iter + 1)

    # Compute initial log-likelihood
    init_model = eval_model(pixel_coords, initial_state, field_center)
    init_ll = poisson_log_likelihood(data, init_model)
    log_likes_array = log_likes_array.at[0].set(init_ll)


    def body_fn(carry, i):
        state, ll_array = carry

        # Update state
        new_state = irls_step(data, pixel_coords, state, field_center,
                             fit_sky=fit_sky, fit_psf=False)

        # Compute log-likelihood after update
        model = eval_model(pixel_coords, new_state, field_center)
        ll = poisson_log_likelihood(data, model)

        # Store log-likelihood (i+1 because index 0 is initial)
        ll_array = ll_array.at[i + 1].set(ll)

        return (new_state, ll_array), None

    # Run optimization with lax.scan
    (final_state, log_likes_array), _ = lax.scan(
        body_fn,
        (initial_state, log_likes_array),
        jnp.arange(n_iter)
    )

    if track_likelihood:
        return final_state, log_likes_array
    else:
        return final_state


@partial(jit, static_argnums=(5, 6, 7, 8))
def irls_step_joint(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    state: ModelState,
    field_center: Float[Array, "2"],
    floor: float = 1.0,
    fit_sky: bool = True,
    fit_amplitudes: bool = True,
    fit_positions: bool = True,
    damping: float = 0.5,
    regularization: float = 1e-6
) -> ModelState:
    """One IRLS iteration solving all parameters jointly.

    Instead of alternating between sky, amplitudes, and positions,
    this solves for all parameter updates simultaneously in a single
    large least squares problem.
    """

    # Current model and weights
    model = eval_model(pixel_coords, state, field_center)
    model = jnp.maximum(model, floor)
    weights = 1.0 / model

    # Residual: data - current model
    residual = data - model

    # Build all design matrices
    S = build_design_matrix_sky(pixel_coords) if fit_sky else None
    M = build_design_matrix_amplitudes(
        pixel_coords, state.positions,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    ) if fit_amplitudes else None
    G = build_design_matrix_positions(
        pixel_coords, state.positions, state.amplitudes,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    ) if fit_positions else None

    # Concatenate design matrices horizontally
    design_matrices = []
    if fit_sky and S is not None:
        design_matrices.append(S)
    if fit_amplitudes and M is not None:
        design_matrices.append(M)
    if fit_positions and G is not None:
        design_matrices.append(G)

    if len(design_matrices) == 0:
        # Nothing to fit, return unchanged
        return state

    # Combined design matrix: [S | M | G]
    X = jnp.concatenate(design_matrices, axis=1)

    # Solve joint least squares: X^T W X delta = X^T W residual
    delta = solve_weighted_lstsq(X, residual, weights, regularization)

    # Extract parameter updates from delta vector
    idx = 0

    # Extract sky update
    if fit_sky and S is not None:
        n_sky = S.shape[1]
        delta_sky = delta[idx:idx + n_sky]
        sky_coeffs = state.sky_coeffs + damping * delta_sky
        idx += n_sky
    else:
        sky_coeffs = state.sky_coeffs

    # Extract amplitude updates
    if fit_amplitudes and M is not None:
        n_amps = M.shape[1]
        delta_amps = delta[idx:idx + n_amps]
        amplitudes = state.amplitudes + damping * delta_amps
        amplitudes = jnp.maximum(amplitudes, 1.0)  # Keep positive
        idx += n_amps
    else:
        amplitudes = state.amplitudes

    # Extract position updates
    if fit_positions and G is not None:
        n_pos = G.shape[1]
        delta_pos = delta[idx:idx + n_pos]
        delta_pos = jnp.clip(delta_pos, -2.0, 2.0)  # Limit step size
        n_stars = state.positions.shape[0]
        positions = state.positions + 0.3 * delta_pos.reshape(n_stars, 2)
    else:
        positions = state.positions

    return ModelState(
        positions=positions,
        amplitudes=amplitudes,
        psf_sigmas=state.psf_sigmas,
        psf_ref_weights=state.psf_ref_weights,
        psf_deviation_coeffs=state.psf_deviation_coeffs,
        psf_coord_scale=state.psf_coord_scale,
        sky_coeffs=sky_coeffs
    )


@partial(jit, static_argnums=(4, 5, 6))
def fit_model_jit_joint(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    initial_state: ModelState,
    field_center: Float[Array, "2"],
    n_iter: int = 30,
    fit_sky: bool = True,
    track_likelihood: bool = False
):
    """Fit model using joint least squares (all parameters updated simultaneously).
    """

    # Pre-allocate array for log-likelihoods
    log_likes_array = jnp.zeros(n_iter + 1)

    # Compute initial log-likelihood
    init_model = eval_model(pixel_coords, initial_state, field_center)
    init_ll = poisson_log_likelihood(data, init_model)
    log_likes_array = log_likes_array.at[0].set(init_ll)

    def body_fn(carry, i):
        state, ll_array = carry

        # Update all parameters jointly
        new_state = irls_step_joint(
            data, pixel_coords, state, field_center,
            fit_sky=fit_sky,
            fit_amplitudes=True,
            fit_positions=True
        )

        # Compute log-likelihood after update
        model = eval_model(pixel_coords, new_state, field_center)
        ll = poisson_log_likelihood(data, model)

        # Store log-likelihood
        ll_array = ll_array.at[i + 1].set(ll)

        return (new_state, ll_array), None

    # Run optimization with lax.scan
    (final_state, log_likes_array), _ = lax.scan(
        body_fn,
        (initial_state, log_likes_array),
        jnp.arange(n_iter)
    )

    if track_likelihood:
        return final_state, log_likes_array
    else:
        return final_state


def apply_parameter_update(
    state: ModelState,
    delta_sky: Float[Array, "6"],
    delta_amps: Float[Array, "n_stars"],
    delta_pos: Float[Array, "2*n_stars"],
    step_size: float,
    fit_sky: bool
) -> ModelState:
    """Apply parameter updates with given step size."""

    # Update sky
    if fit_sky:
        sky_coeffs = state.sky_coeffs + step_size * delta_sky
    else:
        sky_coeffs = state.sky_coeffs

    # Update amplitudes
    amplitudes = state.amplitudes + step_size * delta_amps
    amplitudes = jnp.maximum(amplitudes, 1.0)

    # Update positions
    delta_pos_clipped = jnp.clip(delta_pos, -2.0, 2.0)
    n_stars = state.positions.shape[0]
    positions = state.positions + 0.3 * step_size * delta_pos_clipped.reshape(n_stars, 2)

    return ModelState(
        positions=positions,
        amplitudes=amplitudes,
        psf_sigmas=state.psf_sigmas,
        psf_ref_weights=state.psf_ref_weights,
        psf_deviation_coeffs=state.psf_deviation_coeffs,
        psf_coord_scale=state.psf_coord_scale,
        sky_coeffs=sky_coeffs
    )


@partial(jit, static_argnums=(5,))
def irls_step_joint_linesearch(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    state: ModelState,
    field_center: Float[Array, "2"],
    floor: float = 1.0,
    fit_sky: bool = True,
    regularization: float = 1e-6
) -> ModelState:
    """One IRLS iteration with backtracking line search to ensure LL increases.

    This version guarantees monotonic increase in log-likelihood by
    adaptively reducing the step size if needed.
    """

    # Current model and log-likelihood
    current_model = eval_model(pixel_coords, state, field_center)
    current_model = jnp.maximum(current_model, floor)
    current_ll = poisson_log_likelihood(data, current_model)

    # Compute weights and residual
    weights = 1.0 / current_model
    residual = data - current_model

    # Build all design matrices
    S = build_design_matrix_sky(pixel_coords) if fit_sky else None
    M = build_design_matrix_amplitudes(
        pixel_coords, state.positions,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    )
    G = build_design_matrix_positions(
        pixel_coords, state.positions, state.amplitudes,
        state.psf_sigmas, state.psf_ref_weights,
        state.psf_deviation_coeffs, state.psf_coord_scale,
        field_center
    )

    # Concatenate design matrices
    design_matrices = []
    if fit_sky and S is not None:
        design_matrices.append(S)
    design_matrices.append(M)
    design_matrices.append(G)

    X = jnp.concatenate(design_matrices, axis=1)

    # Solve joint least squares
    delta = solve_weighted_lstsq(X, residual, weights, regularization)

    # Extract individual parameter updates
    idx = 0
    if fit_sky and S is not None:
        n_sky = S.shape[1]
        delta_sky = delta[idx:idx + n_sky]
        idx += n_sky
    else:
        delta_sky = jnp.zeros(6)

    n_amps = M.shape[1]
    delta_amps = delta[idx:idx + n_amps]
    idx += n_amps

    n_pos = G.shape[1]
    delta_pos = delta[idx:idx + n_pos]

    # Backtracking line search
    step_sizes = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])

    def try_step(step_size):
        candidate_state = apply_parameter_update(
            state, delta_sky, delta_amps, delta_pos, step_size, fit_sky
        )
        candidate_model = eval_model(pixel_coords, candidate_state, field_center)
        candidate_model = jnp.maximum(candidate_model, floor)
        candidate_ll = poisson_log_likelihood(data, candidate_model)
        return candidate_state, candidate_ll

    # Try step sizes in order until we find one that increases LL
    def body_fn(carry, step_size):
        best_state, best_ll, found = carry
        candidate_state, candidate_ll = try_step(step_size)

        # If we haven't found a good step yet and this one is better, use it
        improved = candidate_ll > current_ll
        should_update = ~found & improved

        new_state = lax.cond(
            should_update,
            lambda _: candidate_state,
            lambda _: best_state,
            None
        )
        new_ll = lax.cond(
            should_update,
            lambda _: candidate_ll,
            lambda _: best_ll,
            None
        )
        new_found = found | improved

        return (new_state, new_ll, new_found), None

    (final_state, _, _), _ = lax.scan(
        body_fn,
        (state, current_ll, False),
        step_sizes
    )

    return final_state


@partial(jit, static_argnums=(4, 5, 6))
def fit_model_jit_joint_linesearch(
    data: Float[Array, "n_pix"],
    pixel_coords: Float[Array, "n_pix 2"],
    initial_state: ModelState,
    field_center: Float[Array, "2"],
    n_iter: int = 30,
    fit_sky: bool = True,
    track_likelihood: bool = False
):
    """Fit model using joint least squares with line search.

    This version uses backtracking line search to guarantee monotonic
    increase in log-likelihood, fixing the oscillation problem in
    fit_model_jit_joint.
    """

    # Pre-allocate array for log-likelihoods
    log_likes_array = jnp.zeros(n_iter + 1)

    # Compute initial log-likelihood
    init_model = eval_model(pixel_coords, initial_state, field_center)
    init_ll = poisson_log_likelihood(data, init_model)
    log_likes_array = log_likes_array.at[0].set(init_ll)

    def body_fn(carry, i):
        state, ll_array = carry

        # Update all parameters jointly with line search
        new_state = irls_step_joint_linesearch(
            data, pixel_coords, state, field_center,
            fit_sky=fit_sky
        )

        # Compute log-likelihood after update
        model = eval_model(pixel_coords, new_state, field_center)
        ll = poisson_log_likelihood(data, model)

        # Store log-likelihood
        ll_array = ll_array.at[i + 1].set(ll)

        return (new_state, ll_array), None

    # Run optimization with lax.scan
    (final_state, log_likes_array), _ = lax.scan(
        body_fn,
        (initial_state, log_likes_array),
        jnp.arange(n_iter)
    )

    if track_likelihood:
        return final_state, log_likes_array
    else:
        return final_state


def plot_results(
    data: Float[Array, "n_pix"],
    true_model: Float[Array, "n_pix"],
    fitted_model: Float[Array, "n_pix"],
    true_state: ModelState,
    initial_state: ModelState,
    final_state: ModelState,
    image_shape: tuple,
    log_likes: list = None,
    prefix: str = "jit"
):
    """Plot comparison of true, initial, and fitted results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True image (noiseless)
    ax = axes[0, 0]
    im = ax.imshow(true_model.reshape(image_shape), origin='lower', cmap='viridis')
    ax.set_title('True (noiseless)')
    plt.colorbar(im, ax=ax)

    # Noisy data
    ax = axes[0, 1]
    im = ax.imshow(data.reshape(image_shape), origin='lower', cmap='viridis')
    ax.set_title('Noisy data')
    plt.colorbar(im, ax=ax)

    # Fitted model
    ax = axes[1, 0]
    im = ax.imshow(fitted_model.reshape(image_shape), origin='lower', cmap='viridis')
    ax.set_title('Fitted model')
    plt.colorbar(im, ax=ax)

    # Residual
    residual = data - fitted_model
    ax = axes[1, 1]
    vmax = 3 * jnp.std(residual)
    im = ax.imshow(residual.reshape(image_shape), origin='lower', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax)
    ax.set_title('Residual (data - model)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    filename = f'{prefix}_images_comparison.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

    # Plot positions comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    true_pos = true_state.positions
    init_pos = initial_state.positions
    final_pos = final_state.positions

    ax = axes[0]
    ax.scatter(true_pos[:, 0], true_pos[:, 1], s=100, c='green', marker='o',
              label='True', alpha=0.7, edgecolors='black', linewidths=1.5)
    ax.scatter(init_pos[:, 0], init_pos[:, 1], s=80, c='red', marker='x',
              label='Initial', alpha=0.7, linewidths=2)
    ax.scatter(final_pos[:, 0], final_pos[:, 1], s=80, c='blue', marker='+',
              label='Fitted', linewidths=2)

    # Draw lines from initial to fitted
    for i in range(len(true_pos)):
        ax.plot([init_pos[i, 0], final_pos[i, 0]],
               [init_pos[i, 1], final_pos[i, 1]],
               'k--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Star Positions')
    ax.legend()
    ax.set_xlim(-2, image_shape[1] + 2)
    ax.set_ylim(-2, image_shape[0] + 2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Position errors
    ax = axes[1]
    init_errors = jnp.sqrt(jnp.sum((init_pos - true_pos)**2, axis=1))
    final_errors = jnp.sqrt(jnp.sum((final_pos - true_pos)**2, axis=1))

    x = jnp.arange(len(init_errors))
    width = 0.35
    ax.bar(x - width/2, init_errors, width, label='Initial', color='red', alpha=0.7)
    ax.bar(x + width/2, final_errors, width, label='Fitted', color='blue', alpha=0.7)
    ax.set_xlabel('Star index')
    ax.set_ylabel('Position error (pixels)')
    ax.set_title(f'Position Errors (RMSE: init={jnp.sqrt(jnp.mean(init_errors**2)):.3f}, '
                f'final={jnp.sqrt(jnp.mean(final_errors**2)):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'{prefix}_positions_comparison.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

    # Plot amplitude comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    true_amp = true_state.amplitudes
    init_amp = initial_state.amplitudes
    final_amp = final_state.amplitudes

    x = jnp.arange(len(true_amp))
    width = 0.25

    ax.bar(x - width, init_amp, width, label='Initial', color='red', alpha=0.7)
    ax.bar(x, final_amp, width, label='Fitted', color='blue', alpha=0.7)
    ax.bar(x + width, true_amp, width, label='True', color='green', alpha=0.7)

    ax.set_xlabel('Star index')
    ax.set_ylabel('Amplitude (counts)')
    ax.set_title('Star Amplitudes')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'{prefix}_amplitudes_comparison.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

    # Plot convergence if log_likes provided
    if log_likes is not None and len(log_likes) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        import numpy as np
        log_likes = np.array(log_likes)
        n_iter = len(log_likes)

        # Full convergence plot
        ax = axes[0]
        ax.plot(log_likes, 'b-o', markersize=4)
        min_lls = np.min(log_likes[1:])
        max_lls = np.max(log_likes)
        ax.set_ylim(min_lls - 0.1 * (max_lls - min_lls),
                    max_lls + 0.1 * (max_lls - min_lls))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('Full Convergence')
        ax.grid(True, alpha=0.3)

        # Zoomed-in view of first ~50 iterations (skip 1st iteration for better dynamic range)
        ax = axes[1]
        first_n = min(50, n_iter)
        first_iters = list(range(1, first_n))
        first_lls = log_likes[1:first_n]

        ax.plot(first_iters, first_lls, 'b-o', markersize=5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title(f'Zoom: Iterations 2-{first_n}')
        ax.grid(True, alpha=0.3)

        # Add text showing improvement
        if len(first_lls) > 1:
            ll_improvement = first_lls[-1] - first_lls[0]
            ax.text(0.05, 0.95, f'ΔLL = {ll_improvement:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Zoomed-in view of last iterations
        ax = axes[2]
        # Show last 50% of iterations or at least last 10
        zoom_start = max(0, n_iter // 2)
        zoom_iters = list(range(zoom_start, n_iter))
        zoom_lls = log_likes[zoom_start:]

        ax.plot(zoom_iters, zoom_lls, 'b-o', markersize=5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title(f'Zoom: Last {n_iter - zoom_start} Iterations')
        ax.grid(True, alpha=0.3)

        # Add text showing improvement
        if len(zoom_lls) > 1:
            ll_improvement = zoom_lls[-1] - zoom_lls[0]
            ax.text(0.05, 0.95, f'ΔLL = {ll_improvement:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        filename = f'{prefix}_convergence.png'
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close()


if __name__ == '__main__':
    import jax.random as jr

    print("Testing JIT-compatible model")
    print("="*50)

    key = jr.PRNGKey(42)
    keys = jr.split(key, 4)

    n_iter = 20
    # Create test data
    n_pix_side = 4096 # 4096
    n_stars = 256 # 25_600

    #n_pix_side, n_stars = (4096, 25_600)

    pixel_coords = jnp.stack(
        jnp.meshgrid(jnp.arange(n_pix_side), jnp.arange(n_pix_side)),
        axis=-1
    ).reshape(-1, 2).astype(float)
    field_center = jnp.array([n_pix_side/2, n_pix_side/2])

    # Generate random star positions (avoiding edges)
    edge_margin = -3
    true_positions = jr.uniform(
        keys[0],
        (n_stars, 2),
        minval=edge_margin,
        maxval=n_pix_side - edge_margin
    )

    # Generate random star amplitudes (log-uniform distribution)
    log_amps = jr.uniform(keys[1], (n_stars,), minval=jnp.log(500.0), maxval=jnp.log(5000.0))
    true_amplitudes = jnp.exp(log_amps)



    # True state
    true_state = ModelState(
        positions=true_positions,
        amplitudes=true_amplitudes,
        psf_sigmas=jnp.array([2.0]),
        psf_ref_weights=jnp.array([1.0]),
        psf_deviation_coeffs=jnp.zeros((1, 5)),
        psf_coord_scale=50.0,
        sky_coeffs=jnp.array([100., 0., 0., 0., 0., 0.])
    )

    # Generate data
    print(f"Generating {n_stars} stars in {n_pix_side}x{n_pix_side} image...")
    true_model = eval_model(pixel_coords, true_state, field_center)
    data = jr.poisson(keys[2], true_model).astype(float)

    print(f"Data shape: {data.shape}")
    print(f"Data mean: {jnp.mean(data):.2f}")
    print(f"Number of stars: {n_stars}")

    # Initial guess (perturbed)
    # Perturb positions
    init_positions = true_state.positions + jr.normal(keys[3], (n_stars, 2)) * 1.5

    # Estimate amplitudes from pixel values at initial positions
    # Reshape data to 2D image
    data_2d = data.reshape(n_pix_side, n_pix_side)

    # Convert positions to pixel indices (with clipping to stay in bounds)
    pixel_x = jnp.clip(jnp.round(init_positions[:, 0]).astype(int), 0, n_pix_side - 1)
    pixel_y = jnp.clip(jnp.round(init_positions[:, 1]).astype(int), 0, n_pix_side - 1)

    # Sample pixel values at initial positions
    pixel_values = data_2d[pixel_y, pixel_x]

    # Estimate amplitude from pixel value / PSF peak
    # Subtract a rough sky estimate (median of data as rough background)
    rough_sky = jnp.median(data)
    pixel_values_no_sky = jnp.maximum(pixel_values - rough_sky, 1.0)

    # For normalized Gaussian: peak = 1 / (2 * pi * sigma^2)
    psf_sigma = true_state.psf_sigmas[0]
    psf_peak = 1.0 / (2 * jnp.pi * psf_sigma**2)
    init_amplitudes = pixel_values_no_sky / psf_peak
    init_amplitudes = jnp.maximum(init_amplitudes, 100.0)  # Ensure positive and reasonable

    # Initialize sky coefficients randomly (centered on zero with reasonable base level)
    # Keep base level around 100-150, randomize other terms
    init_sky_base = 100.0 + jr.uniform(jr.PRNGKey(888), (), minval=-20.0, maxval=50.0)
    init_sky_other = jr.normal(jr.PRNGKey(777), (5,)) * 5.0  # Small random values
    init_sky_coeffs = jnp.concatenate([jnp.array([init_sky_base]), init_sky_other])

    init_state = ModelState(
        positions=init_positions,
        amplitudes=init_amplitudes,
        psf_sigmas=true_state.psf_sigmas,
        psf_ref_weights=true_state.psf_ref_weights,
        psf_deviation_coeffs=true_state.psf_deviation_coeffs,
        psf_coord_scale=true_state.psf_coord_scale,
        sky_coeffs=init_sky_coeffs
    )

    print(f"\nTrue positions (first 5):\n{true_state.positions[:5]}")
    print(f"Initial positions (first 5):\n{init_state.positions[:5]}")
    print(f"\nTrue amplitudes (first 5): {true_state.amplitudes[:5]}")
    print(f"Initial amplitudes (first 5): {init_state.amplitudes[:5]}")
    print(f"\nTrue sky coeffs: {true_state.sky_coeffs}")
    print(f"Initial sky coeffs: {init_state.sky_coeffs}")

    import time

    t0 = time.time()
    final_state_joint, log_likes_joint = fit_model_jit_joint_linesearch(
        data, pixel_coords, init_state, field_center,
        n_iter=n_iter, fit_sky=True, track_likelihood=True
    )
    final_state_joint.positions[0].block_until_ready()
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")

    t0 = time.time()
    final_state_joint, log_likes_joint = fit_model_jit_joint_linesearch(
        data, pixel_coords, init_state, field_center,
        n_iter=n_iter, fit_sky=True, track_likelihood=True
    )
    final_state_joint.positions[0].block_until_ready()
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")


    pos_errors_joint = jnp.sqrt(jnp.sum((final_state_joint.positions - true_state.positions)**2, axis=1))
    amp_errors_joint = jnp.abs(final_state_joint.amplitudes - true_state.amplitudes) / true_state.amplitudes * 100

    print(f"\nPosition RMSE: {jnp.sqrt(jnp.mean(pos_errors_joint**2)):.4f}")
    print(f"Position median error: {jnp.median(pos_errors_joint):.4f}")
    print(f"Amplitude mean error: {jnp.mean(amp_errors_joint):.1f}%")
    print(f"Amplitude median error: {jnp.median(amp_errors_joint):.1f}%")
    print(f"Final log L: {log_likes_joint[-1]:.2f}")


    # Generate plots for both methods
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)

    print("\nPlots for Joint Least Squares:")
    plot_results(
        data=data,
        true_model=true_model,
        fitted_model=eval_model(pixel_coords, final_state_joint, field_center),
        true_state=true_state,
        initial_state=init_state,
        final_state=final_state_joint,
        image_shape=(n_pix_side, n_pix_side),
        log_likes=[float(ll) for ll in log_likes_joint],
        prefix=f"joint_{n_pix_side}_{n_stars}"
    )
