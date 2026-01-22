"""
Model fitting using iteratively reweighted least squares (IRLS).
"""

import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from jaxtyping import Array, Float

from .models import ModelState
from .forward import eval_model
from .design_matrix import (
    build_design_matrix_amplitudes,
    build_design_matrix_positions,
    build_design_matrix_sky
)
from .utils import solve_weighted_lstsq, poisson_log_likelihood


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
    """Perform one IRLS iteration for single band.

    Uses Fisher scoring: solve for parameter UPDATES, not full parameters.
    For Poisson likelihood: (X^T W X) delta = X^T W (d - mu)
    where W = 1/mu (inverse variance weights).

    Args:
        data: Observed pixel values (n_pix,)
        pixel_coords: Pixel coordinates (n_pix, 2)
        state: Current model state
        field_center: Field center for PSF deviations
        floor: Minimum allowed model value (prevents negative/zero)
        fit_sky: Whether to update sky parameters
        fit_psf: Whether to update PSF parameters (not implemented)
        damping: Damping factor for parameter updates (0=no update, 1=full update)

    Returns:
        Updated model state
    """

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
        delta_sky = solve_weighted_lstsq(S, residual, weights, 1e-3)
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
    delta_A = solve_weighted_lstsq(M, residual, weights, 1e-3)
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
    delta_pos = solve_weighted_lstsq(G, residual, weights, 1e-2)
    delta_pos = jnp.clip(delta_pos, -2.0, 2.0)  # Limit step size
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
        data: Observed pixel values (n_pix,)
        pixel_coords: Pixel coordinates (n_pix, 2)
        initial_state: Initial model state
        field_center: Field center for PSF deviations
        n_iter: Number of IRLS iterations
        fit_sky: Whether to fit sky parameters
        track_likelihood: If True, return (final_state, log_likes). Otherwise just final_state.

    Returns:
        If track_likelihood=False: final_state (ModelState)
        If track_likelihood=True: (final_state, log_likes) tuple
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
        # Return JAX array directly
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

    Args:
        data: Observed pixel values (n_pix,)
        pixel_coords: Pixel coordinates (n_pix, 2)
        state: Current model state
        field_center: Field center for PSF deviations
        floor: Minimum allowed model value
        fit_sky: Whether to update sky parameters
        fit_amplitudes: Whether to update amplitudes
        fit_positions: Whether to update positions
        damping: Damping factor for parameter updates
        regularization: Regularization parameter for least squares

    Returns:
        Updated model state
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

    This is an alternative to the block-coordinate descent in fit_model_jit.
    Instead of alternating between sky, amplitudes, and positions, this
    solves for all parameter updates in a single large least squares problem.

    Args:
        data: Observed pixel values (n_pix,)
        pixel_coords: Pixel coordinates (n_pix, 2)
        initial_state: Initial model state
        field_center: Field center for PSF deviations
        n_iter: Number of iterations
        fit_sky: Whether to fit sky parameters
        track_likelihood: If True, return (final_state, log_likes)

    Returns:
        If track_likelihood=False: final_state (ModelState)
        If track_likelihood=True: (final_state, log_likes) tuple
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
        # Return JAX array directly
        return final_state, log_likes_array
    else:
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

    Args:
        data: Observed pixel values (n_pix,)
        pixel_coords: Pixel coordinates (n_pix, 2)
        initial_state: Initial model state
        field_center: Field center for PSF deviations
        n_iter: Number of iterations
        fit_sky: Whether to fit sky parameters
        track_likelihood: If True, return (final_state, log_likes)

    Returns:
        If track_likelihood=False: final_state (ModelState)
        If track_likelihood=True: (final_state, log_likes) tuple
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
        # Return JAX array directly
        return final_state, log_likes_array
    else:
        return final_state
