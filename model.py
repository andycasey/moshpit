"""
PSF photometry model with iteratively reweighted least squares for Poisson noise.

Simultaneously fits:
- Star positions (shared across bands)
- Star amplitudes (per-band)
- PSF coefficients (per-band, spatially varying)
- Sky background (per-band)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import equinox as eqx
from typing import NamedTuple, Optional
from jaxtyping import Array, Float


class PSFBasis(NamedTuple):
    """PSF represented as sum of Gaussian basis functions with polynomial spatial variation.
    
    The PSF at field position (x, y) is:
        PSF(dx; x, y) = sum_k [ref_weights[k] + deviation(x,y)[k]] * Gaussian(dx; sigma_k)
    
    where deviation(x,y)[k] = sum_m deviation_coeffs[k,m] * poly_m((x-center)/scale, (y-center)/scale)
    
    The reference weights are FIXED (known from calibration).
    Only the deviation_coeffs are learned, and they have no constant term,
    so deviations are zero at field center.
    
    Coordinates are normalized by `coord_scale` before polynomial evaluation to keep
    polynomial terms O(1).
    """
    # Shape: (n_basis,) - widths of Gaussian basis functions (fixed)
    sigmas: Float[Array, "n_basis"]
    # Shape: (n_basis,) - reference weights at field center (fixed)
    ref_weights: Float[Array, "n_basis"]
    # Shape: (n_basis, n_deviation_coeffs) - learnable deviation coefficients
    # n_deviation_coeffs = n_poly_coeffs(poly_order) - 1 (no constant term)
    deviation_coeffs: Float[Array, "n_basis n_deviation"]
    poly_order: int = 2
    # Coordinate scale for normalization (typically half the image size)
    coord_scale: float = 32.0


class SkyModel(NamedTuple):
    """Sky background as 2D polynomial."""
    # Shape: (n_sky_coeffs,) where n_sky_coeffs = (order+1)*(order+2)//2
    coeffs: Float[Array, "n_sky"]
    order: int = 2


class StarParams(NamedTuple):
    """Parameters for all stars."""
    # Shape: (n_stars, 2) - x, y positions
    positions: Float[Array, "n_stars 2"]
    # Shape: (n_bands, n_stars) - amplitudes per band
    amplitudes: Float[Array, "n_bands n_stars"]


class ModelState(NamedTuple):
    """Full model state."""
    stars: StarParams
    psf: list[PSFBasis]  # per band
    sky: list[SkyModel]  # per band
    

def poly_features_2d(x: Float[Array, "2"], order: int) -> Float[Array, "n_coeffs"]:
    """Generate 2D polynomial features up to given order.
    
    For order=2: [1, x, y, x^2, xy, y^2]
    """
    features = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            features.append(x[0]**i * x[1]**j)
    return jnp.array(features)


def poly_features_2d_no_constant(x: Float[Array, "2"], order: int) -> Float[Array, "n_coeffs-1"]:
    """Generate 2D polynomial features WITHOUT constant term.
    
    For order=2: [x, y, x^2, xy, y^2]
    
    This ensures deviations are zero at origin.
    """
    features = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0:
                continue  # skip constant term
            features.append(x[0]**i * x[1]**j)
    return jnp.array(features)


def n_poly_coeffs(order: int) -> int:
    """Number of coefficients in 2D polynomial of given order."""
    return (order + 1) * (order + 2) // 2


def n_deviation_coeffs(order: int) -> int:
    """Number of deviation coefficients (polynomial without constant)."""
    return n_poly_coeffs(order) - 1


def eval_sky(sky: SkyModel, positions: Float[Array, "n_pix 2"]) -> Float[Array, "n_pix"]:
    """Evaluate sky model at pixel positions."""
    order = sky.order
    features = vmap(lambda p: poly_features_2d(p, order))(positions)
    return features @ sky.coeffs


def eval_psf_at_location(
    psf: PSFBasis,
    field_pos: Float[Array, "2"],
    dx: Float[Array, "n_pix 2"],
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix"]:
    """Evaluate PSF at given field position for pixel offsets dx.
    
    Args:
        psf: PSF basis parameters
        field_pos: Position in field where PSF is evaluated (for spatial variation)
        dx: Pixel offsets from PSF center, shape (n_pix, 2)
        field_center: Center of field for deviation calculation (default: origin)
    
    Returns:
        PSF values at each pixel offset
    """
    if field_center is None:
        field_center = jnp.zeros(2)
    
    # Position relative to field center, normalized by coord_scale
    rel_pos = (field_pos - field_center) / psf.coord_scale
    
    # Get deviation polynomial features (no constant term)
    poly_order = psf.poly_order
    deviation_features = poly_features_2d_no_constant(rel_pos, poly_order)
    
    # Compute deviation for each Gaussian basis at this field position
    # Shape: (n_basis,)
    deviations = psf.deviation_coeffs @ deviation_features
    
    # Total weight = reference + deviation
    basis_weights = psf.ref_weights + deviations
    
    # Compute r^2 for each pixel
    r2 = jnp.sum(dx**2, axis=-1)
    
    # Evaluate each Gaussian basis and sum
    # Shape: (n_basis, n_pix)
    gaussians = jnp.exp(-0.5 * r2[None, :] / psf.sigmas[:, None]**2)
    gaussians = gaussians / (2 * jnp.pi * psf.sigmas[:, None]**2)  # normalize
    
    # Weighted sum of basis functions
    return jnp.sum(basis_weights[:, None] * gaussians, axis=0)


def eval_psf_derivative(
    psf: PSFBasis,
    field_pos: Float[Array, "2"],
    dx: Float[Array, "n_pix 2"],
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix 2"]:
    """Evaluate PSF gradient w.r.t. star position.
    
    Returns d(PSF)/d(star_pos) which is -d(PSF)/d(dx)
    """
    if field_center is None:
        field_center = jnp.zeros(2)
    
    rel_pos = (field_pos - field_center) / psf.coord_scale
    poly_order = psf.poly_order
    deviation_features = poly_features_2d_no_constant(rel_pos, poly_order)
    deviations = psf.deviation_coeffs @ deviation_features
    basis_weights = psf.ref_weights + deviations
    
    r2 = jnp.sum(dx**2, axis=-1)
    
    # Gaussian and its derivative
    # d/dx_i exp(-r^2/2s^2) = -x_i/s^2 * exp(-r^2/2s^2)
    gaussians = jnp.exp(-0.5 * r2[None, :] / psf.sigmas[:, None]**2)
    gaussians = gaussians / (2 * jnp.pi * psf.sigmas[:, None]**2)
    
    # Derivative factor: -dx / sigma^2
    # Shape: (n_basis, n_pix, 2)
    deriv_factor = -dx[None, :, :] / psf.sigmas[:, None, None]**2
    
    # Shape: (n_basis, n_pix, 2)
    gaussian_derivs = gaussians[:, :, None] * deriv_factor
    
    # Weighted sum, shape: (n_pix, 2)
    # Negative because we want d/d(star_pos) = -d/d(dx)
    return -jnp.sum(basis_weights[:, None, None] * gaussian_derivs, axis=0)


def build_design_matrix_sky(pixel_coords: Float[Array, "n_pix 2"], order: int) -> Float[Array, "n_pix n_sky"]:
    """Build sky design matrix."""
    return vmap(lambda p: poly_features_2d(p, order))(pixel_coords)


def build_design_matrix_psf(
    pixel_coords: Float[Array, "n_pix 2"],
    star_positions: Float[Array, "n_stars 2"],
    amplitudes: Float[Array, "n_stars"],
    psf: PSFBasis,
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix n_psf_params"]:
    """Build design matrix for PSF deviation coefficients.
    
    We're solving for deviation_coeffs which multiply the deviation polynomial
    features (no constant term). This means we're learning how the PSF changes
    away from the field center, while the reference PSF is fixed.
    """
    if field_center is None:
        field_center = jnp.zeros(2)
    
    n_pix = pixel_coords.shape[0]
    n_basis = len(psf.sigmas)
    n_dev = n_deviation_coeffs(psf.poly_order)
    
    # For each basis function and deviation coefficient, compute the column
    # Q[x, k*n_dev + m] = sum_i A_i * G_k(x - x_i) * deviation_poly_m((x_i - center)/scale)
    
    Q = jnp.zeros((n_pix, n_basis * n_dev))
    
    for i in range(star_positions.shape[0]):
        dx = pixel_coords - star_positions[i]
        r2 = jnp.sum(dx**2, axis=-1)
        
        # Gaussian basis values at this star, shape (n_basis, n_pix)
        gaussians = jnp.exp(-0.5 * r2[None, :] / psf.sigmas[:, None]**2)
        gaussians = gaussians / (2 * jnp.pi * psf.sigmas[:, None]**2)
        
        # Deviation features at star position (no constant term), normalized
        rel_pos = (star_positions[i] - field_center) / psf.coord_scale
        dev_feats = poly_features_2d_no_constant(rel_pos, psf.poly_order)
        
        # Outer product: (n_basis, n_pix) x (n_dev,) -> (n_basis, n_dev, n_pix)
        contrib = amplitudes[i] * gaussians[:, None, :] * dev_feats[None, :, None]
        
        # Reshape to (n_pix, n_basis * n_dev)
        Q = Q + contrib.reshape(n_basis * n_dev, n_pix).T
    
    return Q


def build_design_matrix_amplitudes(
    pixel_coords: Float[Array, "n_pix 2"],
    star_positions: Float[Array, "n_stars 2"],
    psf: PSFBasis,
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix n_stars"]:
    """Build design matrix for star amplitudes."""
    n_pix = pixel_coords.shape[0]
    n_stars = star_positions.shape[0]
    
    M = jnp.zeros((n_pix, n_stars))
    
    for i in range(n_stars):
        dx = pixel_coords - star_positions[i]
        M = M.at[:, i].set(eval_psf_at_location(psf, star_positions[i], dx, field_center))
    
    return M


def build_design_matrix_positions(
    pixel_coords: Float[Array, "n_pix 2"],
    star_positions: Float[Array, "n_stars 2"],
    amplitudes: Float[Array, "n_stars"],
    psf: PSFBasis,
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix 2*n_stars"]:
    """Build design matrix for position updates."""
    n_pix = pixel_coords.shape[0]
    n_stars = star_positions.shape[0]
    
    G = jnp.zeros((n_pix, 2 * n_stars))
    
    for i in range(n_stars):
        dx = pixel_coords - star_positions[i]
        deriv = eval_psf_derivative(psf, star_positions[i], dx, field_center)  # (n_pix, 2)
        G = G.at[:, 2*i].set(amplitudes[i] * deriv[:, 0])
        G = G.at[:, 2*i + 1].set(amplitudes[i] * deriv[:, 1])
    
    return G


def eval_model(
    pixel_coords: Float[Array, "n_pix 2"],
    stars: StarParams,
    psf: PSFBasis,
    sky: SkyModel,
    band: int,
    field_center: Float[Array, "2"] = None
) -> Float[Array, "n_pix"]:
    """Evaluate full model for one band."""
    # Sky contribution
    model = eval_sky(sky, pixel_coords)
    
    # Star contributions
    for i in range(stars.positions.shape[0]):
        dx = pixel_coords - stars.positions[i]
        psf_vals = eval_psf_at_location(psf, stars.positions[i], dx, field_center)
        model = model + stars.amplitudes[band, i] * psf_vals
    
    return model


def poisson_log_likelihood(data: Float[Array, "n_pix"], model: Float[Array, "n_pix"]) -> float:
    """Compute Poisson log-likelihood."""
    model_safe = jnp.maximum(model, 1e-10)
    return jnp.sum(data * jnp.log(model_safe) - model_safe)


def solve_weighted_lstsq(
    A: Float[Array, "n m"],
    b: Float[Array, "n"],
    weights: Float[Array, "n"],
    regularization: float = 1e-6
) -> Float[Array, "m"]:
    """Solve weighted least squares: min_x ||sqrt(W)(Ax - b)||^2"""
    # Form normal equations: A^T W A x = A^T W b
    WA = weights[:, None] * A
    AtwA = A.T @ WA + regularization * jnp.eye(A.shape[1])
    Atwb = A.T @ (weights * b)
    return jnp.linalg.solve(AtwA, Atwb)


def irls_step(
    data: list[Float[Array, "n_pix"]],
    pixel_coords: Float[Array, "n_pix 2"],
    state: ModelState,
    field_center: Float[Array, "2"],
    floor: float = 1.0,
    damping: float = 0.5,
    position_damping: float = 0.3,
    fit_psf: bool = True,
    fit_sky: bool = True
) -> ModelState:
    """One iteration of IRLS across all bands.
    
    Args:
        field_center: Center of field for PSF deviation calculation
        damping: Damping factor for amplitude/sky/psf updates (0-1)
        position_damping: Damping factor for position updates (0-1)
        fit_psf: Whether to update PSF deviations (can disable to test other components)
        fit_sky: Whether to update sky (can disable to test other components)
    """
    
    n_bands = len(data)
    n_stars = state.stars.positions.shape[0]
    
    new_sky = []
    new_psf = []
    new_amplitudes = []
    
    # Per-band updates (could be parallelized)
    for b in range(n_bands):
        # Current model and weights
        model = eval_model(pixel_coords, state.stars, state.psf[b], state.sky[b], b, field_center)
        model = jnp.maximum(model, floor)
        weights = 1.0 / model
        
        # Step 1: Update sky
        S = build_design_matrix_sky(pixel_coords, state.sky[b].order)
        M = build_design_matrix_amplitudes(pixel_coords, state.stars.positions, state.psf[b], field_center)
        source_model = M @ state.stars.amplitudes[b]
        
        if fit_sky:
            residual_for_sky = data[b] - source_model
            sky_coeffs_new = solve_weighted_lstsq(S, residual_for_sky, weights, regularization=1e-3)
            # Damped update
            sky_coeffs = (1 - damping) * state.sky[b].coeffs + damping * sky_coeffs_new
        else:
            sky_coeffs = state.sky[b].coeffs
        
        new_sky.append(SkyModel(coeffs=sky_coeffs, order=state.sky[b].order))
        
        # Update model and weights
        model = source_model + S @ sky_coeffs
        model = jnp.maximum(model, floor)
        weights = 1.0 / model
        
        # Step 2a: Update amplitudes
        residual_for_amps = data[b] - S @ sky_coeffs
        amps_new = solve_weighted_lstsq(M, residual_for_amps, weights, regularization=1e-3)
        amps_new = jnp.maximum(amps_new, 1.0)  # enforce positivity with minimum
        # Damped update
        amps = (1 - damping) * state.stars.amplitudes[b] + damping * amps_new
        new_amplitudes.append(amps)
        
        # Update model and weights
        model = M @ amps + S @ sky_coeffs
        model = jnp.maximum(model, floor)
        weights = 1.0 / model
        
        # Step 2b: Update PSF deviations
        if fit_psf:
            Q = build_design_matrix_psf(pixel_coords, state.stars.positions, amps, state.psf[b], field_center)
            
            # Residual after subtracting sky and reference PSF contribution
            # We need to compute what the reference PSF alone would give
            M_ref = build_design_matrix_amplitudes(
                pixel_coords, state.stars.positions,
                PSFBasis(
                    sigmas=state.psf[b].sigmas,
                    ref_weights=state.psf[b].ref_weights,
                    deviation_coeffs=jnp.zeros_like(state.psf[b].deviation_coeffs),
                    poly_order=state.psf[b].poly_order,
                    coord_scale=state.psf[b].coord_scale
                ),
                field_center
            )
            ref_source_model = M_ref @ amps
            residual_for_psf = data[b] - S @ sky_coeffs - ref_source_model
            
            n_basis = len(state.psf[b].sigmas)
            n_dev = n_deviation_coeffs(state.psf[b].poly_order)
            
            # Solve for deviation coefficients with STRONG regularization toward zero
            # This encodes our prior that deviations should be small
            dev_params_new = solve_weighted_lstsq(Q, residual_for_psf, weights, regularization=1.0)
            dev_coeffs_new = dev_params_new.reshape(n_basis, n_dev)
            
            # Additional constraint: clip to reasonable range
            max_deviation = 0.2  # Deviations should be small relative to ref_weights
            dev_coeffs_new = jnp.clip(dev_coeffs_new, -max_deviation, max_deviation)
            
            # Damped update for PSF deviations (very conservative)
            dev_coeffs = (1 - damping * 0.3) * state.psf[b].deviation_coeffs + (damping * 0.3) * dev_coeffs_new
        else:
            dev_coeffs = state.psf[b].deviation_coeffs
        
        new_psf.append(PSFBasis(
            sigmas=state.psf[b].sigmas,
            ref_weights=state.psf[b].ref_weights,  # Keep reference fixed
            deviation_coeffs=dev_coeffs,
            poly_order=state.psf[b].poly_order,
            coord_scale=state.psf[b].coord_scale
        ))
    
    new_amplitudes = jnp.stack(new_amplitudes, axis=0)
    
    # Step 3: Update positions (joint across bands)
    H = jnp.zeros((2 * n_stars, 2 * n_stars))
    g = jnp.zeros(2 * n_stars)
    
    for b in range(n_bands):
        model = eval_model(pixel_coords, 
                          StarParams(state.stars.positions, new_amplitudes),
                          new_psf[b], new_sky[b], b, field_center)
        model = jnp.maximum(model, floor)
        weights = 1.0 / model
        
        G = build_design_matrix_positions(
            pixel_coords, state.stars.positions, new_amplitudes[b], new_psf[b], field_center
        )
        
        residual = data[b] - model
        
        WG = weights[:, None] * G
        H = H + G.T @ WG
        g = g + G.T @ (weights * residual)
    
    # Solve for position update with stronger regularization
    H = H + 1e-2 * jnp.eye(2 * n_stars)
    delta_pos = jnp.linalg.solve(H, g)
    
    # Clip position updates to prevent wild jumps
    delta_pos = jnp.clip(delta_pos, -2.0, 2.0)
    
    # Damped position update
    new_positions = state.stars.positions + position_damping * delta_pos.reshape(n_stars, 2)
    
    return ModelState(
        stars=StarParams(positions=new_positions, amplitudes=new_amplitudes),
        psf=new_psf,
        sky=new_sky
    )


def fit_model(
    data: list[Float[Array, "n_pix"]],
    pixel_coords: Float[Array, "n_pix 2"],
    initial_state: ModelState,
    field_center: Float[Array, "2"] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = True,
    warmup_iters: int = 10,
    fit_sky: bool = True
) -> tuple[ModelState, list[float]]:
    """Fit model using IRLS.
    
    Uses a staged approach:
    1. First warmup_iters iterations: fix PSF deviations, fit only positions, amplitudes, sky
    2. Remaining iterations: fit all parameters including PSF deviations
    
    Args:
        data: List of data arrays, one per band
        pixel_coords: Pixel coordinates, shape (n_pix, 2)
        initial_state: Initial model parameters
        field_center: Center of field for PSF deviation calculation
        max_iter: Maximum iterations
        tol: Convergence tolerance on log-likelihood
        verbose: Print progress
        warmup_iters: Number of initial iterations with PSF deviations fixed
        fit_sky: Whether to fit sky parameters
    
    Returns:
        Final state and history of log-likelihoods
    """
    if field_center is None:
        # Default to center of pixel coordinate range
        field_center = jnp.array([
            (pixel_coords[:, 0].max() + pixel_coords[:, 0].min()) / 2,
            (pixel_coords[:, 1].max() + pixel_coords[:, 1].min()) / 2
        ])
    
    state = initial_state
    log_likes = []
    
    for iteration in range(max_iter):
        # Compute current log-likelihood
        total_ll = 0.0
        for b in range(len(data)):
            model = eval_model(pixel_coords, state.stars, state.psf[b], state.sky[b], b, field_center)
            total_ll += poisson_log_likelihood(data[b], model)
        log_likes.append(float(total_ll))
        
        # Determine if we're in warmup phase
        fit_psf = iteration >= warmup_iters
        phase = "full" if fit_psf else "warmup"
        
        if verbose:
            print(f"Iteration {iteration} ({phase}): log L = {total_ll:.2f}")
        
        # Check convergence (only after warmup)
        if len(log_likes) > 1 and iteration > warmup_iters:
            delta = log_likes[-1] - log_likes[-2]
            if abs(delta) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # IRLS step
        state = irls_step(data, pixel_coords, state, field_center, fit_psf=fit_psf, fit_sky=fit_sky)
    
    return state, log_likes
