"""
Simple test to verify the PSF fitting algorithm works correctly.
Uses a very simple setup: one band, known PSF, known sky, only fit positions and amplitudes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from model import (
    PSFBasis, SkyModel, StarParams, ModelState,
    eval_psf_at_location, eval_sky, eval_model,
    build_design_matrix_amplitudes, build_design_matrix_positions,
    solve_weighted_lstsq, n_deviation_coeffs
)


def create_simple_psf(coord_scale: float = 32.0) -> PSFBasis:
    """Create a simple Gaussian PSF (no spatial variation)."""
    return PSFBasis(
        sigmas=jnp.array([2.0]),  # Single Gaussian, sigma=2 pixels
        ref_weights=jnp.array([1.0]),  # Unit weight
        deviation_coeffs=jnp.zeros((1, n_deviation_coeffs(2))),  # No deviations
        poly_order=2,
        coord_scale=coord_scale
    )


def create_simple_sky() -> SkyModel:
    """Create a simple flat sky."""
    return SkyModel(
        coeffs=jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Flat sky at 100
        order=2
    )


def create_pixel_coords(image_shape: tuple) -> jnp.ndarray:
    """Create array of pixel coordinates."""
    ny, nx = image_shape
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    return jnp.stack([x.ravel(), y.ravel()], axis=-1).astype(jnp.float32)


def generate_simple_data(key, image_shape=(32, 32), n_stars=5):
    """Generate simple test data."""
    keys = jr.split(key, 4)
    
    pixel_coords = create_pixel_coords(image_shape)
    field_center = jnp.array([image_shape[1] / 2.0, image_shape[0] / 2.0])
    
    psf = create_simple_psf(coord_scale=max(image_shape) / 2.0)
    sky = create_simple_sky()
    
    # Random star positions (avoiding edges)
    positions = jr.uniform(keys[0], (n_stars, 2), minval=5.0, maxval=jnp.array([image_shape[1], image_shape[0]]) - 5.0)
    
    # Star fluxes
    amplitudes = jr.uniform(keys[1], (1, n_stars), minval=1000.0, maxval=5000.0)
    
    stars = StarParams(positions=positions, amplitudes=amplitudes)
    
    # Render true image
    true_image = eval_model(pixel_coords, stars, psf, sky, 0, field_center)
    
    # Add Poisson noise
    noisy_image = jr.poisson(keys[2], true_image).astype(jnp.float32)
    
    return {
        'image': noisy_image,
        'true_image': true_image,
        'pixel_coords': pixel_coords,
        'field_center': field_center,
        'true_stars': stars,
        'psf': psf,
        'sky': sky
    }


def fit_positions_and_amplitudes(
    data: jnp.ndarray,
    pixel_coords: jnp.ndarray,
    initial_positions: jnp.ndarray,
    initial_amplitudes: jnp.ndarray,
    psf: PSFBasis,
    sky: SkyModel,
    field_center: jnp.ndarray,
    n_iter: int = 20,
    damping: float = 0.5
):
    """Simple iterative fitting of positions and amplitudes only."""
    
    positions = initial_positions
    amplitudes = initial_amplitudes
    floor = 1.0
    
    # Subtract sky once (assuming it's known)
    sky_model = eval_sky(sky, pixel_coords)
    residual = data - sky_model
    
    for iteration in range(n_iter):
        # Build amplitude design matrix
        M = build_design_matrix_amplitudes(pixel_coords, positions, psf, field_center)
        
        # Current source model
        source_model = M @ amplitudes
        full_model = source_model + sky_model
        full_model = jnp.maximum(full_model, floor)
        
        # IRLS weights
        weights = 1.0 / full_model
        
        # Update amplitudes
        amps_new = solve_weighted_lstsq(M, residual, weights, regularization=1e-6)
        amps_new = jnp.maximum(amps_new, 1.0)
        amplitudes = (1 - damping) * amplitudes + damping * amps_new
        
        # Update positions
        G = build_design_matrix_positions(pixel_coords, positions, amplitudes, psf, field_center)
        
        # Recompute model with new amplitudes
        source_model = M @ amplitudes
        full_model = source_model + sky_model
        full_model = jnp.maximum(full_model, floor)
        weights = 1.0 / full_model
        
        pos_residual = residual - source_model
        delta_pos = solve_weighted_lstsq(G, pos_residual, weights, regularization=1e-4)
        delta_pos = jnp.clip(delta_pos, -1.0, 1.0)  # Limit step size
        
        n_stars = positions.shape[0]
        positions = positions + 0.3 * delta_pos.reshape(n_stars, 2)
        
        # Log likelihood
        model = M @ amplitudes + sky_model
        model = jnp.maximum(model, 1e-10)
        ll = jnp.sum(data * jnp.log(model) - model)
        print(f"Iter {iteration}: log L = {ll:.2f}")
    
    return positions, amplitudes


if __name__ == '__main__':
    print("Simple PSF fitting test")
    print("="*50)
    
    key = jr.PRNGKey(123)
    key1, key2 = jr.split(key)
    
    # Generate data
    data = generate_simple_data(key1, image_shape=(32, 32), n_stars=5)
    
    print(f"True positions:\n{data['true_stars'].positions}")
    print(f"True amplitudes: {data['true_stars'].amplitudes[0]}")
    print()
    
    # Perturb initial guess
    pos_noise = jr.normal(key2, data['true_stars'].positions.shape) * 1.5
    initial_positions = data['true_stars'].positions + pos_noise
    initial_amplitudes = data['true_stars'].amplitudes[0] * (1 + 0.2 * jr.normal(jr.PRNGKey(456), (5,)))
    
    print(f"Initial positions:\n{initial_positions}")
    print(f"Initial amplitudes: {initial_amplitudes}")
    print()
    
    # Fit
    print("Fitting...")
    final_positions, final_amplitudes = fit_positions_and_amplitudes(
        data['image'],
        data['pixel_coords'],
        initial_positions,
        initial_amplitudes,
        data['psf'],
        data['sky'],
        data['field_center'],
        n_iter=30
    )
    
    print()
    print("Results:")
    print(f"Final positions:\n{final_positions}")
    print(f"True positions:\n{data['true_stars'].positions}")
    print()
    print(f"Final amplitudes: {final_amplitudes}")
    print(f"True amplitudes: {data['true_stars'].amplitudes[0]}")
    print()
    
    # Compute errors
    pos_errors = jnp.sqrt(jnp.sum((final_positions - data['true_stars'].positions)**2, axis=1))
    amp_errors = jnp.abs(final_amplitudes - data['true_stars'].amplitudes[0]) / data['true_stars'].amplitudes[0] * 100
    
    print(f"Position errors (pixels): {pos_errors}")
    print(f"Position RMSE: {jnp.sqrt(jnp.mean(pos_errors**2)):.4f}")
    print(f"Amplitude errors (%): {amp_errors}")
    print(f"Amplitude mean error: {jnp.mean(amp_errors):.1f}%")
