"""
Example: Test JIT-compiled PSF fitting.

This script demonstrates the use of the moshpit package for PSF photometry.
It generates synthetic data with known parameters and fits them using the
JIT-compiled IRLS algorithm.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time

from moshpit import (
    ModelState,
    eval_model,
    fit_model_jit,
    poisson_log_likelihood,
    plot_results
)


def main():
    print("Testing JIT-compatible PSF photometry")
    print("="*50)

    key = jr.PRNGKey(42)
    keys = jr.split(key, 3)

    # Create test data
    n_pix_side = 32
    pixel_coords = jnp.stack(
        jnp.meshgrid(jnp.arange(n_pix_side), jnp.arange(n_pix_side)),
        axis=-1
    ).reshape(-1, 2).astype(float)
    field_center = jnp.array([n_pix_side/2, n_pix_side/2])

    # True state
    true_state = ModelState(
        positions=jnp.array([[10., 10.], [20., 20.], [15., 25.]]),
        amplitudes=jnp.array([2000., 3000., 1500.]),
        psf_sigmas=jnp.array([2.0]),
        psf_ref_weights=jnp.array([1.0]),
        psf_deviation_coeffs=jnp.zeros((1, 5)),
        psf_coord_scale=16.0,
        sky_coeffs=jnp.array([100., 0., 0., 0., 0., 0.])
    )

    # Generate data
    true_model = eval_model(pixel_coords, true_state, field_center)
    data = jr.poisson(keys[0], true_model).astype(float)

    print(f"Data shape: {data.shape}")
    print(f"Data mean: {jnp.mean(data):.2f}")

    # Initial guess (perturbed)
    init_state = ModelState(
        positions=true_state.positions + jr.normal(keys[1], (3, 2)) * 1.5,
        amplitudes=true_state.amplitudes * (1 + 0.3 * jr.normal(keys[2], (3,))),
        psf_sigmas=true_state.psf_sigmas,
        psf_ref_weights=true_state.psf_ref_weights,
        psf_deviation_coeffs=true_state.psf_deviation_coeffs,
        psf_coord_scale=true_state.psf_coord_scale,
        sky_coeffs=true_state.sky_coeffs
    )

    print(f"\nTrue positions:\n{true_state.positions}")
    print(f"Initial positions:\n{init_state.positions}")

    # Test JIT compilation
    print("\nTesting JIT compilation...")

    t0 = time.time()
    # First call compiles
    final_state = fit_model_jit(
        data, pixel_coords, init_state, field_center,
        n_iter=30, fit_sky=False
    )
    t1 = time.time()
    print(f"First call (with compile): {t1-t0:.3f}s")

    # Second call uses cached compilation
    t0 = time.time()
    final_state = fit_model_jit(
        data, pixel_coords, init_state, field_center,
        n_iter=30, fit_sky=False
    )
    t1 = time.time()
    print(f"Second call (cached): {t1-t0:.3f}s")

    # Print results
    print(f"\nFinal positions:\n{final_state.positions}")
    print(f"True positions:\n{true_state.positions}")

    pos_errors = jnp.sqrt(jnp.sum((final_state.positions - true_state.positions)**2, axis=1))
    print(f"\nPosition errors: {pos_errors}")
    print(f"Position RMSE: {jnp.sqrt(jnp.mean(pos_errors**2)):.4f}")

    print(f"\nFinal amplitudes: {final_state.amplitudes}")
    print(f"True amplitudes: {true_state.amplitudes}")
    amp_errors = jnp.abs(final_state.amplitudes - true_state.amplitudes) / true_state.amplitudes * 100
    print(f"Amplitude errors (%): {amp_errors}")

    # Check log-likelihood
    init_model = eval_model(pixel_coords, init_state, field_center)
    final_model = eval_model(pixel_coords, final_state, field_center)
    print(f"\nInitial log L: {poisson_log_likelihood(data, init_model):.2f}")
    print(f"Final log L: {poisson_log_likelihood(data, final_model):.2f}")

    # Generate plots
    print("\n" + "="*50)
    print("Re-running with likelihood tracking for plots...")
    final_state_plot, log_likes = fit_model_jit(
        data, pixel_coords, init_state, field_center,
        n_iter=30, fit_sky=False, track_likelihood=True
    )

    print("\nGenerating plots...")
    plot_results(
        data=data,
        true_model=true_model,
        fitted_model=eval_model(pixel_coords, final_state_plot, field_center),
        true_state=true_state,
        initial_state=init_state,
        final_state=final_state_plot,
        image_shape=(n_pix_side, n_pix_side),
        log_likes=log_likes,
        prefix="example"
    )

    print("\nDone! Check the generated PNG files in the current directory.")


if __name__ == '__main__':
    main()
