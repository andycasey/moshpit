"""
Generate synthetic test images with:
- Spatially varying PSF
- Structured sky background
- Multiple bands with different PSFs and sky levels
- Poisson noise
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from model import (
    PSFBasis, SkyModel, StarParams, ModelState,
    eval_psf_at_location, eval_sky, poly_features_2d, n_poly_coeffs
)


def generate_spatially_varying_psf(
    key: jax.Array,
    n_basis: int = 3,
    poly_order: int = 2,
    base_sigmas: tuple = (1.5, 2.5, 4.0),
    deviation_strength: float = 0.05,
    coord_scale: float = 32.0
) -> PSFBasis:
    """Generate a spatially varying PSF.
    
    The PSF is a sum of Gaussians with different widths.
    The reference weights are fixed, and small deviations vary across the field.
    """
    from model import n_deviation_coeffs
    
    sigmas = jnp.array(base_sigmas[:n_basis])
    n_dev = n_deviation_coeffs(poly_order)
    
    key1, key2 = jr.split(key)
    
    # Reference weights at field center (these are "known" from calibration)
    ref_weights = jnp.zeros(n_basis)
    ref_weights = ref_weights.at[0].set(0.7)   # Primary Gaussian dominates
    ref_weights = ref_weights.at[1].set(0.25)
    if n_basis > 2:
        ref_weights = ref_weights.at[2].set(0.05)
    
    # Small deviations that vary across the field
    # Shape: (n_basis, n_dev) where n_dev = n_poly_coeffs - 1 (no constant)
    deviation_coeffs = jr.normal(key1, (n_basis, n_dev)) * deviation_strength
    
    return PSFBasis(
        sigmas=sigmas, 
        ref_weights=ref_weights,
        deviation_coeffs=deviation_coeffs, 
        poly_order=poly_order,
        coord_scale=coord_scale
    )


def generate_structured_sky(
    key: jax.Array,
    order: int = 2,
    base_level: float = 100.0,
    gradient_strength: float = 0.1,
    quadratic_strength: float = 0.001
) -> SkyModel:
    """Generate a structured sky background with gradients.
    
    Polynomial ordering for order=2 is: [1, y, y², x, xy, x²]
    """
    n_coeffs = n_poly_coeffs(order)
    coeffs = jnp.zeros(n_coeffs)
    
    # Constant term (base sky level)
    coeffs = coeffs.at[0].set(base_level)
    
    # Linear terms (gradient)
    key1, key2 = jr.split(key)
    if order >= 1:
        # Index 1 is y, index 3 is x (for order >= 2)
        # For order=1: [1, y, x]
        # For order=2: [1, y, y², x, xy, x²]
        y_idx = 1
        x_idx = order + 1  # For order=2, this is index 3
        coeffs = coeffs.at[y_idx].set(gradient_strength * jr.uniform(key1, minval=-1, maxval=1))
        if order >= 2:
            coeffs = coeffs.at[x_idx].set(gradient_strength * jr.uniform(key2, minval=-1, maxval=1))
    
    # Quadratic terms for some curvature (only for order >= 2)
    if order >= 2:
        # y² is at index 2, xy at index 4, x² at index 5
        coeffs = coeffs.at[2].set(quadratic_strength * 0.5)  # y²
        coeffs = coeffs.at[5].set(quadratic_strength * 0.3)  # x²
    
    return SkyModel(coeffs=coeffs, order=order)


def generate_star_field(
    key: jax.Array,
    n_stars: int,
    image_shape: tuple,
    n_bands: int,
    flux_range: tuple = (500, 5000),
    color_variation: float = 0.3,
    edge_margin: float = 5.0
) -> StarParams:
    """Generate random star positions and fluxes."""
    key1, key2, key3 = jr.split(key, 3)
    
    # Random positions avoiding edges
    positions = jr.uniform(key1, (n_stars, 2), 
                          minval=edge_margin,
                          maxval=jnp.array([image_shape[1], image_shape[0]]) - edge_margin)
    
    # Base fluxes (log-uniform distribution)
    log_fluxes = jr.uniform(key2, (n_stars,), 
                           minval=jnp.log(flux_range[0]), 
                           maxval=jnp.log(flux_range[1]))
    base_fluxes = jnp.exp(log_fluxes)
    
    # Per-band fluxes with color variation
    # Shape: (n_bands, n_stars)
    color_factors = 1.0 + color_variation * jr.normal(key3, (n_bands, n_stars))
    color_factors = jnp.maximum(color_factors, 0.1)  # keep positive
    
    amplitudes = base_fluxes[None, :] * color_factors
    
    return StarParams(positions=positions, amplitudes=amplitudes)


def render_image(
    pixel_coords: jnp.ndarray,
    stars: StarParams,
    psf: PSFBasis,
    sky: SkyModel,
    band: int,
    field_center: jnp.ndarray = None
) -> jnp.ndarray:
    """Render noiseless image."""
    # Sky
    image = eval_sky(sky, pixel_coords)
    
    # Stars
    for i in range(stars.positions.shape[0]):
        dx = pixel_coords - stars.positions[i]
        psf_vals = eval_psf_at_location(psf, stars.positions[i], dx, field_center)
        image = image + stars.amplitudes[band, i] * psf_vals
    
    return image


def add_poisson_noise(key: jax.Array, image: jnp.ndarray) -> jnp.ndarray:
    """Add Poisson noise to image."""
    # Ensure non-negative
    image = jnp.maximum(image, 0)
    return jr.poisson(key, image).astype(jnp.float32)


def create_pixel_coords(image_shape: tuple) -> jnp.ndarray:
    """Create array of pixel coordinates."""
    ny, nx = image_shape
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    # Flatten and stack as (n_pix, 2) with (x, y) ordering
    return jnp.stack([x.ravel(), y.ravel()], axis=-1).astype(jnp.float32)


def generate_test_data(
    key: jax.Array,
    image_shape: tuple = (128, 128),
    n_stars: int = 15,
    n_bands: int = 2
) -> dict:
    """Generate complete test dataset.
    
    Returns dict with:
        - images: list of noisy images per band
        - true_images: list of noiseless images per band
        - pixel_coords: coordinate array
        - true_state: true ModelState
        - image_shape: shape tuple
        - field_center: center of field for PSF deviations
    """
    keys = jr.split(key, 10)
    
    pixel_coords = create_pixel_coords(image_shape)
    
    # Field center
    field_center = jnp.array([image_shape[1] / 2.0, image_shape[0] / 2.0])
    
    # Generate true parameters
    stars = generate_star_field(keys[0], n_stars, image_shape, n_bands)
    
    psfs = []
    skies = []
    true_images = []
    noisy_images = []
    
    for b in range(n_bands):
        # Different PSF per band (e.g., different seeing)
        psf = generate_spatially_varying_psf(
            keys[1 + b],
            base_sigmas=(1.5 + 0.3*b, 2.5 + 0.3*b, 4.0 + 0.3*b),
            deviation_strength=0.03,  # Small deviations
            coord_scale=max(image_shape) / 2.0  # Normalize to [-1, 1] at edges
        )
        psfs.append(psf)
        
        # Different sky per band
        sky = generate_structured_sky(
            keys[3 + b],
            base_level=100 + 50*b,
            gradient_strength=10.0
        )
        skies.append(sky)
        
        # Render
        true_image = render_image(pixel_coords, stars, psf, sky, b, field_center)
        true_images.append(true_image)
        
        # Add noise
        noisy_image = add_poisson_noise(keys[5 + b], true_image)
        noisy_images.append(noisy_image)
    
    true_state = ModelState(stars=stars, psf=psfs, sky=skies)
    
    return {
        'images': noisy_images,
        'true_images': true_images,
        'pixel_coords': pixel_coords,
        'true_state': true_state,
        'image_shape': image_shape,
        'field_center': field_center
    }


def perturb_initial_guess(
    key: jax.Array,
    true_state: ModelState,
    position_noise: float = 1.0,
    amplitude_noise_frac: float = 0.2,
    psf_deviation_noise_frac: float = 0.5,
    sky_noise_frac: float = 0.2,
    use_true_ref_psf: bool = True
) -> ModelState:
    """Create perturbed initial guess from true parameters.
    
    Args:
        use_true_ref_psf: If True, use the true reference PSF weights (realistic scenario
                          where we know the central PSF well). Only deviations are perturbed.
    """
    from model import n_deviation_coeffs
    
    keys = jr.split(key, 10)
    
    # Perturb positions
    n_stars = true_state.stars.positions.shape[0]
    pos_perturbation = position_noise * jr.normal(keys[0], (n_stars, 2))
    new_positions = true_state.stars.positions + pos_perturbation
    
    # Perturb amplitudes
    amp_perturbation = 1.0 + amplitude_noise_frac * jr.normal(keys[1], true_state.stars.amplitudes.shape)
    new_amplitudes = true_state.stars.amplitudes * amp_perturbation
    new_amplitudes = jnp.maximum(new_amplitudes, 10)  # keep positive
    
    new_stars = StarParams(positions=new_positions, amplitudes=new_amplitudes)
    
    # Perturb PSFs - keep reference weights, perturb or zero out deviations
    new_psfs = []
    for b, psf in enumerate(true_state.psf):
        if use_true_ref_psf:
            # Use true reference, start with zero deviations (will learn them)
            # Or slightly perturbed deviations
            n_dev = n_deviation_coeffs(psf.poly_order)
            if psf_deviation_noise_frac > 0:
                dev_perturbation = psf_deviation_noise_frac * jr.normal(keys[2+b], psf.deviation_coeffs.shape)
                new_dev_coeffs = psf.deviation_coeffs + dev_perturbation
            else:
                new_dev_coeffs = jnp.zeros_like(psf.deviation_coeffs)
            
            new_psfs.append(PSFBasis(
                sigmas=psf.sigmas,
                ref_weights=psf.ref_weights,  # TRUE reference
                deviation_coeffs=new_dev_coeffs,
                poly_order=psf.poly_order,
                coord_scale=psf.coord_scale
            ))
        else:
            # Perturb everything
            ref_perturbation = 1.0 + 0.1 * jr.normal(keys[2+b], psf.ref_weights.shape)
            new_ref = psf.ref_weights * ref_perturbation
            dev_perturbation = psf_deviation_noise_frac * jr.normal(keys[4+b], psf.deviation_coeffs.shape)
            new_dev_coeffs = psf.deviation_coeffs + dev_perturbation
            
            new_psfs.append(PSFBasis(
                sigmas=psf.sigmas,
                ref_weights=new_ref,
                deviation_coeffs=new_dev_coeffs,
                poly_order=psf.poly_order,
                coord_scale=psf.coord_scale
            ))
    
    # Perturb sky
    new_skies = []
    for b, sky in enumerate(true_state.sky):
        sky_perturbation = 1.0 + sky_noise_frac * jr.normal(keys[6+b], sky.coeffs.shape)
        new_coeffs = sky.coeffs * sky_perturbation
        new_skies.append(SkyModel(coeffs=new_coeffs, order=sky.order))
    
    return ModelState(stars=new_stars, psf=new_psfs, sky=new_skies)


def plot_results(data: dict, initial_state: ModelState, final_state: ModelState, log_likes: list):
    """Plot comparison of true, initial, and fitted results."""
    n_bands = len(data['images'])
    image_shape = data['image_shape']
    field_center = data['field_center']
    
    fig, axes = plt.subplots(n_bands, 4, figsize=(16, 4*n_bands))
    if n_bands == 1:
        axes = axes[None, :]
    
    for b in range(n_bands):
        # True image
        ax = axes[b, 0]
        im = ax.imshow(data['true_images'][b].reshape(image_shape), origin='lower', cmap='viridis')
        ax.set_title(f'Band {b}: True (noiseless)')
        plt.colorbar(im, ax=ax)
        
        # Noisy data
        ax = axes[b, 1]
        im = ax.imshow(data['images'][b].reshape(image_shape), origin='lower', cmap='viridis')
        ax.set_title(f'Band {b}: Noisy data')
        plt.colorbar(im, ax=ax)
        
        # Fitted model
        from model import eval_model
        fitted_image = eval_model(
            data['pixel_coords'], final_state.stars, 
            final_state.psf[b], final_state.sky[b], b, field_center
        )
        ax = axes[b, 2]
        im = ax.imshow(fitted_image.reshape(image_shape), origin='lower', cmap='viridis')
        ax.set_title(f'Band {b}: Fitted model')
        plt.colorbar(im, ax=ax)
        
        # Residual
        residual = data['images'][b] - fitted_image
        ax = axes[b, 3]
        im = ax.imshow(residual.reshape(image_shape), origin='lower', cmap='RdBu_r', 
                      vmin=-3*jnp.std(residual), vmax=3*jnp.std(residual))
        ax.set_title(f'Band {b}: Residual')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('images_comparison.png', dpi=150)
    plt.close()
    
    # Plot positions comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    true_pos = data['true_state'].stars.positions
    init_pos = initial_state.stars.positions
    final_pos = final_state.stars.positions
    
    ax = axes[0]
    ax.scatter(true_pos[:, 0], true_pos[:, 1], s=100, c='green', marker='o', label='True', alpha=0.7)
    ax.scatter(init_pos[:, 0], init_pos[:, 1], s=60, c='red', marker='x', label='Initial', alpha=0.7)
    ax.scatter(final_pos[:, 0], final_pos[:, 1], s=60, c='blue', marker='+', label='Fitted', linewidths=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Star Positions')
    ax.legend()
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(0, image_shape[0])
    
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
    ax.set_title('Position Errors')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('positions_comparison.png', dpi=150)
    plt.close()
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_likes, 'b-o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    plt.close()
    
    print("\nPlots saved: images_comparison.png, positions_comparison.png, convergence.png")


def print_comparison(true_state: ModelState, initial_state: ModelState, final_state: ModelState):
    """Print numerical comparison of parameters."""
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)
    
    # Positions
    true_pos = true_state.stars.positions
    init_pos = initial_state.stars.positions
    final_pos = final_state.stars.positions
    
    init_pos_rmse = jnp.sqrt(jnp.mean((init_pos - true_pos)**2))
    final_pos_rmse = jnp.sqrt(jnp.mean((final_pos - true_pos)**2))
    
    print(f"\nPosition RMSE:")
    print(f"  Initial: {init_pos_rmse:.4f} pixels")
    print(f"  Fitted:  {final_pos_rmse:.4f} pixels")
    print(f"  Improvement: {init_pos_rmse/final_pos_rmse:.1f}x")
    
    # Amplitudes
    for b in range(true_state.stars.amplitudes.shape[0]):
        true_amp = true_state.stars.amplitudes[b]
        init_amp = initial_state.stars.amplitudes[b]
        final_amp = final_state.stars.amplitudes[b]
        
        init_amp_err = jnp.mean(jnp.abs(init_amp - true_amp) / true_amp) * 100
        final_amp_err = jnp.mean(jnp.abs(final_amp - true_amp) / true_amp) * 100
        
        print(f"\nBand {b} amplitude mean absolute % error:")
        print(f"  Initial: {init_amp_err:.1f}%")
        print(f"  Fitted:  {final_amp_err:.1f}%")
    
    # Sky
    for b in range(len(true_state.sky)):
        true_sky = true_state.sky[b].coeffs
        init_sky = initial_state.sky[b].coeffs
        final_sky = final_state.sky[b].coeffs
        
        print(f"\nBand {b} sky coefficients:")
        print(f"  True:    {true_sky[:3]}")
        print(f"  Initial: {init_sky[:3]}")
        print(f"  Fitted:  {final_sky[:3]}")


if __name__ == '__main__':
    from model import fit_model
    
    print("Generating synthetic data...")
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)
    
    # Generate data
    data = generate_test_data(
        key1,
        image_shape=(64, 64),
        n_stars=10,
        n_bands=2
    )
    
    print(f"Image shape: {data['image_shape']}")
    print(f"Number of stars: {data['true_state'].stars.positions.shape[0]}")
    print(f"Number of bands: {len(data['images'])}")
    print(f"Field center: {data['field_center']}")
    
    # TEST: Use TRUE PSF and TRUE sky, only fit positions and amplitudes
    print("\n" + "="*60)
    print("TEST: TRUE PSF + TRUE SKY, fit only positions & amplitudes")
    print("="*60)
    
    initial_state = perturb_initial_guess(
        key2,
        data['true_state'],
        position_noise=2.0,
        amplitude_noise_frac=0.3,
        psf_deviation_noise_frac=0.0,
        sky_noise_frac=0.0,  # No sky perturbation
        use_true_ref_psf=True
    )
    
    # Replace with TRUE PSF and TRUE sky
    initial_state = ModelState(
        stars=initial_state.stars,
        psf=data['true_state'].psf,
        sky=data['true_state'].sky  # TRUE sky
    )
    
    final_state, log_likes = fit_model(
        data['images'],
        data['pixel_coords'],
        initial_state,
        field_center=data['field_center'],
        max_iter=30,
        verbose=True,
        warmup_iters=100,  # Never fit PSF
        fit_sky=False  # Don't fit sky either
    )
    
    print_comparison(data['true_state'], initial_state, final_state)
    
    # Plot
    print("\nGenerating plots...")
    plot_results(data, initial_state, final_state, log_likes)
