"""
Visualization utilities for PSF photometry results.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

from .models import ModelState


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
    """Plot comparison of true, initial, and fitted results.

    Args:
        data: Observed pixel values
        true_model: True noiseless model
        fitted_model: Fitted model prediction
        true_state: True model state
        initial_state: Initial model state
        final_state: Final fitted model state
        image_shape: Shape of image (ny, nx)
        log_likes: Optional list of log-likelihoods per iteration
        prefix: Prefix for output filenames
    """
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

        n_iter = len(log_likes)

        # Full convergence plot
        ax = axes[0]
        ax.plot(log_likes, 'b-o', markersize=4)
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
