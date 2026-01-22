"""
Utility functions for PSF photometry.
"""

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float


@jit
def poly_features_2d(x: Float[Array, "2"]) -> Float[Array, "6"]:
    """Compute 2D polynomial features for order=2.

    Returns: [1, y, y², x, xy, x²]
    """
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
    """Compute 2D polynomial features without constant term.

    Returns: [y, y², x, xy, x²]
    """
    return jnp.array([
        x[1],
        x[1]**2,
        x[0],
        x[0] * x[1],
        x[0]**2
    ])


@jit
def solve_weighted_lstsq(
    A: Float[Array, "n m"],
    b: Float[Array, "n"],
    weights: Float[Array, "n"],
    regularization: float = 1e-6
) -> Float[Array, "m"]:
    """Solve weighted least squares: argmin_x ||sqrt(W)(Ax - b)||^2

    Args:
        A: Design matrix (n x m)
        b: Target vector (n,)
        weights: Weights for each observation (n,)
        regularization: Ridge regularization parameter

    Returns:
        Solution vector (m,)
    """
    WA = weights[:, None] * A
    AtwA = A.T @ WA + regularization * jnp.eye(A.shape[1])
    Atwb = A.T @ (weights * b)
    return jnp.linalg.solve(AtwA, Atwb)


@jit
def poisson_log_likelihood(
    data: Float[Array, "n_pix"],
    model: Float[Array, "n_pix"]
) -> float:
    """Compute Poisson log-likelihood.

    LL = sum(data * log(model) - model)

    Args:
        data: Observed counts
        model: Model predictions

    Returns:
        Log-likelihood value
    """
    model_safe = jnp.maximum(model, 1e-10)
    return jnp.sum(data * jnp.log(model_safe) - model_safe)
