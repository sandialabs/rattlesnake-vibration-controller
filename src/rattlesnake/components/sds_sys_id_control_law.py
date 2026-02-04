import numpy as np
from scipy.optimize import minimize


# Define a function to do the optimization
def optimize_phase_targets_pinv(
    A, b_amplitude, weight_accuracy=1.0, weight_magnitude=1.0, rcond=None, phi0=None
):
    """
    Optimize the phase targets of b to balance the accuracy of Ax ≈ b and the magnitude of x,
    using the pseudoinverse of A for computational efficiency.

    Parameters:
        A (ndarray): Complex matrix (m x n).
        b_amplitude (ndarray): Desired amplitudes of b (real-valued, length m).
        weight_accuracy (float): Weight for the accuracy term (default: 1.0).
        weight_magnitude (float): Weight for the magnitude term (default: 1.0).

    Returns:
        x_opt (ndarray): Optimal solution for x.
        b_opt (ndarray): Optimal b with optimized phase targets.
        result (OptimizeResult): Optimization result object from scipy.optimize.
    """
    m, n = A.shape

    # Precompute the pseudoinverse of A
    A_pinv = np.linalg.pinv(A, rcond=rcond)

    # Objective function: balance accuracy of Ax ≈ b and magnitude of x
    def objective(phi):
        # Construct b with the current phase
        b = b_amplitude * np.exp(1j * phi)

        # Compute x using the pseudoinverse
        x = A_pinv @ b

        # Compute accuracy term: ||Ax - b||_2^2
        Ax = A @ x
        accuracy_term = np.sum((np.abs(Ax) - np.abs(b)) ** 2)

        # Compute magnitude term: ||x||_2^2
        magnitude_term = np.sum(np.abs(x) ** 2)

        # Weighted objective function
        return weight_accuracy * accuracy_term + weight_magnitude * magnitude_term

    # Initial guess for phi (zero phase)
    if phi0 is None:
        phi0 = np.zeros(m)

    # Optimize the phase of b
    result = minimize(objective, phi0, method="L-BFGS-B", bounds=[(-2 * np.pi, 2 * np.pi)] * m)

    # Optimal phase and corresponding b
    phi_opt = result.x
    b_opt = b_amplitude * np.exp(1j * phi_opt)

    # Compute the optimal x using the pseudoinverse
    x_opt = A_pinv @ b_opt

    return x_opt, b_opt, result
