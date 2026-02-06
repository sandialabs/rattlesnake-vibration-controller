import numpy as np
from scipy.optimize import minimize
from .sds_sys_id_metadata import SDSMetadata
from enum import Enum


def default_control_law(
    environment_parameters: SDSMetadata,
    transfer_function_frequencies: np.ndarray,
    transfer_function: np.ndarray,
    noise_response_cpsd: np.ndarray,
    noise_reference_cpsd: np.ndarray,
    sysid_response_cpsd: np.ndarray,
    sysid_reference_cpsd: np.ndarray,
    multiple_coherence: np.ndarray,
    frames: int,
    last_response_srs: np.ndarray = None,
    last_drive_amplitudes: np.ndarray = None,
    last_drive_decays: np.ndarray = None,
    last_drive_delays: np.ndarray = None,
    *,
    rcond: float = 1e-10,
    accuracy_weight: float = 100.0,
    input_weight: float = 1.0,
) -> tuple[np.ndarray]:
    """A control law to generate a sum-of-decayed-sines table that will be used to generate a
    voltage signal that will be played from the drive channels.

    Parameters
    ----------
    environment_parameters : SDSMetadata
        The metadata object describing the environment, including the specification, SRS Parameters,
        sine tone information, and other data used to describe the environment.
    transfer_function_frequencies : np.ndarray
        A 1D numpy array containing the frequency lines corresponding to the provided transfer
        function.
    transfer_function : np.ndarray
        A 3D transfer function array with dimension (num_frequencies, num_control_channels,
        num_drive_channels).  This should be used to take into account the system dynamics when
        generating excitation signals.
    noise_response_cpsd : np.ndarray
        A 3D CPSD matrix consisting of the noise on the control channels measured during the
        system identification.  This can be used to tailor the drive signals based on the noise
        level in the test.  The dimensions are (num_frequencies, num_control_channels,
        num_control_channels).
    noise_reference_cpsd : np.ndarray
        A 3D CPSD matrix consisting of the noise on the drive channels measured during the
        system identification.  This can be used to tailor the drive signals based on the noise
        level in the test.  The dimensions are (num_frequencies, num_drive_channels,
        num_drive_channels).
    sysid_response_cpsd : np.ndarray
        A 3D CPSD matrix consisting of the response levels on and correlations between
        the control channels measured during the system identification.  This can be used to
        identify preferred phasing between response channels in the test.  The dimensions are
        (num_frequencies, num_control_channels, num_control_channels).
    sysid_reference_cpsd : np.ndarray
        A 3D CPSD matrix consisting of the drive levels on and correlations between
        the drive channels measured during the system identification.  The dimensions are
        (num_frequencies, num_drive_channels, num_drive_channels).
    multiple_coherence : np.ndarray
        A 2D CPSD matrix consisting of the multiple coherence for each control channel, allowing
        the control law to judge the quality of the measured transfer functions and potentially
        take that into account during development of drive signals.  The dimensions are (
        num_frequencies, num_control_channels).
    frames : int
        The number of frames of data measured in the system identification.
    extra_parameters : str, optional
        A string containing the information typed into the "Control Parameters" box on the
        environment definition user interface panel.  Control laws may parse this string to allow
        additional parameters to be supplied to the control law from the graphical user interface.
        By default, a value of "" is provided.
    last_response_srs : np.ndarray, optional
        A 2D SRS array with dimension (num_frequencies, num_control_channels).  This is the SRS
        developed from the previous control responses that can be used for error corrections.
        For the first hit, this argument will be None, as no previous responses yet exist.
    last_drive_amplitudes : DecayedSineTable, optional
        The amplitudes of the drive signals used to achieve the results in the `last_response_srs`
        argument.  This can be used to make adjustments to the previous drive signals.
    last_drive_decays : DecayedSineTable, optional
        The decays of the drive signals used to achieve the results in the `last_response_srs`
        argument.  This can be used to make adjustments to the previous drive signals.
    last_drive_delays : DecayedSineTable, optional
        The delays of the drive signals used to achieve the results in the `last_response_srs`
        argument.  This can be used to make adjustments to the previous drive signals.
    rcond : float
        An optional

    Returns
    -------
    amplitudes : np.ndarray
        An array of amplitudes at the control_frequencies that is used to
        generate the next drive signals.  The control law must compute the correct amplitudes,
        decay values, and delay values to best match the control SRS.  Should have dimensions
        (num_frequencies, num_drive_channels)
    decays : np.ndarray
        An array of decay values in terms of damping (zeta) at the control_frequencies that is
        used to generate the next drive signals.  The control law must compute the correct
        amplitudes, decay values, and delay values to best match the control SRS.  Should have
        dimensions (num_frequencies, num_drive_channels)
    delays : np.ndarray
        An array of time delays at the control_frequencies that is used to
        generate the next drive signals.  The control law must compute the correct amplitudes,
        decay values, and delay values to best match the control SRS.  Should have dimensions
        (num_frequencies, num_drive_channels)
    """
    print("Running the default control law!")
    frequencies = environment_parameters.get_sds_frequencies_w_compensation_pulse()
    num_frequencies = len(frequencies)
    num_drive_signals = environment_parameters.num_reference_channels
    decays = environment_parameters.get_sds_decays_w_compensation_pulse()
    amplitudes = np.ones((num_frequencies, num_drive_signals)) * np.linspace(
        1, 2, num_drive_signals
    )
    amplitudes[1::2] = -1 * amplitudes[1::2]
    delays = np.zeros((num_frequencies, num_drive_signals))
    decays = np.ones((num_frequencies, num_drive_signals)) * decays[:, np.newaxis]
    return amplitudes, decays, delays


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
