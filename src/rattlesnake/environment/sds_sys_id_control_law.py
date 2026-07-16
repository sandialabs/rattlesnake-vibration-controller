import numpy as np
from scipy.optimize import minimize
from rattlesnake.environment.sds_sys_id_metadata import SDSMetadata
from rattlesnake.process.abstract_sysid_data_analysis import SysIdDataPackage
import os
import pickle
from datetime import datetime

DEBUG = True
DEBUG_FILENAME = None


def optimize_phase_targets_pinv(
    A,
    b_amplitude,
    weight_accuracy=1.0,
    weight_magnitude=1.0,
    rcond=None,
    phi0=None,
):
    """
    Optimize the phase targets of b to balance the accuracy of Ax ≈ b and the magnitude of x,
    using the pseudoinverse of A.

    Parameters
    ----------
    A : ndarray
        Complex FRF matrix of shape (num_control_channels, num_drive_channels)
    b_amplitude : ndarray
        Desired response amplitudes, shape (num_control_channels,)
    weight_accuracy : float
        Weight for accuracy of Ax ≈ b
    weight_magnitude : float
        Weight for small drive magnitude
    rcond : float or None
        Regularization parameter for pseudoinverse
    phi0 : ndarray or None
        Optional initial phase guess for the responses

    Returns
    -------
    x_opt : ndarray
        Optimal complex drive vector
    b_opt : ndarray
        Optimal complex response target vector
    result : OptimizeResult
        Result object from scipy.optimize.minimize
    """
    m, _ = A.shape
    A_pinv = np.linalg.pinv(A, rcond=rcond)

    def objective(phi):
        b = b_amplitude * np.exp(1j * phi)
        x = A_pinv @ b
        Ax = A @ x

        accuracy_term = np.sum((np.abs(Ax) - np.abs(b)) ** 2)
        magnitude_term = np.sum(np.abs(x) ** 2)

        return weight_accuracy * accuracy_term + weight_magnitude * magnitude_term

    if phi0 is None:
        phi0 = np.zeros(m)

    result = minimize(
        objective,
        phi0,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * m,
    )

    phi_opt = result.x
    b_opt = b_amplitude * np.exp(1j * phi_opt)
    x_opt = A_pinv @ b_opt

    return x_opt, b_opt, result


def default_control_law(
    environment_metadata: SDSMetadata,
    sysid_data: SysIdDataPackage,
    last_response_srs: np.ndarray = None,
    last_response_signals: np.ndarray = None,
    last_drive_amplitudes: np.ndarray = None,
    last_drive_decays: np.ndarray = None,
    last_drive_delays: np.ndarray = None,
    last_drive_signals: np.ndarray = None,
    *,
    rcond: float = 1e-10,
    accuracy_weight: float = 100.0,
    input_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Default SDS control law based on optimized phase-target pseudoinverse MIMO solution.

    Returns
    -------
    amplitudes : ndarray
        Shape (num_frequencies_with_comp, num_drive_channels)
    decays : ndarray
        Shape (num_frequencies_with_comp, num_drive_channels)
    delays : ndarray
        Shape (num_frequencies_with_comp, num_drive_channels)
    """
    print("Running the default control law!")

    if DEBUG:
        if DEBUG_FILENAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_filename = f"sds_default_control_law_debug_{timestamp}.pkl"
        else:
            debug_filename = DEBUG_FILENAME

        debug_data = {
            "environment_metadata": environment_metadata,
            "sysid_data": sysid_data,
            "last_response_srs": last_response_srs,
            "last_response_signals": last_response_signals,
            "last_drive_amplitudes": last_drive_amplitudes,
            "last_drive_decays": last_drive_decays,
            "last_drive_delays": last_drive_delays,
            "last_drive_signals": last_drive_signals,
            "rcond": rcond,
            "accuracy_weight": accuracy_weight,
            "input_weight": input_weight,
        }

        with open(debug_filename, "wb") as f:
            pickle.dump(debug_data, f)

        print(f"Saved SDS control law debug data to {os.path.abspath(debug_filename)}")

    # Main control frequencies (not including compensation pulse)
    control_frequencies = environment_metadata.get_sds_frequencies()
    all_frequencies = environment_metadata.get_sds_frequencies_w_compensation_pulse()

    num_control_freqs = len(control_frequencies)
    num_all_freqs = len(all_frequencies)
    num_drive_channels = environment_metadata.num_reference_channels
    num_control_channels = environment_metadata.num_response_channels

    # Use metadata-defined decays
    control_decays = environment_metadata.get_sds_decays()
    all_decays = environment_metadata.get_sds_decays_w_compensation_pulse()

    # Pull target SRS from metadata
    target_srs = np.array(environment_metadata.specification_data.srs_spec, dtype=float)

    # Sanity check shape
    if target_srs.shape[0] != num_control_freqs:
        raise ValueError(
            f"Specification SRS frequency dimension ({target_srs.shape[0]}) does not match "
            f"number of SDS frequencies ({num_control_freqs})."
        )

    if target_srs.shape[1] != num_control_channels:
        raise ValueError(
            f"Specification SRS channel dimension ({target_srs.shape[1]}) does not match "
            f"number of control channels ({num_control_channels})."
        )

    # FRFs should align to these frequencies already through system ID frame spacing.
    frf_frequencies = np.array(sysid_data.frequencies)
    frf_matrix = np.array(sysid_data.sysid_frf)

    # Allocate outputs for the main SRS frequencies
    drive_amplitudes = np.zeros((num_control_freqs, num_drive_channels), dtype=float)
    drive_delays = np.zeros((num_control_freqs, num_drive_channels), dtype=float)

    previous_response_phases = None

    for i_freq, freq in enumerate(control_frequencies):
        frf_index = np.argmin(np.abs(frf_frequencies - freq))
        A_full = frf_matrix[frf_index]  # shape: (num_control_channels, num_drive_channels)

        target_amplitudes_full = target_srs[i_freq]  # shape: (num_control_channels,)

        # Ignore unconstrained channels at this frequency
        valid = np.isfinite(target_amplitudes_full) & (target_amplitudes_full > 0)

        if not np.any(valid):
            continue

        A = A_full[valid, :]
        b_amplitude = target_amplitudes_full[valid]

        phi0 = None
        if (
            previous_response_phases is not None
            and previous_response_phases.size == b_amplitude.size
        ):
            phi0 = previous_response_phases

        x_opt, b_opt, _ = optimize_phase_targets_pinv(
            A,
            b_amplitude,
            weight_accuracy=accuracy_weight,
            weight_magnitude=input_weight,
            rcond=rcond,
            phi0=phi0,
        )

        previous_response_phases = np.angle(b_opt)

        drive_amplitudes[i_freq, :] = np.abs(x_opt)

        # Convert drive phase to delay
        # delay = -phase / (2*pi*f)
        if freq > 0:
            drive_delays[i_freq, :] = -np.angle(x_opt) / (2 * np.pi * freq)
        else:
            drive_delays[i_freq, :] = 0.0

    # Build full arrays including compensation pulse row
    amplitudes = np.zeros((num_all_freqs, num_drive_channels), dtype=float)
    delays = np.zeros((num_all_freqs, num_drive_channels), dtype=float)
    decays = np.ones((num_all_freqs, num_drive_channels), dtype=float) * all_decays[:, np.newaxis]

    amplitudes[:-1, :] = drive_amplitudes
    delays[:-1, :] = drive_delays

    # Compensation pulse initialized to zero amplitude and zero delay.
    # Synthesis machinery will still see the row and can handle compensation consistently.
    amplitudes[-1, :] = 0.0
    delays[-1, :] = 0.0

    return amplitudes, decays, delays
