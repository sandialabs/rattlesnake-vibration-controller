import numpy as np
from scipy.optimize import minimize
from rattlesnake.environment.sds_sys_id_metadata import SDSMetadata
from rattlesnake.environment.sds_sys_id_utilities import sum_decayed_sines
from rattlesnake.process.abstract_sysid_data_analysis import SysIdDataPackage
import os
import pickle
from datetime import datetime
from scipy.interpolate import interp1d

DEBUG = False
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
        Weight for accuracy of Ax = b
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

    # Use metadata-defined decays
    control_decays = environment_metadata.get_sds_decays()

    # Pull target SRS from metadata
    target_srs = np.array(environment_metadata.specification_data.srs_spec, dtype=float)
    target_frequencies = np.array(environment_metadata.specification_data.frequencies, dtype=float)

    sds_amplitudes = []
    sds_decays = []
    for index, specification in enumerate(target_srs.T):
        breakpoint_table = np.concatenate(
            (target_frequencies[:, np.newaxis], specification[:, np.newaxis]), axis=-1
        )
        _, _, _, _, sine_amplitudes, sine_decays, _ = sum_decayed_sines(
            environment_metadata.sample_rate,
            environment_metadata.block_size,
            sine_frequencies=control_frequencies,
            sine_decays=control_decays,
            srs_breakpoints=breakpoint_table,
            srs_damping=environment_metadata.srs_data.srs_damping,
            srs_type=environment_metadata.srs_data.srs_type.value
            * environment_metadata.srs_data.srs_displacement.value,
            ignore_compensation_pulse=True,
        )
        sds_amplitudes.append(sine_amplitudes)
        sds_decays.append(sine_decays)
    sds_amplitudes = np.array(sds_amplitudes)
    sds_decays = np.array(sds_decays)

    # Set up the initial optimization problem
    x_opt = []
    b_opt = []
    result = []

    # Get the transfer functions
    sysid_frequencies = sysid_data.frequencies
    frfs = sysid_data.sysid_frf
    frf_interpolator = interp1d(sysid_frequencies, frfs, axis=0, kind="linear", bounds_error=True)
    A_all = frf_interpolator(control_frequencies)
    b_all = sds_amplitudes[:, :-1].T

    # Solve for the specification phases that result in the best accuracy and force
    for A, b_amplitude in zip(A_all, b_all):
        x_o, b_o, r_o = optimize_phase_targets_pinv(
            A,
            b_amplitude,
            rcond=rcond,
            weight_accuracy=accuracy_weight,
            weight_magnitude=input_weight,
        )
        x_opt.append(x_o)
        b_opt.append(b_o)
        result.append(r_o)

    x_opt = np.array(x_opt)
    b_opt = np.array(b_opt)

    # Now that we know the phases, recompute the SRSs with adjusted phases
    # to get better amplitude estimates
    phases = np.angle(b_opt).T
    delays = -phases / (2 * np.pi * control_frequencies)
    decays = sds_decays[:, :-1]

    sds_amplitudes = []
    sds_decays = []
    for index, specification in enumerate(target_srs.T):
        breakpoint_table = np.concatenate(
            (target_frequencies[:, np.newaxis], specification[:, np.newaxis]), axis=-1
        )
        _, _, _, _, sine_amplitudes, sine_decays, _ = sum_decayed_sines(
            environment_metadata.sample_rate,
            environment_metadata.block_size,
            sine_frequencies=control_frequencies,
            sine_decays=decays[index],
            sine_delays=delays[index],
            srs_breakpoints=breakpoint_table,
            srs_damping=environment_metadata.srs_data.srs_damping,
            srs_type=environment_metadata.srs_data.srs_type.value
            * environment_metadata.srs_data.srs_displacement.value,
            ignore_compensation_pulse=True,
        )
        sds_amplitudes.append(sine_amplitudes)
        sds_decays.append(sine_decays)
    sds_amplitudes = np.array(sds_amplitudes)
    sds_decays = np.array(sds_decays)

    # Now again solve for the drive signals to match this preferred phasing
    x_opt = []
    result = []
    angle_guess = np.angle(b_opt)
    b_all = sds_amplitudes[:, :-1].T
    b_opt2 = []

    for A, b, phi0 in zip(A_all, b_all, angle_guess):
        x_o, b_o, r_o = optimize_phase_targets_pinv(
            A,
            np.abs(b),
            rcond=rcond,
            phi0=phi0,
            weight_accuracy=accuracy_weight,
            weight_magnitude=input_weight,
        )
        x_opt.append(x_o)
        b_opt2.append(b_o)
        result.append(r_o)

    x_opt = np.array(x_opt).T
    b_opt2 = np.array(b_opt2).T

    # Extract the drive amplitudes and phases
    amplitudes = np.abs(x_opt).T
    drive_phases = np.angle(x_opt)
    delays = (-drive_phases / (2 * np.pi * control_frequencies)).T
    decays = np.tile(sds_decays[:1, :-1], [amplitudes.shape[-1], 1]).T

    # Add back the compensation pulse if necessary
    if environment_metadata.compensation_pulse_data.use_compensation_pulse:
        amplitudes = np.concatenate((amplitudes, np.zeros((1, amplitudes.shape[-1]))))
        delays = np.concatenate((delays, np.zeros((1, delays.shape[-1]))))
        decays = np.concatenate((decays, np.zeros((1, decays.shape[-1]))))

    return amplitudes, decays, delays
