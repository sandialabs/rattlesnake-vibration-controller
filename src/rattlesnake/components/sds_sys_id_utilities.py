from .sds_sys_id_metadata import DecayStrategy
import numpy as np


def octspace(low, high, points_per_octave):
    """
    Constructs octave spacing between low and high values

    Parameters
    ----------
    low : float
        Starting value for the spacing
    high : float
        Upper value for the spacing
    points_per_octave : int
        Number of points per octave

    Returns
    -------
    octave_points : np.ndarray
        Octave-spaced points
    """
    num_octaves = np.log2(high / low)
    num_steps = np.ceil(num_octaves * points_per_octave)
    point_indices = np.arange(num_steps + 1)
    log_points = np.log2(low) + num_octaves / num_steps * point_indices
    points = 2**log_points
    return points


def convert_damping_strategy(
    old_values: np.ndarray,
    frequencies: np.ndarray,
    block_length: float,
    old_strategy: DecayStrategy,
    new_strategy: DecayStrategy,
):
    """Convert between different damping strategies

    Parameters
    ----------
    old_values : np.ndarray
        Damping values defined in the stratgy given in `old_strategy`
    frequencies : np.ndarray
        Frequencies (in Hz) corresponding to the damping values in `old_values`
    block_length : float
        Length of the time block (in seconds) over which the number of time constants will be
        evaluated
    old_strategy : DecayStrategy
        The decay strategy in which the the `old_values` are defined.
    new_strategy : DecayStrategy
        The new strategy in which the decay values will be returned.

    Returns
    -------
    new_values : np.ndarray
        Decay values defined in the form specified by `new_strategy`

    Raises
    ------
    ValueError
        If invalid decay strategies are provided
    """
    if old_strategy == new_strategy:
        # If the strategies are the same, return the old values directly
        return old_values

    # Convert frequencies to angular frequencies (omega)
    omega = 2 * np.pi * frequencies

    # Conversion logic
    if old_strategy == DecayStrategy.TIME_CONSTANT:
        tau = old_values
        if new_strategy == DecayStrategy.DAMPING:
            # Convert tau to zeta
            zeta = 1 / (tau * omega)
            return zeta
        if new_strategy == DecayStrategy.NUM_TIME_CONSTANTS:
            # Convert tau to number of time constants per block
            num_time_constants = block_length / tau
            return num_time_constants

    elif old_strategy == DecayStrategy.DAMPING:
        zeta = old_values
        tau = 1 / (zeta * omega)
        if new_strategy == DecayStrategy.TIME_CONSTANT:
            # Convert zeta to tau
            return tau
        if new_strategy == DecayStrategy.NUM_TIME_CONSTANTS:
            # Convert zeta to number of time constants per block
            num_time_constants = block_length / tau
            return num_time_constants

    elif old_strategy == DecayStrategy.NUM_TIME_CONSTANTS:
        num_time_constants = old_values
        tau = block_length / num_time_constants
        if new_strategy == DecayStrategy.TIME_CONSTANT:
            # Convert number of time constants per block to tau
            return tau
        if new_strategy == DecayStrategy.DAMPING:
            # Convert number of time constants per block to zeta
            zeta = 1 / (tau * omega)
            return zeta

    # If no valid conversion is found, raise an error
    raise ValueError("Invalid conversion between damping strategies.")
