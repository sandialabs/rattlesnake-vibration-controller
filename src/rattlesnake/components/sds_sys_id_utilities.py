import numpy as np
from enum import Enum
from .utilities import VerboseMessageQueue
import multiprocessing as mp
from multiprocessing.queues import Queue

from .sds_sys_id_metadata import DecayStrategy


# %% Commands
class SDSCommands(Enum):
    """Valid commands for the SDS environment"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    PERFORM_CONTROL_PREDICTION = 2
    SDS_TABLE_PREDICTION = 3
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 4


# %% Queues


class SDSQueues:
    """A container class for the queues that this environment will manage."""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """A container class for the queues that SDS will manage.

        The environment uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.

        Parameters
        ----------
        environment_name : str
            Name of the environment
        environment_command_queue : VerboseMessageQueue
            Queue that is read by the environment for environment commands
        gui_update_queue : mp.queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        controller_communication_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        data_in_queue : mp.queues.Queue
            Multiprocessing queue that connects the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        data_out_queue : mp.queues.Queue
            Multiprocessing queue that connects the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.
        log_file_queue : VerboseMessageQueue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.data_analysis_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Analysis Command Queue"
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Signal Generation Command Queue"
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Spectral Computation Command Queue"
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Collector Command Queue"
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.time_history_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


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


class DecayedSineTable(np.ndarray):

    def __new__(cls, shape, num_signals, buffer=None, offset=0, strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to __array_finalize__
        data_dtype = [
            ("frequency", "float64"),
            ("amplitude", "float64", num_signals),
            ("decay", "float64", num_signals),
            ("delay", "float64", num_signals),
        ]
        obj = super().__new__(cls, shape, data_dtype, buffer, offset, strides, order)
        # Finally, we must return the newly created object:
        return obj


def decayed_sine_table(
    frequency,
    amplitude,
    decay,
    delay,
):
    """
    Helper function to create a DecayedSineTable object.

    Parameters
    ----------
    frequency : np.ndarray
        Frequencies of the decaying sine waves.
    amplitude : np.ndarray
        Amplitudes of the decaying sine waves.
    decay : np.ndarray
        Damping values of the decaying sine waves (zeta, not time constants)
    delay : np.ndarray
        Delay values of the decaying sine waves.

    Returns
    -------
    DecayedSineTable :
        A DecayedSineTable object containing the specified information

    """
    num_frequencies, num_signals = amplitude.shape
    st = DecayedSineTable(num_frequencies, num_signals)
    st["frequency"] = frequency
    st["amplitude"] = amplitude
    st["decay"] = decay
    st["delay"] = delay
    return st
