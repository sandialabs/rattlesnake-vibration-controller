# -*- coding: utf-8 -*-
"""
This file defines a Modal Testing Environment where users can perform
hammer or shaker modal tests and export FRFs and other relevant data.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import time
from enum import Enum
from glob import glob
from multiprocessing.queues import Queue
import netCDF4 as nc4
import numpy as np
from rattlesnake.environment.abstract_environment import AbstractEnvironment, AbstractMetadata
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.process.signal_generation import (
    BurstRandomSignalGenerator,
    ChirpSignalGenerator,
    PseudorandomSignalGenerator,
    RandomSignalGenerator,
    SineSignalGenerator,
    SquareSignalGenerator,
)
from rattlesnake.utilities import (
    DataAcquisitionParameters,
    GlobalCommands,
    VerboseMessageQueue,
    flush_queue,
)

CONTROL_TYPE = ControlTypes.MODAL
WAIT_TIME = 0.02


class ModalCommands(Enum):
    """Valid commands for the modal environment"""

    START_CONTROL = 0
    STOP_CONTROL = 1
    ACCEPT_FRAME = 2
    RUN_CONTROL = 3
    CHECK_FOR_COMPLETE_SHUTDOWN = 4


class ModalUICommands(Enum):
    SPECTRAL_UPDATE = 0
    FINISHED = 1


# region: Queues
class ModalQueues:
    """A set of queues used by the modal environment"""

    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        gui_update_queue: mp.queues.Queue,
        controller_communication_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        log_file_queue: VerboseMessageQueue,
    ):
        """
        Creates a namespace to store all the queues used by the Modal Environment

        Parameters
        ----------
        environment_command_queue : VerboseMessageQueue
            Queue from which the environment will receive instructions.
        gui_update_queue : mp.queues.Queue
            Queue to which the environment will put GUI updates.
        controller_communication_queue : VerboseMessageQueue
            Queue to which the environment will put global contorller instructions.
        data_in_queue : mp.queues.Queue
            Queue from which the environment will receive data from acquisition.
        data_out_queue : mp.queues.Queue
            Queue to which the environment will write data for output.
        log_file_queue : VerboseMessageQueue
            Queue to which the environment will write log file messages.
        """
        self.environment_command_queue = environment_command_queue
        self.gui_update_queue = gui_update_queue
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.log_file_queue = log_file_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.signal_generation_update_queue = mp.Queue()
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Spectral Computation Command Queue"
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Data Collector Command Queue"
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue, environment_name + " Signal Generation Command Queue"
        )


# region: Metadata
class ModalMetadata(AbstractMetadata):
    """Class for storing metadata for an environment.

    This class is used as a storage container for parameters used by an
    environment.  It is returned by the environment UI's
    ``collect_environment_definition_parameters`` function as well as its
    ``initialize_environment`` function.  Various parts of the controller and
    environment will query the class's data members for parameter values.
    """

    def __init__(
        self,
        sample_rate: float,
        samples_per_frame: int,
        averaging_type: str,
        num_averages: int,
        averaging_coefficient: float,
        frf_technique: str,
        frf_window: str,
        overlap_percent: float,
        trigger_type: str,
        accept_type: str,
        wait_for_steady_state: float,
        trigger_channel: int,
        pretrigger_percent: float,
        trigger_slope_positive: bool,
        trigger_level_percent: float,
        hysteresis_level_percent: float,
        hysteresis_frame_percent: float,
        signal_generator_type: str,
        signal_generator_level: float,
        signal_generator_min_frequency: float,
        signal_generator_max_frequency: float,
        signal_generator_on_percent: float,
        acceptance_function,
        reference_channel_indices,
        response_channel_indices,
        output_channel_indices,
        data_acquisition_parameters: DataAcquisitionParameters,
        exponential_window_value_at_frame_end: float,
    ):
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.averaging_type = averaging_type
        self.num_averages = num_averages
        self.averaging_coefficient = averaging_coefficient
        self.frf_technique = frf_technique
        self.frf_window = frf_window
        self.overlap = overlap_percent / 100
        self.trigger_type = trigger_type
        self.accept_type = accept_type
        self.wait_for_steady_state = wait_for_steady_state
        self.trigger_channel = trigger_channel
        self.pretrigger = pretrigger_percent / 100
        self.trigger_slope_positive = trigger_slope_positive
        self.trigger_level = trigger_level_percent / 100
        self.hysteresis_level = hysteresis_level_percent / 100
        self.hysteresis_length = hysteresis_frame_percent / 100
        self.signal_generator_type = signal_generator_type
        self.signal_generator_level = signal_generator_level
        self.signal_generator_min_frequency = signal_generator_min_frequency
        self.signal_generator_max_frequency = signal_generator_max_frequency
        self.signal_generator_on_fraction = signal_generator_on_percent / 100
        self.acceptance_function = acceptance_function
        self.reference_channel_indices = reference_channel_indices
        self.response_channel_indices = response_channel_indices
        self.output_channel_indices = output_channel_indices
        self.exponential_window_value_at_frame_end = exponential_window_value_at_frame_end
        # Set up signal generator
        self.output_oversample = data_acquisition_parameters.output_oversample
        self.signal_generator = self.get_signal_generator()

    def get_signal_generator(self):
        """Gets a signal generator object that the modal environment will use to generate signals"""
        if self.signal_generator_type == "none":
            signal_generator = PseudorandomSignalGenerator(
                rms=0.0,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                low_frequency_cutoff=self.signal_generator_min_frequency,
                high_frequency_cutoff=self.signal_generator_max_frequency,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "random":
            signal_generator = RandomSignalGenerator(
                rms=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                low_frequency_cutoff=self.signal_generator_min_frequency,
                high_frequency_cutoff=self.signal_generator_max_frequency,
                cola_overlap=0.5,
                cola_window="hann",
                cola_exponent=0.5,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "pseudorandom":
            signal_generator = PseudorandomSignalGenerator(
                rms=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                low_frequency_cutoff=self.signal_generator_min_frequency,
                high_frequency_cutoff=self.signal_generator_max_frequency,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "burst":
            signal_generator = BurstRandomSignalGenerator(
                rms=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                low_frequency_cutoff=self.signal_generator_min_frequency,
                high_frequency_cutoff=self.signal_generator_max_frequency,
                on_fraction=self.signal_generator_on_fraction,
                ramp_fraction=0.05,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "chirp":
            signal_generator = ChirpSignalGenerator(
                level=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                low_frequency_cutoff=self.signal_generator_min_frequency,
                high_frequency_cutoff=self.signal_generator_max_frequency,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "square":
            signal_generator = SquareSignalGenerator(
                level=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                frequency=self.signal_generator_min_frequency,
                phase=0,
                on_fraction=self.signal_generator_on_fraction,
                output_oversample=self.output_oversample,
            )
        elif self.signal_generator_type == "sine":
            signal_generator = SineSignalGenerator(
                level=self.signal_generator_level,
                sample_rate=self.sample_rate,
                num_samples_per_frame=self.samples_per_frame,
                num_signals=len(self.output_channel_indices),
                frequency=self.signal_generator_min_frequency,
                phase=0,
                output_oversample=self.output_oversample,
            )
        else:
            raise ValueError(f"Invalid Signal Type {self.signal_generator_type}")
        return signal_generator

    @property
    def samples_per_acquire(self):
        """Property returning the samples per acquisition step given the overlap"""
        return int(self.samples_per_frame * (1 - self.overlap))

    @property
    def frame_time(self):
        """Property returning the time per measurement frame"""
        return self.samples_per_frame / self.sample_rate

    @property
    def nyquist_frequency(self):
        """Property returning half the sample rate"""
        return self.sample_rate / 2

    @property
    def fft_lines(self):
        """Property returning the frequency lines given the sampling parameters"""
        return self.samples_per_frame // 2 + 1

    @property
    def skip_frames(self):
        """Property returning the number of frames to skip while waiting for steady state"""
        return int(
            np.ceil(
                self.wait_for_steady_state
                * self.sample_rate
                / (self.samples_per_frame * (1 - self.overlap))
            )
        )

    @property
    def frequency_spacing(self):
        """Property returning frequency line spacing given the sampling parameters"""
        return self.sample_rate / self.samples_per_frame

    def get_trigger_levels(self, channels):
        """Gets the trigger levels for a channel based on the channel table information

        Parameters
        ----------
        channels : list of Channel
            A list of channels in the environment

        Returns
        -------
        trigger_level_v
            The trigger level in volts
        trigger_level_eu
            The trigger level in engineering units defined in the channel table
        hysterisis_level_v
            The level that the signal must return below before another trigger can be accepted,
            in volts
        hysterisis_level_eu
            The level that the signal must return below before another trigger can be accepted,
            in engineering units defined in the channel table
        """
        channel = channels[self.trigger_channel]
        try:
            volt_range = float(channel.maximum_value)
            if volt_range == 0.0:
                volt_range = 10.0
        except (ValueError, TypeError):
            volt_range = 10.0
        try:
            mv_per_eu = float(channel.sensitivity)
            if mv_per_eu == 0.0:
                mv_per_eu = 1000.0
        except (ValueError, TypeError):
            mv_per_eu = 1000.0
        v_per_eu = mv_per_eu / 1000.0
        trigger_level_v = self.trigger_level * volt_range
        trigger_level_eu = trigger_level_v / v_per_eu
        hysterisis_level_v = self.hysteresis_level * volt_range
        hysterisis_level_eu = hysterisis_level_v / v_per_eu
        return (
            trigger_level_v,
            trigger_level_eu,
            hysterisis_level_v,
            hysterisis_level_eu,
        )

    @property
    def disabled_signals(self):
        """Returns a list of indices corresponding to output signals that have been disabled"""
        return [
            i
            for i, index in enumerate(self.output_channel_indices)
            if not (
                index in self.response_channel_indices or index in self.reference_channel_indices
            )
        ]

    @property
    def hysteresis_samples(self):
        """Property returning the number of samples that a signal must be below the hysterisis
        level"""
        return int(self.hysteresis_length * self.samples_per_frame)

    def store_to_netcdf(
        self, netcdf_group_handle: nc4._netCDF4.Group  # pylint: disable=c-extension-no-member
    ):
        """Store parameters to a group in a netCDF streaming file.

        This function stores parameters from the environment into the netCDF
        file in a group with the environment's name as its name.  The function
        will receive a reference to the group within the dataset and should
        store the environment's parameters into that group in the form of
        attributes, dimensions, or variables.

        This function is the "write" counterpart to the retrieve_metadata
        function in the AbstractUI class, which will read parameters from
        the netCDF file to populate the parameters in the user interface.

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A reference to the Group within the netCDF dataset where the
            environment's metadata is stored.

        """
        netcdf_group_handle.samples_per_frame = self.samples_per_frame
        netcdf_group_handle.averaging_type = self.averaging_type
        netcdf_group_handle.num_averages = self.num_averages
        netcdf_group_handle.averaging_coefficient = self.averaging_coefficient
        netcdf_group_handle.frf_technique = self.frf_technique
        netcdf_group_handle.frf_window = self.frf_window
        netcdf_group_handle.overlap = self.overlap
        netcdf_group_handle.trigger_type = self.trigger_type
        netcdf_group_handle.accept_type = self.accept_type
        netcdf_group_handle.wait_for_steady_state = self.wait_for_steady_state
        netcdf_group_handle.trigger_channel = self.trigger_channel
        netcdf_group_handle.pretrigger = self.pretrigger
        netcdf_group_handle.trigger_slope_positive = 1 if self.trigger_slope_positive else 0
        netcdf_group_handle.trigger_level = self.trigger_level
        netcdf_group_handle.hysteresis_level = self.hysteresis_level
        netcdf_group_handle.hysteresis_length = self.hysteresis_length
        netcdf_group_handle.signal_generator_type = self.signal_generator_type
        netcdf_group_handle.signal_generator_level = self.signal_generator_level
        netcdf_group_handle.signal_generator_min_frequency = self.signal_generator_min_frequency
        netcdf_group_handle.signal_generator_max_frequency = self.signal_generator_max_frequency
        netcdf_group_handle.signal_generator_on_fraction = self.signal_generator_on_fraction
        netcdf_group_handle.exponential_window_value_at_frame_end = (
            self.exponential_window_value_at_frame_end
        )
        netcdf_group_handle.acceptance_function = (
            self.acceptance_function[0] + ":" + self.acceptance_function[1]
            if self.acceptance_function is not None
            else "None"
        )
        # Reference channels
        netcdf_group_handle.createDimension(
            "reference_channels", len(self.reference_channel_indices)
        )
        var = netcdf_group_handle.createVariable(
            "reference_channel_indices", "i4", ("reference_channels")
        )
        var[...] = self.reference_channel_indices
        # Response channels
        netcdf_group_handle.createDimension("response_channels", len(self.response_channel_indices))
        var = netcdf_group_handle.createVariable(
            "response_channel_indices", "i4", ("response_channels")
        )
        var[...] = self.response_channel_indices

    @classmethod
    def from_ui(cls, ui):
        """
        Creates a ModalMetadata object from the user interface

        Parameters
        ----------
        ui : ModalUI
            A Modal User Interface.

        Returns
        -------
        test_parameters : ModalMetadata
            Parameters corresponding to the data in the user interface

        """
        signal_generator_level = 0
        signal_generator_min_frequency = 0
        signal_generator_max_frequency = 0
        signal_generator_on_percent = 0
        if ui.definition_widget.signal_generator_selector.currentIndex() == 0:  # None
            signal_generator_type = "none"
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 1:  # Random
            signal_generator_type = "random"
            signal_generator_level = ui.definition_widget.random_rms_selector.value()
            signal_generator_min_frequency = (
                ui.definition_widget.random_min_frequency_selector.value()
            )
            signal_generator_max_frequency = (
                ui.definition_widget.random_max_frequency_selector.value()
            )
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 2:  # Burst Random
            signal_generator_type = "burst"
            signal_generator_level = ui.definition_widget.burst_rms_selector.value()
            signal_generator_min_frequency = (
                ui.definition_widget.burst_min_frequency_selector.value()
            )
            signal_generator_max_frequency = (
                ui.definition_widget.burst_max_frequency_selector.value()
            )
            signal_generator_on_percent = ui.definition_widget.burst_on_percentage_selector.value()
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 3:  # Pseudorandom
            signal_generator_type = "pseudorandom"
            signal_generator_level = ui.definition_widget.pseudorandom_rms_selector.value()
            signal_generator_min_frequency = (
                ui.definition_widget.pseudorandom_min_frequency_selector.value()
            )
            signal_generator_max_frequency = (
                ui.definition_widget.pseudorandom_max_frequency_selector.value()
            )
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 4:  # Chirp
            signal_generator_type = "chirp"
            signal_generator_level = ui.definition_widget.chirp_level_selector.value()
            signal_generator_min_frequency = (
                ui.definition_widget.chirp_min_frequency_selector.value()
            )
            signal_generator_max_frequency = (
                ui.definition_widget.chirp_max_frequency_selector.value()
            )
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 5:  # Square
            signal_generator_type = "square"
            signal_generator_level = ui.definition_widget.square_level_selector.value()
            signal_generator_min_frequency = ui.definition_widget.square_frequency_selector.value()
            signal_generator_on_percent = ui.definition_widget.square_percent_on_selector.value()
        elif ui.definition_widget.signal_generator_selector.currentIndex() == 6:  # Sine
            signal_generator_type = "sine"
            signal_generator_level = ui.definition_widget.sine_level_selector.value()
            signal_generator_min_frequency = ui.definition_widget.sine_frequency_selector.value()
        else:
            index = ui.definition_widget.signal_generator_selector.currentIndex()
            raise ValueError(f"Invalid Signal Generator {index} (How did you get here?)")
        return cls(
            ui.definition_widget.sample_rate_display.value(),
            ui.definition_widget.samples_per_frame_selector.value(),
            ui.definition_widget.system_id_averaging_scheme_selector.itemText(
                ui.definition_widget.system_id_averaging_scheme_selector.currentIndex()
            ),
            ui.definition_widget.system_id_frames_to_average_selector.value(),
            ui.definition_widget.system_id_averaging_coefficient_selector.value(),
            ui.definition_widget.system_id_frf_technique_selector.itemText(
                ui.definition_widget.system_id_frf_technique_selector.currentIndex()
            ),
            ui.definition_widget.system_id_transfer_function_computation_window_selector.itemText(
                ui.definition_widget.system_id_transfer_function_computation_window_selector.currentIndex()
            ).lower(),
            ui.definition_widget.system_id_overlap_percentage_selector.value(),
            ui.definition_widget.triggering_type_selector.itemText(
                ui.definition_widget.triggering_type_selector.currentIndex()
            ),
            ui.definition_widget.acceptance_selector.itemText(
                ui.definition_widget.acceptance_selector.currentIndex()
            ),
            ui.definition_widget.wait_for_steady_selector.value(),
            ui.definition_widget.trigger_channel_selector.currentIndex(),
            ui.definition_widget.pretrigger_selector.value(),
            ui.definition_widget.trigger_slope_selector.currentIndex() == 0,
            ui.definition_widget.trigger_level_selector.value(),
            ui.definition_widget.hysteresis_selector.value(),
            ui.definition_widget.hysteresis_length_selector.value(),
            signal_generator_type,
            signal_generator_level,
            signal_generator_min_frequency,
            signal_generator_max_frequency,
            signal_generator_on_percent,
            ui.acceptance_function,
            ui.reference_indices,
            ui.response_indices,
            ui.all_output_channel_indices,
            ui.data_acquisition_parameters,
            ui.definition_widget.window_value_selector.value() / 100,
        )

    def generate_signal(self):
        """Generates a single frame of data"""
        if self.signal_generator is None:
            return np.zeros(
                (
                    len(self.output_channel_indices),
                    self.samples_per_frame * self.output_oversample,
                )
            )
        else:
            return self.signal_generator.generate_frame()[0]


from ..process.data_collector import (  # noqa # pylint: disable=wrong-import-position
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    TriggerSlope,
    Window,
    data_collector_process,
)
from ..process.signal_generation_process import (  # noqa # pylint: disable=wrong-import-position
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from ..process.spectral_processing import (  # noqa # pylint: disable=wrong-import-position
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
    spectral_processing_process,
)


# region: Environment
class ModalEnvironment(AbstractEnvironment):
    """Modal Environment class defining the interface with the controller"""

    def __init__(
        self,
        environment_name: str,
        queues: ModalQueues,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
        super().__init__(
            environment_name,
            queues.environment_command_queue,
            queues.gui_update_queue,
            queues.controller_communication_queue,
            queues.log_file_queue,
            queues.data_in_queue,
            queues.data_out_queue,
            acquisition_active,
            output_active,
        )
        self.queue_container = queues
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.frame_number = 0
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False

        # Map commands
        self.map_command(ModalCommands.ACCEPT_FRAME, self.accept_frame)
        self.map_command(ModalCommands.START_CONTROL, self.start_environment)
        self.map_command(ModalCommands.RUN_CONTROL, self.run_control)
        self.map_command(ModalCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN, self.check_for_shutdown)
        self.map_command(
            SignalGenerationCommands.SHUTDOWN_ACHIEVED, self.siggen_shutdown_achieved_fn
        )
        self.map_command(
            DataCollectorCommands.SHUTDOWN_ACHIEVED, self.collector_shutdown_achieved_fn
        )
        self.map_command(
            SpectralProcessingCommands.SHUTDOWN_ACHIEVED,
            self.spectral_shutdown_achieved_fn,
        )

    def initialize_data_acquisition_parameters(
        self, data_acquisition_parameters: DataAcquisitionParameters
    ):
        """Initialize the data acquisition parameters in the environment.

        The environment will receive the global data acquisition parameters from
        the controller, and must set itself up accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            A container containing data acquisition parameters, including
            channels active in the environment as well as sampling parameters.
        """
        self.data_acquisition_parameters = data_acquisition_parameters

    def initialize_environment_test_parameters(self, environment_parameters: ModalMetadata):
        """
        Initialize the environment parameters specific to this environment

        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : ModalMetadata
            A container containing the parameters defining the environment

        """
        self.environment_parameters = environment_parameters

        # Set up the collector
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.INITIALIZE_COLLECTOR,
                self.get_data_collector_metadata(),
            ),
        )
        # Set up the signal generation
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_signal_generation_metadata(),
            ),
        )
        # Set up the spectral processing
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_spectral_processing_metadata(),
            ),
        )

    def get_data_collector_metadata(self) -> CollectorMetadata:
        """Collects metadata used to define the data collector"""
        num_channels = len(self.data_acquisition_parameters.channel_list)
        reference_channel_indices = self.environment_parameters.reference_channel_indices
        response_channel_indices = self.environment_parameters.response_channel_indices
        if self.environment_parameters.trigger_type == "Free Run":
            acquisition_type = AcquisitionType.FREE_RUN
        elif self.environment_parameters.trigger_type == "First Frame":
            acquisition_type = AcquisitionType.TRIGGER_FIRST_FRAME
        elif self.environment_parameters.trigger_type == "Every Frame":
            acquisition_type = AcquisitionType.TRIGGER_EVERY_FRAME
        else:
            raise ValueError(
                f"Invalid Acquisition Type: {self.environment_parameters.trigger_type}"
            )
        if self.environment_parameters.accept_type == "Accept All":
            acceptance = Acceptance.AUTOMATIC
            acceptance_function = None
        elif self.environment_parameters.accept_type == "Manual":
            acceptance = Acceptance.MANUAL
            acceptance_function = None
        elif self.environment_parameters.accept_type == "Autoreject...":
            acceptance = Acceptance.AUTOMATIC
            acceptance_function = self.environment_parameters.acceptance_function
        else:
            raise ValueError(f"Invalid Acceptance Type: {self.environment_parameters.accept_type}")
        overlap_fraction = self.environment_parameters.overlap
        trigger_channel_index = self.environment_parameters.trigger_channel
        trigger_slope = (
            TriggerSlope.POSITIVE
            if self.environment_parameters.trigger_slope_positive
            else TriggerSlope.NEGATIVE
        )
        (_, trigger_level, _, trigger_hysteresis) = self.environment_parameters.get_trigger_levels(
            self.data_acquisition_parameters.channel_list
        )
        trigger_hysteresis_samples = self.environment_parameters.hysteresis_samples
        pretrigger_fraction = self.environment_parameters.pretrigger
        frame_size = self.environment_parameters.samples_per_frame
        if self.environment_parameters.frf_window == "hann":
            window = Window.HANN
        elif self.environment_parameters.frf_window == "rectangle":
            window = Window.RECTANGLE
        elif self.environment_parameters.frf_window == "exponential":
            window = Window.EXPONENTIAL
        else:
            raise ValueError(f"Invalid Window Type: {self.environment_parameters.frf_window}")
        window_parameter = -(frame_size) / np.log(
            self.environment_parameters.exponential_window_value_at_frame_end
        )
        return CollectorMetadata(
            num_channels,
            response_channel_indices,
            reference_channel_indices,
            acquisition_type,
            acceptance,
            acceptance_function,
            overlap_fraction,
            trigger_channel_index,
            trigger_slope,
            trigger_level,
            trigger_hysteresis,
            trigger_hysteresis_samples,
            pretrigger_fraction,
            frame_size,
            window,
            response_transformation_matrix=None,
            reference_transformation_matrix=None,
            window_parameter_2=window_parameter,
        )

    def get_spectral_processing_metadata(self) -> SpectralProcessingMetadata:
        """Collects metadata to define the spectral processing"""
        averaging_type = (
            AveragingTypes.LINEAR
            if self.environment_parameters.averaging_type == "Linear"
            else AveragingTypes.EXPONENTIAL
        )
        averages = self.environment_parameters.num_averages
        exponential_averaging_coefficient = self.environment_parameters.averaging_coefficient
        if self.environment_parameters.frf_technique == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_parameters.frf_technique == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_parameters.frf_technique == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_parameters.frf_technique == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(
                f"Invalid FRF Estimator {self.environment_parameters.frf_technique}. "
                "How did you get here?"
            )
        num_response_channels = len(self.environment_parameters.response_channel_indices)
        num_reference_channels = len(self.environment_parameters.reference_channel_indices)
        frequency_spacing = self.environment_parameters.frequency_spacing
        sample_rate = self.environment_parameters.sample_rate
        num_frequency_lines = self.environment_parameters.fft_lines
        return SpectralProcessingMetadata(
            averaging_type,
            averages,
            exponential_averaging_coefficient,
            frf_estimator,
            num_response_channels,
            num_reference_channels,
            frequency_spacing,
            sample_rate,
            num_frequency_lines,
            compute_cpsd=False,
            compute_apsd=True,
        )

    def get_signal_generation_metadata(self) -> SignalGenerationMetadata:
        """Collects metadata to define the signal generator"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=1,
            output_transformation_matrix=None,
            disabled_signals=self.environment_parameters.disabled_signals,
        )

    def get_signal_generator(self):
        """Gets the signal generator object used to generate signals for the environment"""
        return self.environment_parameters.get_signal_generator()

    def start_environment(self, data):  # pylint: disable=unused-argument
        """Starts the environment

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        self.log("Starting Modal")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False

        # Set up the collector
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
                self.get_data_collector_metadata(),
            ),
        )

        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.skip_frames, 1),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_signal_generation_metadata(),
            ),
        )

        self.queue_container.signal_generation_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                self.get_signal_generator(),
            ),
        )

        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, 1.0)
        )

        # Tell the collector to start acquiring data
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # Set up the spectral processing
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_spectral_processing_metadata(),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

        self.queue_container.environment_command_queue.put(
            self.environment_name, (ModalCommands.RUN_CONTROL, None)
        )

    def run_control(self, data):  # pylint: disable=unused-argument
        """Runs the environment

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        # Pull data off the spectral queue
        spectral_data = flush_queue(
            self.queue_container.updated_spectral_quantities_queue, timeout=WAIT_TIME
        )
        if len(spectral_data) > 0:
            self.log("Received Data")
            (
                frames,
                frequencies,
                frf,
                coherence,
                response_cpsd,
                reference_cpsd,
                condition,
            ) = spectral_data[-1]
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        ModalUICommands.SPECTRAL_UPDATE,
                        (
                            frames,
                            self.environment_parameters.num_averages,
                            frequencies,
                            frf,
                            coherence,
                            response_cpsd,
                            reference_cpsd,
                            condition,
                        ),
                    ),
                )
            )
        else:
            time.sleep(WAIT_TIME)
        self.queue_container.environment_command_queue.put(
            self.environment_name, (ModalCommands.RUN_CONTROL, None)
        )

    def siggen_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the signal generation shutdown flag to True

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        self.siggen_shutdown_achieved = True

    def collector_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the collector shutdown flag to True

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        self.collector_shutdown_achieved = True

    def spectral_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the spectral processing shutdown flag to True

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        self.spectral_shutdown_achieved = True

    def check_for_shutdown(self, data):  # pylint: disable=unused-argument
        """Checks if all environment subprocesses have shut down successfully.

        Parameters
        ----------
        data : NoneType
            Requred by the message/data data-passing strategy in Rattlesnake, but not needed by
            this method
        """
        if (
            self.siggen_shutdown_achieved
            and self.collector_shutdown_achieved
            and self.spectral_shutdown_achieved
        ):
            self.log("Shutdown Achieved")
            self.gui_update_queue.put((self.environment_name, (ModalUICommands.FINISHED, None)))
        else:
            # Recheck some time later
            time.sleep(1)
            self.environment_command_queue.put(
                self.environment_name, (ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)
            )

    def accept_frame(self, data):
        """Accepts or rejects the previous measurement frame"""
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACCEPT, data)
        )

    def stop_environment(self, data):
        """Stop the environment gracefully

        This function defines the operations to shut down the environment
        gracefully so there is no hard stop that might damage test equipment
        or parts.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.log("Stopping Control")
        flush_queue(self.queue_container.environment_command_queue)
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.SET_TEST_LEVEL, (1000, 1))
        )
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING, None),
        )
        self.queue_container.environment_command_queue.put(
            self.environment_name, (ModalCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)
        )

    def quit(self, data):
        """Returns True to stop the ``run`` while loop and exit the process

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        Returns
        -------
        True :
            This function returns True to signal to the ``run`` while loop
            that it is time to close down the environment.

        """
        for queue in [
            self.queue_container.spectral_command_queue,
            self.queue_container.signal_generation_command_queue,
            self.queue_container.collector_command_queue,
        ]:
            queue.put(self.environment_name, (GlobalCommands.QUIT, None))
        return True


# region: Process
def modal_process(
    environment_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: Queue,
    controller_communication_queue: VerboseMessageQueue,
    log_file_queue: Queue,
    data_in_queue: Queue,
    data_out_queue: Queue,
    acquisition_active: mp.sharedctypes.Synchronized,
    output_active: mp.sharedctypes.Synchronized,
):
    """Modal environment process function called by multiprocessing

    This function defines the Modal Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a ModalEnvironment object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment
    input_queue : VerboseMessageQueue :
        Queue containing instructions for the environment
    gui_update_queue : Queue :
        Queue where GUI updates are put
    controller_communication_queue : Queue :
        Queue for global communications with the controller
    log_file_queue : Queue :
        Queue for writing log file messages
    data_in_queue : Queue :
        Queue from which data will be read by the environment
    data_out_queue : Queue :
        Queue to which data will be written that will be output by the hardware.

    """
    queue_container = ModalQueues(
        environment_name,
        input_queue,
        gui_update_queue,
        controller_communication_queue,
        data_in_queue,
        data_out_queue,
        log_file_queue,
    )

    spectral_proc = mp.Process(
        target=spectral_processing_process,
        args=(
            environment_name,
            queue_container.spectral_command_queue,
            queue_container.data_for_spectral_computation_queue,
            queue_container.updated_spectral_quantities_queue,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.log_file_queue,
        ),
    )
    spectral_proc.start()
    siggen_proc = mp.Process(
        target=signal_generation_process,
        args=(
            environment_name,
            queue_container.signal_generation_command_queue,
            queue_container.signal_generation_update_queue,
            queue_container.data_out_queue,
            queue_container.environment_command_queue,
            queue_container.log_file_queue,
            queue_container.gui_update_queue,
        ),
    )
    siggen_proc.start()
    collection_proc = mp.Process(
        target=data_collector_process,
        args=(
            environment_name,
            queue_container.collector_command_queue,
            queue_container.data_in_queue,
            [queue_container.data_for_spectral_computation_queue],
            queue_container.environment_command_queue,
            queue_container.log_file_queue,
            queue_container.gui_update_queue,
        ),
    )

    collection_proc.start()

    process_class = ModalEnvironment(
        environment_name, queue_container, acquisition_active, output_active
    )
    process_class.run()

    # Rejoin all the processes
    process_class.log("Joining Subprocesses")
    process_class.log("Joining Spectral Computation")
    spectral_proc.join()
    process_class.log("Joining Signal Generation")
    siggen_proc.join()
    process_class.log("Joining Data Collection")
    collection_proc.join()
