# -*- coding: utf-8 -*-
"""
Abstract environment that can be used to create new environment control strategies
in the controller that use system identification.

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
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from multiprocessing.queues import Queue
import netCDF4 as nc4
import numpy as np

from rattlesnake.process.abstract_sysid_data_analysis import SysIDDataAnalysisCommands
from rattlesnake.environment.abstract_environment import AbstractEnvironment, AbstractMetadata
from rattlesnake.process.data_collector import (
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    TriggerSlope,
    Window,
)
from rattlesnake.process.signal_generation import (
    BurstRandomSignalGenerator,
    ChirpSignalGenerator,
    PseudorandomSignalGenerator,
    RandomSignalGenerator,
    SignalGenerator,
)
from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
)
from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
)
from rattlesnake.utilities import DataAcquisitionParameters, GlobalCommands, VerboseMessageQueue


class SystemIdCommands(Enum):
    """Enumeration of commands that could be sent to the system identification environment"""

    PREVIEW_NOISE = 0
    PREVIEW_TRANSFER_FUNCTION = 1
    START_SYSTEM_ID = 2
    STOP_SYSTEM_ID = 3
    CHECK_FOR_COMPLETE_SHUTDOWN = 4


class SystemIdUICommands(Enum):
    ENABLE_SYSTEM_ID = 0
    DISABLE_SYSTEM_ID = 1


# region: Metadata
class AbstractSysIdMetadata(AbstractMetadata):
    """Abstract class for storing metadata for an environment.

    This class is used as a storage container for parameters used by an
    environment.  It is returned by the environment UI's
    ``collect_environment_definition_parameters`` function as well as its
    ``initialize_environment`` function.  Various parts of the controller and
    environment will query the class's data members for parameter values.

    Classes inheriting from AbstractMetadata must define:
      1. store_to_netcdf - A function defining the way the parameters are
         stored to a netCDF file saved during streaming operations.
    """

    def __init__(self):
        self.sysid_frame_size = None
        self.sysid_averaging_type = None
        self.sysid_noise_averages = None
        self.sysid_averages = None
        self.sysid_exponential_averaging_coefficient = None
        self.sysid_estimator = None
        self.sysid_level = None
        self.sysid_level_ramp_time = None
        self.sysid_signal_type = None
        self.sysid_window = None
        self.sysid_overlap = None
        self.sysid_burst_on = None
        self.sysid_pretrigger = None
        self.sysid_burst_ramp_fraction = None
        self.sysid_low_frequency_cutoff = None
        self.sysid_high_frequency_cutoff = None

    @property
    @abstractmethod
    def number_of_channels(self):
        """Number of channels in the environment"""

    @property
    @abstractmethod
    def response_channel_indices(self):
        """Indices corresponding to the response or control channels in the environment"""

    @property
    @abstractmethod
    def reference_channel_indices(self):
        """Indices corresponding to the excitation channels in the environment"""

    @property
    def num_response_channels(self):
        """Gets the total number of control channels including transformation effects"""
        return (
            len(self.response_channel_indices)
            if self.response_transformation_matrix is None
            else self.response_transformation_matrix.shape[0]
        )

    @property
    def num_reference_channels(self):
        """Gets the total number of excitation channels including transformation effects"""
        return (
            len(self.reference_channel_indices)
            if self.reference_transformation_matrix is None
            else self.reference_transformation_matrix.shape[0]
        )

    @property
    @abstractmethod
    def response_transformation_matrix(self):
        """Gets the response transformation matrix"""

    @property
    @abstractmethod
    def reference_transformation_matrix(self):
        """Gets the excitation transformation matrix"""

    @property
    def sysid_frequency_spacing(self):
        """Frequency spacing in spectral quantities computed by system identification"""
        return self.sample_rate / self.sysid_frame_size

    @property
    @abstractmethod
    def sample_rate(self):
        """Sample rate (not oversampled) of the data acquisition system"""

    @property
    def sysid_fft_lines(self):
        """Number of frequency lines in the FFT"""
        return self.sysid_frame_size // 2 + 1

    @property
    def sysid_skip_frames(self):
        """Number of frames to skip in the time stream due to ramp time"""
        return int(
            np.ceil(
                self.sysid_level_ramp_time
                * self.sample_rate
                / (self.sysid_frame_size * (1 - self.sysid_overlap))
            )
        )

    @abstractmethod
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
        netcdf_group_handle.sysid_frame_size = self.sysid_frame_size
        netcdf_group_handle.sysid_averaging_type = self.sysid_averaging_type
        netcdf_group_handle.sysid_noise_averages = self.sysid_noise_averages
        netcdf_group_handle.sysid_averages = self.sysid_averages
        netcdf_group_handle.sysid_exponential_averaging_coefficient = (
            self.sysid_exponential_averaging_coefficient
        )
        netcdf_group_handle.sysid_estimator = self.sysid_estimator
        netcdf_group_handle.sysid_level = self.sysid_level
        netcdf_group_handle.sysid_level_ramp_time = self.sysid_level_ramp_time
        netcdf_group_handle.sysid_signal_type = self.sysid_signal_type
        netcdf_group_handle.sysid_window = self.sysid_window
        netcdf_group_handle.sysid_overlap = self.sysid_overlap
        netcdf_group_handle.sysid_burst_on = self.sysid_burst_on
        netcdf_group_handle.sysid_pretrigger = self.sysid_pretrigger
        netcdf_group_handle.sysid_burst_ramp_fraction = self.sysid_burst_ramp_fraction
        netcdf_group_handle.sysid_low_frequency_cutoff = self.sysid_low_frequency_cutoff
        netcdf_group_handle.sysid_high_frequency_cutoff = self.sysid_high_frequency_cutoff

    def __eq__(self, other):
        try:
            return np.all(
                [np.all(value == other.__dict__[field]) for field, value in self.__dict__.items()]
            )
        except (AttributeError, KeyError):
            return False


# region: Environment
class AbstractSysIdEnvironment(AbstractEnvironment):
    """Abstract Environment class defining the interface with the controller

    This class is used to define the operation of an environment within the
    controller, which must be completed by subclasses inheriting from this
    class.  Children of this class will sit in a While loop in the
    ``AbstractEnvironment.run()`` function.  While in this loop, the
    Environment will pull instructions and data from the
    ``command_queue`` and then use the ``command_map`` to map those instructions
    to functions in the class.

    All child classes inheriting from AbstractEnvironment will require functions
    to be defined for global operations of the controller, which are already
    mapped in the ``command_map``.  Any additional operations must be defined
    by functions and then added to the command_map when initilizing the child
    class.

    All functions called via the ``command_map`` must accept one input argument
    which is the data passed along with the command.  For functions that do not
    require additional data, this argument can be ignored, but it must still be
    present in the function's calling signature.

    The run function will continue until one of the functions called by
    ``command_map`` returns a truthy value, which signifies the controller to
    quit.  Therefore, any functions mapped to ``command_map`` that should not
    instruct the program to quit should not return any value that could be
    interpreted as true."""

    def __init__(
        self,
        environment_name: str,
        command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
        collector_command_queue: VerboseMessageQueue,
        signal_generator_command_queue: VerboseMessageQueue,
        spectral_processing_command_queue: VerboseMessageQueue,
        data_analysis_command_queue: VerboseMessageQueue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        acquisition_active: mp.sharedctypes.Synchronized,
        output_active: mp.sharedctypes.Synchronized,
    ):
        super().__init__(
            environment_name,
            command_queue,
            gui_update_queue,
            controller_communication_queue,
            log_file_queue,
            data_in_queue,
            data_out_queue,
            acquisition_active,
            output_active,
        )
        self.map_command(SystemIdCommands.PREVIEW_NOISE, self.preview_noise)
        self.map_command(SystemIdCommands.PREVIEW_TRANSFER_FUNCTION, self.preview_transfer_function)
        self.map_command(SystemIdCommands.START_SYSTEM_ID, self.start_noise)
        self.map_command(SystemIdCommands.STOP_SYSTEM_ID, self.stop_system_id)
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
        self.map_command(
            SysIDDataAnalysisCommands.SHUTDOWN_ACHIEVED,
            self.analysis_shutdown_achieved_fn,
        )
        self.map_command(SysIDDataAnalysisCommands.START_SHUTDOWN, self.stop_system_id)
        self.map_command(
            SysIDDataAnalysisCommands.START_SHUTDOWN_AND_RUN_SYSID,
            self.start_shutdown_and_run_sysid,
        )
        self.map_command(SysIDDataAnalysisCommands.SYSTEM_ID_COMPLETE, self.system_id_complete)
        self.map_command(SysIDDataAnalysisCommands.LOAD_NOISE, self.load_noise)
        self.map_command(
            SysIDDataAnalysisCommands.LOAD_TRANSFER_FUNCTION,
            self.load_transfer_function,
        )
        self.map_command(
            SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, self.check_for_sysid_shutdown
        )
        self._waiting_to_start_transfer_function = False
        self.collector_command_queue = collector_command_queue
        self.signal_generator_command_queue = signal_generator_command_queue
        self.spectral_processing_command_queue = spectral_processing_command_queue
        self.data_analysis_command_queue = data_analysis_command_queue
        self.data_acquisition_parameters = None
        self.environment_parameters = None
        self.collector_shutdown_achieved = True
        self.spectral_shutdown_achieved = True
        self.siggen_shutdown_achieved = True
        self.analysis_shutdown_achieved = True
        self._sysid_stream_name = None

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

    def initialize_environment_test_parameters(self, environment_parameters: AbstractSysIdMetadata):
        """
        Initialize the environment parameters specific to this environment

        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_parameters : AbstractMetadata
            A container containing the parameters defining the environment

        """
        self.environment_parameters = environment_parameters

    def get_sysid_data_collector_metadata(self) -> CollectorMetadata:
        """Collects metadata to send to the data collector"""
        num_channels = self.environment_parameters.number_of_channels
        response_channel_indices = self.environment_parameters.response_channel_indices
        reference_channel_indices = self.environment_parameters.reference_channel_indices
        if self.environment_parameters.sysid_signal_type in [
            "Random",
            "Pseudorandom",
            "Chirp",
        ]:
            acquisition_type = AcquisitionType.FREE_RUN
        else:
            acquisition_type = AcquisitionType.TRIGGER_FIRST_FRAME
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        if self.environment_parameters.sysid_signal_type == "Random":
            overlap_fraction = self.environment_parameters.sysid_overlap
        else:
            overlap_fraction = 0
        if self.environment_parameters.sysid_signal_type == "Burst Random":
            trigger_channel_index = reference_channel_indices[0]
        else:
            trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = self.environment_parameters.sysid_level / 100
        trigger_hysteresis = self.environment_parameters.sysid_level / 200
        trigger_hysteresis_samples = (
            (1 - self.environment_parameters.sysid_burst_on)
            * self.environment_parameters.sysid_frame_size
        ) // 2
        pretrigger_fraction = self.environment_parameters.sysid_pretrigger
        frame_size = self.environment_parameters.sysid_frame_size
        window = (
            Window.HANN if self.environment_parameters.sysid_window == "Hann" else Window.RECTANGLE
        )
        kurtosis_buffer_length = self.environment_parameters.sysid_averages

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
            kurtosis_buffer_length=kurtosis_buffer_length,
            response_transformation_matrix=self.environment_parameters.response_transformation_matrix,
            reference_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def get_sysid_spectral_processing_metadata(self, is_noise=False) -> SpectralProcessingMetadata:
        """Collects metadata to send to the spectral processing process"""
        averaging_type = (
            AveragingTypes.LINEAR
            if self.environment_parameters.sysid_averaging_type == "Linear"
            else AveragingTypes.EXPONENTIAL
        )
        averages = (
            self.environment_parameters.sysid_noise_averages
            if is_noise
            else self.environment_parameters.sysid_averages
        )
        exponential_averaging_coefficient = (
            self.environment_parameters.sysid_exponential_averaging_coefficient
        )
        if self.environment_parameters.sysid_estimator == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_parameters.sysid_estimator == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_parameters.sysid_estimator == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_parameters.sysid_estimator == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(f"Invalid FRF Estimator {self.environment_parameters.sysid_estimator}")
        num_response_channels = self.environment_parameters.num_response_channels
        num_reference_channels = self.environment_parameters.num_reference_channels
        frequency_spacing = self.environment_parameters.sysid_frequency_spacing
        sample_rate = self.environment_parameters.sample_rate
        num_frequency_lines = self.environment_parameters.sysid_fft_lines
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
        )

    def get_sysid_signal_generation_metadata(self) -> SignalGenerationMetadata:
        """Collects metadata to send to the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_parameters.sysid_level_ramp_time
            * self.environment_parameters.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_parameters.reference_transformation_matrix,
        )

    def get_sysid_signal_generator(self) -> SignalGenerator:
        """Creates a signal generator object that will generate the signals"""
        if self.environment_parameters.sysid_signal_type == "Random":
            return RandomSignalGenerator(
                rms=self.environment_parameters.sysid_level,
                sample_rate=self.environment_parameters.sample_rate,
                num_samples_per_frame=self.environment_parameters.sysid_frame_size,
                num_signals=self.environment_parameters.num_reference_channels,
                low_frequency_cutoff=self.environment_parameters.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_parameters.sysid_high_frequency_cutoff,
                cola_overlap=0.5,
                cola_window="hann",
                cola_exponent=0.5,
                output_oversample=self.data_acquisition_parameters.output_oversample,
            )
        elif self.environment_parameters.sysid_signal_type == "Pseudorandom":
            return PseudorandomSignalGenerator(
                rms=self.environment_parameters.sysid_level,
                sample_rate=self.environment_parameters.sample_rate,
                num_samples_per_frame=self.environment_parameters.sysid_frame_size,
                num_signals=self.environment_parameters.num_reference_channels,
                low_frequency_cutoff=self.environment_parameters.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_parameters.sysid_high_frequency_cutoff,
                output_oversample=self.data_acquisition_parameters.output_oversample,
            )
        elif self.environment_parameters.sysid_signal_type == "Burst Random":
            return BurstRandomSignalGenerator(
                rms=self.environment_parameters.sysid_level,
                sample_rate=self.environment_parameters.sample_rate,
                num_samples_per_frame=self.environment_parameters.sysid_frame_size,
                num_signals=self.environment_parameters.num_reference_channels,
                low_frequency_cutoff=self.environment_parameters.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_parameters.sysid_high_frequency_cutoff,
                on_fraction=self.environment_parameters.sysid_burst_on,
                ramp_fraction=self.environment_parameters.sysid_burst_ramp_fraction,
                output_oversample=self.data_acquisition_parameters.output_oversample,
            )
        elif self.environment_parameters.sysid_signal_type == "Chirp":
            return ChirpSignalGenerator(
                level=self.environment_parameters.sysid_level,
                sample_rate=self.environment_parameters.sample_rate,
                num_samples_per_frame=self.environment_parameters.sysid_frame_size,
                num_signals=self.environment_parameters.num_reference_channels,
                low_frequency_cutoff=np.max(
                    [
                        self.environment_parameters.sysid_frequency_spacing,
                        self.environment_parameters.sysid_low_frequency_cutoff,
                    ]
                ),
                high_frequency_cutoff=np.min(
                    [
                        self.environment_parameters.sample_rate / 2,
                        self.environment_parameters.sysid_high_frequency_cutoff,
                    ]
                ),
                output_oversample=self.data_acquisition_parameters.output_oversample,
            )

    def load_noise(self, data):
        """Sends noise data to the data analysis process"""
        self.data_analysis_command_queue.put(
            self.environment_name, (SysIDDataAnalysisCommands.LOAD_NOISE, data)
        )

    def load_transfer_function(self, data):
        """Sends transfer function data to the data analysis process"""
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.LOAD_TRANSFER_FUNCTION, data),
        )

    def preview_noise(self, data):
        """Starts up the noise preview with the defined metadata"""
        self.log("Starting Noise Preview")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        self.controller_communication_queue.put(
            self.environment_name, (GlobalCommands.RUN_HARDWARE, None)
        )
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )

        # Set up the collector
        collector_metadata = deepcopy(self.get_sysid_data_collector_metadata())
        collector_metadata.acquisition_type = AcquisitionType.FREE_RUN
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR, collector_metadata),
        )

        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.sysid_skip_frames, 1),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_signal_generation_metadata(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                self.get_sysid_signal_generator(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_parameters,
            ),
        )

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name, (SysIDDataAnalysisCommands.RUN_NOISE, False)
        )

        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_spectral_processing_metadata(is_noise=True),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

        # Tell data collector to clear the kurtosis buffer
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.CLEAR_KURTOSIS_BUFFER, None)
        )

    def preview_transfer_function(self, data):
        """Starts up a transfer function preview with the provided environment metadata"""
        self.log("Starting System ID Preview")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        self.controller_communication_queue.put(
            self.environment_name, (GlobalCommands.RUN_HARDWARE, None)
        )
        # Wait for the environment to start up
        while not (self.acquisition_active and self.output_active):
            # print('Waiting for Acquisition and Output to Start up')
            time.sleep(0.1)
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )

        # Set up the collector
        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
                self.get_sysid_data_collector_metadata(),
            ),
        )

        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.sysid_skip_frames, 1),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_signal_generation_metadata(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                self.get_sysid_signal_generator(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, 1.0)
        )

        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_parameters,
            ),
        )

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION, False),
        )

        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_spectral_processing_metadata(is_noise=False),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

        # Tell data collector to clear the kurtosis buffer
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.CLEAR_KURTOSIS_BUFFER, None)
        )

    def start_noise(self, data):
        """Starts the noise measurement with the provided metadata"""
        self.log("Starting Noise Measurement for System ID")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters, self._sysid_stream_name = data
        self.controller_communication_queue.put(
            self.environment_name,
            (
                GlobalCommands.UPDATE_METADATA,
                (self.environment_name, self.environment_parameters),
            ),
        )
        # Start up controller
        if self._sysid_stream_name is not None:
            self.controller_communication_queue.put(
                self.environment_name,
                (GlobalCommands.INITIALIZE_STREAMING, self._sysid_stream_name),
            )
            self.controller_communication_queue.put(
                self.environment_name, (GlobalCommands.START_STREAMING, None)
            )
        self.controller_communication_queue.put(
            self.environment_name, (GlobalCommands.RUN_HARDWARE, None)
        )
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )

        # Set up the collector
        collector_metadata = deepcopy(self.get_sysid_data_collector_metadata())
        collector_metadata.acquisition_type = AcquisitionType.FREE_RUN
        self.collector_command_queue.put(
            self.environment_name,
            (DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR, collector_metadata),
        )

        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.sysid_skip_frames, 1),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_signal_generation_metadata(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                self.get_sysid_signal_generator(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_parameters,
            ),
        )

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name, (SysIDDataAnalysisCommands.RUN_NOISE, True)
        )

        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_spectral_processing_metadata(is_noise=True),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

        # Tell data collector to clear the kurtosis buffer
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.CLEAR_KURTOSIS_BUFFER, None)
        )

    def start_transfer_function(self, data):
        """Starts the transfer function measurement with the provided metadata"""
        self.log("Starting Transfer Function for System ID")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.environment_parameters = data
        # Start up controller
        if self._sysid_stream_name is not None:
            self.controller_communication_queue.put(
                self.environment_name, (GlobalCommands.START_STREAMING, None)
            )

        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )

        # Set up the collector
        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.FORCE_INITIALIZE_COLLECTOR,
                self.get_sysid_data_collector_metadata(),
            ),
        )

        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.sysid_skip_frames, 1),
            ),
        )
        time.sleep(0.01)

        # Set up the signal generation
        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_signal_generation_metadata(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name,
            (
                SignalGenerationCommands.INITIALIZE_SIGNAL_GENERATOR,
                self.get_sysid_signal_generator(),
            ),
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.MUTE, None)
        )

        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, 1.0)
        )

        # Tell the collector to start acquiring data
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_parameters,
            ),
        )

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION, True),
        )

        # Set up the spectral processing
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (
                SpectralProcessingCommands.INITIALIZE_PARAMETERS,
                self.get_sysid_spectral_processing_metadata(is_noise=False),
            ),
        )

        # Tell the spectral analysis to clear and start acquiring
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING, None),
        )

        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING, None),
        )

        # Tell data collector to clear the kurtosis buffer
        self.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.CLEAR_KURTOSIS_BUFFER, None)
        )

    def stop_system_id(self, stop_tasks):
        """Starts the shutdown process for the system identification"""
        stop_data_analysis, stop_hardware = stop_tasks
        self.log("Stop Transfer Function")
        if stop_hardware:
            self.controller_communication_queue.put(
                self.environment_name, (GlobalCommands.STOP_HARDWARE, None)
            )
        elif self._sysid_stream_name is not None:
            self.controller_communication_queue.put(
                self.environment_name, (GlobalCommands.STOP_STREAMING, None)
            )
        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.sysid_skip_frames * 10, 1),
            ),
        )
        self.signal_generator_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )
        self.spectral_processing_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING, None),
        )
        if stop_data_analysis:
            self.data_analysis_command_queue.put(
                self.environment_name, (SysIDDataAnalysisCommands.STOP_SYSTEM_ID, None)
            )
        self.environment_command_queue.put(
            self.environment_name, (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)
        )

    def siggen_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the sshutdown flag to denote the signal generation has shut down successfully"""
        self.siggen_shutdown_achieved = True

    def collector_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the shutdown flag to denote the data collector has shut down successfully"""
        self.collector_shutdown_achieved = True

    def spectral_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the shutdown flag to denote the spectral computation has shut down successfully"""
        self.spectral_shutdown_achieved = True

    def analysis_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the shutdown flag to denote the data analysis has shut down successfully"""
        self.analysis_shutdown_achieved = True

    def check_for_sysid_shutdown(self, data):  # pylint: disable=unused-argument
        """Checks that all of the relevant system identification processes have shut down"""
        if (
            self.siggen_shutdown_achieved
            and self.collector_shutdown_achieved
            and self.spectral_shutdown_achieved
            and self.analysis_shutdown_achieved
            and ((not self.acquisition_active) or self._waiting_to_start_transfer_function)
            and ((not self.output_active) or self._waiting_to_start_transfer_function)
        ):
            self.log("Shutdown Achieved")
            if self._waiting_to_start_transfer_function:
                self.start_transfer_function(self.environment_parameters)
            else:
                self.gui_update_queue.put(
                    (self.environment_name, (SystemIdUICommands.ENABLE_SYSTEM_ID, None))
                )
                self._sysid_stream_name = None
            self._waiting_to_start_transfer_function = False
        else:
            # Recheck some time later
            time.sleep(1)
            waiting_for = []
            if not self.siggen_shutdown_achieved:
                waiting_for.append("Signal Generation")
            if not self.collector_shutdown_achieved:
                waiting_for.append("Collector")
            if not self.spectral_shutdown_achieved:
                waiting_for.append("Spectral Processing")
            if not self.analysis_shutdown_achieved:
                waiting_for.append("Data Analysis")
            if self.output_active and (not self._waiting_to_start_transfer_function):
                waiting_for.append("Output Shutdown")
            if self.acquisition_active and (not self._waiting_to_start_transfer_function):
                waiting_for.append("Acquisition Shutdown")
            self.log(f"Waiting for {' and '.join(waiting_for)}")
            self.environment_command_queue.put(
                self.environment_name,
                (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
            )

    def start_shutdown_and_run_sysid(self, data):  # pylint: disable=unused-argument
        """After successful noise run, shut down and start up system identification"""
        self.log("Shutting down and then Running System ID Afterwards")
        self._waiting_to_start_transfer_function = True
        self.stop_system_id((False, False))

    def system_id_complete(self, data):
        """Sends a message to the controller that this environment has completed system id"""
        self.log("Finished System Identification")
        self.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.COMPLETED_SYSTEM_ID, (self.environment_name, data)),
        )

    @abstractmethod
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

    def quit(self, data):
        """Closes down the environment when quitting the software"""
        for queue in [
            self.collector_command_queue,
            self.signal_generator_command_queue,
            self.spectral_processing_command_queue,
            self.data_analysis_command_queue,
        ]:
            queue.put(self.environment_name, (GlobalCommands.QUIT, None))
        # Return true to stop the task
        return True
