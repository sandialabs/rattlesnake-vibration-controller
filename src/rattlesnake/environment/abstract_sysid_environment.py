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
import time
import os
from pathlib import Path
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from multiprocessing.queues import Queue
from typing import List

import netCDF4 as nc4
import numpy as np
import openpyxl
from scipy.io import savemat

from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
    read_transformation_matrix_from_worksheet,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.abstract_environment import (
    EnvironmentMetadata,
    Environment,
)
from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdDataAnalysisCommands,
    SysIdDataAnalysisUICommands,
    SysIdMetadata,
    SysIdDataPackage,
)
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
from rattlesnake.user_interface.ui_utilities import UICommands


# region Commands
class SystemIdCommands(Enum):
    """Enumeration of commands that could be sent to the system identification environment"""

    CHECK_FOR_COMPLETE_SHUTDOWN = 0


class SysIdUICommands(Enum):
    SYSID_STARTED = 0
    SYSID_ENDED = 1

    @property
    def label(self):
        """Used by UI as names for"""
        return self.name.replace("_", " ").title()


# endregion


class SysIdEnvironmentMetadata(EnvironmentMetadata):
    # region Metadata
    def __init__(
        self,
        environment_type,
        environment_name,
        channel_list_bools,
        sample_rate,
        sysid_metadata=None,
    ):
        super().__init__(
            environment_type,
            environment_name,
            channel_list_bools,
            sample_rate,
        )
        # I default initialize this because a lot of sysid environments use it to
        # check the validity of the control class during initialize_environment.
        # (ex. SineEnvironment needs self.environment_metadata.sysid_metadata.sysid_frequency_spacing)
        # It is always overwritten with initialize_sysid after sysid
        # is made so it is never used for actual calculations.
        if isinstance(sysid_metadata, SysIdMetadata):
            self.sysid_metadata = sysid_metadata
        else:
            self.sysid_metadata = SysIdMetadata.default_metadata(sample_rate)

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

    # endregion

    # region Validation
    @abstractmethod
    def validate(self, hardware_metadata):
        return super().validate(hardware_metadata)

    def __eq__(self, other):
        try:
            return np.all(
                [
                    np.all(value == other.__dict__[field])
                    for field, value in self.__dict__.items()
                ]
            )
        except (AttributeError, KeyError):
            return False

    # endregion

    # region Loading
    @abstractmethod
    def save_metadata_to_netcdf(
        self, netcdf_group_handle: nc4._netCDF4.Group
    ):  # pylint: disable=c-extension-no-member
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
        self.sysid_metadata.save_metadata_to_netcdf(netcdf_group_handle)

    @classmethod
    @abstractmethod
    def load_metadata_from_netcdf(
        cls,
        netcdf_group_handle: nc4._netCDF4.Group,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):  # pylint: disable=c-extension-no-member
        """Collects environment parameters from a netCDF dataset.

        This function retrieves parameters from a netCDF dataset that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_netcdf
        function in the AbstractMetadata class, which will write parameters to
        the netCDF file to document the metadata.

        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a Group
        in the dataset sharing the environment's name, e.g.

        ``group = netcdf_handle.groups[self.environment_name]``
        ``self.definition_widget.parameter_selector.setValue(group.parameter)``

        Parameters
        ----------
        netcdf_handle : nc4._netCDF4.Dataset :
            The netCDF dataset from which the data will be read.  It should have
            a group name with the enviroment's name.
        """
        return SysIdMetadata.load_metadata_from_netcdf(
            netcdf_group_handle, hardware_metadata
        )

    @classmethod
    @abstractmethod
    def create_blank_worksheet_template(cls, worksheet):
        """
        Creates blank worksheet template for an excel worksheet
        """
        super().create_blank_worksheet_template(worksheet)

    @abstractmethod
    def save_metadata_to_worksheet(
        self, worksheet: openpyxl.worksheet.worksheet.Worksheet
    ):
        """
        Store parameters to a worksheet in an netCDF streaming file.

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
        super().save_metadata_to_worksheet(worksheet)

    @classmethod
    @abstractmethod
    def load_metadata_from_worksheet(
        cls,
        worksheet: openpyxl.worksheet.worksheet.Worksheet,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):  # pylint: disable=c-extension-no-member
        """Collects environment parameters from an Excel worksheet.

        This function retrieves parameters from an Excel worksheet that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_worksheet
        function in the AbstractMetadata class, which will write parameters to
        the netCDF file to document the metadata.

        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a worksheet
        in the worksheet sharing the environment's name, e.g.

        """
        super().load_metadata_from_worksheet(
            worksheet, environment_name, channel_list_bools, hardware_metadata
        )

    @classmethod
    def save_sysid_matrix_to_worksheet(
        cls,
        worksheet,
        response_matrix,
        output_matrix,
        start_row,
    ):
        response_row = start_row
        output_row = start_row + 1
        if response_matrix is not None:
            worksheet.cell(start_row + 1, 1).value = None
            worksheet.cell(start_row + 1, 2).value = None
            for i, row in enumerate(response_matrix):
                for j, value in enumerate(row):
                    worksheet.cell(i + response_row, j + 2, value)
            # Shift output transfomation matrix down
            output_row = response_row + i + 1
            worksheet.cell(output_row, 1, "Output Transformation Matrix:")
            worksheet.cell(
                output_row,
                2,
                "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
                "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should be "
                "the number of physical output channels in the environment.",
            )
        else:
            worksheet.cell(
                response_row,
                2,
                "None",
            )
        if output_matrix is not None:
            for i, row in enumerate(output_matrix):
                for j, value in enumerate(row):
                    worksheet.cell(i + output_row, j + 2, value)
        else:
            worksheet.cell(output_row, 2, "None")

    @classmethod
    def load_sysid_matrix_from_worksheet(cls, worksheet, start_row):
        start_response_row = start_row
        num_response_row = 1
        if (
            isinstance(worksheet.cell(start_response_row, 2).value, str)
            and worksheet.cell(start_response_row, 2).value.lower() == "none"
        ):
            response_transformation_matrix = None
        elif isinstance(
            worksheet.cell(start_response_row, 2).value, str
        ) and worksheet.cell(start_response_row, 2).value.lower().startswith(
            "# transformation matrix"
        ):
            response_transformation_matrix = None
        else:

            while True:
                first_col_value = worksheet.cell(
                    start_response_row + num_response_row, 2
                ).value
                if worksheet.cell(
                    start_response_row + num_response_row, 1
                ).value == "Output Transformation Matrix:" or (
                    first_col_value is None
                    or (
                        (
                            isinstance(first_col_value, str)
                            and (
                                first_col_value.startswith("#")
                                or first_col_value.strip() == ""
                            )
                        )
                    )
                ):
                    break
                num_response_row += 1
            response_transformation_matrix = read_transformation_matrix_from_worksheet(
                worksheet,
                start_row=start_response_row,
                num_rows=num_response_row,
                start_col=2,
            )
        # Output transformation matrix
        start_output_row = start_response_row + num_response_row
        num_output_row = 1
        if (
            isinstance(worksheet.cell(start_output_row, 2).value, str)
            and worksheet.cell(start_output_row, 2).value.lower() == "none"
        ):
            output_transformation_matrix = None
        elif isinstance(
            worksheet.cell(start_output_row, 2).value, str
        ) and worksheet.cell(start_output_row, 2).value.lower().startswith(
            "# transformation matrix"
        ):
            output_transformation_matrix = None
        else:
            while True:
                first_col_value = worksheet.cell(
                    start_output_row + num_output_row, 2
                ).value
                if first_col_value is None or (
                    isinstance(first_col_value, str) and first_col_value.strip() == ""
                ):
                    break
                num_output_row += 1
            output_transformation_matrix = read_transformation_matrix_from_worksheet(
                worksheet,
                start_row=start_output_row,
                num_rows=num_output_row,
                start_col=2,
            )

        return response_transformation_matrix, output_transformation_matrix

    # endregion


class SysIdEnvironment(Environment):
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

    # region Environment
    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        command_queue: VerboseMessageQueue,
        gui_update_queue: Queue,
        controller_command_queue: VerboseMessageQueue,
        log_file_queue: Queue,
        collector_command_queue: VerboseMessageQueue,
        signal_generator_command_queue: VerboseMessageQueue,
        spectral_processing_command_queue: VerboseMessageQueue,
        data_analysis_command_queue: VerboseMessageQueue,
        data_in_queue: Queue,
        data_out_queue: Queue,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        sysid_active_event: mp.synchronize.Event,
        sysid_stored_event: mp.synchronize.Event,
    ):
        super().__init__(
            environment_name,
            queue_name,
            command_queue,
            gui_update_queue,
            controller_command_queue,
            log_file_queue,
            data_in_queue,
            data_out_queue,
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
        )
        self._sysid_active_event = sysid_active_event
        self._sysid_stored_event = sysid_stored_event
        self.map_command(GlobalCommands.INITIALIZE_SYSTEM_ID, self.initialize_sysid)
        self.map_command(GlobalCommands.START_SYSTEM_ID_NOISE, self.start_noise)
        self.map_command(
            GlobalCommands.START_SYSTEM_ID_TRANSFER, self.start_transfer_function
        )
        self.map_command(GlobalCommands.STOP_SYSTEM_ID, self.stop_system_id)
        self.map_command(GlobalCommands.SAVE_SYSTEM_ID, self.save_system_id_to_file)
        self.map_command(
            GlobalCommands.LOAD_SYSTEM_ID, self.load_system_id_from_package
        )
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
            SysIdDataAnalysisCommands.SHUTDOWN_ACHIEVED,
            self.analysis_shutdown_achieved_fn,
        )
        self.map_command(SysIdDataAnalysisCommands.START_SHUTDOWN, self.stop_system_id)
        self.map_command(
            SysIdDataAnalysisCommands.SYSTEM_ID_NOISE_COMPLETE,
            self.system_id_noise_complete,
        )
        self.map_command(
            SysIdDataAnalysisCommands.SYSTEM_ID_COMPLETE, self.system_id_complete
        )
        self.map_command(SysIdDataAnalysisCommands.LOAD_NOISE, self.load_noise)
        self.map_command(
            SysIdDataAnalysisCommands.LOAD_TRANSFER_FUNCTION,
            self.load_transfer_function,
        )
        self.map_command(
            SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, self.check_for_sysid_shutdown
        )
        self.collector_command_queue = collector_command_queue
        self.signal_generator_command_queue = signal_generator_command_queue
        self.spectral_processing_command_queue = spectral_processing_command_queue
        self.data_analysis_command_queue = data_analysis_command_queue
        self.hardware_metadata = None
        self.environment_metadata = None
        self.sysid_data = SysIdDataPackage()
        self.collector_shutdown_achieved = True
        self.spectral_shutdown_achieved = True
        self.siggen_shutdown_achieved = True
        self.analysis_shutdown_achieved = True

    # endregion

    # region Events
    @property
    def sysid_active(self):
        return self._sysid_active_event.is_set()

    def set_sysid_active(self):
        self._sysid_active_event.set()

    def clear_sysid_active(self):
        self._sysid_active_event.clear()

    @property
    def sysid_stored(self):
        return self._sysid_stored_event.is_set()

    def set_sysid_stored(self):
        self._sysid_stored_event.set()

    def clear_sysid_stored(self):
        self._sysid_stored_event.clear()

    # endregion

    # region State Sync
    @abstractmethod
    def initialize_hardware(self, hardware_metadata: HardwareMetadata):
        """Initialize the data acquisition parameters in the environment.

        The environment will receive the global data acquisition parameters from
        the controller, and must set itself up accordingly.

        Parameters
        ----------
        hardware_metadata : DataAcquisitionParameters :
            A container containing data acquisition parameters, including
            channels active in the environment as well as sampling parameters.
        """
        self.hardware_metadata = hardware_metadata
        self.set_ready()

    @abstractmethod
    def initialize_environment(self, environment_metadata: SysIdEnvironmentMetadata):
        """
        Initialize the environment parameters specific to this environment

        The environment will recieve parameters defining itself from the
        user interface and must set itself up accordingly.

        Parameters
        ----------
        environment_metadata : AbstractMetadata
            A container containing the parameters defining the environment

        """
        self.data_analysis_command_queue.put(
            self.environment_name,
            (GlobalCommands.INITIALIZE_ENVIRONMENT, self.environment_name),
        )
        super().initialize_environment(environment_metadata)

    @abstractmethod
    def initialize_sysid(self, sysid_metadata: SysIdMetadata):
        self.environment_metadata.sysid_metadata = sysid_metadata

        # Set up the data analysis
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIdDataAnalysisCommands.INITIALIZE_PARAMETERS,
                sysid_metadata,
            ),
        )
        # self.set_ready() # Call this at the end of your function

    def get_sysid_data_collector_metadata(self) -> CollectorMetadata:
        """Collects metadata to send to the data collector"""
        num_channels = self.environment_metadata.number_of_channels
        response_channel_indices = self.environment_metadata.response_channel_indices
        reference_channel_indices = self.environment_metadata.reference_channel_indices
        if self.environment_metadata.sysid_metadata.sysid_signal_type in [
            "Random",
            "Pseudorandom",
            "Chirp",
        ]:
            acquisition_type = AcquisitionType.FREE_RUN
        else:
            acquisition_type = AcquisitionType.TRIGGER_FIRST_FRAME
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        if self.environment_metadata.sysid_metadata.sysid_signal_type == "Random":
            overlap_fraction = self.environment_metadata.sysid_metadata.sysid_overlap
        else:
            overlap_fraction = 0
        if self.environment_metadata.sysid_metadata.sysid_signal_type == "Burst Random":
            trigger_channel_index = reference_channel_indices[0]
        else:
            trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = self.environment_metadata.sysid_metadata.sysid_level / 100
        trigger_hysteresis = self.environment_metadata.sysid_metadata.sysid_level / 200
        trigger_hysteresis_samples = (
            (1 - self.environment_metadata.sysid_metadata.sysid_burst_on)
            * self.environment_metadata.sysid_metadata.sysid_frame_size
        ) // 2
        pretrigger_fraction = self.environment_metadata.sysid_metadata.sysid_pretrigger
        frame_size = self.environment_metadata.sysid_metadata.sysid_frame_size
        window = (
            Window.HANN
            if self.environment_metadata.sysid_metadata.sysid_window == "Hann"
            else Window.RECTANGLE
        )
        kurtosis_buffer_length = self.environment_metadata.sysid_metadata.sysid_averages

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
            response_transformation_matrix=self.environment_metadata.response_transformation_matrix,
            reference_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def get_sysid_spectral_processing_metadata(
        self, is_noise=False
    ) -> SpectralProcessingMetadata:
        """Collects metadata to send to the spectral processing process"""
        averaging_type = (
            AveragingTypes.LINEAR
            if self.environment_metadata.sysid_metadata.sysid_averaging_type == "Linear"
            else AveragingTypes.EXPONENTIAL
        )
        averages = (
            self.environment_metadata.sysid_metadata.sysid_noise_averages
            if is_noise
            else self.environment_metadata.sysid_metadata.sysid_averages
        )
        exponential_averaging_coefficient = (
            self.environment_metadata.sysid_metadata.sysid_exponential_averaging_coefficient
        )
        if self.environment_metadata.sysid_metadata.sysid_estimator == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_metadata.sysid_metadata.sysid_estimator == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(
                f"Invalid FRF Estimator {self.environment_metadata.sysid_metadata.sysid_estimator}"
            )
        num_response_channels = self.environment_metadata.num_response_channels
        num_reference_channels = self.environment_metadata.num_reference_channels
        frequency_spacing = (
            self.environment_metadata.sysid_metadata.sysid_frequency_spacing
        )
        sample_rate = self.environment_metadata.sample_rate
        num_frequency_lines = self.environment_metadata.sysid_metadata.sysid_fft_lines
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
            samples_per_write=self.hardware_metadata.samples_per_write,
            level_ramp_samples=self.environment_metadata.sysid_metadata.sysid_level_ramp_time
            * self.environment_metadata.sample_rate
            * self.hardware_metadata.output_oversample,
            output_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def get_sysid_signal_generator(self) -> SignalGenerator:
        """Creates a signal generator object that will generate the signals"""
        if self.environment_metadata.sysid_metadata.sysid_signal_type == "Random":
            return RandomSignalGenerator(
                rms=self.environment_metadata.sysid_metadata.sysid_level,
                sample_rate=self.environment_metadata.sample_rate,
                num_samples_per_frame=self.environment_metadata.sysid_metadata.sysid_frame_size,
                num_signals=self.environment_metadata.num_reference_channels,
                low_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_high_frequency_cutoff,
                cola_overlap=0.5,
                cola_window="hann",
                cola_exponent=0.5,
                output_oversample=self.hardware_metadata.output_oversample,
            )
        elif (
            self.environment_metadata.sysid_metadata.sysid_signal_type == "Pseudorandom"
        ):
            return PseudorandomSignalGenerator(
                rms=self.environment_metadata.sysid_metadata.sysid_level,
                sample_rate=self.environment_metadata.sample_rate,
                num_samples_per_frame=self.environment_metadata.sysid_metadata.sysid_frame_size,
                num_signals=self.environment_metadata.num_reference_channels,
                low_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_high_frequency_cutoff,
                output_oversample=self.hardware_metadata.output_oversample,
            )
        elif (
            self.environment_metadata.sysid_metadata.sysid_signal_type == "Burst Random"
        ):
            return BurstRandomSignalGenerator(
                rms=self.environment_metadata.sysid_metadata.sysid_level,
                sample_rate=self.environment_metadata.sample_rate,
                num_samples_per_frame=self.environment_metadata.sysid_metadata.sysid_frame_size,
                num_signals=self.environment_metadata.num_reference_channels,
                low_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_low_frequency_cutoff,
                high_frequency_cutoff=self.environment_metadata.sysid_metadata.sysid_high_frequency_cutoff,
                on_fraction=self.environment_metadata.sysid_metadata.sysid_burst_on,
                ramp_fraction=self.environment_metadata.sysid_metadata.sysid_burst_ramp_fraction,
                output_oversample=self.hardware_metadata.output_oversample,
            )
        elif self.environment_metadata.sysid_metadata.sysid_signal_type == "Chirp":
            return ChirpSignalGenerator(
                level=self.environment_metadata.sysid_metadata.sysid_level,
                sample_rate=self.environment_metadata.sample_rate,
                num_samples_per_frame=self.environment_metadata.sysid_metadata.sysid_frame_size,
                num_signals=self.environment_metadata.num_reference_channels,
                low_frequency_cutoff=np.max(
                    [
                        self.environment_metadata.sysid_metadata.sysid_frequency_spacing,
                        self.environment_metadata.sysid_metadata.sysid_low_frequency_cutoff,
                    ]
                ),
                high_frequency_cutoff=np.min(
                    [
                        self.environment_metadata.sample_rate / 2,
                        self.environment_metadata.sysid_metadata.sysid_high_frequency_cutoff,
                    ]
                ),
                output_oversample=self.hardware_metadata.output_oversample,
            )

    # endregion

    # region Loading
    def load_noise(self, data):
        """Sends noise data to the data analysis process"""
        self.data_analysis_command_queue.put(
            self.environment_name, (SysIdDataAnalysisCommands.LOAD_NOISE, data)
        )

    def load_transfer_function(self, data):
        """Sends transfer function data to the data analysis process"""
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIdDataAnalysisCommands.LOAD_TRANSFER_FUNCTION, data),
        )

    def save_system_id_to_file(self, data):
        filepath = Path(data)
        file_extension = filepath.suffix

        match file_extension:
            case ".nc4":
                netcdf_dataset = nc4.Dataset(  # pylint: disable=no-member
                    filepath, "w", format="NETCDF4", clobber=True
                )
                if self.environment_name not in netcdf_dataset.groups:
                    netcdf_handle = netcdf_dataset.createGroup(self.environment_name)
                else:
                    netcdf_handle = netcdf_dataset.groups[self.environment_name]
                self.environment_metadata.save_metadata_to_netcdf(netcdf_handle)
                self.sysid_data.save_package_to_netcdf(netcdf_handle)
                netcdf_dataset.close()
            case ".mat":
                field_dict = {}
                self.sysid_data.save_package_to_mat_field(field_dict)
                savemat(filepath, field_dict)
            case ".npz":
                field_dict = {}
                self.sysid_data.save_package_to_numpy_field(field_dict)
                np.savez(filepath, **field_dict)

        self.set_ready()

    def load_system_id_from_package(self, data: SysIdDataPackage):
        # Store SystemIdDataPackage to data_analysis process and environment
        sysid_data = data
        self.data_analysis_command_queue.put(
            self.environment_name,
            (SysIdDataAnalysisCommands.LOAD_SYSTEM_ID, sysid_data),
        )

        self.gui_update_queue.put(
            (
                self.environment_name,
                (
                    SysIdDataAnalysisUICommands.SYSID_UPDATE,
                    (
                        sysid_data.sysid_frames,
                        sysid_data.sysid_frames,
                        sysid_data.frequencies,
                        sysid_data.sysid_frf,
                        sysid_data.sysid_coherence,
                        sysid_data.sysid_response_cpsd,
                        sysid_data.sysid_reference_cpsd,
                        sysid_data.sysid_condition,
                    ),
                ),
            )
        )

        # This seems counterintuitive but lots of environments overwrite the
        # system_id_complete function so it is easier to do it this way
        self.queue_container.environment_command_queue.put(
            self.environment_name,
            (
                SysIdDataAnalysisCommands.SYSTEM_ID_COMPLETE,
                (self.environment_metadata.sysid_metadata, sysid_data),
            ),
        )

    # endregion

    # region Process
    def start_noise(self, data):
        """Starts the noise measurement with the provided metadata"""
        self.log("Starting Noise Measurement for System ID")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False

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
                (self.environment_metadata.sysid_metadata.sysid_skip_frames, 1),
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

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIdDataAnalysisCommands.RUN_NOISE,
                self.environment_metadata.sysid_metadata.auto_shutdown,
            ),
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

        self.set_sysid_active()
        self.gui_update_queue.put(
            (self.environment_name, (SysIdUICommands.SYSID_STARTED, None))
        )

    def start_transfer_function(self, data):
        """Starts the transfer function measurement with the provided metadata"""
        self.log("Starting Transfer Function for System ID")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False

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
                (self.environment_metadata.sysid_metadata.sysid_skip_frames, 1),
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

        # Start the data analysis running
        self.data_analysis_command_queue.put(
            self.environment_name,
            (
                SysIdDataAnalysisCommands.RUN_TRANSFER_FUNCTION,
                (self.environment_metadata.sysid_metadata.auto_shutdown),
            ),
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

        self.set_sysid_active()
        self.gui_update_queue.put(
            (self.environment_name, (SysIdUICommands.SYSID_STARTED, None))
        )

    # endregion

    # region Shutdown
    def stop_system_id(self, stop_tasks):
        """Starts the shutdown process for the system identification"""
        stop_data_analysis = stop_tasks  # This is so that data analysis class can stop itself then tell environment to stop for quicker responses
        self.log("Stop Transfer Function")
        self.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_metadata.sysid_metadata.sysid_skip_frames * 10, 1),
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
                self.environment_name, (SysIdDataAnalysisCommands.STOP_SYSTEM_ID, None)
            )
        self.environment_command_queue.put(
            self.environment_name, (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None)
        )

    def siggen_shutdown_achieved_fn(self, data):  # pylint: disable=unused-argument
        """Sets the shutdown flag to denote the signal generation has shut down successfully"""
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
        ):
            self._sysid_stream_name = None
            self.clear_sysid_active()
            self.gui_update_queue.put(
                (self.environment_name, (SysIdUICommands.SYSID_ENDED, None))
            )
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
            self.log(f"Waiting for {' and '.join(waiting_for)}")
            self.environment_command_queue.put(
                self.environment_name,
                (SystemIdCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
            )

    def system_id_noise_complete(self, data):
        """Sends a message to the gui that this environment has completed system id noise and should start
        system id transfer function"""
        self.log("System Identification Noise Compelte")
        self.gui_update_queue.put(
            (self.environment_name, (SysIdDataAnalysisUICommands.NOISE_COMPLETED, data))
        )

    def system_id_complete(self, data):
        """Sends a message to the controller that this environment has completed system id"""
        self.log("Finished System Identification")
        _, self.sysid_data = data
        self.gui_update_queue.put(
            (
                self.environment_name,
                (SysIdDataAnalysisUICommands.TRANSFER_COMPLETED, data),
            )
        )
        self.gui_update_queue.put(
            (UICommands.COMPLETED_SYSTEM_ID, (self.environment_name, data))
        )  # Enable tabs
        self.set_sysid_stored()

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

    # endregion
