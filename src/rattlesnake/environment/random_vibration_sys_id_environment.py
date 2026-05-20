# -*- coding: utf-8 -*-
"""
This file defines a Random Vibration Environment where a specification is
defined and the controller solves for excitations that will cause the test
article to match the specified response.

This environment has a number of subprocesses, including CPSD and FRF
computation, data analysis, and signal generation.

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
import threading
import multiprocessing as mp
import multiprocessing.sharedctypes  # pylint: disable=unused-import
import time
from enum import Enum
from multiprocessing.queues import Queue
from typing import List

import netCDF4 as nc4
import numpy as np
import openpyxl

from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.environment.abstract_environment import EnvironmentInstructions, EnvironmentCommands
from rattlesnake.environment.abstract_sysid_environment import (
    SysIdEnvironment,
    SysIdEnvironmentMetadata,
)
from rattlesnake.utilities import (
    GlobalCommands,
    VerboseMessageQueue,
)
from rattlesnake.environment.abstract_interactive_control_law import ControlLawCommands
from rattlesnake.process.data_collector import (
    Acceptance,
    AcquisitionType,
    CollectorMetadata,
    DataCollectorCommands,
    TriggerSlope,
    Window,
    data_collector_process,
)
from rattlesnake.process.signal_generation import (
    CPSDSignalGenerator,
)
from rattlesnake.process.signal_generation_process import (
    SignalGenerationCommands,
    SignalGenerationMetadata,
    signal_generation_process,
)
from rattlesnake.process.spectral_processing import (
    AveragingTypes,
    Estimator,
    SpectralProcessingCommands,
    SpectralProcessingMetadata,
    spectral_processing_process,
)

CONTROL_TYPE = EnvironmentType.RANDOM


# region Commands
class RandomVibrationCommands(EnvironmentCommands):
    """Valid random vibration commands"""

    ADJUST_TEST_LEVEL = 0
    START_CONTROL = 1
    STOP_CONTROL = 2
    CHECK_FOR_COMPLETE_SHUTDOWN = 3
    RECOMPUTE_PREDICTION = 4
    # UPDATE_INTERACTIVE_CONTROL_PARAMETERS = 5

    VALID_PROFILE_COMMANDS = ()
    VALID_DATA = {
        START_CONTROL: type(None),
        STOP_CONTROL: type(None),
    }


class RandomVibrationUICommands(Enum):
    ENABLE_CONTROL = 0


# endregion


# region Metadata
class RandomVibrationMetadata(SysIdEnvironmentMetadata):
    """Container to hold the signal processing parameters of the environment"""

    def __init__(
        self,
        *,
        environment_name: str,
        channel_list_bools: list,
        sample_rate: int,
        number_of_channels,
        samples_per_frame,
        test_level_ramp_time,
        cola_window,
        cola_overlap,
        cola_window_exponent,
        sigma_clip,
        update_tf_during_control,
        frames_in_cpsd,
        cpsd_window,
        cpsd_overlap,
        percent_lines_out,
        allow_automatic_aborts,
        control_python_script,
        control_python_function,
        control_python_function_type,
        control_python_function_parameters,
        control_channel_indices,
        output_channel_indices,
        specification_frequency_lines,
        specification_cpsd_matrix,
        specification_warning_matrix,
        specification_abort_matrix,
        response_transformation_matrix,
        output_transformation_matrix,
        sysid_metadata = None,
    ):
        super().__init__(CONTROL_TYPE, environment_name, channel_list_bools, sample_rate, sysid_metadata)
        self.number_of_channels = number_of_channels
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.test_level_ramp_time = test_level_ramp_time
        self.cpsd_overlap = cpsd_overlap
        self.update_tf_during_control = update_tf_during_control
        self.cola_window = cola_window
        self.cola_overlap = cola_overlap
        self.cola_window_exponent = cola_window_exponent
        self.sigma_clip = sigma_clip
        self.frames_in_cpsd = frames_in_cpsd
        self.cpsd_window = cpsd_window
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = output_transformation_matrix
        self.control_python_script = control_python_script
        self.control_python_function = control_python_function
        self.control_python_function_type = control_python_function_type
        self.control_python_function_parameters = control_python_function_parameters
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.specification_frequency_lines = specification_frequency_lines
        self.specification_cpsd_matrix = specification_cpsd_matrix
        self.specification_warning_matrix = specification_warning_matrix
        self.specification_abort_matrix = specification_abort_matrix
        self.percent_lines_out = percent_lines_out
        self.allow_automatic_aborts = allow_automatic_aborts

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def number_of_channels(self):
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        self._number_of_channels = value

    @property
    def reference_channel_indices(self):
        return self.output_channel_indices

    @property
    def response_channel_indices(self):
        return self.control_channel_indices

    @property
    def response_transformation_matrix(self):
        return self._response_transformation_matrix

    @response_transformation_matrix.setter
    def response_transformation_matrix(self, value):
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        self._reference_transformation_matrix = value

    @property
    def samples_per_acquire(self):
        """Property returning the samples per acquisition step given the overlap"""
        return int(self.samples_per_frame * (1 - self.cpsd_overlap))

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
    def frequency_spacing(self):
        """Property returning frequency line spacing given the sampling parameters"""
        return self.sample_rate / self.samples_per_frame

    @property
    def samples_per_output(self):
        """Property returning the samples per output given the COLA overlap"""
        return int(self.samples_per_frame * (1 - self.cola_overlap))

    @property
    def overlapped_output_samples(self):
        """Property returning the number of output samples that are overlapped."""
        return self.samples_per_frame - self.samples_per_output

    @property
    def skip_frames(self):
        """Property returning the number of frames to skip when changing levels"""
        return int(
            np.ceil(
                self.test_level_ramp_time
                * self.sample_rate
                / (self.samples_per_frame * (1 - self.cpsd_overlap))
            )
        )

    # endregion

    # region Validation
    def validate(self, hardware_metadata):
        return super().validate(hardware_metadata)

    # endregion

    # region Loading
    def save_metadata_to_netcdf(
        self,
        netcdf_group_handle: nc4._netCDF4.Group,  # pylint: disable=c-extension-no-member
    ):
        """Store parameters to a group in a netCDF streaming file.

        This function stores parameters from the environment into the netCDF
        file in a group with the environment's name as its name.  The function
        will receive a reference to the group within the dataset and should
        store the environment's parameters into that group in the form of
        attributes, dimensions, or variables.

        This function is the "write" counterpart to the retrieve_metadata
        function in the RandomVibrationUI class, which will read parameters from
        the netCDF file to populate the parameters in the user interface.

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A reference to the Group within the netCDF dataset where the
            environment's metadata is stored.

        """
        super().store_to_netcdf(netcdf_group_handle)
        netcdf_group_handle.samples_per_frame = self.samples_per_frame
        netcdf_group_handle.test_level_ramp_time = self.test_level_ramp_time
        netcdf_group_handle.cpsd_overlap = self.cpsd_overlap
        netcdf_group_handle.update_tf_during_control = (
            1 if self.update_tf_during_control else 0
        )
        netcdf_group_handle.cola_window = self.cola_window
        netcdf_group_handle.cola_overlap = self.cola_overlap
        netcdf_group_handle.cola_window_exponent = self.cola_window_exponent
        netcdf_group_handle.frames_in_cpsd = self.frames_in_cpsd
        netcdf_group_handle.cpsd_window = self.cpsd_window
        netcdf_group_handle.control_python_script = self.control_python_script
        netcdf_group_handle.control_python_function = self.control_python_function
        netcdf_group_handle.control_python_function_type = (
            self.control_python_function_type
        )
        netcdf_group_handle.control_python_function_parameters = (
            self.control_python_function_parameters
        )
        netcdf_group_handle.allow_automatic_aborts = (
            1 if self.allow_automatic_aborts else 0
        )
        # Specifications
        netcdf_group_handle.createDimension("fft_lines", self.fft_lines)
        netcdf_group_handle.createDimension("two", 2)
        netcdf_group_handle.createDimension(
            "specification_channels", self.specification_cpsd_matrix.shape[-1]
        )
        var = netcdf_group_handle.createVariable(
            "specification_frequency_lines", "f8", ("fft_lines",)
        )
        var[...] = self.specification_frequency_lines
        var = netcdf_group_handle.createVariable(
            "specification_cpsd_matrix_real",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.specification_cpsd_matrix.real
        var = netcdf_group_handle.createVariable(
            "specification_cpsd_matrix_imag",
            "f8",
            ("fft_lines", "specification_channels", "specification_channels"),
        )
        var[...] = self.specification_cpsd_matrix.imag
        var = netcdf_group_handle.createVariable(
            "specification_warning_matrix",
            "f8",
            ("two", "fft_lines", "specification_channels"),
        )
        var[...] = self.specification_warning_matrix.real
        var = netcdf_group_handle.createVariable(
            "specification_abort_matrix",
            "f8",
            ("two", "fft_lines", "specification_channels"),
        )
        var[...] = self.specification_abort_matrix.real
        # Transformation matrices
        if self.response_transformation_matrix is not None:
            netcdf_group_handle.createDimension(
                "response_transformation_rows",
                self.response_transformation_matrix.shape[0],
            )
            netcdf_group_handle.createDimension(
                "response_transformation_cols",
                self.response_transformation_matrix.shape[1],
            )
            var = netcdf_group_handle.createVariable(
                "response_transformation_matrix",
                "f8",
                ("response_transformation_rows", "response_transformation_cols"),
            )
            var[...] = self.response_transformation_matrix
        if self.reference_transformation_matrix is not None:
            netcdf_group_handle.createDimension(
                "reference_transformation_rows",
                self.reference_transformation_matrix.shape[0],
            )
            netcdf_group_handle.createDimension(
                "reference_transformation_cols",
                self.reference_transformation_matrix.shape[1],
            )
            var = netcdf_group_handle.createVariable(
                "reference_transformation_matrix",
                "f8",
                ("reference_transformation_rows", "reference_transformation_cols"),
            )
            var[...] = self.reference_transformation_matrix
        # Control channels
        netcdf_group_handle.createDimension(
            "control_channels", len(self.control_channel_indices)
        )
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices

    @classmethod
    def load_metadata_from_netcdf(
        cls,
        netcdf_group_handle: nc4._netCDF4.Group,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        pass

    @staticmethod
    def create_blank_worksheet_template(worksheet):
        worksheet.cell(1, 1, "Control Type")
        worksheet.cell(1, 2, "Random")
        worksheet.cell(2, 1, "Samples Per Frame:")
        worksheet.cell(2, 2, "# Number of Samples per Measurement Frame")
        worksheet.cell(3, 1, "Test Level Ramp Time:")
        worksheet.cell(3, 2, "# Time taken to Ramp between test levels")
        worksheet.cell(4, 1, "COLA Window:")
        worksheet.cell(4, 2, "# Window used for Constant Overlap and Add process")
        worksheet.cell(5, 1, "COLA Overlap %:")
        worksheet.cell(5, 2, "# Overlap used in Constant Overlap and Add process")
        worksheet.cell(6, 1, "COLA Window Exponent:")
        worksheet.cell(
            6,
            2,
            "# Exponent Applied to the COLA Window (use 0.5 unless you "
            "are sure you don't want to!)",
        )
        worksheet.cell(7, 1, "Update System ID During Control:")
        worksheet.cell(
            7,
            2,
            "# Continue updating transfer function while the controller is controlling (Y/N)",
        )
        worksheet.cell(8, 1, "Frames in CPSD:")
        worksheet.cell(8, 2, "# Frames used to compute the CPSD matrix")
        worksheet.cell(9, 1, "CPSD Window:")
        worksheet.cell(9, 2, "# Window used to compute the CPSD matrix")
        worksheet.cell(10, 1, "CPSD Overlap %:")
        worksheet.cell(10, 2, "# Overlap percentage for CPSD calculations")
        worksheet.cell(11, 1, "Allow Automatic Aborts")
        worksheet.cell(12, 1, "Control Python Script:")
        worksheet.cell(12, 2, "# Path to the Python script containing the control law")
        worksheet.cell(13, 1, "Control Python Function:")
        worksheet.cell(
            13,
            2,
            "# Function or class name within the Python Script that will serve as the control law",
        )
        worksheet.cell(14, 1, "Control Parameters:")
        worksheet.cell(14, 2, "# Extra parameters used in the control law")
        worksheet.cell(15, 1, "Control Channels (1-based):")
        worksheet.cell(16, 1, "System ID Averaging:")
        worksheet.cell(
            16,
            2,
            "# Averaging Type used for system ID.  Should be Linear or Exponential",
        )
        worksheet.cell(17, 1, "Noise Averages:")
        worksheet.cell(17, 2, "# Number of Averages used when characterizing noise")
        worksheet.cell(18, 1, "System ID Averages:")
        worksheet.cell(18, 2, "# Number of Averages used when computing the FRF")
        worksheet.cell(19, 1, "Exponential Averaging Coefficient:")
        worksheet.cell(
            19, 2, "# Averaging Coefficient for Exponential Averaging (if used)"
        )
        worksheet.cell(20, 1, "System ID Estimator:")
        worksheet.cell(
            20,
            2,
            "# Technique used to compute system ID.  Should be one of H1, H2, H3, or Hv.",
        )
        worksheet.cell(21, 1, "System ID Level (V RMS):")
        worksheet.cell(
            21,
            2,
            "# RMS Value of Flat Voltage Spectrum used for System Identification.",
        )
        worksheet.cell(22, 1, "System ID Signal Type:")
        worksheet.cell(23, 1, "System ID Window:")
        worksheet.cell(
            23,
            2,
            "# Window used to compute FRFs during system ID.  Should be one of Hann or None",
        )
        worksheet.cell(24, 1, "System ID Overlap %:")
        worksheet.cell(24, 2, "# Overlap to use in the system identification")
        worksheet.cell(25, 1, "System ID Burst On %:")
        worksheet.cell(25, 2, "# Percentage of a frame that the burst random is on for")
        worksheet.cell(26, 1, "System ID Burst Pretrigger %:")
        worksheet.cell(
            26,
            2,
            "# Percentage of a frame that occurs before the burst starts in a burst random signal",
        )
        worksheet.cell(27, 1, "System ID Ramp Fraction %:")
        worksheet.cell(
            27,
            2,
            '# Percentage of the "System ID Burst On %" that will be used to ramp up to full level',
        )
        worksheet.cell(28, 1, "Specification File:")
        worksheet.cell(28, 2, "# Path to the file containing the Specification")
        worksheet.cell(29, 1, "Response Transformation Matrix:")
        worksheet.cell(
            29,
            2,
            "# Transformation matrix to apply to the response channels.  Type None if there "
            "is none.  Otherwise, make this a 2D array in the spreadsheet and move the Output "
            "Transformation Matrix line down so it will fit.  The number of columns should be the "
            "number of physical control channels.",
        )
        worksheet.cell(30, 1, "Output Transformation Matrix:")
        worksheet.cell(
            30,
            2,
            "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
            "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should be "
            "the number of physical output channels in the environment.",
        )

    def save_metadata_to_worksheet(
        self, worksheet: openpyxl.worksheet.worksheet.Worksheet
    ):
        pass

    @classmethod
    def load_metadata_from_worksheet(
        cls,
        worksheet: openpyxl.worksheet.worksheet.Worksheet,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        pass


    # def set_parameters_from_template(
    #     self, worksheet: openpyxl.worksheet.worksheet.Worksheet
    # ):
    #     """
    #     Collects parameters for the user interface from the Excel template file

    #     This function reads a filled out template worksheet to create an
    #     environment.  Cells on this worksheet contain parameters needed to
    #     specify the environment, so this function should read those cells and
    #     update the UI widgets with those parameters.

    #     This function is the "read" counterpart to the
    #     ``create_environment_template`` function in the ``RandomVibrationUI`` class,
    #     which writes a template file that can be filled out by a user.


    #     Parameters
    #     ----------
    #     worksheet : openpyxl.worksheet.worksheet.Worksheet
    #         An openpyxl worksheet that contains the environment template.
    #         Cells on this worksheet should contain the parameters needed for the
    #         user interface.

    #     """
    #     self.definition_widget.samples_per_frame_selector.setValue(
    #         int(worksheet.cell(2, 2).value)
    #     )
    #     self.definition_widget.ramp_time_spinbox.setValue(
    #         float(worksheet.cell(3, 2).value)
    #     )
    #     self.definition_widget.cola_window_selector.setCurrentIndex(
    #         self.definition_widget.cola_window_selector.findText(
    #             worksheet.cell(4, 2).value
    #         )
    #     )
    #     self.definition_widget.cola_overlap_percentage_selector.setValue(
    #         float(worksheet.cell(5, 2).value)
    #     )
    #     self.definition_widget.cola_exponent_selector.setValue(
    #         float(worksheet.cell(6, 2).value)
    #     )
    #     self.definition_widget.update_transfer_function_during_control_selector.setChecked(
    #         worksheet.cell(7, 2).value.upper() == "Y"
    #     )
    #     self.definition_widget.cpsd_frames_selector.setValue(
    #         int(worksheet.cell(8, 2).value)
    #     )
    #     self.definition_widget.cpsd_computation_window_selector.setCurrentIndex(
    #         self.definition_widget.cpsd_computation_window_selector.findText(
    #             worksheet.cell(9, 2).value
    #         )
    #     )
    #     self.definition_widget.cpsd_overlap_selector.setValue(
    #         float(worksheet.cell(10, 2).value)
    #     )
    #     self.definition_widget.auto_abort_checkbox.setChecked(
    #         worksheet.cell(11, 2).value.upper() == "Y"
    #     )
    #     self.select_python_module(None, worksheet.cell(12, 2).value)
    #     self.definition_widget.control_function_input.setCurrentIndex(
    #         self.definition_widget.control_function_input.findText(
    #             worksheet.cell(13, 2).value
    #         )
    #     )
    #     self.definition_widget.control_parameters_text_input.setText(
    #         ""
    #         if worksheet.cell(14, 2).value is None
    #         else str(worksheet.cell(14, 2).value)
    #     )
    #     column_index = 2
    #     while True:
    #         value = worksheet.cell(15, column_index).value
    #         if value is None or (isinstance(value, str) and value.strip() == ""):
    #             break
    #         item = self.definition_widget.control_channels_selector.item(int(value) - 1)
    #         item.setCheckState(Qt.Checked)
    #         column_index += 1
    #     self.system_id_widget.averagingTypeComboBox.setCurrentIndex(
    #         self.system_id_widget.averagingTypeComboBox.findText(
    #             worksheet.cell(16, 2).value
    #         )
    #     )
    #     self.system_id_widget.noiseAveragesSpinBox.setValue(
    #         int(worksheet.cell(17, 2).value)
    #     )
    #     self.system_id_widget.systemIDAveragesSpinBox.setValue(
    #         int(worksheet.cell(18, 2).value)
    #     )
    #     self.system_id_widget.averagingCoefficientDoubleSpinBox.setValue(
    #         float(worksheet.cell(19, 2).value)
    #     )
    #     self.system_id_widget.estimatorComboBox.setCurrentIndex(
    #         self.system_id_widget.estimatorComboBox.findText(
    #             worksheet.cell(20, 2).value
    #         )
    #     )
    #     self.system_id_widget.levelDoubleSpinBox.setValue(
    #         float(worksheet.cell(21, 2).value)
    #     )
    #     # this should be a temporary solution - template file rework needed
    #     low, high = worksheet.cell(21, 3).value, worksheet.cell(21, 4).value
    #     sigma = worksheet.cell(21, 5).value
    #     if low is not None:
    #         self.system_id_widget.lowFreqCutoffSpinBox.setValue(int(low))
    #     if high is not None:
    #         self.system_id_widget.highFreqCutoffSpinBox.setValue(int(high))
    #     if sigma is not None:
    #         self.definition_widget.sigma_clipping_selector.setValue(
    #             float(sigma)
    #         )  # TODO: sigma clipping and bandwidths should get
    #         # their own rows, but how to maintain backward compatibility?
    #     self.system_id_widget.signalTypeComboBox.setCurrentIndex(
    #         self.system_id_widget.signalTypeComboBox.findText(
    #             worksheet.cell(22, 2).value
    #         )
    #     )
    #     self.system_id_widget.windowComboBox.setCurrentIndex(
    #         self.system_id_widget.windowComboBox.findText(worksheet.cell(23, 2).value)
    #     )
    #     self.system_id_widget.overlapDoubleSpinBox.setValue(
    #         float(worksheet.cell(24, 2).value)
    #     )
    #     self.system_id_widget.onFractionDoubleSpinBox.setValue(
    #         float(worksheet.cell(25, 2).value)
    #     )
    #     self.system_id_widget.pretriggerDoubleSpinBox.setValue(
    #         float(worksheet.cell(26, 2).value)
    #     )
    #     self.system_id_widget.rampFractionDoubleSpinBox.setValue(
    #         float(worksheet.cell(27, 2).value)
    #     )

    #     # Now we need to find the transformation matrices' sizes
    #     response_channels = self.definition_widget.control_channels_display.value()
    #     output_channels = self.definition_widget.output_channels_display.value()
    #     output_transform_row = 30
    #     if (
    #         isinstance(worksheet.cell(29, 2).value, str)
    #         and worksheet.cell(29, 2).value.lower() == "none"
    #     ):
    #         self.response_transformation_matrix = None
    #     else:
    #         while True:
    #             if (
    #                 worksheet.cell(output_transform_row, 1).value
    #                 == "Output Transformation Matrix:"
    #             ):
    #                 break
    #             output_transform_row += 1
    #         response_size = output_transform_row - 29
    #         response_transformation = []
    #         for i in range(response_size):
    #             response_transformation.append([])
    #             for j in range(response_channels):
    #                 response_transformation[-1].append(
    #                     float(worksheet.cell(29 + i, 2 + j).value)
    #                 )
    #         self.response_transformation_matrix = np.array(response_transformation)
    #     if (
    #         isinstance(worksheet.cell(output_transform_row, 2).value, str)
    #         and worksheet.cell(output_transform_row, 2).value.lower() == "none"
    #     ):
    #         self.output_transformation_matrix = None
    #     else:
    #         output_transformation = []
    #         i = 0
    #         while True:
    #             if worksheet.cell(output_transform_row + i, 2).value is None or (
    #                 isinstance(worksheet.cell(output_transform_row + i, 2).value, str)
    #                 and worksheet.cell(output_transform_row + i, 2).value.strip() == ""
    #             ):
    #                 break
    #             output_transformation.append([])
    #             for j in range(output_channels):
    #                 output_transformation[-1].append(
    #                     float(worksheet.cell(output_transform_row + i, 2 + j).value)
    #                 )
    #             i += 1
    #         self.output_transformation_matrix = np.array(output_transformation)
    #     self.define_transformation_matrices(None, dialog=False)
    #     self.select_spec_file(None, worksheet.cell(28, 2).value)

    # def retrieve_metadata(
    #     self,
    #     netcdf_handle: nc4._netCDF4.Dataset = None,  # pylint: disable=c-extension-no-member
    #     environment_name: str = None,
    # ):
    #     """Collects environment parameters from a netCDF dataset.

    #     This function retrieves parameters from a netCDF dataset that was written
    #     by the controller during streaming.  It must populate the widgets
    #     in the user interface with the proper information.

    #     This function is the "read" counterpart to the store_to_netcdf
    #     function in the AbstractMetadata class, which will write parameters to
    #     the netCDF file to document the metadata.

    #     Note that the entire dataset is passed to this function, so the function
    #     should collect parameters pertaining to the environment from a Group
    #     in the dataset sharing the environment's name, e.g.

    #     ``group = netcdf_handle.groups[self.environment_name]``
    #     ``self.definition_widget.parameter_selector.setValue(group.parameter)``

    #     Parameters
    #     ----------
    #     netcdf_handle : nc4._netCDF4.Dataset
    #         The netCDF dataset from which the data will be read.  It should have
    #         a group name with the enviroment's name.
    #     environment_name : str (optional)
    #         name of environment from which to retrieve metadata. Only needed if
    #         different from current environment.

    #     """
    #     group = super().retrieve_metadata(netcdf_handle, environment_name)

    #     # Control channels
    #     try:
    #         for i in group.variables["control_channel_indices"][...]:
    #             item = self.definition_widget.control_channels_selector.item(i)
    #             item.setCheckState(Qt.Checked)
    #     except KeyError:
    #         print(
    #             "no variable control_channel_indices, please select control channels manually"
    #         )
    #     # Other data
    #     try:
    #         self.response_transformation_matrix = group.variables[
    #             "response_transformation_matrix"
    #         ][...].data
    #     except KeyError:
    #         self.response_transformation_matrix = None
    #     try:
    #         self.output_transformation_matrix = group.variables[
    #             "reference_transformation_matrix"
    #         ][...].data
    #     except KeyError:
    #         self.output_transformation_matrix = None
    #     self.define_transformation_matrices(None, dialog=False)

    #     # environment_name is passed when the saved environment doesn't match the
    #     # current environment
    #     if environment_name is None:
    #         # Spinboxes
    #         self.definition_widget.samples_per_frame_selector.setValue(
    #             group.samples_per_frame
    #         )
    #         self.definition_widget.ramp_time_spinbox.setValue(
    #             group.test_level_ramp_time
    #         )
    #         self.definition_widget.cola_overlap_percentage_selector.setValue(
    #             group.cola_overlap * 100
    #         )
    #         self.definition_widget.cola_exponent_selector.setValue(
    #             group.cola_window_exponent
    #         )
    #         self.definition_widget.cpsd_overlap_selector.setValue(
    #             group.cpsd_overlap * 100
    #         )
    #         self.definition_widget.cpsd_frames_selector.setValue(group.frames_in_cpsd)
    #         # Checkboxes
    #         self.definition_widget.update_transfer_function_during_control_selector.setChecked(
    #             bool(group.update_tf_during_control)
    #         )
    #         self.definition_widget.auto_abort_checkbox.setChecked(
    #             bool(group.allow_automatic_aborts)
    #         )
    #         # Comboboxes
    #         self.definition_widget.cola_window_selector.setCurrentIndex(
    #             self.definition_widget.cola_window_selector.findText(group.cola_window)
    #         )
    #         self.definition_widget.cpsd_computation_window_selector.setCurrentIndex(
    #             self.definition_widget.cpsd_computation_window_selector.findText(
    #                 group.cpsd_window
    #             )
    #         )
    #         # Specification
    #         self.specification_frequency_lines = group.variables[
    #             "specification_frequency_lines"
    #         ][...].data
    #         self.specification_cpsd_matrix = (
    #             group.variables["specification_cpsd_matrix_real"][...].data
    #             + 1j * group.variables["specification_cpsd_matrix_imag"][...].data
    #         )
    #         self.specification_warning_matrix = group.variables[
    #             "specification_warning_matrix"
    #         ][...].data
    #         self.specification_abort_matrix = group.variables[
    #             "specification_abort_matrix"
    #         ][...].data
    #         self.select_python_module(None, group.control_python_script)
    #         index = self.definition_widget.control_function_input.findText(
    #             group.control_python_function
    #         )
    #         if (
    #             index == -1
    #         ):  # error handling (older revisions of rattlesnake may be missing newer control laws)
    #             index = 0
    #             default = self.definition_widget.control_function_input.itemText(index)
    #             print(
    #                 f'Warning: control function "{group.control_python_function}" not found, '
    #                 f'defaulting to "{default}"'
    #             )
    #         self.definition_widget.control_function_input.setCurrentIndex(index)
    #         self.definition_widget.control_parameters_text_input.setText(
    #             group.control_python_function_parameters
    #         )
    #         self.show_specification()

    # endregion

# region Instructions
class RandomInstructions(EnvironmentInstructions):
    def __init__(self, environment_name, control_test_level):
        super().__init__(CONTROL_TYPE, environment_name)
        self.control_test_level = control_test_level

    def validate(self):
        return super().validate()


# endregion


# region Queues
class RandomVibrationQueues:
    """A container class for the queues that random vibration will manage."""

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
        """A container class for the queues that random vibration will manage.

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
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Analysis Command Queue",
        )
        self.signal_generation_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Signal Generation Command Queue",
        )
        self.spectral_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Spectral Computation Command Queue",
        )
        self.collector_command_queue = VerboseMessageQueue(
            log_file_queue,
            mp.Queue(),
            environment_name + " Data Collector Command Queue",
        )
        self.controller_communication_queue = controller_communication_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.data_for_spectral_computation_queue = mp.Queue()
        self.updated_spectral_quantities_queue = mp.Queue()
        self.cpsd_to_generate_queue = mp.Queue()
        self.log_file_queue = log_file_queue


# endregion

from rattlesnake.process.random_vibration_sys_id_data_analysis import (  # noqa: E402 pylint: disable=wrong-import-position
    RandomVibrationDataAnalysisCommands,
    random_data_analysis_process,
)


class RandomVibrationEnvironment(SysIdEnvironment):
    """Random Environment class defining the interface with the controller"""

    # region Environment
    def __init__(
        self,
        environment_name: str,
        queue_name: str,
        queue_container: RandomVibrationQueues,
        acquisition_active_event: mp.synchronize.Event,
        output_active_event: mp.synchronize.Event,
        active_event: mp.synchronize.Event,
        ready_event: mp.synchronize.Event,
        sysid_active_event: mp.synchronize.Event,
        sysid_stored_event: mp.synchronize.Event,
    ):
        """
        Random Vibration Environment Constructor that fills out the ``command_map``

        Parameters
        ----------
        environment_name : str
            Name of the environment.
        queue_container : RandomVibrationQueues
            Container of queues used by the Random Vibration Environment.

        """
        super().__init__(
            environment_name,
            queue_name,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.controller_communication_queue,
            queue_container.log_file_queue,
            queue_container.collector_command_queue,
            queue_container.signal_generation_command_queue,
            queue_container.spectral_command_queue,
            queue_container.data_analysis_command_queue,
            queue_container.data_in_queue,
            queue_container.data_out_queue,
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
            sysid_active_event,
            sysid_stored_event,
        )
        self.map_command(RandomVibrationCommands.START_CONTROL, self.start_control)
        self.map_command(RandomVibrationCommands.STOP_CONTROL, self.stop_environment)
        self.map_command(
            RandomVibrationCommands.ADJUST_TEST_LEVEL, self.adjust_test_level
        )
        self.map_command(
            RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN,
            self.check_for_control_shutdown,
        )
        self.map_command(
            RandomVibrationCommands.RECOMPUTE_PREDICTION, self.recompute_prediction
        )
        self.map_command(
            ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS,
            self.update_interactive_control_parameters,
        )
        self.map_command(
            ControlLawCommands.SEND_INTERACTIVE_COMMAND, self.send_interactive_command
        )
        self.queue_container = queue_container

    # endregion

    # region StateSync
    def initialize_hardware(self, hardware_metadata):
        return super().initialize_hardware(hardware_metadata)

    def initialize_environment(self, environment_metadata: RandomVibrationMetadata):
        super().initialize_environment(environment_metadata)
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
        # Set up the data analysis
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (
                RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
                self.environment_metadata,
            ),
        )
        self.set_ready()

    def initialize_sysid(self, sysid_metadata):
        return super().initialize_sysid(sysid_metadata)

    def update_interactive_control_parameters(self, parameters):
        """Sends updated parameters to the interactive control law on the data analysis process"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (ControlLawCommands.UPDATE_INTERACTIVE_CONTROL_PARAMETERS, parameters),
        )

    def get_data_collector_metadata(self):
        """Gets relevant metadata for the data collector process"""
        num_channels = self.environment_metadata.number_of_channels
        response_channel_indices = self.environment_metadata.response_channel_indices
        reference_channel_indices = self.environment_metadata.reference_channel_indices
        acquisition_type = AcquisitionType.FREE_RUN
        acceptance = Acceptance.AUTOMATIC
        acceptance_function = None
        overlap_fraction = self.environment_metadata.cpsd_overlap
        trigger_channel_index = 0
        trigger_slope = TriggerSlope.POSITIVE
        trigger_level = 0
        trigger_hysteresis = 0
        trigger_hysteresis_samples = 0
        pretrigger_fraction = 0
        frame_size = self.environment_metadata.samples_per_frame
        window = (
            Window.HANN if self.environment_metadata.cpsd_window == "Hann" else None
        )
        # use number of sysid averages as kurtosis buffer size
        # (could maybe make this match the test duration if user is using the "Time at Level"
        # function, would need to pass info from the RandomVibrationUI object)
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

    def get_signal_generation_metadata(self):
        """Gets relevant metadata for the signal generation process"""
        return SignalGenerationMetadata(
            samples_per_write=self.data_acquisition_parameters.samples_per_write,
            level_ramp_samples=self.environment_metadata.test_level_ramp_time
            * self.environment_metadata.sample_rate
            * self.data_acquisition_parameters.output_oversample,
            output_transformation_matrix=self.environment_metadata.reference_transformation_matrix,
        )

    def get_signal_generator(self):
        """Gets the signal generator object that will generate signals for the environment"""
        return CPSDSignalGenerator(
            self.environment_metadata.sample_rate,
            self.environment_metadata.samples_per_frame,
            self.environment_metadata.num_reference_channels,
            None,
            self.environment_metadata.cola_overlap,
            self.environment_metadata.cola_window,
            self.environment_metadata.cola_window_exponent,
            self.environment_metadata.sigma_clip,
            self.data_acquisition_parameters.output_oversample,
        )

    def get_spectral_processing_metadata(self):
        """Gets the required metadata for the spectral processing process"""
        averaging_type = AveragingTypes.LINEAR
        averages = self.environment_metadata.frames_in_cpsd
        exponential_averaging_coefficient = 0
        if self.environment_parameters.sysid_estimator == "H1":
            frf_estimator = Estimator.H1
        elif self.environment_parameters.sysid_estimator == "H2":
            frf_estimator = Estimator.H2
        elif self.environment_parameters.sysid_estimator == "H3":
            frf_estimator = Estimator.H3
        elif self.environment_parameters.sysid_estimator == "Hv":
            frf_estimator = Estimator.HV
        else:
            raise ValueError(
                f"Invalid FRF Estimator {self.environment_parameters.sysid_estimator}"
            )
        num_response_channels = self.environment_parameters.num_response_channels
        num_reference_channels = self.environment_parameters.num_reference_channels
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
        )

    # endregion

    # region Commands
    def adjust_test_level(self, data):
        """Adjusts the test level of the environment to the specified level"""
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, data)
        )
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_metadata.skip_frames, data),
            ),
        )

    def send_interactive_command(self, command):
        """General method that can be used by an interactive UI object to pass commands and data to
        its corresponding computation object"""
        if self.environment_metadata.control_python_function_type == 3:  # Interactive
            self.queue_container.data_analysis_command_queue.put(
                self.environment_name,
                (ControlLawCommands.SEND_INTERACTIVE_COMMAND, command),
            )
        else:
            raise ValueError(
                "Received an SEND_INTERACTIVE_COMMAND signal without an interactive control law.  "
                "How did this happen?"
            )

    def system_id_complete(self, data):
        """Triggered when system identification has been completed, starting control predictions"""
        super().system_id_complete(data)
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION, None),
        )
        self.set_sysid_stored()

    def recompute_prediction(self, data):  # pylint: disable=unused-argument
        """Sends a signal to the data analysis process to recompute test predictions"""
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.PERFORM_CONTROL_PREDICTION, None),
        )

    def start_control(self, data: RandomInstructions):
        """Starts the environment at the specified test level"""
        self.log("Starting Control")
        self.siggen_shutdown_achieved = False
        self.collector_shutdown_achieved = False
        self.spectral_shutdown_achieved = False
        self.analysis_shutdown_achieved = False
        self.queue_container.controller_communication_queue.put(
            self.environment_name,
            (GlobalCommands.START_ENVIRONMENT, self.environment_name),
        )
        # Set up the collector
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.INITIALIZE_COLLECTOR,
                self.get_data_collector_metadata(),
            ),
        )

        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_metadata.skip_frames, data.control_test_level),
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
            self.environment_name, (SignalGenerationCommands.ADJUST_TEST_LEVEL, data.control_test_level)
        )

        # Tell the collector to start acquiring data
        self.queue_container.collector_command_queue.put(
            self.environment_name, (DataCollectorCommands.ACQUIRE, None)
        )

        # Tell the signal generation to start generating signals
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.GENERATE_SIGNALS, None)
        )

        # # Set up the data analysis
        # self.queue_container.data_analysis_command_queue.put(
        #     self.environment_name,
        #     (RandomVibrationDataAnalysisCommands.INITIALIZE_PARAMETERS,
        #      self.environment_parameters))

        # Start the data analysis running
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.RUN_CONTROL, None),
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

        self.set_active()

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
        self.queue_container.collector_command_queue.put(
            self.environment_name,
            (
                DataCollectorCommands.SET_TEST_LEVEL,
                (self.environment_parameters.skip_frames * 10, 1),
            ),
        )
        self.queue_container.signal_generation_command_queue.put(
            self.environment_name, (SignalGenerationCommands.START_SHUTDOWN, None)
        )
        self.queue_container.spectral_command_queue.put(
            self.environment_name,
            (SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING, None),
        )
        self.queue_container.data_analysis_command_queue.put(
            self.environment_name,
            (RandomVibrationDataAnalysisCommands.STOP_CONTROL, None),
        )
        self.queue_container.environment_command_queue.put(
            self.environment_name,
            (RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
        )

    # region Shutdown
    def check_for_control_shutdown(self, data):  # pylint: disable=unused-argument
        """Checks the different processes to see if the controller has shut down gracefully"""
        if (
            self.siggen_shutdown_achieved
            and self.collector_shutdown_achieved
            and self.spectral_shutdown_achieved
            and self.analysis_shutdown_achieved
        ):
            self.log("Shutdown Achieved")
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (RandomVibrationUICommands.ENABLE_CONTROL, None),
                )
            )
            self.clear_active()
        else:
            # Recheck some time later
            time.sleep(1)
            self.environment_command_queue.put(
                self.environment_name,
                (RandomVibrationCommands.CHECK_FOR_COMPLETE_SHUTDOWN, None),
            )

    def quit(self, data):
        """Closes down the environment permanently as the software is exiting"""
        for queue in [
            self.queue_container.spectral_command_queue,
            self.queue_container.data_analysis_command_queue,
            self.queue_container.signal_generation_command_queue,
            self.queue_container.collector_command_queue,
        ]:
            queue.put(self.environment_name, (GlobalCommands.QUIT, None))
        # Return true to stop the task
        return True

    # endregion


# region Process
def random_vibration_process(
    environment_name: str,
    queue_name: str,
    input_queue: VerboseMessageQueue,
    gui_update_queue: mp.Queue,
    controller_command_queue: VerboseMessageQueue,
    log_file_queue: mp.Queue,
    data_in_queue: mp.Queue,
    data_out_queue: mp.Queue,
    acquisition_active_event: mp.synchronize.Event,
    output_active_event: mp.synchronize.Event,
    active_event: mp.synchronize.Event,
    ready_event: mp.synchronize.Event,
    shutdown_event: mp.synchronize.Event,
    sysid_active_event: mp.synchronize.Event,
    sysid_stored_event: mp.synchronize.Event,
    threaded: bool,
):
    """Random vibration environment process function called by multiprocessing

    This function defines the Random Vibration Environment process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a RandomVibrationEnvironment object and runs it.

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
    acquisition_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the acquisition is active
    output_active : mp.sharedctypes.Synchronized
        A synchronized value that indicates when the output is active
    """
    # Create vibration queues
    if threaded:
        new_process = threading.Thread  # worker threads
    else:
        new_process = mp.Process  # worker processes
    queue_container = RandomVibrationQueues(
        environment_name,
        input_queue,
        gui_update_queue,
        controller_command_queue,
        data_in_queue,
        data_out_queue,
        log_file_queue,
    )

    spectral_proc = new_process(
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
    analysis_proc = new_process(
        target=random_data_analysis_process,
        args=(
            environment_name,
            queue_container.data_analysis_command_queue,
            queue_container.updated_spectral_quantities_queue,
            queue_container.cpsd_to_generate_queue,
            queue_container.environment_command_queue,
            queue_container.gui_update_queue,
            queue_container.log_file_queue,
        ),
    )
    analysis_proc.start()
    siggen_proc = new_process(
        target=signal_generation_process,
        args=(
            environment_name,
            queue_container.signal_generation_command_queue,
            queue_container.cpsd_to_generate_queue,
            queue_container.data_out_queue,
            queue_container.environment_command_queue,
            queue_container.log_file_queue,
            queue_container.gui_update_queue,
        ),
    )
    siggen_proc.start()
    collection_proc = new_process(
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
    process_class = RandomVibrationEnvironment(
        environment_name,
        queue_name,
        queue_container,
        acquisition_active_event,
        output_active_event,
        active_event,
        ready_event,
        sysid_active_event,
        sysid_stored_event,
    )
    process_class.run()

    # Rejoin all the processes
    process_class.log("Joining Subprocesses")
    process_class.log("Joining Spectral Computation")
    spectral_proc.join()
    process_class.log("Joining Data Analysis")
    analysis_proc.join()
    process_class.log("Joining Signal Generation")
    siggen_proc.join()
    process_class.log("Joining Data Collection")
    collection_proc.join()


# endregion
