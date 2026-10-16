"""
This file defines the metadata that describes a MIMO Random Vibration environment

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

import inspect
from enum import Enum
from typing import List

import netCDF4 as nc4
import numpy as np
import openpyxl
import scipy.signal as sig

from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawComputation,
)
from rattlesnake.environment.abstract_sysid_environment import SysIdEnvironmentMetadata
from rattlesnake.environment.environment_utilities import (
    EnvironmentType,
)
from rattlesnake.environment.random_vibration_sys_id_utilities import (
    load_specification,
)
from rattlesnake.hardware.abstract_hardware import HardwareMetadata
from rattlesnake.process.abstract_sysid_data_analysis import SysIdMetadata
from rattlesnake.utilities import (
    RattlesnakeError,
    _direction_map,
    load_python_module,
    worksheet_cell_str,
)

CONTROL_TYPE = EnvironmentType.RANDOM
# region Control Law Utilities


class RandomControlLawType(Enum):
    """Enumeration containing acceptable types of objects to use for a control law"""

    FUNCTION = 0
    GENERATOR = 1
    CLASS = 2
    INTERACTIVE_CLASS = 3


class RandomControlParameters:
    """Class to store control law data"""

    def __init__(
        self,
        control_script: str | None,
        control_object: str | None,
        control_type: RandomControlLawType | None,
        control_parameters: dict | None,
    ):
        """Initializes an object to store information about the custom control law

        Parameters
        ----------
        control_script : str
            The path to the python script containing the control law
        control_object : str
            The name of the item (function, generator, class) containing the control law
        control_type : ControlLawType
            The type of item defining the control law
        control_parameters : dict
            Any extra keyword arguments to pass to the control law.
        """
        self.control_script = control_script
        self.control_object = control_object
        self.control_type = control_type
        if isinstance(
            control_parameters, str
        ):  # attempting to retain backward compatibility
            try:
                self.control_parameters = {
                    s.split("=")[0].strip(): s.split("=")[1].strip()
                    for s in control_parameters.splitlines()
                }
            except Exception as e:
                print(
                    f"Could not interpret legacy `extra_parameters` as `key=value` pairs: {e}"
                )
                self.control_parameters = None
        else:
            self.control_parameters = control_parameters


# endregion


def _parse_scalar_string(value):
    if isinstance(value, (int, float, bool)):
        return value
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return ""

    if text.lower() in ("true", "y", "yes"):
        return True
    if text.lower() in ("false", "n", "no"):
        return False

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text


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
        sysid_metadata=None,
    ):
        super().__init__(
            CONTROL_TYPE,
            environment_name,
            channel_list_bools,
            sample_rate,
            sysid_metadata,
        )
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
        self._specification_file = None  # This is only used for saving purposes

    @property
    def specification_file(self):
        return self._specification_file

    def set_file(self, filepath):
        self._specification_file = filepath

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

    # region Validation
    def validate(self, hardware_metadata):
        super().validate(hardware_metadata)
        self.sysid_metadata.validate(hardware_metadata)

        if self.samples_per_frame <= 0:
            raise RattlesnakeError("samples_per_frame must be greater than 0")

        if not (0 <= self.cpsd_overlap < 1):
            raise RattlesnakeError(
                "CPSD overlap must be a percentage from 0 up to but not including 100"
            )

        if not (0 <= self.cola_overlap < 1):
            raise RattlesnakeError(
                "COLA overlap must be a percentage from 0 up to but not including 100"
            )

        if self.cola_window_exponent < 0:
            raise RattlesnakeError(
                "COLA window exponent must be greater than or equal to 0"
            )

        try:
            sig.get_window(str(self.cola_window).lower(), 2)
        except ValueError as e:
            raise RattlesnakeError(
                f"Invalid COLA window {self.cola_window}: {e}"
            ) from e

        if self.sigma_clip <= 0:
            raise RattlesnakeError("Sigma clipping must be greater than 0")

        if self.test_level_ramp_time < 0:
            raise RattlesnakeError(
                "Test level ramp time must be greater than or equal to 0"
            )

        if len(self.control_channel_indices) == 0:
            raise RattlesnakeError(
                "control_channel_indices must contain at least one channel"
            )
        for index in self.control_channel_indices:
            if not (0 <= index < self.number_of_channels):
                raise RattlesnakeError(
                    f"control_channel_indices contains invalid channel index "
                    f"{index} (must be between 0 and {self.number_of_channels - 1})"
                )

        if len(self.output_channel_indices) == 0:
            raise RattlesnakeError(
                "output_channel_indices must contain at least one channel"
            )
        for index in self.output_channel_indices:
            if not (0 <= index < self.number_of_channels):
                raise RattlesnakeError(
                    f"output_channel_indices contains invalid channel index "
                    f"{index} (must be between 0 and {self.number_of_channels - 1})"
                )

        if self.response_transformation_matrix is not None and (
            self.response_transformation_matrix.ndim != 2
            or self.response_transformation_matrix.shape[1]
            != len(self.control_channel_indices)
        ):
            raise RattlesnakeError(
                "response_transformation_matrix must be 2D with a number of "
                "columns matching the number of control channels "
                f"({len(self.control_channel_indices)}), got shape "
                f"{self.response_transformation_matrix.shape}"
            )

        if self.reference_transformation_matrix is not None and (
            self.reference_transformation_matrix.ndim != 2
            or self.reference_transformation_matrix.shape[1]
            != len(self.output_channel_indices)
        ):
            raise RattlesnakeError(
                "reference_transformation_matrix must be 2D with a number of "
                "columns matching the number of output channels "
                f"({len(self.output_channel_indices)}), got shape "
                f"{self.reference_transformation_matrix.shape}"
            )

        self._validate_specification()
        self._validate_control_law()

    def _validate_specification(self):
        if (
            self.specification_cpsd_matrix is None
            or self.specification_warning_matrix is None
            or self.specification_abort_matrix is None
            or self.specification_frequency_lines is None
        ):
            raise RattlesnakeError(
                f"{self.environment_name} has no specification loaded"
            )

        num_response_channels = self.num_response_channels
        expected_cpsd_shape = (
            self.fft_lines,
            num_response_channels,
            num_response_channels,
        )
        if self.specification_cpsd_matrix.shape != expected_cpsd_shape:
            raise RattlesnakeError(
                f"{self.environment_name} specification CPSD matrix has "
                f"shape {self.specification_cpsd_matrix.shape}, expected "
                f"{expected_cpsd_shape} for the current samples per frame "
                "and control channels; reload the specification"
            )

        expected_limit_shape = (2, self.fft_lines, num_response_channels)
        if self.specification_warning_matrix.shape != expected_limit_shape:
            raise RattlesnakeError(
                f"{self.environment_name} specification warning matrix has "
                f"shape {self.specification_warning_matrix.shape}, expected "
                f"{expected_limit_shape} for the current samples per frame "
                "and control channels; reload the specification"
            )
        if self.specification_abort_matrix.shape != expected_limit_shape:
            raise RattlesnakeError(
                f"{self.environment_name} specification abort matrix has "
                f"shape {self.specification_abort_matrix.shape}, expected "
                f"{expected_limit_shape} for the current samples per frame "
                "and control channels; reload the specification"
            )

    def _validate_control_law(self):
        if not self.control_python_script:
            raise RattlesnakeError(
                f"{self.environment_name} has no control_python_script set"
            )
        if not self.control_python_function:
            raise RattlesnakeError(
                f"{self.environment_name} has no control_python_function set"
            )

        try:
            module = load_python_module(self.control_python_script)
        except Exception as e:
            raise RattlesnakeError(
                f"{self.environment_name} could not load control script "
                f"{self.control_python_script}: {e}"
            ) from e

        function = getattr(module, self.control_python_function, None)
        if function is None:
            raise RattlesnakeError(
                f"{self.environment_name} control script "
                f"{self.control_python_script} has no function or class "
                f"named {self.control_python_function}"
            )

        if inspect.isgeneratorfunction(function):
            actual_type = RandomControlLawType.GENERATOR
        elif inspect.isclass(function) and issubclass(
            function, AbstractControlLawComputation
        ):
            actual_type = RandomControlLawType.INTERACTIVE_CLASS
        elif inspect.isclass(function):
            actual_type = RandomControlLawType.CLASS
        else:
            if not callable(function):
                raise RattlesnakeError(
                    f"{self.environment_name} control function "
                    f"{self.control_python_function} is not callable"
                )
            actual_type = RandomControlLawType.FUNCTION

        if actual_type != RandomControlLawType(self.control_python_function_type):
            raise RattlesnakeError(
                f"{self.environment_name} control_python_function_type "
                f"({self.control_python_function_type}) does not match what "
                f"{self.control_python_function} actually is (detected type "
                f"{actual_type}); reload the control script"
            )

    # endregion

    # region Loading
    def load_specification(self, environment_channel_list, filename):
        coord_dtype = np.dtype([("node", "<u8"), ("direction", "i1")])
        if self.response_transformation_matrix is not None:
            control_coordinate = None
        else:
            control_coordinate = np.array(
                [
                    (
                        environment_channel_list[i].node_number,
                        _direction_map[environment_channel_list[i].node_direction],
                    )
                    for i in self.control_channel_indices
                ],
                dtype=coord_dtype,
            )

        (
            self.specification_frequency_lines,
            self.specification_cpsd_matrix,
            self.specification_warning_matrix,
            self.specification_abort_matrix,
        ) = load_specification(
            filename,
            self.fft_lines,
            self.frequency_spacing,
            control_coordinate,
        )

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
        super().save_metadata_to_netcdf(netcdf_group_handle)
        netcdf_group_handle.samples_per_frame = self.samples_per_frame
        netcdf_group_handle.test_level_ramp_time = self.test_level_ramp_time
        netcdf_group_handle.cpsd_overlap = self.cpsd_overlap
        netcdf_group_handle.update_tf_during_control = (
            1 if self.update_tf_during_control else 0
        )
        netcdf_group_handle.cola_window = self.cola_window
        netcdf_group_handle.cola_overlap = self.cola_overlap
        netcdf_group_handle.percent_lines_out = self.percent_lines_out
        netcdf_group_handle.cola_window_exponent = self.cola_window_exponent
        netcdf_group_handle.frames_in_cpsd = self.frames_in_cpsd
        netcdf_group_handle.cpsd_window = self.cpsd_window
        netcdf_group_handle.allow_automatic_aborts = (
            1 if self.allow_automatic_aborts else 0
        )
        # Control group
        control_grp = netcdf_group_handle.createGroup("control_parameters")
        control_grp.control_type = self.control_python_function_type.value
        control_grp.control_script = self.control_python_script
        control_grp.control_object = self.control_python_function
        control_grp_params = control_grp.createGroup("control_extra_parameters")
        for key, value in self.control_python_function_parameters.items():
            setattr(control_grp_params, key, value)
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
        """Collect environment parameters from a netCDF group."""

        sample_rate = hardware_metadata.sample_rate
        number_of_channels = sum(channel_list_bools)

        environment_channel_list = [
            channel
            for channel, channel_bool in zip(
                hardware_metadata.channel_list, channel_list_bools
            )
            if channel_bool
        ]

        output_channel_indices = [
            index
            for index, channel in enumerate(environment_channel_list)
            if channel.feedback_device is not None
        ]

        samples_per_frame = netcdf_group_handle.samples_per_frame
        test_level_ramp_time = netcdf_group_handle.test_level_ramp_time
        cpsd_overlap = netcdf_group_handle.cpsd_overlap
        update_tf_during_control = bool(netcdf_group_handle.update_tf_during_control)
        cola_window = netcdf_group_handle.cola_window
        cola_overlap = netcdf_group_handle.cola_overlap
        cola_window_exponent = netcdf_group_handle.cola_window_exponent
        frames_in_cpsd = netcdf_group_handle.frames_in_cpsd
        cpsd_window = netcdf_group_handle.cpsd_window
        if hasattr(netcdf_group_handle, "percent_lines_out"):
            percent_lines_out = netcdf_group_handle.percent_lines_out
        else:
            percent_lines_out = 0.1
        allow_automatic_aborts = bool(netcdf_group_handle.allow_automatic_aborts)

        # Control Data
        if "control_parameters" in netcdf_group_handle.groups:
            control_grp = netcdf_group_handle.groups["control_parameters"]
            control_type = RandomControlLawType(control_grp.control_type)
            control_script = control_grp.control_script
            control_object = control_grp.control_object

            control_parameters = {}
            if "control_extra_parameters" in control_grp.groups:
                control_grp_params = control_grp.groups["control_extra_parameters"]
                for key in control_grp_params.ncattrs():
                    control_parameters[key] = getattr(control_grp_params, key)

            control_script_data = RandomControlParameters(
                control_script=control_script,
                control_object=control_object,
                control_type=control_type,
                control_parameters=control_parameters,
            )
        else:
            control_script_data = RandomControlParameters(
                control_script=netcdf_group_handle.control_python_script,
                control_object=netcdf_group_handle.control_python_function,
                control_type=RandomControlLawType(netcdf_group_handle.control_python_function_type),
                control_parameters=netcdf_group_handle.control_python_function_parameters,
            )

        control_channel_indices = netcdf_group_handle.variables[
            "control_channel_indices"
        ][...]

        specification_frequency_lines = netcdf_group_handle.variables[
            "specification_frequency_lines"
        ][...]
        specification_cpsd_matrix = (
            netcdf_group_handle.variables["specification_cpsd_matrix_real"][...]
            + 1j * netcdf_group_handle.variables["specification_cpsd_matrix_imag"][...]
        )
        specification_warning_matrix = netcdf_group_handle.variables[
            "specification_warning_matrix"
        ][...]
        specification_abort_matrix = netcdf_group_handle.variables[
            "specification_abort_matrix"
        ][...]

        response_transformation_matrix = None
        if "response_transformation_matrix" in netcdf_group_handle.variables:
            response_transformation_matrix = netcdf_group_handle.variables[
                "response_transformation_matrix"
            ][...]

        reference_transformation_matrix = None
        if "reference_transformation_matrix" in netcdf_group_handle.variables:
            reference_transformation_matrix = netcdf_group_handle.variables[
                "reference_transformation_matrix"
            ][...]

        sysid_metadata = SysIdMetadata.load_metadata_from_netcdf(
            netcdf_group_handle, hardware_metadata
        )

        return cls(
            environment_name=environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=sample_rate,
            number_of_channels=number_of_channels,
            samples_per_frame=samples_per_frame,
            test_level_ramp_time=test_level_ramp_time,
            cola_window=cola_window,
            cola_overlap=cola_overlap,
            cola_window_exponent=cola_window_exponent,
            sigma_clip=5,
            update_tf_during_control=update_tf_during_control,
            frames_in_cpsd=frames_in_cpsd,
            cpsd_window=cpsd_window,
            cpsd_overlap=cpsd_overlap,
            percent_lines_out=percent_lines_out,
            allow_automatic_aborts=allow_automatic_aborts,
            control_python_script=control_script_data.control_script,
            control_python_function=control_script_data.control_object,
            control_python_function_type=control_script_data.control_type,
            control_python_function_parameters=control_script_data.control_parameters,
            control_channel_indices=control_channel_indices,
            output_channel_indices=output_channel_indices,
            specification_frequency_lines=specification_frequency_lines,
            specification_cpsd_matrix=specification_cpsd_matrix,
            specification_warning_matrix=specification_warning_matrix,
            specification_abort_matrix=specification_abort_matrix,
            response_transformation_matrix=response_transformation_matrix,
            output_transformation_matrix=reference_transformation_matrix,
            sysid_metadata=sysid_metadata,
        )

    @classmethod
    def create_blank_worksheet_template(cls, worksheet):
        super().create_blank_worksheet_template(worksheet)
        worksheet.cell(1, 2, "Random")
        worksheet.cell(2, 1, "Samples Per Frame:")
        worksheet.cell(2, 3, "# Number of Samples per Measurement Frame")
        worksheet.cell(3, 1, "Test Level Ramp Time:")
        worksheet.cell(3, 3, "# Time taken to Ramp between test levels")
        worksheet.cell(4, 1, "COLA Window:")
        worksheet.cell(4, 3, "# Window used for Constant Overlap and Add process")
        worksheet.cell(5, 1, "COLA Overlap %:")
        worksheet.cell(5, 3, "# Overlap used in Constant Overlap and Add process")
        worksheet.cell(6, 1, "COLA Window Exponent:")
        worksheet.cell(
            6,
            3,
            "# Exponent Applied to the COLA Window (use 0.5 unless you "
            "are sure you don't want to!)",
        )
        worksheet.cell(7, 1, "Update System ID During Control:")
        worksheet.cell(
            7,
            3,
            "# Continue updating transfer function while the controller is controlling (Y/N)",
        )
        worksheet.cell(8, 1, "Frames in CPSD:")
        worksheet.cell(8, 3, "# Frames used to compute the CPSD matrix")
        worksheet.cell(9, 1, "CPSD Window:")
        worksheet.cell(9, 3, "# Window used to compute the CPSD matrix")
        worksheet.cell(10, 1, "CPSD Overlap %:")
        worksheet.cell(10, 3, "# Overlap percentage for CPSD calculations")
        worksheet.cell(11, 1, "Percent Lines Out")
        worksheet.cell(
            11,
            3,
            "# The Percent of Frequency Lines that the Signal Must be Over for Warning/Abort Levels",
        )
        worksheet.cell(12, 1, "Allow Automatic Aborts")
        worksheet.cell(
            12,
            3,
            "# Shut down the test automatically if an abort level is reached (Y/N)",
        )
        worksheet.cell(13, 1, "Control Python Script:")
        worksheet.cell(13, 3, "# Path to the Python script containing the control law")
        worksheet.cell(14, 1, "Control Python Function:")
        worksheet.cell(
            14,
            3,
            "# Function or class name within the Python Script that will serve as the control law",
        )
        worksheet.cell(15, 1, "Control Parameters:")
        worksheet.cell(15, 3, '# key=value pairs, one per cell, e.g. "rcond=1e-10"')
        worksheet.cell(16, 1, "Control Channels (1-based):")
        worksheet.cell(17, 1, "Sigma Clipping")
        worksheet.cell(
            17, 3, "# Standard-deviation threshold used to reject outlier data."
        )
        SysIdMetadata.create_blank_worksheet_template(worksheet, start_row=18)
        worksheet.cell(34, 1, "Specification File:")
        worksheet.cell(34, 3, "# Path to the file containing the Specification")
        worksheet.cell(35, 1, "Response Transformation Matrix:")
        worksheet.cell(
            35,
            2,
            "# Transformation matrix to apply to the response channels.  Type None if there "
            "is none.  Otherwise, make this a 2D array in the spreadsheet and move the Output "
            "Transformation Matrix line down so it will fit.  The number of columns should be the "
            "number of physical control channels.",
        )
        worksheet.cell(36, 1, "Output Transformation Matrix:")
        worksheet.cell(
            36,
            2,
            "# Transformation matrix to apply to the outputs.  Type None if there is none.  "
            "Otherwise, make this a 2D array in the spreadsheet.  The number of columns should be "
            "the number of physical output channels in the environment.",
        )

    def save_metadata_to_worksheet(
        self, worksheet: openpyxl.worksheet.worksheet.Worksheet
    ):
        super().save_metadata_to_worksheet(worksheet)

        if self.samples_per_frame is not None:
            worksheet.cell(2, 2, self.samples_per_frame)
        if self.test_level_ramp_time is not None:
            worksheet.cell(3, 2, self.test_level_ramp_time)
        if self.cola_window is not None:
            worksheet.cell(4, 2, self.cola_window)
        if self.cola_overlap is not None:
            worksheet.cell(5, 2, self.cola_overlap)
        if self.cola_window_exponent is not None:
            worksheet.cell(6, 2, self.cola_window_exponent)
        if self.update_tf_during_control is not None:
            worksheet.cell(7, 2, "Y" if self.update_tf_during_control else "N")
        if self.frames_in_cpsd is not None:
            worksheet.cell(8, 2, self.frames_in_cpsd)
        if self.cpsd_window is not None:
            worksheet.cell(9, 2, self.cpsd_window)
        if self.cpsd_overlap is not None:
            worksheet.cell(10, 2, self.cpsd_overlap)
        if self.percent_lines_out is not None:
            worksheet.cell(11, 2, self.percent_lines_out)
        if self.allow_automatic_aborts is not None:
            worksheet.cell(12, 2, "Y" if self.allow_automatic_aborts else "N")
        if self.control_python_script is not None:
            worksheet.cell(13, 2, self.control_python_script)
        if self.control_python_function is not None:
            worksheet.cell(14, 2, self.control_python_function)
        if self.control_python_function_parameters is not None:
            for idx, (key, value) in enumerate(
                self.control_python_function_parameters.items()
            ):
                worksheet.cell(15, 2 + idx, f"{key}={value}")
        if self.control_channel_indices is not None:
            for idx, channel_ind in enumerate(self.control_channel_indices):
                col_idx = idx + 2
                worksheet.cell(16, col_idx, channel_ind + 1)
        if self.sigma_clip is not None:
            worksheet.cell(17, 2, self.sigma_clip)
        self.sysid_metadata.save_metadata_to_worksheet(worksheet, start_row=18)
        if self.specification_file:
            worksheet.cell(34, 2, str(self.specification_file))
        self.save_sysid_matrix_to_worksheet(
            worksheet,
            self.response_transformation_matrix,
            self.reference_transformation_matrix,
            start_row=35,
        )

    @classmethod
    def load_metadata_from_worksheet(
        cls,
        worksheet: openpyxl.worksheet.worksheet.Worksheet,
        environment_name: str,
        channel_list_bools: List[bool],
        hardware_metadata: HardwareMetadata,
    ):
        sample_rate = hardware_metadata.sample_rate
        number_of_channels = sum(channel_list_bools)
        environment_channel_list = [
            channel
            for channel, channel_bool in zip(
                hardware_metadata.channel_list, channel_list_bools
            )
            if channel_bool
        ]

        output_channel_indices = [
            index
            for index, channel in enumerate(environment_channel_list)
            if channel.feedback_device is not None
        ]

        samples_per_frame = int(worksheet.cell(2, 2).value)
        test_level_ramp_time = float(worksheet.cell(3, 2).value)
        cola_window = worksheet.cell(4, 2).value
        cola_overlap = float(worksheet.cell(5, 2).value)
        cola_window_exponent = float(worksheet.cell(6, 2).value)
        update_tf_during_control = (
            worksheet_cell_str(worksheet.cell(7, 2).value).upper() == "Y"
        )
        frames_in_cpsd = int(worksheet.cell(8, 2).value)
        cpsd_window = worksheet.cell(9, 2).value
        cpsd_overlap = float(worksheet.cell(10, 2).value)
        percent_lines_out = float(worksheet.cell(11, 2).value)
        allow_automatic_aborts = (
            worksheet_cell_str(worksheet.cell(12, 2).value).upper() == "Y"
        )

        control_script = (
            worksheet.cell(13, 2).value
            if worksheet.cell(13, 2).value is not None
            else ""
        )
        control_object = (
            worksheet.cell(14, 2).value
            if worksheet.cell(14, 2).value is not None
            else ""
        )
        control_parameters = {}
        col = 2
        while True:
            value = worksheet.cell(15, col).value
            if value is None or (isinstance(value, str) and value.strip() == ""):
                break
            text = str(value)
            if "=" not in text:
                raise ValueError(
                    f'Invalid control parameter entry "{text}", expected key=value'
                )
            key, raw_val = text.split("=", 1)
            control_parameters[key.strip()] = _parse_scalar_string(raw_val.strip())
            col += 1

        # Find python module type
        if control_script:
            python_control_module = load_python_module(control_script)
            function = getattr(python_control_module, control_object)
            control_type = None
            if inspect.isgeneratorfunction(function):
                control_type = RandomControlLawType.GENERATOR
            elif inspect.isclass(function) and issubclass(
                function, AbstractControlLawComputation
            ):
                control_type = RandomControlLawType.INTERACTIVE_CLASS
            elif inspect.isclass(function):
                control_type = RandomControlLawType.CLASS
            else:
                control_type = RandomControlLawType.FUNCTION
        else:
            control_type = None

        control_script_data = RandomControlParameters(
            control_script=control_script,
            control_object=control_object,
            control_type=control_type,
            control_parameters=control_parameters,
        )
        control_channel_indices = []
        column_index = 2
        while True:
            channel_ind = worksheet.cell(16, column_index).value
            if channel_ind is None or (
                isinstance(channel_ind, str)
                and (channel_ind.startswith("#") or channel_ind.strip() == "")
            ):
                break
            try:
                control_channel_indices.append(int(channel_ind) - 1)
            except:
                break
            column_index += 1
        sigma_clip = float(worksheet.cell(17, 2).value)

        sysid_metadata = SysIdMetadata.load_metadata_from_worksheet(
            worksheet, hardware_metadata, 18
        )

        response_transformation_matrix, output_transformation_matrix = (
            cls.load_sysid_matrix_from_worksheet(worksheet, start_row=35)
        )

        coord_dtype = np.dtype([("node", "<u8"), ("direction", "i1")])
        if response_transformation_matrix is not None:
            control_coordinate = None
        else:
            control_coordinate = np.array(
                [
                    (
                        environment_channel_list[i].node_number,
                        _direction_map[environment_channel_list[i].node_direction],
                    )
                    for i in control_channel_indices
                ],
                dtype=coord_dtype,
            )

        metadata = cls(
            environment_name=environment_name,
            channel_list_bools=channel_list_bools,
            sample_rate=sample_rate,
            number_of_channels=number_of_channels,
            samples_per_frame=samples_per_frame,
            test_level_ramp_time=test_level_ramp_time,
            cola_window=cola_window,
            cola_overlap=cola_overlap,
            cola_window_exponent=cola_window_exponent,
            sigma_clip=sigma_clip,
            update_tf_during_control=update_tf_during_control,
            frames_in_cpsd=frames_in_cpsd,
            cpsd_window=cpsd_window,
            cpsd_overlap=cpsd_overlap,
            percent_lines_out=percent_lines_out,
            allow_automatic_aborts=allow_automatic_aborts,
            control_python_script=control_script_data.control_script,
            control_python_function=control_script_data.control_object,
            control_python_function_type=control_script_data.control_type,
            control_python_function_parameters=control_script_data.control_parameters,
            control_channel_indices=control_channel_indices,
            output_channel_indices=output_channel_indices,
            specification_frequency_lines=None,
            specification_cpsd_matrix=None,
            specification_warning_matrix=None,
            specification_abort_matrix=None,
            response_transformation_matrix=response_transformation_matrix,
            output_transformation_matrix=output_transformation_matrix,
            sysid_metadata=sysid_metadata,
        )

        # Load specification
        specification_file = worksheet.cell(34, 2).value
        if specification_file is not None:
            (
                metadata.specification_frequency_lines,
                metadata.specification_cpsd_matrix,
                metadata.specification_warning_matrix,
                metadata.specification_abort_matrix,
            ) = load_specification(
                specification_file,
                metadata.fft_lines,
                metadata.frequency_spacing,
                control_coordinate,
            )
            metadata.set_file(specification_file)

        return metadata

    # endregion


# endregion
