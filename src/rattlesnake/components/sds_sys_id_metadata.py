"""
This file defines the metadata that describes a MIMO Shock environment

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

from enum import Enum

import netCDF4 as nc4
import numpy as np

from .abstract_sysid_environment import AbstractSysIdMetadata


class ToneStrategy(Enum):
    """Enumeration specifying different ways of specifying sine tones"""

    FROM_SPEC = 0
    OCTAVE = 1
    MANUAL = 2


class ToneParameters:
    """Class to contain tone information"""

    def __init__(self, tone_strategy: ToneStrategy, tone_data: None | np.ndarray):
        """Initialize tone data information for storing with metadata

        Parameters
        ----------
        tone_strategy : ToneStrategy
            A tone strategy describing how to interpret the `tone_data` argument.
        tone_data : None | np.ndarray
            If `tone_strategy` is FROM_SPEC, this should be None and will be ignored.
            If `tone_stragegy` is OCTAVE, then this should be a size 3 NumPy array with values
            [min_frequency, max_frequency, tones_per_octave].
            If `tone_strategy` is MANUAL, then this should be a 1D NumPy array consisting of
            the individual sine tones, not including the compensation pulse if used.
        """
        if not isinstance(tone_strategy, ToneStrategy):
            raise ValueError("`tone_strategy` must be one of the `ToneStrategy` enumeration values")
        self.tone_strategy = tone_strategy
        if self.tone_strategy == ToneStrategy.FROM_SPEC:
            self.tone_data = None
        elif self.tone_strategy == ToneStrategy.OCTAVE:
            tone_data = np.array(tone_data).flatten()
            if tone_data.size != 3:
                raise ValueError("`tone_data` must be length-3 for `OCTAVE` tone strategy.")
            self.tone_data = tone_data
        else:
            self.tone_data = np.array(tone_data).flatten()


class CompPulseParameters:
    """Class to contain compenstation pulse data"""

    def __init__(
        self,
        use_compensation_pulse: bool,
        compensation_frequency: None | float = None,
        compensation_decay: None | float = None,
    ):
        """Initialize compensation pulse information for storing with metadata

        Parameters
        ----------
        use_compensation_pulse : bool
            True if a compensation pulse is used.
        compensation_frequency : None | float
            The frequency at which the compensation tone is defined.
            If None, it will be selected automatically.   Will be ignored if not used.
        compensation_decay : None | float
            The compensation decay as a fraction (0.95) rather than a percentage (95%).  Can be
            None if a compensation pulse is not used.  Will be ignored if not used.
        """
        self.use_compensation_pulse = use_compensation_pulse
        if not self.use_compensation_pulse:
            compensation_frequency = None
            compensation_decay = None
        self.compensation_frequency = compensation_frequency
        self.compensation_decay = compensation_decay


class DecayStrategy(Enum):
    """Enumeration containing different ways to define decay in a sum of decayed sines table"""

    DAMPING = 0
    TIME_CONSTANT = 1
    NUM_TIME_CONSTANTS = 2


class DecayParameters:
    """A class to store data defining how the sine terms decay"""

    def __init__(
        self, decay_strategy: DecayStrategy, common_decay: bool, decay_data: float | np.ndarray
    ):
        """Initializes the decay data

        Parameters
        ----------
        decay_strategy : DecayStrategy
            The strategy used to define the decay parameters in the sum of decayed sine table
        common_decay : bool
            If True, a single decay parameter is used for all tones.  If False, one must be
            specified for each sine tone
        decay_data : float | np.ndarray
            If `common_decay` is True, this should be a single floating point number defining the
            decay value.  If `common_decay` is False, this should be a np.ndarray with the same
            number of values as sine tone frequencies in the environment.
        """
        self.decay_strategy = decay_strategy
        self.common_decay = common_decay
        self.decay_data = np.array(decay_data).flatten()


class SRSType(Enum):
    """Enumeration containing different ways to compute an SRS"""

    PRIMARY_POS = 1
    PRIMARY_NEG = 2
    PRIMARY_ABSMAX = 3
    RESIDUAL_POS = 4
    RESIDUAL_NEG = 5
    RESIDUAL_ABSMAX = 6
    MAXIMUM_POS = 7
    MAXIMUM_NEG = 8
    MAXIMUM_ABSMAX = 9


class SRSDisplacementType(Enum):
    """Enumeration containing different types of displacement to compute SRSs from"""

    ABSOLUTE = 1
    RELATIVE = -1


class SRSParameters:
    """Class defining parameters for how SRSs are computed."""

    def __init__(
        self, srs_type: SRSType, srs_displacement: SRSDisplacementType, srs_damping: float
    ):
        """Initializes an object to store SRS computation parameters

        Parameters
        ----------
        srs_type : SRSType
            The method for computing maximum response for the SRS
        srs_displacement : SRSDisplacementType
            The response type to compute maximums from
        srs_damping : float
            The damping value used for the single-degree-of-freedom oscilators in the SRS.
        """
        self.srs_type = srs_type
        self.srs_displacement = srs_displacement
        self.srs_damping = srs_damping


class SDSParameters:
    """Class defining how sum-of-decayed-sine tables are computed"""

    def __init__(
        self, iterations: int, convergence: float, scale_factor: float, error_tolerance: float
    ):
        self.iterations = iterations
        self.convervence = convergence
        self.scale_factor = scale_factor
        self.error_tolerance = error_tolerance


class SpecParameters:
    """A class for storing SRS values that define the environment specification"""

    def __init__(
        self,
        frequencies: np.ndarray,
        srs_spec: np.ndarray,
        srs_lower_limit: np.ndarray,
        srs_upper_limit: np.ndarray,
        num_hits: int,
    ):
        """Initializes specification parameters

        Parameters
        ----------
        frequencies : np.ndarray
            The frequency tones defined in the specification.  This should be a 1D numpy array
            with each entry representing a different frequency value.
        srs_spec : np.ndarray
            The values of the SRS at each frequency defined in the `frequencies` array for each
            of the control degrees of freedom (physical or virtual) in the environment.  This should
            be 2-dimensional, with the rows equal to the number of frequencies and the columns equal
            to the number of control channels.  If control is not required for specific sine tones
            or channels, a NaN can be placed at that row and column.
        srs_lower_limit : np.ndarray
            The values of the lower limit SRS at each frequency defined in the `frequencies` array
            for each
            of the control degrees of freedom (physical or virtual) in the environment.  This should
            be 2-dimensional, with the rows equal to the number of frequencies and the columns equal
            to the number of control channels.  If a limit is not required for specific sine tones
            or channels, a NaN can be placed at that row and column.
        srs_upper_limit : np.ndarray
            The values of the upper limit SRS at each frequency defined in the `frequencies` array
            for each
            of the control degrees of freedom (physical or virtual) in the environment.  This should
            be 2-dimensional, with the rows equal to the number of frequencies and the columns equal
            to the number of control channels.  If a limit is not required for specific sine tones
            or channels, a NaN can be placed at that row and column.
        num_hits : int
            The number of hits to apply to the test article in the environment.
        """
        self.frequencies = frequencies
        self.srs_spec = srs_spec
        self.srs_lower_limit = srs_lower_limit
        self.srs_upper_limit = srs_upper_limit
        self.num_hits = num_hits


class ControlLawType(Enum):
    """Enumeration containing acceptable types of objects to use for a control law"""

    FUNCTION = 0
    GENERATOR = 1
    CLASS = 2
    INTERACTIVE_CLASS = 3


class ControlParameters:
    """Class to store control law data"""

    def __init__(
        self,
        control_script: str | None,
        control_object: str | None,
        control_type: ControlLawType | None,
        control_parameters: str | None,
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
        control_parameters : str
            Any extra parameters to pass to the control law.
        """
        self.control_script = control_script
        self.control_object = control_object
        self.control_type = control_type
        self.control_parameters = control_parameters


from .sds_sys_id_utilities import octspace, convert_damping_strategy


class SDSMetadata(AbstractSysIdMetadata):
    """Metadata required to define a Shock control law in rattlesnake."""

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        block_size: int,
        tone_data: ToneParameters,
        compensation_pulse_data: CompPulseParameters,
        decay_data: DecayParameters,
        srs_data: SRSParameters,
        sds_data: SDSParameters,
        control_script_data: ControlParameters,
        control_channel_indices: np.ndarray,
        output_channel_indices: np.ndarray,
        response_transformation_matrix: None | np.ndarray,
        excitation_transformation_matrix: None | np.ndarray,
        specification_data: SpecParameters,
    ):
        super().__init__()
        self.block_size = block_size
        self.number_of_channels = num_channels
        self.compensation_pulse_data = compensation_pulse_data
        self.decay_data = decay_data
        self.srs_data = srs_data
        self.sds_data = sds_data
        self.tone_data = tone_data
        self.sample_rate = sample_rate
        self.control_script_data = control_script_data
        self.control_channel_indices = control_channel_indices
        self.output_channel_indices = output_channel_indices
        self.response_transformation_matrix = response_transformation_matrix
        self.reference_transformation_matrix = excitation_transformation_matrix
        self.specification_data = specification_data

    @property
    def number_of_channels(self):
        """Total number of channels in the environment"""
        return self._number_of_channels

    @number_of_channels.setter
    def number_of_channels(self, value):
        """Sets the total number of channels in the environment"""
        self._number_of_channels = value

    @property
    def response_channel_indices(self):
        """Indices identifying which channels are control channels"""
        return self.control_channel_indices

    @property
    def reference_channel_indices(self):
        """Indices identifying which channels are reference or excitation channels"""
        return self.output_channel_indices

    @property
    def response_transformation_matrix(self):
        """Transformation matrix applied to the control channels"""
        return self._response_transformation_matrix

    @response_transformation_matrix.setter
    def response_transformation_matrix(self, value):
        """Sets the transformation matrix for the control channels"""
        self._response_transformation_matrix = value

    @property
    def reference_transformation_matrix(self):
        """Transformation matrix applied to the excitation channels"""
        return self._reference_transformation_matrix

    @reference_transformation_matrix.setter
    def reference_transformation_matrix(self, value):
        """Sets the transformation matrix applied to the excitation channels"""
        self._reference_transformation_matrix = value

    @property
    def sample_rate(self):
        """Gets the sample rate of the data acquisition system"""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        """Sets the sample rate of the data acquisition system"""
        self._sample_rate = value

    def store_to_netcdf(
        self, netcdf_group_handle: nc4._netCDF4.Group  # pylint: disable=c-extension-no-member
    ):
        """Stores the metadata in a netcdf group

        Parameters
        ----------
        netcdf_group_handle : nc4._netCDF4.Group
            A group in a NetCDF4 group defining the environment's medatadata
        """
        super().store_to_netcdf(netcdf_group_handle)
        # netcdf_group_handle.control_python_script = self.control_python_script
        # netcdf_group_handle.control_python_function = self.control_python_function
        # netcdf_group_handle.control_python_function_type = self.control_python_function_type
        # netcdf_group_handle.control_python_function_parameters = (
        #     self.control_python_function_parameters
        # )
        netcdf_group_handle.block_size = self.block_size
        # Create groups for different portions of the metadata
        tone_grp = netcdf_group_handle.createGroup("tone_parameters")
        decay_grp = netcdf_group_handle.createGroup("decay_parameters")
        srs_grp = netcdf_group_handle.createGroup("srs_parameters")
        comp_grp = netcdf_group_handle.createGroup("compensation_pulse_parameters")
        control_grp = netcdf_group_handle.createGroup("control_parameters")
        spec_grp = netcdf_group_handle.createGroup("specification_parameters")
        # Tone group
        tone_grp.strategy = self.tone_data.tone_strategy.value
        if self.tone_data.tone_data is not None:
            tone_grp.createDimension("tone_data_size", self.tone_data.tone_data)
            var = tone_grp.createVariable("tone_data", "f8", ("tone_data_size"))
            var[...] = self.tone_data.tone_data
        # Compensation pulse
        comp_grp.use_compensation_pulse = self.compensation_pulse_data.use_compensation_pulse
        if self.compensation_pulse_data.compensation_frequency is not None:
            comp_grp.compensation_frequency = self.compensation_pulse_data.compensation_frequency
        if self.compensation_pulse_data.compensation_decay is not None:
            comp_grp.compensation_decay = self.compensation_pulse_data.compensation_decay
        # Decay parameters
        decay_grp.decay_strategy = self.decay_data.decay_strategy.value
        decay_grp.common_decay = self.decay_data.common_decay
        if self.decay_data.common_decay:
            decay_grp.decay_data = self.decay_data.decay_data
        else:
            decay_grp.createDimension("num_decays", self.decay_data.decay_data.size)
            var = decay_grp.createVariable("decay_data", "f8", ("num_decays"))
            var[...] = self.decay_data.decay_data
        # SRS Group
        srs_grp.srs_type = self.srs_data.srs_type.value
        srs_grp.srs_displacement = self.srs_data.srs_displacement.value
        srs_grp.srs_damping = self.srs_data.srs_damping
        # Specification
        spec_grp.num_hits = self.specification_data.num_hits
        spec_grp.createDimension("num_frequencies", self.specification_data.frequencies.size)
        spec_grp.createDimension("num_spec_signals", self.specification_data.srs_spec.shape[1])
        var = spec_grp.createVariable("frequencies", "f8", ("num_frequencies"))
        var[...] = self.specification_data.frequencies
        var = spec_grp.createVariable("srs_spec", "f8", ("num_frequencies", "num_spec_signals"))
        var[...] = self.specification_data.srs_spec
        var = spec_grp.createVariable(
            "srs_lower_limit", "f8", ("num_frequencies", "num_spec_signals")
        )
        var[...] = self.specification_data.srs_lower_limit
        var = spec_grp.createVariable(
            "srs_upper_limit", "f8", ("num_frequencies", "num_spec_signals")
        )
        var[...] = self.specification_data.srs_upper_limit
        # Control group
        control_grp.control_type = self.control_script_data.control_type
        control_grp.control_script = self.control_script_data.control_script
        control_grp.control_object = self.control_script_data.control_object
        control_grp.control_parameters = self.control_script_data.control_parameters
        netcdf_group_handle.createDimension("control_channels", len(self.control_channel_indices))
        if self.response_transformation_matrix is None:
            netcdf_group_handle.createDimension(
                "specification_channels", len(self.control_channel_indices)
            )
        else:
            netcdf_group_handle.createDimension(
                "specification_channels", self.response_transformation_matrix.shape[0]
            )
        # Control Channels
        var = netcdf_group_handle.createVariable(
            "control_channel_indices", "i4", ("control_channels")
        )
        var[...] = self.control_channel_indices
        # Transformation Matrix
        if self.response_transformation_matrix is not None:
            var = netcdf_group_handle.createVariable(
                "response_transformation_matrix",
                "f8",
                ("specification_channels", "control_channels"),
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

    def get_sds_frequencies(self):
        if self.tone_data.tone_strategy == ToneStrategy.FROM_SPEC:
            return self.specification_data.frequencies
        if self.tone_data.tone_strategy == ToneStrategy.OCTAVE:
            return octspace(*self.tone_data.tone_data)
        if self.tone_data.tone_strategy == ToneStrategy.MANUAL:
            return self.tone_data.tone_data

    def get_sds_decays(self):
        frequencies = self.get_sds_frequencies()
        if self.decay_data.common_decay:
            decay_values = self.decay_data.decay_data[0] * np.ones(len(frequencies))
        else:
            decay_values = self.decay_data.decay_data
        return convert_damping_strategy(
            decay_values,
            frequencies,
            self.block_size / self.sample_rate,
            self.decay_data.decay_strategy,
            DecayStrategy.DAMPING,
        )
