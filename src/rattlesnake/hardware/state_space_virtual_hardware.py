# -*- coding: utf-8 -*-
"""
Synthetic "hardware" that allows the responses to be simulated by integrating
linear equations of motion using state space matrices, A, B, C, and D.

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
import os
import time
from typing import List

import openpyxl
import netCDF4 as nc4
import numpy as np
import scipy.signal as signal
from scipy.io import loadmat

from rattlesnake.hardware.abstract_hardware import (
    HardwareMetadata,
    HardwareAcquisition,
    HardwareOutput,
)
from rattlesnake.utilities import flush_queue, RattlesnakeError
from rattlesnake.hardware.hardware_utilities import Channel, HardwareType
from rattlesnake.user_interface.ui_utilities import HardwareAssistModules

HARDWARE_TYPE = HardwareType.STATE_SPACE


# region Metadata
class StateSpaceMetadata(HardwareMetadata):
    def __init__(
        self,
        channel_list: List[Channel],
        sample_rate: int,
        time_per_read: float,
        time_per_write: float,
        output_oversample: int,
        hardware_file: str,
    ):
        super().__init__(
            HARDWARE_TYPE,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample=output_oversample,
        )
        self.hardware_file = hardware_file

    # region Validation
    def validate(self):
        super().validate()
        a_matrix, b_matrix, c_matrix, d_matrix = self._load_state_space_matrices()

        if a_matrix.ndim != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
            raise RattlesnakeError(
                f"State space matrix 'A' in {self.hardware_file} must be "
                f"square, got shape {a_matrix.shape}"
            )
        n_states = a_matrix.shape[0]
        if b_matrix.ndim != 2 or b_matrix.shape[0] != n_states:
            raise RattlesnakeError(
                f"State space matrix 'B' in {self.hardware_file} must have "
                f"{n_states} rows to match 'A', got shape {b_matrix.shape}"
            )
        if c_matrix.ndim != 2 or c_matrix.shape[1] != n_states:
            raise RattlesnakeError(
                f"State space matrix 'C' in {self.hardware_file} must have "
                f"{n_states} columns to match 'A', got shape {c_matrix.shape}"
            )
        n_outputs = c_matrix.shape[0]
        n_inputs = b_matrix.shape[1]
        if d_matrix.shape != (n_outputs, n_inputs):
            raise RattlesnakeError(
                f"State space matrix 'D' in {self.hardware_file} must have "
                f"shape ({n_outputs}, {n_inputs}) to match 'B' and 'C', got "
                f"shape {d_matrix.shape}"
            )

        for row, channel in enumerate(self.channel_list):
            # TODO: Discuss with Cody: I want to be able to mirror real channel tables
            # with state space hardware.
            # if channel.physical_device not in self.valid_physical_device:
            #     raise RattlesnakeError(
            #         f"Physical device should be 'Virtual' in channel table row {row + 1}"
            #     )
            if str(channel.channel_type).lower() not in self.accepted_channel_type_strings:
                raise RattlesnakeError(
                    f"Invalid channel type in channel table row {row + 1}. "
                    f"Valid channel types include {', '.join(self.valid_channel_types)}"
                )
            # TODO: Discuss with Cody: you can construct a state-space system
            # where the inputs are voltages if you use shaker models.
            # if str(
            #     channel.channel_type
            # ).lower() == "force" and not self._has_feedback_device(channel):
            #     raise RattlesnakeError(
            #         "Force channel types require an 'Input' feedback device "
            #         f"in channel table row {row+1}"
            #     )
            # TODO: Discuss with Cody: I want to be able to mirror real channel tables
            # with state space hardware.
            # if (
            #     self._has_feedback_device(channel)
            #     and channel.feedback_device not in self.valid_feedback_device
            # ):
            #     raise RattlesnakeError(
            #         f"Invalid feedback device in channel table row {row + 1}. "
            #         "Valid feedback devices include 'Input' or blank"
            #     )

        if len(self.channel_list) != n_outputs:
            raise RattlesnakeError(
                f"Channel table has {len(self.channel_list)} channels but "
                f"the state space 'C'/'D' matrices in {self.hardware_file} "
                f"define {n_outputs} outputs; they must match"
            )

        n_force_channels = sum(
            1 for channel in self.channel_list if self._has_feedback_device(channel)
        )
        if n_force_channels != n_inputs:
            raise RattlesnakeError(
                f"Channel table has {n_force_channels} channels with a "
                f"feedback device assigned but the state space 'B'/'D' "
                f"matrices in {self.hardware_file} define {n_inputs} inputs; "
                "they must match"
            )

    @staticmethod
    def _has_feedback_device(channel: Channel):
        return channel.feedback_device is not None and channel.feedback_device != ""

    def _load_state_space_matrices(self):
        """Loads the A, B, C, and D matrices from ``self.hardware_file``,
        mirroring the loading logic in ``StateSpaceAcquisition.initialize_hardware``."""
        _, extension = os.path.splitext(str(self.hardware_file))
        if extension.lower() == ".npz":
            try:
                data = np.load(self.hardware_file)
            except:
                raise RattlesnakeError(f"Invalid state space file {self.hardware_file}")
        elif extension.lower() == ".mat":
            try:
                data = loadmat(self.hardware_file)
            except:
                raise RattlesnakeError(f"Invalid state space file {self.hardware_file}")
        else:
            raise RattlesnakeError(
                f"Unknown extension on state space file {self.hardware_file}, "
                f"should be .npz or .mat, not {extension}"
            )

        try:
            a_matrix = np.asarray(data["A"])
            b_matrix = np.asarray(data["B"])
            c_matrix = np.asarray(data["C"])
            d_matrix = np.asarray(data["D"])
        except KeyError as e:
            raise RattlesnakeError(
                f"State space file {self.hardware_file} is missing matrix {e}"
            ) from e

        return a_matrix, b_matrix, c_matrix, d_matrix

    @property
    def accepted_channel_type_strings(self):
        return [
            "accel",
            "acceleration",
            "acc",
            "force",
            "disp",
            "displacement",
            "velocity",
            "vel",
            "voltage",
            "volt",
            "current",
            "stress",
            "strain",
        ]

    @property
    def assist_mode_modules(self):
        assist_mode_modules = super().assist_mode_modules
        assist_mode_modules["physical_device"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["channel_type"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["feedback_device"] = HardwareAssistModules.COMBOBOX
        return assist_mode_modules

    def valid_channel_dict(self, channel: Channel):
        valid_dict = super().valid_channel_dict(channel)

        # TODO: Discuss with Cody what these do?
        # valid_dict["physical_device"] = self.valid_physical_device
        valid_dict["channel_type"] = self.valid_channel_types
        # valid_dict["feedback_device"] = self.valid_feedback_device

        return valid_dict

    # TODO: Discuss with Cody: I added a few different types of channels based on
    # what you could concievably put into a state-space model.  Even this might be
    # a bit too constrained.
    @property
    def valid_channel_types(self):
        return [
            "Acceleration",
            "Force",
            "Displacement",
            "Velocity",
            "Voltage",
            "Current",
            "Strain",
            "Stress",
        ]

    # TODO: Discuss with Cody: I kind of want state space to be able to mirror other
    # hardware devices so we could use it in teaching situations where I want the
    # students' channel tables to mirror mine.
    # @property
    # def valid_physical_device(self):
    #     return ["Virtual"]
    # @property
    # def valid_feedback_device(self):
    #     return ["Input"]

    # endregion

    # region Loading
    def save_metadata_to_netcdf(self, netcdf_dataset: nc4.Dataset):
        super().save_metadata_to_netcdf(netcdf_dataset)

        netcdf_dataset.hardware_file = self.hardware_file

    @classmethod
    def load_metadata_from_netcdf(cls, netcdf_dataset: nc4.Dataset):
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_netcdf(netcdf_dataset)

        hardware_file = netcdf_dataset.hardware_file

        return cls(
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
            hardware_file,
        )

    def save_metadata_to_workbook(self, workbook: openpyxl.workbook.workbook.Workbook):
        super().save_metadata_to_workbook(workbook)

        hardware_worksheet = workbook["Hardware"]
        hardware_worksheet.cell(2, 2, self.hardware_file)
        hardware_worksheet.cell(7, 2, self.output_oversample)

    @classmethod
    def load_metadata_from_workbook(cls, workbook: openpyxl.workbook.workbook.Workbook):
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_workbook(workbook)

        hardware_file = None

        hardware_worksheet = workbook["Hardware"]
        for row in hardware_worksheet.rows:
            name = str(row[0].value).lower().strip().replace(" ", "_")
            value = row[1].value
            if value is None or value == "":
                continue
            match name:
                case "hardware_file":
                    hardware_file = value
                case _:
                    continue

        return cls(
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
            hardware_file,
        )

    # endregion


# endregion


# region Acqusition
class StateSpaceAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition

    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  In this case, the hardware is
    actually simulated by integrating state space matrices, A, B, C, and D.
    It is run by the Acquisition process, and must define how to get data from
    the test hardware into the controller.
    """

    def __init__(self, ping_alive_event: mp.synchronize.Event, queue: mp.queues.Queue):
        """Loads in the state space file and sets initial parameters to null values


        Parameters
        ----------
        state_space_file : str :
            Path to the file containing state space matrices A, B, C, and D.
        queue : mp.queues.Queue
            A queue that passes input data from the StateSpaceOutput class to
            this class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the integration.

        """
        self.frame_time = None
        self.queue = queue
        self.force_buffer = None
        self.integration_oversample = None
        self.acquisition_delay = None
        self.response_channels: np.ndarray
        self.response_channels = None

    def initialize_hardware(self, test_data: StateSpaceMetadata):
        """
        Initialize the hardware and set up channels and sampling properties

        The function must create channels on the hardware corresponding to
        the channels in the test.  It must also set the sampling rates.

        Parameters
        ----------
        test_data : HardwareMetadata :
            A container containing the data acquisition parameters for the
            controller set by the user.
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        Returns
        -------
        None.

        """
        _, extension = os.path.splitext(test_data.hardware_file)

        if extension.lower() == ".npz":
            data = np.load(test_data.hardware_file)
        elif extension.lower() == ".mat":
            data = loadmat(test_data.hardware_file)
        else:
            raise ValueError(
                f"Unknown extension to file {test_data.hardware_file}, "
                f"should be .npz or .mat, not {extension}"
            )
        self.system = signal.StateSpace(data["A"], data["B"], data["C"], data["D"])
        self.times = None
        self.state = np.zeros(data["A"].shape[0])

        self.create_response_channels(test_data.channel_list)
        self.set_parameters(test_data)

    def create_response_channels(self, channel_data: List[Channel]):
        """Method to set up response channels

        This function takes channels from the supplied list of channels and
        extracts the mode shape coefficients corresponding to those channels.

        Parameters
        ----------
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        """
        #        print('{:} Channels'.format(len(channel_data)))
        self.response_channels = np.array(
            [
                channel.feedback_device is None or channel.feedback_device == ""
                for channel in channel_data
            ],
            dtype="bool",
        )
        # Need to add a signal buffer in case the write size is not equal to
        # the read size
        self.force_buffer = np.zeros((0, np.sum(~self.response_channels)))

    def set_parameters(self, test_data: HardwareMetadata):
        """Method to set up sampling rate and other test parameters

        For the synthetic case, we will set up the integration parameters using
        the sample rates provided.

        Parameters
        ----------
        test_data : HardwareMetadata :
            A container containing the data acquisition parameters for the
            controller set by the user.

        """
        self.integration_oversample = test_data.output_oversample
        # Need to get one more sample than you would think because lsim doesn't bridge the gap
        # between integrations
        self.times = np.arange(test_data.samples_per_read * self.integration_oversample + 1) / (
            test_data.sample_rate * self.integration_oversample
        )
        self.frame_time = test_data.samples_per_read / test_data.sample_rate
        self.acquisition_delay = test_data.samples_per_write / test_data.output_oversample

    def start(self):
        """Method to start acquiring data.

        For the synthetic case, it simply initializes the state of the system to zero"""
        self.state[:] = 0

    def get_acquisition_delay(self) -> int:
        """
        Get the number of samples between output and acquisition.

        This function returns the number of samples that need to be read to
        ensure that the last output is read by the acquisition.  If there is
        buffering in the output, this delay should be adjusted accordingly.

        Returns
        -------
        int
            Number of samples between when a dataset is written to the output
            and when it has finished playing.

        """
        return self.acquisition_delay

    def read(self):
        """Method to read a frame of data from the hardware

        This function gets the force from the output queue and adds it to the
        buffer of time signals that represents the force.  It then integrates
        a frame of time and sends it to the acquisition.

        Returns
        -------
        read_data :
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``

        """
        start_time = time.time()
        while self.force_buffer.shape[0] < self.times.size:
            try:
                forces = self.queue.get(timeout=self.frame_time)
            except mp.queues.Empty:
                # If we don't get an output in time, this likely means output
                # has stopped so just put zeros.
                forces = np.zeros((self.force_buffer.shape[-1], self.times.size))
            self.force_buffer = np.concatenate((self.force_buffer, forces.T), axis=0)

        # Now extract a force that is the correct size
        this_force = self.force_buffer[: self.times.size]
        # And leave the rest for next time
        # Note we have to keep the last force sample still on the
        # buffer because it will be the next force sample we use
        self.force_buffer = self.force_buffer[self.times.size - 1 :]

        _, sys_out, x_out = signal.lsim(self.system, this_force, self.times, self.state)

        self.state[:] = x_out[-1]

        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)

        # We don't want to return the last sample because it
        # will be the initial state for the next sample
        return sys_out.T[..., : -1 : self.integration_oversample]

    def read_remaining(self):
        """Method to read the rest of the data on the acquisition

        This function simply returns one sample of zeros.

        Returns
        -------
        read_data :
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``
        """
        return np.zeros((len(self.response_channels), 1))

    def stop(self):
        """Method to stop the acquisition.

        This simply sets the state to zero."""
        self.state[:] = 0

    def close(self):
        """Method to close down the hardware"""


# endregion


# region Output
class StateSpaceOutput(HardwareOutput):
    """Class defining the interface between the controller and synthetic output

    Note that the only thing that this class does is pass data to the acquisition
    hardware task which actually performs the integration.  Therefore, many of
    the functions here are actually empty."""

    def __init__(self, ping_alive_event: mp.synchronize.Event, queue: mp.queues.Queue):
        """
        Initializes the hardware by simply storing the data passing queue.

        Parameters
        ----------
        queue : mp.queues.Queue
            Queue used to pass data from output to acquisition for integration.
            See ``StateSpaceAcquisition.__init__``

        """
        self.queue = queue

    def initialize_hardware(self, test_data: StateSpaceMetadata):
        """
        Initialize the hardware and set up sources and sampling properties

        This does nothing for the synthetic hardware

        Parameters
        ----------
        test_data : HardwareMetadata :
            A container containing the data acquisition parameters for the
            controller set by the user.
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        Returns
        -------
        None.

        """

    def start(self):
        """Method to start acquiring data

        Does nothing for synthetic hardware."""

    def write(self, data: np.ndarray):
        """Method to write a frame of data

        For the synthetic excitation, this simply puts the data into the data-
        passing queue.

        Parameters
        ----------
        data : np.ndarray
            Data to write to the output.

        """
        self.queue.put(data)

    def stop(self):
        """Method to stop the acquisition

        Does nothing for synthetic hardware."""
        flush_queue(self.queue)

    def close(self):
        """Method to close down the hardware

        Does nothing for synthetic hardware."""

    def ready_for_new_output(self):
        """Signals that the hardware is ready for new output

        Returns ``True`` if the data-passing queue is empty.
        """
        return self.queue.empty()


# endregion
