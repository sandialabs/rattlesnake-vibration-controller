"""
Synthetic "hardware" that allows the responses to be simulated by integrating
linear equations of motion.

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
from typing import List
from collections import defaultdict

import numpy as np
import scipy.signal as signal
import netCDF4 as nc4
import openpyxl

from rattlesnake.utilities import (
    flush_queue,
    RattlesnakeError,
    _direction_map,
    _direction_inv_map,
)
from rattlesnake.hardware.abstract_hardware import (
    HardwareMetadata,
    HardwareAcquisition,
    HardwareOutput,
)
from rattlesnake.hardware.hardware_utilities import HardwareType, Channel
from rattlesnake.user_interface.ui_utilities import HardwareAssistModules

HARDWARE_TYPE = HardwareType.SDYNPY_SYSTEM
DEBUG = False

if DEBUG:
    from glob import glob

    FILE_OUTPUT = "debug_data/sdynpy_hardware_{:}.npz"


# region Metadata
class SDynPySystemMetadata(HardwareMetadata):
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
        self._node_dict = None  # Dont set this

    # endregion

    # region Validation
    def validate(self):
        super().validate()
        # Validate node number
        self.detect_devices()
        has_physical = False
        for row, channel in enumerate(self.channel_list):
            if str(channel.node_number) not in self.valid_node_numbers:
                raise RattlesnakeError(
                    f"Invalid node number in channel table row {row+1}"
                )
            if channel.node_direction not in self.valid_node_directions(
                str(channel.node_number)
            ):
                raise RattlesnakeError(
                    f"Invalid node direction in channel table row {row+1}"
                )
            if channel.physical_device is None:
                raise RattlesnakeError(
                    f"Physical device should be 'Virtual' in channel table row {row+1}"
                )
            if (
                str(channel.channel_type).lower() == "force"
                and channel.feedback_device is None
            ):
                raise RattlesnakeError(
                    f"Force channel types require 'Virtual' feedback device in channel table row {row+1}"
                )
            if (
                str(channel.channel_type).lower()
                not in self.accepted_channel_type_strings
            ):
                raise RattlesnakeError(
                    f"Invalid channel type in channel table row {row+1}. Valid channel types include 'Displacement', 'Velocity', 'Acceleration', 'Force'"
                )
            if channel.physical_device is not None and channel.feedback_device is None:
                has_physical = True

        if not has_physical:
            raise RattlesnakeError(
                "SDynPy channel table requires atleast 1 physical device without an assigned feedback device"
            )

    @property
    def assist_mode_modules(self):
        assist_mode_modules = super().assist_mode_modules
        assist_mode_modules["node_number"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["node_direction"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["physical_device"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["channel_type"] = HardwareAssistModules.COMBOBOX
        assist_mode_modules["feedback_device"] = HardwareAssistModules.COMBOBOX
        return assist_mode_modules

    def valid_channel_dict(self, channel: Channel):
        valid_dict = super().valid_channel_dict(channel)

        if not self.node_dict:
            self.detect_devices()

        valid_dict["node_number"] = self.valid_node_numbers
        valid_dict["node_direction"] = self.valid_node_directions(channel.node_number)
        valid_dict["physical_device"] = self.valid_physical_device
        valid_dict["channel_type"] = self.valid_channel_types
        valid_dict["feedback_device"] = self.valid_feedback_device

        return valid_dict

    def detect_devices(self):
        try:
            sdynpy_system_data = {
                key: val for key, val in np.load(self.hardware_file).items()
            }
            channel_indices = {
                tuple([abs(v) for v in val]) for val in sdynpy_system_data["coordinate"]
            }
        except:
            raise RattlesnakeError("Invalid SDynPy system file")

        # Map node directions to node numbers
        self._node_dict = defaultdict(set)
        for node_num, dir_ind in channel_indices:
            node_dir = _direction_inv_map[dir_ind]
            neg_node_dir = _direction_inv_map[-dir_ind]
            self._node_dict[str(node_num)].add(node_dir)
            self._node_dict[str(node_num)].add(neg_node_dir)

    @property
    def node_dict(self):
        return self._node_dict

    @property
    def valid_node_numbers(self):
        node_numbers = list(self.node_dict.keys())
        node_numbers.sort()
        return node_numbers

    def valid_node_directions(self, node_number: str = ""):
        if node_number in list(self.node_dict.keys()):
            node_directions = list(self.node_dict[node_number])
            node_directions.sort()
        else:
            node_directions = []
        return node_directions

    @property
    def valid_channel_types(self):
        channel_types = ["Acceleration", "Velocity", "Displacement", "Force"]
        return channel_types

    @property
    def accepted_channel_type_strings(self):
        channel_types = [
            "accel",
            "acceleration",
            "acc",
            "force",
            "vel",
            "velocity",
            "disp",
            "displacement",
        ]
        return channel_types

    @property
    def valid_physical_device(self):
        return ["Virtual"]

    @property
    def valid_feedback_device(self):
        return ["Virtual"]

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
                case "Hardware File":
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


# region Acquisition
class SDynPySystemAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition

    This class defines the interfaces between the controller and the data
    acquisition portion of the hardware.  In this case, the hardware is simulated
    by integrating state space matrices derived from a SDynPy system object.
    It is run by the acquisition process, and must define how to get data from
    the test hardware into the controller.
    """

    def __init__(self, queue: mp.Queue, sleep: bool = True):
        """
        Loads in the SDynPy system file and sets initial parameters to null
        values.

        Parameters
        ----------
        system_file : str
            Path to the file containing state space the SDynPy system object
        queue : mp.Queue
            A queue that passes input data from the SDynPySystemOutput class to
            this class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the integration.
        sleep : bool
            If True, the integrator will wait the amount of time the calculation
            would have took if it were real life, which adds a realistic delay
            to simulate an actual measurement.  If False, the integration will
            proceed as fast as possible.

        Returns
        -------
        None.

        """
        self.system = None
        self.times = None
        self.state = None
        self.frame_time = None
        self.queue = queue
        self.force_buffer = None
        self.integration_oversample = None
        self.response_channels: np.ndarray
        self.output_channels: np.ndarray
        self.response_channels = None
        self.output_channels = None
        self.acquisition_delay = None
        self.sleep = sleep

    def initialize_hardware(self, test_data: SDynPySystemMetadata):
        """
        Initialize the hardware and set up channels and sampling properties

        The function must create channels on the hardware corresponding to
        the channels in the test.  It must also set the sampling rates.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        Returns
        -------
        None.

        """
        self.sdynpy_system_data = {
            key: val for key, val in np.load(test_data.hardware_file).items()
        }
        # Create a dictionary of channels for faster lookup
        self.channel_indices = {
            tuple([abs(v) for v in val]): index
            for index, val in enumerate(self.sdynpy_system_data["coordinate"])
        }
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
        # pylint: disable=invalid-name
        #        print('{:} Channels'.format(len(channel_data)))
        self.response_channels = np.array(
            [
                channel.feedback_device is None or channel.feedback_device == ""
                for channel in channel_data
            ],
            dtype="bool",
        )
        self.output_channels = ~self.response_channels
        # Need to add a signal buffer in case the write size is not equal to
        # the read size
        self.force_buffer = np.zeros((0, np.sum(~self.response_channels)))

        # Figure out which channels go with which indices
        channel_indices = []
        channel_signs = []
        for channel in channel_data:
            node_number = int(channel.node_number)
            direction = _direction_map[channel.node_direction]
            channel_index = self.channel_indices[(node_number, abs(direction))]
            channel_indices.append(channel_index)
            channel_signs.append(
                np.sign(direction)
                * np.sign(
                    self.sdynpy_system_data["coordinate"][channel_index]["direction"]
                )
            )
        channel_indices = np.array(channel_indices)
        channel_signs = np.array(channel_signs)

        # Now we need to actually go through and set up the A, B, C, D state matrices
        M = self.sdynpy_system_data["mass"]
        C = self.sdynpy_system_data["damping"]
        K = self.sdynpy_system_data["stiffness"]

        # Now we need to pull out the transformation matrices
        phi = self.sdynpy_system_data["transformation"][channel_indices, :]
        # Multiply by the signs
        phi *= channel_signs[:, np.newaxis]

        # Separate into responses and excitations; here input is into the system
        phi_excitation = phi[self.output_channels, :].copy()
        phi_response = phi[self.response_channels, :].copy()

        # Set up some other parameters
        ndofs = M.shape[0]
        tdofs_response = phi_response.shape[0]
        tdofs_input = phi_excitation.shape[0]

        # Assembly the full state matrices

        # A = [[     0,     I],
        #      [M^-1*K,M^-1*C]]

        A_state = np.block(
            [
                [np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                [-np.linalg.solve(M, K), -np.linalg.solve(M, C)],
            ]
        )

        # B = [[     0],
        #      [  M^-1]]

        B_state = np.block(
            [[np.zeros((ndofs, tdofs_input))], [np.linalg.solve(M, phi_excitation.T)]]
        )

        # C = [[     I,     0],   # Displacements
        #      [     0,     I],   # Velocities
        #      [M^-1*K,M^-1*C],   # Accelerations
        #      [     0,     0]]   # Forces

        C_all = np.block(
            [
                [phi_response, np.zeros((tdofs_response, ndofs))],
                [np.zeros((tdofs_response, ndofs)), phi_response],
                [
                    -phi_response @ np.linalg.solve(M, K),
                    -phi_response @ np.linalg.solve(M, C),
                ],
                [np.zeros((tdofs_input, ndofs)), np.zeros((tdofs_input, ndofs))],
            ]
        )

        # D = [[     0],   # Displacements
        #      [     0],   # Velocities
        #      [  M^-1],   # Accelerations
        #      [     I]]   # Forces

        D_all = np.block(
            [
                [np.zeros((tdofs_response, tdofs_input))],
                [np.zeros((tdofs_response, tdofs_input))],
                [phi_response @ np.linalg.solve(M, phi_excitation.T)],
                [np.eye(tdofs_input)],
            ]
        )

        # Split into different types
        displacement_indices = np.arange(tdofs_response)
        velocity_indices = np.arange(tdofs_response) + tdofs_response
        acceleration_indices = np.arange(tdofs_response) + 2 * tdofs_response
        force_indices = np.arange(tdofs_input) + 3 * tdofs_response

        C_disp = C_all[displacement_indices]
        C_vel = C_all[velocity_indices]
        C_accel = C_all[acceleration_indices]
        C_force = C_all[force_indices]

        D_disp = D_all[displacement_indices]
        D_vel = D_all[velocity_indices]
        D_accel = D_all[acceleration_indices]
        D_force = D_all[force_indices]

        # Now assemble the full response C and D matrices based on the data type
        C_response = []
        D_response = []
        response_index = 0
        for i, channel in enumerate(channel_data):
            if self.output_channels[i]:
                continue
            if channel.channel_type.lower() in ["disp", "displacement"]:
                C_response.append(C_disp[response_index])
                D_response.append(D_disp[response_index])
            elif channel.channel_type.lower() in ["vel", "velocity"]:
                C_response.append(C_vel[response_index])
                D_response.append(D_vel[response_index])
            elif channel.channel_type.lower() in ["accel", "acceleration", "acc"]:
                C_response.append(C_accel[response_index])
                D_response.append(D_accel[response_index])
            else:
                print(
                    f"Unknown Channel Type for Channel {i + 1}: {channel.channel_type}"
                )
                C_response.append(C_disp[response_index])
                D_response.append(D_disp[response_index])
            response_index += 1
        C_response = np.array(C_response)
        D_response = np.array(D_response)

        # Now assemble the final C and D matrices
        C_state = np.empty((len(channel_data), C_response.shape[-1]))
        C_state[self.response_channels, :] = C_response
        C_state[self.output_channels, :] = C_force
        D_state = np.empty((len(channel_data), D_response.shape[-1]))
        D_state[self.response_channels, :] = D_response
        D_state[self.output_channels, :] = D_force
        self.system = signal.StateSpace(A_state, B_state, C_state, D_state)
        self.state = np.zeros(A_state.shape[0])
        # np.savez('SDynPy_State.npz', A=A_state, B=B_state, C = C_state, D = D_state)

    def set_parameters(self, test_data: SDynPySystemMetadata):
        """Method to set up sampling rate and other test parameters

        For the synthetic case, we will set up the integration parameters using
        the sample rates provided.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.

        """
        self.integration_oversample = test_data.output_oversample
        # Need to get one more sample than you would think because lsim doesn't bridge the gap
        # between integrations
        self.times = np.arange(
            test_data.samples_per_read * self.integration_oversample + 1
        ) / (test_data.sample_rate * self.integration_oversample)
        self.frame_time = test_data.samples_per_read / test_data.sample_rate
        self.acquisition_delay = (
            test_data.samples_per_write / test_data.output_oversample
        )

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
                # If we don't get an output in time, this likely means output has stopped
                # so just put zeros.
                print("Warning! SDynPy integrator ran out of samples!")
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

        if DEBUG:
            num_files = len(glob(FILE_OUTPUT.format("*")))
            np.savez(
                FILE_OUTPUT.format(num_files),
                force_in=this_force.T,
                response_out_full_resolution=sys_out.T[
                    ..., : -1 : self.integration_oversample
                ],
                response_out_downsampled=sys_out.T[..., :-1],
            )

        integration_time = time.time() - start_time
        remaining_time = self.frame_time - integration_time
        if remaining_time > 0.0 and self.sleep:
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


# region Output
class SDynPySystemOutput(HardwareOutput):
    """Class defining the interface between the controller and synthetic output

    Note that the only thing that this class does is pass data to the acquisition
    hardware task which actually performs the integration.  Therefore, many of
    the functions here are actually empty."""

    def __init__(self, queue: mp.Queue):
        """
        Initializes the hardware by simply storing the data passing queue.

        Parameters
        ----------
        queue : mp.Queue
            Queue used to pass data from output to acquisition for integration.
            See ``StateSpaceAcquisition.__init__``

        """
        self.queue = queue

    def initialize_hardware(self, test_data: SDynPySystemMetadata):
        """
        Initialize the hardware and set up sources and sampling properties

        This does nothing for the synthetic hardware

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
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
