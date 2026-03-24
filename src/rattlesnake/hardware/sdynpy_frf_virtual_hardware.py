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

import numpy as np

from rattlesnake.hardware.abstract_hardware import HardwareAcquisition, HardwareOutput
from rattlesnake.utilities import (
    Channel,
    DataAcquisitionParameters,
    flush_queue,
    reduce_array_by_coordinate,
)

try:
    # cupy may not exist if correct modules aren't installed
    import cupy as cp  # type: ignore
    from cupyx.scipy.signal import oaconvolve  # type: ignore

    xp = cp
    CUDA = True
except ModuleNotFoundError:
    from scipy.signal import oaconvolve

    xp = np
    CUDA = False

_direction_map = {
    "X+": 1,
    "X": 1,
    "+X": 1,
    "Y+": 2,
    "Y": 2,
    "+Y": 2,
    "Z+": 3,
    "Z": 3,
    "+Z": 3,
    "RX+": 4,
    "RX": 4,
    "+RX": 4,
    "RY+": 5,
    "RY": 5,
    "+RY": 5,
    "RZ+": 6,
    "RZ": 6,
    "+RZ": 6,
    "X-": -1,
    "-X": -1,
    "Y-": -2,
    "-Y": -2,
    "Z-": -3,
    "-Z": -3,
    "RX-": -4,
    "-RX": -4,
    "RY-": -5,
    "-RY": -5,
    "RZ-": -6,
    "-RZ": -6,
    "": 0,
    None: 0,
}


# region: Acqusition
class SDynPyFRFAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and synthetic acquisition

    This class defines the interfaces between the controller and the data
    acquisition portion of the hardware.  In this case, the hardware is simulated
    by convolving an IRF with each new frame of data, where the IRF is supplied from
    either a SDynPy TransferFunctionArray or ImpulseResponseFunctionArray object.
    It is run by the acquisition process, and must define how to get data from
    the test hardware into the controller.
    """

    def __init__(self, frf_file: str, queue: mp.queues.Queue):
        """
        Loads in the SDynPy file and sets initial parameters to null
        values.

        Parameters
        ----------
        system_file : str
            Path to the file containing state space the SDynPy object
        queue : mp.queues.Queue
            A queue that passes input data from the SDynPyFRFOutput class to
            this class.  Normally, this data transfer would occur through
            the physical test object: the exciters would excite the test object
            with the specified excitation and the Acquisition would record the
            responses to that excitation.  In the synthetic case, we need to
            pass the output data to the acquisition which does the convolution.

        Returns
        -------
        None.

        """
        self.sdynpy_data, self.function_type = np.load(frf_file).values()
        if self.function_type.item() not in [4, 29]:
            raise ValueError(
                "File must be SDynPy TransferFunctionArray or ImpulseResponseFunctionArray"
            )
        self.system = None
        self.times = None
        self.sample_rate = None
        self.samples_per_read = None
        self.samples_per_write = None
        self.frame_time = None
        self.convolution_samples = None
        self.queue = queue
        self.force_buffer = None
        self.output_signal_time = None
        self.sys_out = None
        self.integration_oversample = None
        self.response_channels: np.ndarray
        self.output_channels: np.ndarray
        self.response_channels = None
        self.output_channels = None
        self.irf = None
        self.acquisition_delay = None

    # region: Abstract Methods
    def set_up_data_acquisition_parameters_and_channels(
        self, test_data: DataAcquisitionParameters, channel_data: List[Channel]
    ):
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
        self.set_parameters(test_data)
        self.create_response_channels(channel_data)

    def create_response_channels(self, channel_data: List[Channel]):
        """Method to set up response channels

        This function takes channels from the supplied list of channels and
        extracts the mode shape coefficients corresponding to those channels.

        Parameters
        ----------
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        """
        self.response_channels = np.array(
            [
                channel.feedback_device is None or channel.feedback_device == ""
                for channel in channel_data
            ],
            dtype="bool",
        )
        self.output_channels = ~self.response_channels
        # Need to add a signal buffer in case the write size is not equal to the read size
        self.force_buffer = np.zeros((0, np.sum(~self.response_channels)))

        # Figure out which channels go with which indices
        response_coord = []
        excitation_coord = []
        for channel in channel_data:
            node_number = int(channel.node_number)
            direction = _direction_map[channel.node_direction]
            channel_coord = (node_number, direction)
            if channel.feedback_device is None or channel.feedback_device == "":
                response_coord.append(channel_coord)
            else:
                excitation_coord.append(channel_coord)
        coord_dtype = np.dtype([("node", "<u8"), ("direction", "i1")])
        response_coord = np.array(response_coord, dtype=coord_dtype)
        excitation_coord = np.array(excitation_coord, dtype=coord_dtype)

        # check for even abscissa spacing
        spacing = np.diff(self.sdynpy_data["abscissa"], axis=-1)
        mean_spacing = np.mean(spacing)
        if not np.allclose(spacing, mean_spacing):
            raise ValueError("SDynPy array does not have evenly spaced abscissa")
        # index array by coordinate. `reduce_array_by_coordinate` expects frequency on first axis
        array = reduce_array_by_coordinate(
            np.moveaxis(self.sdynpy_data["ordinate"], -1, 0),
            self.sdynpy_data["coordinate"],
            response_coord,
            excitation_coord,
        )
        # convert to irf if needed
        if self.function_type == 4:
            # compute irf and transpose so that shape becomes (nref, nresp, nsamples)
            self.irf = np.fft.irfft(array, axis=0).T
            num_samples = self.irf.shape[-1]
            dt = 1 / (self.sdynpy_data["abscissa"].max() * num_samples / np.floor(num_samples / 2))
        elif self.function_type == 29:
            self.irf = array.T
            dt = mean_spacing
        else:
            raise ValueError(
                "SDynPy FRFs should have type TransferFunctionArray or ImpulseResponseFunctionArray"
            )
        if CUDA:
            self.irf = cp.asarray(self.irf)

        # Checking to see if the transfer function sampling rate matches the acquisition rate
        if not np.isclose(self.sample_rate, 1 / dt):
            raise ValueError(
                f"The transfer function sampling rate {1 / dt} "
                f"does not match the hardware sampling rate {self.sample_rate}."
            )

        # check that all channels from channel table will have a corresponding irf
        _, number_responses, model_order = self.irf.shape
        if number_responses != np.sum(self.response_channels):
            raise ValueError(
                f"Number of responses in FRF ({number_responses}) does not "
                f"match channel table ({np.sum(self.response_channels)})"
            )
        # each frame of the convolution must include M - 1 samples of
        # previous data to maintain causality (where M is length of impulse response)
        self.convolution_samples = self.samples_per_read + model_order - 1
        # initialize convolution and output arrays (read function will overwrite rather than
        # re-allocate)
        self.output_signal_time = xp.zeros(
            (number_responses, self.convolution_samples), dtype=xp.float64
        )
        self.sys_out = np.zeros((len(channel_data), self.times.size), dtype=np.float64)

    def set_parameters(self, test_data: DataAcquisitionParameters):
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
        self.sample_rate = test_data.sample_rate
        self.times = np.arange(test_data.samples_per_read) / (test_data.sample_rate)
        self.frame_time = test_data.samples_per_read / test_data.sample_rate
        self.acquisition_delay = test_data.samples_per_write
        self.samples_per_read = test_data.samples_per_read
        self.samples_per_write = test_data.samples_per_write

    def start(self):
        """Method to start acquiring data.

        For the synthetic case, doesn't need to do anything"""

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
        buffer of time signals that represents the force. It then convolves
        a frame of time and sends it to the acquisition.

        For large datasets, computation time may exceed the acquisition
        time in which this function is expected to return. This may result in
        slower than real-time execution. GPU hardware acceleration is
        available to increase computation speed if a CuPy installation is found.
        (requires Nvidia CUDA toolkit and CUDA compatible GPU,
        see https://docs.cupy.dev/en/stable/install.html)

        Returns
        -------
        read_data :
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``

        """
        start_time = time.time()
        while self.force_buffer.shape[0] < self.convolution_samples:
            try:
                forces = self.queue.get(timeout=self.frame_time)
            except mp.queues.Empty:
                # If we don't get an output in time, this likely means output has stopped
                # so just put zeros.
                forces = np.zeros((self.force_buffer.shape[-1], self.times.size))
            self.force_buffer = np.concatenate((self.force_buffer, forces.T), axis=0)

        # Now extract a force that is the correct size (including past samples for convolution)
        this_force = self.force_buffer[: self.convolution_samples].T
        # And leave the rest for next time
        self.force_buffer = self.force_buffer[self.times.size :]

        if np.any(this_force):
            if CUDA:
                this_force = cp.asarray(this_force)
            # reset the output signal array to zero
            self.output_signal_time[:] = 0
            # Setting up and doing the convolution (using GPU if possible)
            # (see sdynpy.data.TimeHistoryArray.mimo_forward)
            for reference_irfs, inputs in zip(self.irf, this_force):
                self.output_signal_time += oaconvolve(reference_irfs, inputs[np.newaxis, :])[
                    :, : self.convolution_samples
                ]

            # assign latest frame of data to correct channels
            # (transfer from GPU to CPU if necessary)
            if CUDA:
                self.sys_out[self.response_channels, :] = self.output_signal_time[
                    :, -self.times.size :
                ].get()
                self.sys_out[self.output_channels, :] = this_force[:, -self.times.size :].get()
            else:
                self.sys_out[self.response_channels, :] = self.output_signal_time[
                    :, -self.times.size :
                ]
                self.sys_out[self.output_channels, :] = this_force[:, -self.times.size :]
        else:
            self.sys_out[:] = 0

        computation_time = time.time() - start_time
        remaining_time = self.frame_time - computation_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)

        return self.sys_out

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
        """Method to stop the acquisition."""

    def close(self):
        """Method to close down the hardware"""


# region: Output
class SDynPyFRFOutput(HardwareOutput):
    """Class defining the interface between the controller and synthetic output

    Note that the only thing that this class does is pass data to the acquisition
    hardware task which actually performs the integration.  Therefore, many of
    the functions here are actually empty."""

    def __init__(self, queue: mp.queues.Queue):
        """
        Initializes the hardware by simply storing the data passing queue.

        Parameters
        ----------
        queue : mp.queues.Queue
            Queue used to pass data from output to acquisition for integration.
            See ``StateSpaceAcquisition.__init__``

        """
        self.queue = queue

    # region: Abstract Methods
    def set_up_data_output_parameters_and_channels(
        self, test_data: DataAcquisitionParameters, channel_data: List[Channel]
    ):
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
