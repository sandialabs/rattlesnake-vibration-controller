# -*- coding: utf-8 -*-
"""
Hardware definition that allows for the Data Physics DP900 series hardware to
be run with Rattlesnake.

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

import ctypes
import datetime
from ctypes import c_bool, c_char_p, c_double, c_float, c_int
from enum import Enum

import numpy as np
from numpy.ctypeslib import ndpointer

DEBUG = False

if DEBUG:
    __log_file__ = "DataPhysics_Log.txt"
    _PRINT_MESSAGE = True
    _WRITE_MESSAGE = True
    if _WRITE_MESSAGE:
        log_file = open(__log_file__, "w", encoding="utf-8")

    def debug_fn(message):
        """Reports the message via print or log file

        Parameters
        ----------
        message : str
            The message to be reported
        """
        now = datetime.datetime.now()
        if _PRINT_MESSAGE:
            print(f"{now} -- {message}")
        if _WRITE_MESSAGE:
            log_file.write(f"{now} -- {message}\n")


class DP900Status(Enum):
    """Valid DP900 statuses"""

    DISCONNECTED = -1
    IDLE = 0
    INIT = 1
    RUNNING = 2
    STOPPED = 3


class DP900Coupling(Enum):
    """Valid DP900 Couplings"""

    AC_DIFFERENTIAL = 0
    DC_DIFFERENTIAL = 1
    AC_SINGLE_ENDED = 2
    DC_SINGLE_ENDED = 3
    AC_COUPLED_IEPE = 4


class DP900:
    """An interface to the data physics C API for the DP900 hardware"""

    def __init__(self, library_path: str):
        """
        Connects to the library

        Parameters
        ----------
        library_path : str
            Path to the Dp900Matlab.dll file that is used to run the DP900 device

        Returns
        -------
        None.

        """
        self._api = ctypes.WinDLL(library_path)
        self._valid_input_ranges = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        self._valid_output_ranges = np.array([2.0, 10.0])
        self._num_inputs = 0
        self._num_outputs = 0

        # Set up prototypes for the various function calls
        # Define the argument and return types for each function

        # DP900_API int      IsHwConnected();
        # self._api.IsHwConnected.argtypes = []
        # self._api.IsHwConnected.restype = c_int

        # DP900_API int      Connect(char* testName);
        self._api.Connect.argtypes = [c_char_p]
        self._api.Connect.restype = c_int

        # DP900_API int      Disconnect();
        self._api.Disconnect.argtypes = []
        self._api.Disconnect.restype = c_int

        # DP900_API int      SetSampleRate(double sampleRate);
        self._api.SetSampleRate.argtypes = [c_double]
        self._api.SetSampleRate.restype = c_int

        # DP900_API int      Init();
        self._api.Init.argtypes = []
        self._api.Init.restype = c_int

        # DP900_API int      SetInpParams(
        # int* coupling, int* chNum, float* sensitivity, float* range, int numInps);
        self._api.SetInpParams.argtypes = [
            ndpointer(c_int),
            ndpointer(c_int),
            ndpointer(c_float),
            ndpointer(c_float),
            c_int,
        ]
        self._api.SetInpParams.restype = c_int

        # DP900_API int      SetOutParams(
        # float* sensitivity, float* range, int* chNums,int numOuts);
        self._api.SetOutParams.argtypes = [
            ndpointer(c_float),
            ndpointer(c_float),
            ndpointer(c_int),
            c_int,
        ]
        self._api.SetOutParams.restype = c_int

        # DP900_API int      Start();
        self._api.Start.argtypes = []
        self._api.Start.restype = c_int

        # DP900_API int      Stop();
        self._api.Stop.argtypes = []
        self._api.Stop.restype = c_int

        # DP900_API int      End();
        self._api.End.argtypes = []
        self._api.End.restype = c_int

        # DP900_API int      EmergencyEnd();
        self._api.EmergencyEnd.argtypes = []
        self._api.EmergencyEnd.restype = c_int

        # DP900_API char*      GetSystemList(bool online);
        self._api.GetSystemList.argtypes = [c_bool]
        self._api.GetSystemList.restype = c_char_p

        # DP900_API char*      GetTestList();
        self._api.GetTestList.argtypes = []
        self._api.GetTestList.restype = c_char_p

        # DP900_API int       SetSystemList(char* sysList);
        self._api.SetSystemList.argtypes = [c_char_p]
        self._api.SetSystemList.restype = c_int

        # DP900_API int       SaveTest(char* testName);
        self._api.SaveTest.argtypes = [c_char_p]
        self._api.SaveTest.restype = c_int

        # DP900_API int       DeleteTest(char* testName);
        self._api.DeleteTest.argtypes = [c_char_p]
        self._api.DeleteTest.restype = c_int

        # DP900_API int      GetData(float* outputBuf, int dataType, int length);
        self._api.GetData.argtypes = [ndpointer(c_float), c_int, c_int]
        self._api.GetData.restype = c_int

        # DP900_API int      GetAvailableDataLength();
        self._api.GetAvailableDataLength.argtypes = []
        self._api.GetAvailableDataLength.restype = c_int

        # DP900_API int      GetSpaceInOutBuffer();
        self._api.GetSpaceInOutBuffer.argtypes = []
        self._api.GetSpaceInOutBuffer.restype = c_int

        # DP900_API int      GetTotalSamplesInOutputBuffer();
        self._api.GetTotalSamplesInOutputBuffer.argtypes = []
        self._api.GetTotalSamplesInOutputBuffer.restype = c_int

        # DP900_API int      PutOutData(float* outputBuf, int length);
        self._api.PutOutData.argtypes = [ndpointer(c_float), c_int]
        self._api.PutOutData.restype = c_int

        # DP900_API char*     GetErrorList();
        self._api.GetErrorList.argtypes = []
        self._api.GetErrorList.restype = c_char_p

        # DP900_API int      SetCBufSize(int buffSz);
        self._api.SetCBufSize.argtypes = [c_int]
        self._api.SetCBufSize.restype = c_int

        # DP900_API int      SetSaveRecording(int bEnabled);
        self._api.SetSaveRecording.argtypes = [c_int]
        self._api.SetSaveRecording.restype = c_int

        # DP900_API int      GetNumInpsAvailable();
        self._api.GetNumInpsAvailable.argtypes = []
        self._api.GetNumInpsAvailable.restype = c_int

        # DP900_API int      GetNumInpsSelected();
        self._api.GetNumInpsSelected.argtypes = []
        self._api.GetNumInpsSelected.restype = c_int

        # DP900_API int      GetNumOutsAvailable();
        self._api.GetNumOutsAvailable.argtypes = []
        self._api.GetNumOutsAvailable.restype = c_int

        # DP900_API int      GetNumOutsSelected();
        self._api.GetNumOutsSelected.argtypes = []
        self._api.GetNumOutsSelected.restype = c_int

        # DP900_API int      GetCBufSize();
        self._api.GetCBufSize.argtypes = []
        self._api.GetCBufSize.restype = c_int

        # DP900_API int      GetInputChannelBNCs(int* bncs);
        self._api.GetInputChannelBNCs.argtypes = [ndpointer(c_int)]
        self._api.GetInputChannelBNCs.restype = c_int

        # DP900_API int      GetOutputChannelBNCs(int* bncs);
        self._api.GetOutputChannelBNCs.argtypes = [ndpointer(c_int)]
        self._api.GetOutputChannelBNCs.restype = c_int

        # if self.is_hardware_connected():
        #     self.status = DP900Status.IDLE
        # else:
        self.status = DP900Status.DISCONNECTED

    @property
    def num_outputs(self):
        """Gets the number of output channels"""
        return self._num_outputs

    @property
    def num_inputs(self):
        """Gets the number of acquisition channels"""
        return self._num_inputs

    def raise_error(self):
        """
        Collects the data physics error and raises it in Python.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        error = self.get_error_list()
        if DEBUG:
            debug_fn(f"DP Error: {error}")
        raise RuntimeError(f"DP Error: {error}")

    # def is_hw_connected(self):
    #     """
    #     Checks if the hardware is connected

    #     Returns
    #     -------
    #     bool
    #         Returns True if the hardware is already connected, otherwise returns
    #         False.

    #     """
    #     if DEBUG:
    #         debug_fn('Calling IsHwConnected\n')
    #     return bool(self._api.IsHwConnected())

    def connect(self, test_name):
        """
        Launches background processes if not running, and connects the 900 API software to them.

        Parameters
        ----------
        test_name : str
            Pointer to a null-terminated (C-style) char array representing the test name.
            Pass an empty string (pointer to an empty char) to start from a new test.
            Unless recalling a Saved test setup (with save_test()), pass an empty string.
            Commas are not allowed in the string.

        """
        # if not self.is_hw_connected():
        if DEBUG:
            debug_fn(f"Calling Connect with\n    test_name = {test_name}")
        success = self._api.Connect(test_name.encode("utf-8"))
        # else:
        #     raise RuntimeError('Hardware is already connected')
        if not success == 1:
            self.raise_error()
        else:
            self.status = DP900Status.IDLE

    def disconnect(self):
        """
        Disconnects from the API.

        Returns
        -------
        int
            1 if disconnection to the API was successful, 0 if not successful.

        """
        # if self.is_hw_connected():
        if DEBUG:
            debug_fn("Calling Disconnect")
        success = self._api.Disconnect()
        # else:
        #     raise RuntimeError('Hardware is not connected')
        if not success == 1:
            self.raise_error()
        else:
            self.status = DP900Status.DISCONNECTED

    def set_sample_rate(self, sample_rate):
        """
        Sets the sample rate to be used for the next test. The sample rate can
        not be changed while a test is running; it must be set before the
        init() command.

        The sample rate can be arbitrarily set to any value between 256Hz and 216kHz.

        Parameters
        ----------
        sample_rate : float
            The sample rate for the test.
        """
        if DEBUG:
            debug_fn(f"Calling SetSampleRate with\n    sample_rate = {sample_rate}")
        success = self._api.SetSampleRate(sample_rate)
        if not success == 1:
            self.raise_error()

    def init(self):
        """
        Initializes the test

        Returns
        -------
        int
            0 if the command was not successful; 1 if the command was
            successful.
        """
        if self.status == DP900Status.IDLE:
            if DEBUG:
                debug_fn("Calling Init")
            success = self._api.Init()
            if not success == 1:
                self.raise_error()
            else:
                self.status = DP900Status.INIT
        else:
            raise RuntimeError(
                f"Hardware status must be IDLE to initialize.  "
                f"Current status is {self.status.name}."
            )

    def setup_input_parameters(self, coupling_array, channel_array, sensitivity_array, range_array):
        """
        Tells the API how many channels will be used; as well as the hardware
        settings for those parameters.

        Parameters
        ----------
        coupling_array : array of DP900Coupling
            Pointer to an integer array, of length num_inps representing the
            hardware coupling for each channel
        channel_array : array of integers
            Array of integers of length num_inps where the nth value represents
            the nth input channel number.  If channels are to be skipped
            (eg. enabling channels 1, 2, 5, 6) the array will contain
            non-consecutive numbers (eg [1, 2, 5, 6] with num_inps=4)
        sensitivity_array : array of float
            Pointer to a float array, of length numInps where the nth value
            represents the sensitivity of the nth input channel. After voltages
            are read from the hardware, each channel is scaled by its
            sensitivity before being recorded or passed to the API.
        range_array : array of float
            Pointer to a float array, of length numInps where the nth value
            represents the voltage range of the nth input channel.  Can be
            0.1, 0.3, 1.0, 3.0, 10.0, 30.0
        """
        # Set up the channel arrays
        if len(coupling_array) != len(sensitivity_array):
            raise ValueError("Coupling array must have same size as Sensitivity Array")
        if len(range_array) != len(sensitivity_array):
            raise ValueError("Range array must have same size as Sensitivity Array")
        if len(channel_array) != len(sensitivity_array):
            raise ValueError("Channel array must have same size as Sensitivity Array")
        self._num_inputs = len(coupling_array)
        coupling_array = np.array(
            [int(coupling.value) for coupling in coupling_array], dtype=np.int32
        )
        sensitivity_array = np.array([float(val) for val in sensitivity_array], dtype=np.float32)
        channel_array = np.array([int(channel) for channel in channel_array], dtype=np.int32)
        validated_range_array = []
        for rng in range_array:
            close_ranges = self._valid_input_ranges[np.isclose(self._valid_input_ranges, rng)]
            if len(close_ranges) == 0:
                raise ValueError(
                    f"Range {rng} is not valid.  Valid sample rates are "
                    f"{', '.join([f'{v:0.1f}' for v in self._valid_input_ranges])}"
                )
            elif len(close_ranges) > 1:
                raise ValueError(
                    f"Multiple Ranges are close to the specified rate ({rng}, {close_ranges}).  "
                    f"This shouldn't happen!"
                )
            validated_range_array.append(close_ranges[0])
        validated_range_array = np.array(validated_range_array, dtype=np.float32)
        # Call the API function
        if DEBUG:
            debug_fn(
                f"Calling SetInpParams with\n    "
                f"coupling_array = {coupling_array.tolist()}\n    "
                f"channel_array = {channel_array.tolist()}\n    "
                f"sensitivity_array = {sensitivity_array.tolist()}\n    "
                f"range_array = {validated_range_array.tolist()}\n    "
                f"num_inputs = {self._num_inputs}"
            )
        success = self._api.SetInpParams(
            coupling_array,
            channel_array,
            sensitivity_array,
            validated_range_array,
            ctypes.c_int(self._num_inputs),
        )
        if not success == 1:
            self.raise_error()

    def setup_output_parameters(self, sensitivity_array, range_array, channel_array):
        """
        Configures the settings for the output channels to use during test.

        This must be run before a test is Initialized (Init()).

        Parameters
        ----------
        sensitivity_array : array of float
            A floating-point array, where the Nth element representing the
            sensitivity (scaling) for the Nth output channel, in EU/V
            (EU=Engineering Unit).  Each sample, when acquired, is divided by
            this value. A value of 1 essentially applies no scaling (outputs
            are voltages)
        range_array : array of float
            A floating point array containing the output range for each output
            channel.
        channel_array : array of integers
            Array of integers of length num_outputs where the nth value represents
            the nth output channel number.  If channels are to be skipped
            (eg. enabling channels 1, 2, 5, 6) the array will contain
            non-consecutive numbers (eg [1, 2, 5, 6] with num_outputs=4)

        Raises
        ------
        ValueError
            If input arrays are not the same size or if ranges are not valid.

        """
        if len(range_array) != len(sensitivity_array):
            raise ValueError("Range array must have same size as Sensitivity Array")
        if len(channel_array) != len(sensitivity_array):
            raise ValueError("Channel number array must have same size as Sensitivity Array")
        self._num_outputs = len(sensitivity_array)
        sensitivity_array = np.array([float(val) for val in sensitivity_array], dtype=np.float32)
        channel_array = np.array([int(val) for val in channel_array], dtype=np.int32)
        validated_range_array = []
        for rng in range_array:
            close_ranges = self._valid_output_ranges[np.isclose(self._valid_output_ranges, rng)]
            if len(close_ranges) == 0:
                raise ValueError(
                    f"Range {rng} is not valid.  Valid sample rates are "
                    f"{', '.join([f'{v:0.1f}' for v in self._valid_output_ranges])}"
                )
            elif len(close_ranges) > 1:
                raise ValueError(
                    f"Multiple Ranges are close to the specified rate ({rng}, {close_ranges}).  "
                    f"This shouldn't happen!"
                )
            validated_range_array.append(close_ranges[0])
        validated_range_array = np.array(validated_range_array, dtype=np.float32)
        # Call the API function
        if DEBUG:
            debug_fn(
                f"Calling SetOutParams with \n    "
                f"sensitivity_array = {sensitivity_array.tolist()}\n    "
                f"range_array = {validated_range_array.tolist()}\n    "
                f"channel_array = {channel_array.tolist()}\n    "
                f"num_outputs = {self._num_outputs}"
            )
        success = self._api.SetOutParams(
            sensitivity_array,
            validated_range_array,
            channel_array,
            ctypes.c_int(self._num_outputs),
        )
        if not success == 1:
            self.raise_error()

    def start(self):
        """
        Starts data acquisition (and output through output channel). The test
        must be initialized (init()) before running start().
        """
        if self.status in [DP900Status.INIT, DP900Status.STOPPED]:
            if DEBUG:
                debug_fn("Calling Start")
            success = self._api.Start()
            if not success == 1:
                self.raise_error()
            else:
                self.status = DP900Status.RUNNING
        else:
            raise RuntimeError(
                f"Current hardware status is {self.status.name}.  Hardware must be initialized or "
                f"stopped prior to starting a measurement"
            )

    def stop(self):
        """
        Stops data acquisition (and output through output channel). The test
        must be started (start()) before running stop().
        """
        if self.status == DP900Status.RUNNING:
            if DEBUG:
                debug_fn("Calling Stop")
            success = self._api.Stop()
            if not success == 1:
                self.raise_error()
            else:
                self.status = DP900Status.STOPPED
        else:
            raise RuntimeError(
                f"Current hardware status is {self.status.name}.  Hardware must be "
                "running prior to stopping a measurement"
            )

    def end(self):
        """
        Ends the current test. The test must be stopped (stop()) before running
        Stop().
        """
        if self.status in [DP900Status.STOPPED, DP900Status.INIT]:
            if DEBUG:
                debug_fn("Calling End")
            success = self._api.End()
            if not success == 1:
                self.raise_error()
            else:
                self.status = DP900Status.IDLE

    def emergency_end(self):
        """
        If a test is Initialized or Started; but the client (your) code running
        the API crashes or goes into a bad state; the client code can be
        relaunched and execute this function to Stop/End a test that is in
        progress.

        Stops or Ends a test that is currently running, but the API is not
        connected to. This is intended to be used if your code crashes or goes
        into a bad state while the API is running a test.
        """
        if DEBUG:
            debug_fn("Calling EmergencyEnd")
        success = self._api.EmergencyEnd()
        if not success == 1:
            self.raise_error()
        else:
            self.status = DP900Status.IDLE

    def get_system_list(self, online):
        """
        Gets a list of hardware units available for selection
        in the test.

        Note: this command also returns Abacus0 blocks (Data Server), which
        can not be selected for use in a test.

        Parameters
        ----------
        online : bool
            If True, the returned list will only contain systems that are all
            online (detected by the 900 API).

        Returns
        -------
        list
            A list of systems that are selectable for use in
            the 900 software

        """
        if DEBUG:
            debug_fn(f"Calling GetSystemList with\n    online = {online}")
        return self._api.GetSystemList(online).decode("utf-8").split(",")

    def get_test_list(self):
        """
        Gets a comma separated list of tests available for opening with the
        connect() command.

        Note: This will return a list of all tests in the Manage Tests screen
        of DP900. If DP900 contains tests which were not created with the API,
        they should not be opened with the API.

        Returns
        -------
        str
            A comma separated list of tests available for opening with the
            connect() command

        """
        if DEBUG:
            debug_fn("Calling GetTestList")
        return self._api.GetTestList().decode("utf-8")

    def set_system_list(self, sys_list):
        """
        Sets the hardware units available to be used in the test.

        Parameters
        ----------
        sys_list : array of strings
            A list of the names of systems to include in the test (eg.
            ["dp912-98012","dp901-94019"]).
        """
        if isinstance(sys_list, str):
            sys_list = [sys_list]
        sys_list = ",".join(sys_list)
        if DEBUG:
            debug_fn(f"Calling SetSystemList with\n    system_list = {sys_list}")
        success = self._api.SetSystemList(sys_list.encode("utf-8"))
        if not success == 1:
            self.raise_error()

    def save_test(self, test_name):
        """
        Saves the test setup for recalling with the connect() command.

        The test will be saved in the 900 Series database, and can be recalled
        with the connect() call.

        Parameters
        ----------
        test_name : str
            String to use as the name for the test to be saved as.

        """
        if DEBUG:
            debug_fn(f"Calling SaveTest with\n    test_name = {test_name}")
        success = self._api.SaveTest(test_name.encode("utf-8"))
        if not success == 1:
            self.raise_error()

    def delete_test(self, test_name):
        """
        Deletes a test setup from the archived database.

        Parameters
        ----------
        test_name : str
            String containing the name of the test to be deleted
        """
        if DEBUG:
            debug_fn(f"Calling DeleteTest with\n    test_name = {test_name}")
        success = self._api.DeleteTest(test_name.encode("utf-8"))
        if not success == 1:
            self.raise_error()

    def read_input_data(self, num_samples, newest_data=False):
        """
        Reads data from the data acquisition software

        Parameters
        ----------
        num_samples : int
            The number of samples to read
        newest_data : TYPE, optional
            If True, gets the newest data in the buffer, otherwise gets the
            oldest data in the buffer. The default is False.

        Returns
        -------
        np.ndarray
            An array of data with shape (num_inputs + num_outputs) x num_samples

        """
        read_array = np.zeros(
            (self._num_inputs + self._num_outputs) * num_samples, dtype=np.float32
        )
        read_type = ctypes.c_int(0 if newest_data else 1)
        if DEBUG:
            debug_fn(f"Calling GetData with\n    length = {num_samples}")
        _ = self._api.GetData(read_array, read_type, ctypes.c_int(num_samples))
        # if not success == 1:
        #     self.raise_error()
        return read_array.reshape((self._num_inputs + self._num_outputs, num_samples))

    def get_available_input_data_samples(self):
        """
        Gets the number of available samples in the input channel circular
        buffers.

        Returns
        -------
        samples : int
            The number of samples that are available to be read out of the input
            channel buffer.

        """
        if DEBUG:
            debug_fn("Calling GetAvailableDataLength")
        samples = self._api.GetAvailableDataLength()
        if DEBUG:
            debug_fn(f"{samples} Samples Available")
        return samples

    def get_space_in_out_buffer(self):
        """
        Gets the amount of free space in the output buffer (how much data can
        be sent to the output buffer before it is full).

        Returns
        -------
        int
            The number of samples that can be sent to the output buffer
            before it is full

        """
        if DEBUG:
            debug_fn("Calling GetSpaceInOutBuffer")
        samples = self._api.GetSpaceInOutBuffer()
        if DEBUG:
            debug_fn(f"    {samples} Samples Available in Buffer")
        return samples

    def get_total_output_samples_on_buffer(self):
        """
        Gets the number of samples in the output channel circular buffers.

        Returns
        -------
        samples : int
            The number of samples in line to be output through the output
            channels.

        """
        if DEBUG:
            debug_fn("Calling GetTotalSamplesInOutputBuffer")
        samples = self._api.GetTotalSamplesInOutputBuffer()
        if DEBUG:
            debug_fn(f"    {samples} Output Samples Available")
        return samples

    def write_output_data(self, output_data):
        """
        The function will fill the output buffer with data specified, which will
        result in it eventually getting output from the system.

        Parameters
        ----------
        output_data : array of float
            2D data array with shape num_outputs x num_samples containing the
            data to be output from the signal generator.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if output_data.ndim != 2:
            raise ValueError("`output_data` should have 2 dimensions (num_outputs x num_samples)")
        if output_data.shape[0] != self._num_outputs:
            raise ValueError(
                f"`output_data` must have number of rows equal to the number of "
                f"outputs ({self._num_outputs})"
            )
        num_samples = output_data.shape[-1]
        this_output_data = np.zeros(np.prod(output_data.shape), dtype=np.float32)
        this_output_data[:] = output_data.flatten().astype(np.float32)
        # debug_fn(this_output_data.shape, num_samples, self._num_outputs)
        if DEBUG:
            debug_fn(f"Calling PutOutData with\n    length {num_samples}")
        _ = self._api.PutOutData(this_output_data, ctypes.c_int(num_samples))
        # if not success == 1:
        #     self.raise_error()

    def get_raw_error_list(self):
        """Gets the raw bytes from the error list"""
        if DEBUG:
            debug_fn("Calling GetErrorList")
        return self._api.GetErrorList()

    def get_error_list(self):
        """Gets the decoded error list"""
        if DEBUG:
            debug_fn("Calling GetErrorList")
        data = self._api.GetErrorList()
        return data.decode()

    def set_buffer_size(self, buff_sz):
        """
        Sets the number of samples, per channel, in the input and output
        channel circular buffers.

        Parameters
        ----------
        buff_sz : int
            The desired size of the circular buffers, in samples.  Must be greater
            than 4096.  The upper limit of buff_sz is limited by available
            memory in the PC.
        """
        if DEBUG:
            debug_fn("Calling SetCBufSize")
        success = self._api.SetCBufSize(buff_sz)
        if not success == 1:
            self.raise_error()

    def set_save_recording(self, enabled):
        """
        Sets the API to archive (or not archive) the complete time history
        being sent through the API. If enabled, this happens in the background
        as the test is running; and can be accessed by later launching the 900
        Series software.

        Parameters
        ----------
        enabled : bool
            If True, enables the recording.  If False, disables it.
        """
        if DEBUG:
            debug_fn("Calling SetSaveRecording")
        success = self._api.SetSaveRecording(enabled)
        if not success == 1:
            self.raise_error()

    def get_num_inps_available(self):
        """
        Returns the number of available input channels on the currently
        selected hardware.

        Returns
        -------
        int
            The number of input channels available for usage, with the
            currently selected hardware units.

        """
        if DEBUG:
            debug_fn("Calling GetNumInpsAvailable")
        inps_available = self._api.GetNumInpsAvailable()
        if DEBUG:
            debug_fn(f"    {inps_available} Inputs Available")
        return inps_available

    def get_num_inps_selected(self):
        """
        Returns the number of selected input channels

        Returns
        -------
        int
            The number of input channels selected in the current setup.

        """
        if DEBUG:
            debug_fn("Calling GetNumInpsSelected")
        inps_selected = self._api.GetNumInpsSelected()
        if DEBUG:
            debug_fn(f"    {inps_selected} Inputs Selected")
        return inps_selected

    def get_num_outs_available(self):
        """
        Returns the number of available output channels on the currently
        selected hardware.

        Returns
        -------
        int
            The number of output channels available for use, with the currently
            selected hardware units

        """
        if DEBUG:
            debug_fn("Calling GetNumOutsAvailable")
        outs_available = self._api.GetNumOutsAvailable()
        if DEBUG:
            debug_fn(f"    {outs_available} Outputs Available")
        return outs_available

    def get_num_outs_selected(self):
        """
        Returns the number of output channels selected in the current test

        Returns
        -------
        int
            The number of output channels selected in the current setup

        """
        if DEBUG:
            debug_fn("Calling GetNumOutsSelected")
        outs_selected = self._api.GetNumOutsSelected()
        if DEBUG:
            debug_fn(f"    {outs_selected} Outputs Selected")
        return outs_selected

    def get_cbuf_size(self):
        """
        Gets the number of samples, per channel, in the input and output
        channel circular buffers.

        Returns
        -------
        int
            The circular buffer size.

        """
        if DEBUG:
            debug_fn("Calling GetCBufSize")
        cbuf_size = self._api.GetCBufSize()
        if DEBUG:
            debug_fn(f"    CBuf Size {cbuf_size}")
        return cbuf_size

    def get_input_channel_bncs(self):
        """
        Gets the BNC numbers of available input channels on the currently selected hardware.

        Returns
        -------
        bncs : array of int
            BNC numbers corresponding to input channels

        """
        num_inputs = self.get_num_inps_available()
        bncs = np.zeros((num_inputs,), dtype=np.int32)
        if DEBUG:
            debug_fn("Calling GetInputChannelBNCs")
        success = self._api.GetInputChannelBNCs(bncs)
        if not success == 1:
            self.raise_error()
        if DEBUG:
            debug_fn(f"    Input BNCs: {bncs.tolist()}")
        return bncs

    def get_output_channel_bncs(self):
        """


        Parameters
        ----------
        bncs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        num_outputs = self.get_num_outs_available()
        bncs = np.zeros((num_outputs,), dtype=np.int32)
        if DEBUG:
            debug_fn("Calling GetOutputChannelBNCs")
        success = self._api.GetOutputChannelBNCs(bncs)
        if not success == 1:
            self.raise_error()
        if DEBUG:
            debug_fn(f"    Output BNCs: {bncs.tolist()}")
        return bncs

    def __del__(self):
        if DEBUG:
            log_file.close()


# Example usage:
# wrapper = DP900("path_to_your_dll.dll")
# print(wrapper.is_hw_connected())
