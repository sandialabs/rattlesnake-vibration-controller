# -*- coding: utf-8 -*-
"""
Hardware definition that allows for the Data Physics Quattro Device to be run
with Rattlesnake.

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
from enum import Enum
import numpy as np
from numpy.ctypeslib import ndpointer
from time import sleep,time

DEBUG = False

if DEBUG:
    __log_file__ = 'DataPhysics_Log.txt'

class QuattroStatus(Enum):
    DISCONNECTED = -1
    IDLE = 0
    INIT = 1
    RUNNING = 2
    STOPPED = 3
    
class QuattroCoupling(Enum):
    AC_DIFFERENTIAL = 0
    DC_DIFFERENTIAL = 1
    AC_SINGLE_ENDED = 2
    DC_SINGLE_ENDED = 3
    AC_COUPLED_IEPE = 4

class DPQuattro:
    """An interface to the data physics C API for the quattro hardware"""
    
    def __init__(self, library_path : str):
        """
        Connects to the library
        
        Parameters
        ----------
        library_path : str
            Path to the DpQuattro.dll file that is used to run the Quattro device

        Returns
        -------
        None.

        """
        if DEBUG:
            self.log_file = open(__log_file__,'w')
        self.input_channel_parameters = None
        self.output_channel_parameters = None
        self._api = ctypes.WinDLL(library_path)
        self._valid_sample_rates = np.array([
            10.24, 12.5, 12.8, 13.1072, 16, 16.384, 20, 20.48, 25, 25.6, 32, 
            32.768, 40, 40.96, 50, 51.2, 64, 65.536, 80, 81.92, 100, 102.4, 
            128, 160, 163.84, 200, 204.8, 256, 320, 327.68, 400, 409.6, 512, 
            640, 800, 819.2, 1024, 1280, 1600, 1638.4, 2048, 2560, 3200, 4096, 
            5120, 6400, 8192, 10240, 12800, 20480, 25600, 40960, 51200, 102400, 
            204800])
        self._valid_input_ranges = np.array([0.1, 1.0, 10.0, 20.0])
        self._valid_output_ranges = np.array([2.0,10.0])
        self._num_inputs = 0
        self._num_outputs = 0
        
        # Set up prototypes for the various function calls
        # Comments are from the quattro API header file
        # DPCOMM_API int  IsHwConnected();
        self._api.IsHwConnected.restype = ctypes.c_int
        # DPCOMM_API int  Connect();
        self._api.Connect.restype = ctypes.c_int
        # DPCOMM_API int  Disconnect();
        self._api.Disconnect.restype = ctypes.c_int
        # DPCOMM_API int  SetSampleRate(double sampleRate);
        self._api.SetSampleRate.argtypes = ctypes.c_double,
        # DPCOMM_API int  IsLicensed();
        self._api.IsLicensed.restype = ctypes.c_int
        # DPCOMM_API int  Init();
        self._api.Init.restype = ctypes.c_int
        # DPCOMM_API int  SetInpParams(int* coupling, float* sensitivity, float* range, int numInps);
        self._api.SetInpParams.argtypes = (ndpointer(ctypes.c_int),
                                           ndpointer(ctypes.c_float),
                                           ndpointer(ctypes.c_float),
                                           ctypes.c_int)
        # DPCOMM_API int  SetOutParams(float* sensitivity, float* range, int numOuts);
        self._api.SetOutParams.argtypes = (ndpointer(ctypes.c_float),
                                           ndpointer(ctypes.c_float),
                                           ctypes.c_int)
        # DPCOMM_API int  SetTacParams(int* coupling, float* holdOffTime, float* hysteresis, float* preScaler, float* PPR,
        #         float* smoothing, float* speedRatio, float* trigLevel, int* trigSlope, int numTacs);
        # DPCOMM_API int  Start();
        self._api.Start.restype = ctypes.c_int
        # DPCOMM_API int  Stop();
        self._api.Stop.restype = ctypes.c_int
        # DPCOMM_API int  End();
        self._api.End.restype = ctypes.c_int
        # DPCOMM_API int  SerialNumber();
        self._api.SerialNumber.restype = ctypes.c_int
        # DPCOMM_API int  GetData(float* outputBuf, int dataType, int length);
        self._api.GetData.argtypes = (ndpointer(ctypes.c_float),
                                      ctypes.c_int,
                                      ctypes.c_int)
        self._api.GetData.restype = ctypes.c_int
        # DPCOMM_API int  GetAvailableDataLength();
        self._api.GetAvailableDataLength.restype = ctypes.c_int
        # # DPCOMM_API int  GetAvailableOutData();
        # self._api.GetAvailableOutData.restype = ctypes.c_int
        # DPCOMM_API int       GetTotalSamplesInOutputBuffer();
        self._api.GetTotalSamplesInOutputBuffer.restype = ctypes.c_int
        # DPCOMM_API int  PutOutData(float* outputBuf, int length);
        self._api.PutOutData.argtypes = (ndpointer(ctypes.c_float),
                                         ctypes.c_int)
        self._api.PutOutData.restype = ctypes.c_int
        # DPCOMM_API char* GetErrorList();
        self._api.GetErrorList.restype = ctypes.c_char_p
        # DPCOMM_API int  SetCBufSize(int buffSz);
        self._api.SetCBufSize.argtypes = ctypes.c_int,
        # DPCOMM_API int  GetCBufSize();
        self._api.GetCBufSize.restype = ctypes.c_int
        
        
        if self.is_hardware_connected():
            self.status = QuattroStatus.IDLE
        else:
            self.status = QuattroStatus.DISCONNECTED
            
        
    def connect(self):
        if not self.is_hardware_connected():
            if DEBUG:
                self.log_file.write('Calling Connect\n')
            success = self._api.Connect()
        else:
            raise RuntimeError('Hardware is already connected')
        if not success:
            self.raise_error()
        else:
            self.status = QuattroStatus.IDLE
        
        
    def disconnect(self):
        if self.is_hardware_connected():
            if DEBUG:
                self.log_file.write('Calling Disconnect\n')
            success = self._api.Disconnect()
        else:
            raise RuntimeError('Hardware is not connected')
        if not success:
            self.raise_error()
        else:
            self.status = QuattroStatus.DISCONNECTED
            
    def is_hardware_connected(self):
        if DEBUG:
            self.log_file.write('Calling IsHwConnected\n')
        return bool(self._api.IsHwConnected())
    
    def get_raw_error_list(self):
        if DEBUG:
            self.log_file.write('Calling GetErrorList\n')
        return self._api.GetErrorList()
    
    def get_error_list(self):
        if DEBUG:
            self.log_file.write('Calling GetErrorList\n')
        data = self._api.GetErrorList()
        return data.decode()
    
    def set_sample_rate(self,sample_rate):
        close_rates = np.isclose(self._valid_sample_rates,sample_rate)
        close_sample_rates = self._valid_sample_rates[close_rates]
        if len(close_sample_rates) == 0:
            raise ValueError('Sample Rate {:} is not valid.  Valid sample rates are {:}'.format(
                sample_rate,', '.join(['{:0.2f}'.format(v) for v in self._valid_sample_rates])))
        elif len(close_sample_rates) > 1:
            raise ValueError("Multiple Sample Rates are close to the specified rate ({:}, {:}).  This shouldn't happen!".format(
                sample_rate,close_sample_rates))
        if DEBUG:
            self.log_file.write('Calling SetSampleRate\n')
        success = self._api.SetSampleRate(ctypes.c_double(close_sample_rates[0]))
        if not success:
            self.raise_error()
            
    def is_licensed(self):
        if DEBUG:
            self.log_file.write('Calling IsLicensed\n')
        return bool(self._api.IsLicensed())
    
    def initialize(self):
        if self.status == QuattroStatus.IDLE:
            if DEBUG:
                self.log_file.write('Calling Init\n')
            success = self._api.Init()
            if not success:
                self.raise_error()
            else:
                self.status = QuattroStatus.INIT
        else:
            raise RuntimeError('Hardware status must be IDLE to initialize.  Current status is {:}.'.format(self.status.name))
            
    def setup_input_parameters(
            self,coupling_array,sensitivity_array,range_array):
        # Set up the channel arrays
        if len(coupling_array) != len(sensitivity_array):
            raise ValueError('Coupling array must have same size as Sensitivity Array')
        if len(range_array) != len(sensitivity_array):
            raise ValueError('Range array must have same size as Sensitivity Array')
        self._num_inputs = len(coupling_array)
        coupling_array = np.array([int(coupling.value) for coupling in coupling_array],dtype=np.int32)
        sensitivity_array = np.array([float(val) for val in sensitivity_array],dtype=np.float32)
        validated_range_array = []
        for rng in range_array:
            close_ranges = self._valid_input_ranges[np.isclose(self._valid_input_ranges,rng)]
            if len(close_ranges) == 0:
                raise ValueError('Range {:} is not valid.  Valid sample rates are {:}'.format(
                    rng,', '.join(['{:0.1f}'.format(v) for v in self._valid_input_ranges])))
            elif len(close_ranges) > 1:
                raise ValueError("Multiple Ranges are close to the specified rate ({:}, {:}).  This shouldn't happen!".format(
                    rng,close_ranges))
            validated_range_array.append(close_ranges[0])
        validated_range_array = np.array(validated_range_array,dtype=np.float32)
        # Call the API function
        if DEBUG:
            self.log_file.write('Calling SetInpParams\n')
        success = self._api.SetInpParams(
            coupling_array,
            sensitivity_array,
            validated_range_array,
            ctypes.c_int(self._num_inputs)
            )
        if not success:
            self.raise_error()
    
    def setup_output_parameters(
            self, sensitivity_array, range_array):
        if len(range_array) != len(sensitivity_array):
            raise ValueError('Range array must have same size as Sensitivity Array')
        self._num_outputs = len(sensitivity_array)
        sensitivity_array = np.array([float(val) for val in sensitivity_array],dtype=np.float32)
        validated_range_array = []
        for rng in range_array:
            close_ranges = self._valid_output_ranges[np.isclose(self._valid_output_ranges,rng)]
            if len(close_ranges) == 0:
                raise ValueError('Range {:} is not valid.  Valid sample rates are {:}'.format(
                    rng,', '.join(['{:0.1f}'.format(v) for v in self._valid_output_ranges])))
            elif len(close_ranges) > 1:
                raise ValueError("Multiple Ranges are close to the specified rate ({:}, {:}).  This shouldn't happen!".format(
                    rng,close_ranges))
            validated_range_array.append(close_ranges[0])
        validated_range_array = np.array(validated_range_array,dtype=np.float32)
        # Call the API function
        if DEBUG:
            self.log_file.write('Calling SetOutParams\n')
        success = self._api.SetOutParams(
            sensitivity_array,
            validated_range_array,
            ctypes.c_int(self._num_outputs)
            )
        if not success:
            self.raise_error()
        
    def raise_error(self):
        if DEBUG:
            self.log_file.write('DP Error: {:}\n'.format(self.get_error_list()))
        # raise RuntimeError(self.get_error_list())
        
    def start(self):
        if self.status in [QuattroStatus.INIT,QuattroStatus.STOPPED]:
            if DEBUG:
                self.log_file.write('Calling Start\n')
            success = self._api.Start()
            if not success:
                self.raise_error()
            else:
                self.status = QuattroStatus.RUNNING
        else:
            raise RuntimeError('Current hardware status is {:}.  Hardware must be initialized or stopped prior to starting a measurement'.format(self.status.name))
    
    def stop(self):
        if self.status == QuattroStatus.RUNNING:
            if DEBUG:
                self.log_file.write('Calling Stop\n')
            success = self._api.Stop()
            if not success:
                self.raise_error()
            else:
                self.status = QuattroStatus.STOPPED
        else:
            raise RuntimeError('Current hardware status is {:}.  Hardware must be running prior to stopping a measurement'.format(self.status.name))
    
    def end(self):
        if self.status in [QuattroStatus.STOPPED,QuattroStatus.INIT]:
            if DEBUG:
                self.log_file.write('Calling End\n')
            success = self._api.End()
            if not success:
                self.raise_error()
            else:
                self.status = QuattroStatus.IDLE
                
    @property
    def serial_number(self):
        if DEBUG:
            self.log_file.write('Warning, this gives out the wrong value!\n')
        if DEBUG:
            self.log_file.write('Calling SerialNumber\n')
        return self._api.SerialNumber()
    
    def set_buffer_size(self,buffer_size):
        if DEBUG:
            self.log_file.write('Calling SetCBufSize\n')
        success = self._api.SetCBufSize(ctypes.c_int(buffer_size))
        if not success:
            self.raise_error()
            
    def get_buffer_size(self):
        if DEBUG:
            self.log_file.write('Calling GetCBufSize\n')
        return self._api.GetCBufSize()
    
    def get_available_input_data_samples(self):
        if DEBUG:
            self.log_file.write('Calling GetAvailableDataLength\n')
        samples =  self._api.GetAvailableDataLength()
        if DEBUG:
            self.log_file.write('{:} Samples Available\n'.format(samples))
        return samples
    
    # def get_output_samples_on_buffer(self):
    #     self.log_file.write('Calling GetAvailableOutData\n')
    #     samples = self._api.GetAvailableOutData()
    #     self.log_file.write('{:} Output Samples Available\n'.format(samples))
    #     return samples
    
    def get_total_output_samples_on_buffer(self):
        if DEBUG:
            self.log_file.write('Calling GetTotalSamplesInOutputBuffer\n')
        samples = self._api.GetTotalSamplesInOutputBuffer()
        if DEBUG:
            self.log_file.write('{:} Output Samples Available\n'.format(samples))
        return samples
    
    def read_input_data(self,num_samples,newest_data=False):
        read_array = np.zeros(self._num_inputs*num_samples,dtype=np.float32)
        read_type = ctypes.c_int(0 if newest_data else 1)
        if DEBUG:
            self.log_file.write('Calling GetData with length {:}\n'.format(ctypes.c_int(num_samples)))
        success = self._api.GetData(
            read_array,
            read_type,
            ctypes.c_int(num_samples))
        if not success:
            self.raise_error()
        return read_array.reshape((self._num_inputs,num_samples))
        
    def write_output_data(self,output_data):
        if output_data.ndim != 2:
            raise ValueError('`output_data` should have 2 dimensions (num_outputs x num_samples)')
        if output_data.shape[0] != self._num_outputs:
            raise ValueError('`output_data` must have number of rows equal to the number of outputs ({:})'.format(self._num_outputs))
        num_samples = output_data.shape[-1]
        this_output_data = np.zeros(np.prod(output_data.shape),dtype=np.float32)
        this_output_data[:] = output_data.flatten().astype(np.float32)
        # self.log_file.write(this_output_data.shape, num_samples, self._num_outputs)
        if DEBUG:
            self.log_file.write('Calling PutOutData with length {:}\n'.format(ctypes.c_int(num_samples)))
        success = self._api.PutOutData(
            this_output_data,
            ctypes.c_int(num_samples))
        # if not success:
        #     self.raise_error()

    def __del__(self):
        if DEBUG:
            self.log_file.close()