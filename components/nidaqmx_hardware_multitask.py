# -*- coding: utf-8 -*-
"""
This file defines an interface to the NIDAQmx hardware, and is used to set up
and interact with read and write tasks on the hardware.

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

from .abstract_hardware import HardwareAcquisition,HardwareOutput
import nidaqmx as ni
import nidaqmx.stream_readers as ni_read
import nidaqmx.stream_writers as ni_write
import nidaqmx.constants as nic
from .utilities import Channel,DataAcquisitionParameters
import numpy as np
from typing import List
import time

BUFFER_SIZE_FACTOR = 3

class NIDAQmxAcquisition(HardwareAcquisition):
    """Class defining the interface between the controller and NI hardware
    
    This class defines the interfaces between the controller and National
    Instruments Hardware that runs the NI-DAQmx library.  It is run by the
    Acquisition process, and must define how to get data from the test
    hardware into the controller."""
    def __init__(self):
        """
        Constructs the NIDAQmx Acquisition class and specifies values to null.
        """
        self.task = None
        self.read_data = None
        self.reader = None
        self.acquisition_delay = None
    
    def set_up_data_acquisition_parameters_and_channels(self,
                                                        test_data : DataAcquisitionParameters,
                                                        channel_data : List[Channel]):
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
        self.initialize()
        self.create_response_channels(channel_data)
        self.set_parameters(test_data)
    
    def initialize(self):
        """Method to start up the hardware
        
        This function starts the NIDAQmx Acquisition task on the hardware."""
        self.task = ni.Task()
    
    def create_response_channels(self,channel_data : List[Channel]):
        """Method to set up response channels
        
        This function takes channels from the supplied list of channels and
        creates analog inputs on the hardware.

        Parameters
        ----------
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test

        """
        for channel in channel_data:
            self._create_channel(channel)
    
    def set_parameters(self,test_data : DataAcquisitionParameters):
        """Method to set up sampling rate and other test parameters
        
        This function sets the clock configuration on the NIDAQmx hardware.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.

        """
        self.task.timing.cfg_samp_clk_timing(test_data.sample_rate,sample_mode=nic.AcquisitionType.CONTINUOUS,samps_per_chan=test_data.samples_per_read)
        self.task.in_stream.wait_mode = nic.WaitMode.POLL
        self.reader = ni_read.AnalogMultiChannelReader(self.task.in_stream)
        self.read_data = np.zeros((len(self.task.ai_channels),test_data.samples_per_read))
        self.acquisition_delay = BUFFER_SIZE_FACTOR*test_data.samples_per_write
        print('Actual Acquisition Sample Rate: {:}'.format(self.task.timing.samp_clk_rate))
    
    def start(self):
        """Start acquiring data"""
        self.task.start()
    
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
        """Method to read a frame of data from the controller
        
        Returns
        -------
        read_data : 
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``
        """
        self.reader.read_many_sample(self.read_data,number_of_samples_per_channel=self.read_data.shape[-1])
        return self.read_data
    
    def read_remaining(self):
        """Method to read the rest of the data on the acquisition
        
        Returns
        -------
        read_data : 
            2D Data read from the controller with shape ``n_channels`` x
            ``n_samples``
            """
        read_data = np.zeros((len(self.task.ai_channels),self.task.in_stream.avail_samp_per_chan))
        self.reader.read_many_sample(read_data,number_of_samples_per_channel=read_data.shape[-1])
        return read_data
    
    def stop(self):
        """Method to stop the acquisition"""
        self.task.stop()
    
    def close(self):
        """Method to close down the hardware"""
        if not self.task is None:
            self.task.close()
    
    def _create_channel(self,channel_data: Channel):
        """Helper function to construct a channel on the hardware.

        Parameters
        ----------
        channel_data: Channel :
            Channel object specifying the channel parameters.

        Returns
        -------
            channel : 
                A reference to the NIDAQmx channel created by the function
        """
        physical_channel = channel_data.physical_device+'/'+channel_data.physical_channel
        # Parse the channel structure to make sure datatypes are correct
        # Sensitivity
        try:
            sensitivity = float(channel_data.sensitivity)
        except (TypeError,ValueError):
            raise ValueError('{:} not a valid sensitivity'.format(channel_data.sensitivity))
        # Minimum Value
        try:
            minimum_value = float(channel_data.minimum_value)
        except (TypeError,ValueError):
            raise ValueError('{:} not a valid minimum value'.format(channel_data.minimum_value))
        # Maximum Value
        try:
            maximum_value = float(channel_data.maximum_value)
        except (TypeError,ValueError):
            raise ValueError('{:} not a valid maximum value'.format(channel_data.maximum_value))
        # Channel Type and Units
        if channel_data.channel_type.lower() in ['accelerometer','acceleration','accel']:
            channel_type = nic.UsageTypeAI.ACCELERATION_ACCELEROMETER_CURRENT_INPUT
            if channel_data.unit.lower() in ['g','gs']:
                unit = nic.AccelUnits.G
            else:
                raise ValueError('Accelerometer units must be in G, not {:}'.format(channel_data.unit))
        elif channel_data.channel_type.lower() == 'force':
            channel_type = nic.UsageTypeAI.FORCE_IEPE_SENSOR
            if channel_data.unit.lower() in ['lb','pound','pounds','lbf','lbs','lbfs']:
                unit = nic.ForceUnits.POUNDS
            elif channel_data.unit.lower() in ['n','newton','newtons','ns']:
                unit = nic.ForceUnits.NEWTONS
            else:
                raise ValueError('Unrecognized Force Unit {:}'.format(channel_data.unit))
        elif channel_data.channel_type.lower() in ['voltage','volt']:
            channel_type = nic.UsageTypeAI.VOLTAGE
            unit = None
        else:
            raise ValueError('{:} not a valid channel type.  Must be one of ["acceleration","accelerometer","accel","force","voltage","volt"]'.format(channel_type))
        # Excitation Source
        if channel_data.excitation_source.lower() == 'internal':
            excitation_source = nic.ExcitationSource.INTERNAL
            try:
                excitation = float(channel_data.excitation)
            except (TypeError,ValueError):
                raise ValueError('{:} not a valid excitation'.format(channel_data.excitation))
        elif channel_data.excitation_source.lower() == 'none':
            excitation_source = nic.ExcitationSource.NONE
            excitation = 0
        else:
            raise ValueError('{:} not a valid excitation source.  Must be one of ["internal","none"]'.format(channel_data.excitation_source))
        # Now go and create the channel
        if channel_type != nic.UsageTypeAI.VOLTAGE:
            min_val = minimum_value*1000/sensitivity
            max_val = maximum_value*1000/sensitivity
        else:
            min_val = minimum_value
            max_val = maximum_value
        if channel_type == nic.UsageTypeAI.ACCELERATION_ACCELEROMETER_CURRENT_INPUT:
            try:
                channel = self.task.ai_channels.add_ai_accel_chan(
                            physical_channel,min_val = min_val,max_val = max_val,
                            units=unit,sensitivity=sensitivity,
                            sensitivity_units=nic.AccelSensitivityUnits.M_VOLTS_PER_G,
                            current_excit_source = excitation_source,
                            current_excit_val = excitation
                        )
            except AttributeError:
                channel = self.task.ai_channels.add_ai_accel_chan(
                            physical_channel,min_val = min_val,max_val = max_val,
                            units=unit,sensitivity=sensitivity,
                            sensitivity_units=nic.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                            current_excit_source = excitation_source,
                            current_excit_val = excitation
                        )
        elif channel_type == nic.UsageTypeAI.FORCE_IEPE_SENSOR:
            try:
                channel = self.task.ai_channels.add_ai_force_iepe_chan(
                            physical_channel,min_val = min_val,max_val = max_val,
                            units=unit,sensitivity=sensitivity,
                            sensitivity_units = nic.ForceIEPESensorSensitivityUnits.M_VOLTS_PER_NEWTON 
                                if unit == nic.ForceUnits.NEWTONS
                                else nic.ForceIEPESensorSensitivityUnits.M_VOLTS_PER_POUND,
                            current_excit_source= excitation_source,
                            current_excit_val = excitation
                        )
            except AttributeError:
                channel = self.task.ai_channels.add_ai_force_iepe_chan(
                            physical_channel,min_val = min_val,max_val = max_val,
                            units=unit,sensitivity=sensitivity,
                            sensitivity_units = nic.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON 
                                if unit == nic.ForceUnits.NEWTONS
                                else nic.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_POUND,
                            current_excit_source= excitation_source,
                            current_excit_val = excitation
                        )
        elif channel_type == nic.UsageTypeAI.VOLTAGE:
            channel = self.task.ai_channels.add_ai_voltage_chan(
                        physical_channel,min_val = min_val,max_val=max_val,
                        units=nic.VoltageUnits.VOLTS
                    )
        return channel
    
class NIDAQmxOutput(HardwareOutput):
    """Class defining the interface between the controller and NI hardware
    
    This class defines the interfaces between the controller and National
    Instruments Hardware that runs the NI-DAQmx library.  It is run by the
    Output process, and must define how to get data from the controller to the
    output hardware."""
    def __init__(self):
        """
        Constructs the NIDAQmx Output class and initializes values to null.
        """
        self.tasks = None
        self.channel_task_map = None
        self.writers = None
        self.write_trigger = None
        self.signal_samples = None
        self.sample_rate = None
        self.buffer_size_factor = BUFFER_SIZE_FACTOR
    
    def set_up_data_output_parameters_and_channels(self,
                                                   test_data : DataAcquisitionParameters,
                                                   channel_data : List[Channel]):
        """
        Initialize the hardware and set up sources and sampling properties
        
        The function must create channels on the hardware corresponding to
        the sources in the test.  It must also set the sampling rates.

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
        self.create_sources(channel_data)
        self.set_parameters(test_data)
    
    def create_sources(self,channel_data : List[Channel]):
        """Method to set up excitation sources
        
        This function takes channels from the supplied list of channels and
        creates analog outputs on the hardware.

        Parameters
        ----------
        channel_data : List[Channel] :
            A list of ``Channel`` objects defining the channels in the test
        """
        # Get the physical devices
        physical_devices = list(set([ni.system.device.Device(channel.feedback_device).product_type
                                     for channel in channel_data
                                     if not (channel.feedback_device is None)
                                     and not (channel.feedback_device.strip() == '')]))
        # Check if it's a CDAQ device
        try:
            devices = [ni.system.device.Device(channel.feedback_device)
                       for channel in channel_data
                       if not (channel.feedback_device is None)
                       and not (channel.feedback_device.strip() == '')]
            if len(devices) == 0:
                self.write_trigger = None # No output device
            else:
                chassis_device = devices[0].compact_daq_chassis_device
                self.write_trigger = [trigger for trigger in chassis_device.terminals if 'ai/StartTrigger' in trigger][0]
        except ni.DaqError:
            self.write_trigger = '/'+channel_data[0].physical_device+'/ai/StartTrigger'
        print('Output Devices: {:}'.format(physical_devices))
        self.tasks = [ni.Task() for device in physical_devices]
        index = 0
        self.channel_task_map = [[] for device in physical_devices]
        for channel in channel_data:
            if not (channel.feedback_device is None) and not (channel.feedback_device.strip() == ''):
                device_index = physical_devices.index(ni.system.device.Device(channel.feedback_device).product_type)
                self.channel_task_map[device_index].append(index)
                index += 1
                self._create_channel(channel,device_index)
        print('Output Mapping: {:}'.format(self.channel_task_map))
                    
    
    def set_parameters(self,test_data : DataAcquisitionParameters):
        """Method to set up sampling rate and other test parameters
        
        This function sets the clock configuration on the NIDAQmx hardware.

        Parameters
        ----------
        test_data : DataAcquisitionParameters :
            A container containing the data acquisition parameters for the
            controller set by the user.
        """
        self.signal_samples = test_data.samples_per_write
        self.sample_rate = test_data.sample_rate
        self.writers = []
        for task in self.tasks:
            task.timing.cfg_samp_clk_timing(test_data.sample_rate,
                                            sample_mode=nic.AcquisitionType.CONTINUOUS,
                                            samps_per_chan=test_data.samples_per_write)
            task.out_stream.regen_mode = nic.RegenerationMode.DONT_ALLOW_REGENERATION
            #task.out_stream.relative_to = nic.WriteRelativeTo.CURRENT_WRITE_POSITION
            task.triggers.start_trigger.dig_edge_src = self.write_trigger
            task.triggers.start_trigger.dig_edge_edge = ni.constants.Edge.RISING
            task.triggers.start_trigger.trig_type = ni.constants.TriggerType.DIGITAL_EDGE
            task.out_stream.output_buf_size = self.buffer_size_factor*test_data.samples_per_write
            self.writers.append(ni_write.AnalogMultiChannelWriter(task.out_stream,auto_start=False))
            print('Actual Output Sample Rate: {:}'.format(task.timing.samp_clk_rate))
    
    def start(self):
        """Method to start acquiring data"""
        for task in self.tasks:
            task.start()
    
    def write(self,data):
        """Method to write a frame of data

        Parameters
        ----------
        data : np.ndarray
            2D Data to be written to the controller with shape ``n_sources`` x
            ``n_samples``

        """
        for i,writer in enumerate(self.writers):
            writer.write_many_sample(data[self.channel_task_map[i]])
    
    def stop(self):
        """Method to stop the output"""
        # Need to output everything in the buffer and then some zeros and we'll
        # shut down during the zeros portion
        for i,writer in enumerate(self.writers):
            writer.write_many_sample(np.zeros((len(self.channel_task_map[i]),self.signal_samples)))
        # Now figure out how many samples are remaining
        samples_remaining = (self.tasks[0].out_stream.curr_write_pos
                             - self.tasks[0].out_stream.total_samp_per_chan_generated
                             - self.signal_samples) # Subtract off the zeros
        time_remaining = samples_remaining/self.sample_rate
        time.sleep(time_remaining)
        for task in self.tasks:
            task.stop()
    
    def close(self):
        """Method to close down the hardware"""
        if not self.tasks is None:
            for task in self.tasks:
                task.close()

    def ready_for_new_output(self):
        """Returns true if the system is ready for new outputs
        
        Returns
        -------
        bool : 
            True if the hardware is accepting the next data to write."""
        return self.tasks[0].out_stream.curr_write_pos - self.tasks[0].out_stream.total_samp_per_chan_generated < (self.buffer_size_factor-1)*self.signal_samples

    def _create_channel(self, channel_data: Channel,device_index):
        """
        Helper function to construct a channel on the hardware.

        Parameters
        ----------
        channel_data: Channel :
            Channel object specifying the channel parameters.

        Returns
        -------
            channel : 
                A reference to the NIDAQmx channel created by the function
        """
        # Minimum Value
        try:
            minimum_value = float(channel_data.minimum_value)
        except (TypeError,ValueError):
            raise ValueError('{:} not a valid minimum value'.format(channel_data.minimum_value))
        # Maximum Value
        try:
            maximum_value = float(channel_data.maximum_value)
        except (TypeError,ValueError):
            raise ValueError('{:} not a valid maximum value'.format(channel_data.maximum_value))
        physical_channel = channel_data.feedback_device+'/'+channel_data.feedback_channel
        channel = self.tasks[device_index].ao_channels.add_ao_voltage_chan(
                    physical_channel, min_val=minimum_value, max_val=maximum_value)
        return channel
    
    