# -*- coding: utf-8 -*-
"""
Implementation for the HBK LAN-XI Hardware

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

import requests
import socket
from typing import List,Tuple
import time
import numpy as np
import multiprocessing as mp
import os
from math import ceil

from .abstract_hardware import HardwareAcquisition,HardwareOutput
from .utilities import Channel,DataAcquisitionParameters

from .lanxi_stream import OpenapiMessage,OpenapiHeader

OUTPUT_RATE = 131072
LANXI_STATE_TIMEOUT = 255.0
LANXI_STATE_REPORT = 10
VALID_FILTERS = ['DC', '0.7 Hz', '7.0 Hz', '22.4 Hz', 'Intensity']
VALID_RANGES = ['0.316', '1', '10', '31.6']
HEADER_LENGTH = 28
BUFFER_SIZE = 2.5

LANXI_STATE_SHUTDOWN = {
    'RecorderRecording':'/rest/rec/measurements/stop',
    'RecorderStreaming':'/rest/rec/finish',
    'RecorderOpened':'/rest/rec/close',
    'Idle':None}

# TODO List
# TODO Get responses each time a get or put is done so we know if it was successful
# TODO Shut down the data acquisition more quickly

class LanXIError(Exception):
    pass

def read_lanxi(socket_handle : socket.socket):
    """
    Reads and interprets data from a Lan-XI

    Parameters
    ----------
    socket_handle : socket.socket
        Handle to the socket that the communication is happening over.

    Returns
    -------
    message_type : OpenapiStream.Header.EMessageType
        Enum determining what the data type is
    data : np.ndarray or dict
        The data read from the device.  Will be a np.ndarray for a signal and
        a dictionary for a interpretation

    """
    # print('Reading data from {:}:{:}'.format(*socket_handle.getpeername()))
    data = socket_handle.recv(HEADER_LENGTH,socket.MSG_WAITALL)
    if len(data) == 0:
        raise LanXIError('Socket is not connected anymore')
    wstream = OpenapiHeader.from_bytes(data)
    content_length = wstream.message_length + HEADER_LENGTH
    # We use the header's content_length to collect the rest of a package
    while len(data) < content_length:
        packet = socket_handle.recv(content_length - len(data))
        data += packet
    # Now we parse the package
    try:
        package = OpenapiMessage.from_bytes(data)
    except EOFError as e:
        print('Data Invalid {:}'.format(data))
        raise e
    if(package.header.message_type == OpenapiMessage.Header.EMessageType.e_interpretation):
        interpretation_dict = {}
        for interpretation in package.message.interpretations:
            interpretation_dict[interpretation.descriptor_type] = interpretation.value 
        return (package.header.message_type,interpretation_dict)
    elif (package.header.message_type == OpenapiMessage.Header.EMessageType.e_signal_data): # If the data contains signal data
        array = []
        for signal in package.message.signals: # For each signal in the package
            array.append(np.array([x.calc_value for x in signal.values])/2**23)
        return package.header.message_type,np.concatenate(array,axis=-1)
    # If 'quality data' message, then record information on data quality issues
    elif package.header.message_type == OpenapiMessage.Header.EMessageType.e_data_quality:
        for q in package.message.qualities:
            if q.validity_flags.overload:
                print('Overload Detected on {:}:{:}'.format(*socket_handle.getpeername()))
            if q.validity_flags.invalid:
                print('Invalid Data Detected on {:}:{:}'.format(*socket_handle.getpeername()))
            if q.validity_flags.overrun:
                print('Overrun Detected on {:}:{:}'.format(*socket_handle.getpeername()))
        return None,None
    else:
        raise LanXIError("Unknown Message Type: {:}".format(package.header.message_type))

def lanxi_multisocket_reader(socket_handles : List[socket.socket], active_channels_list : List[int], data_queues : List[mp.queues.Queue]):
    """
    Reads data from all channels on multiple modules.
    
    This function is designed to be run by a multiprocessing Process

    Parameters
    ----------
    socket_handles : List[socket.socket]
        A list of the sockets for the cards on this module.
    active_channels_list : List[int]
        A list of number of active channels on each module.
    data_queues : List[mp.queues.Queue]
        A set of queues to pass data back to the main process.

    """
    print('Starting to record from:\n  {:}'.format(
        '\n  '.join(['{:}:{:}'.format(*socket_handle.getpeername()) 
                    for socket_handle in socket_handles])))
    try:
        while True:
            for socket_handle,active_channels,data_queue in zip(socket_handles,active_channels_list,data_queues):
                socket_data = []
                socket_data_types = []
                while len(socket_data) < active_channels:
                    message_type,data = read_lanxi(socket_handle)
                    # print('Reading {:}:{:} Data Type {:}'.format(*socket_handle.getpeername(),message_type))
                    if not message_type is None:
                        socket_data.append(data)
                        socket_data_types.append(message_type)
                # Make sure they are all the same type
                assert all([data_type == socket_data_types[0] for data_type in socket_data_types])
                if socket_data_types[0] == OpenapiMessage.Header.EMessageType.e_interpretation:
                    # print('{:}:{:} Putting Interpretation to Queue'.format(*socket_handle.getpeername()))
                    data_queue.put(('Interpretation',socket_data))
                elif socket_data_types[0] == OpenapiMessage.Header.EMessageType.e_signal_data:
                    # print('{:}:{:} Putting Signal to Queue'.format(*socket_handle.getpeername()))
                    data_queue.put(('Signal',socket_data))
                else:
                    raise ValueError('Unknown Signal Type {:} in {:}:{:}'.format(socket_data_types[0],*socket_handle.getpeername()))
    except LanXIError:
        for socket_handle,data_queue in zip(socket_handles,data_queues):
            # The socket has closed, so gracefully close down
            print('Closing Socket {:}:{:}'.format(*socket_handle.getpeername()))
            while True:
                try:
                    print('Emptying Queue {:}:{:}'.format(*socket_handle.getpeername()))
                    data_queue.get(False)
                except mp.queues.Empty:
                    print('Returning {:}:{:}'.format(*socket_handle.getpeername()))
                    break
        return

def create_harware_maps(acquisition_map,output_map,channel_list):
    for i,channel in enumerate(channel_list):
        if not channel.physical_device in acquisition_map:
            acquisition_map[channel.physical_device] = {}
        acquisition_map[channel.physical_device][int(channel.physical_channel)] = i,channel
    for i,channel in enumerate([channel for channel in channel_list if channel.feedback_device is not None]):
        if not channel.feedback_device in output_map:
            output_map[channel.feedback_device] = {}
        output_map[channel.feedback_device][int(channel.feedback_channel)] = i,channel

def wait_for_ptp_state(host: str, state : str):
    '''Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    '''
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get('http://'+host+'/rest/rec/onchange')
        state_data = response.json()
        current_state = state_data['ptpStatus']
        if current_state == state:
            result = True
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print('Host {:} at {:} state, waiting for {:}'.format(host, current_state,state))
    if not result:
        raise LanXIError('Wait for PTP State {:} timed out on host {:}.  Last retrieved state: {:}'.format(state,host,current_state))
    return result

def wait_for_recorder_state(host: str, state : str):
    '''Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    '''
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get('http://'+host+'/rest/rec/onchange')
        state_data = response.json()
        current_state = state_data['moduleState']
        if current_state == state:
            result = True
            break
        if current_state == 'PostFailed':
            result = False
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print('Host {:} at {:} state, waiting for {:}'.format(host, current_state,state))
    if not result:
        raise LanXIError('Wait for Recorder State {:} timed out on host {:}.  Last retrieved state: {:}'.format(state,host,current_state))
    return result

def wait_for_input_state(host: str, state : str):
    '''Waits until hardware is at a current state

    Parameters
    ----------
    host : str
        The address of the host to wait for
    state : str
        The name of the state to wait until.

    Returns
    -------
    bool
        True if the state has changed, False if the hardware timed out.

    '''
    start_time = time.time()
    current_state = ""
    iteration = 0
    while True:
        response = requests.get('http://'+host+'/rest/rec/onchange')
        state_data = response.json()
        current_state = state_data['inputStatus']
        if current_state == state:
            result = True
            break
        if time.time() - start_time > LANXI_STATE_TIMEOUT:
            result = False
            break
        time.sleep(1)
        iteration += 1
        if iteration % LANXI_STATE_REPORT == 0:
            print('Host {:} at {:} state, waiting for {:}'.format(host, current_state,state))
    if not result:
        raise LanXIError('Wait for Input State {:} timed out on host {:}.  Last retrieved state: {:}'.format(state,host,current_state))
    return result

def close_recorder(host):
    response = requests.get('http://'+host+'/rest/rec/onchange')
    state_data = response.json()
    current_state = state_data['moduleState']
    if current_state == 'RecorderRecording':
        print('Stopping Measurement on {:}'.format(host))
        requests.put('http://'+host+'/rest/rec/measurements/stop')
        wait_for_recorder_state(host, 'RecorderStreaming')
        close_recorder(host)
    elif current_state == 'RecorderConfiguring':
        response = requests.get('http://'+host+'/rest/rec/channels/input/default')
        channel_settings = response.json()
        response = requests.put('http://'+host+"/rest/rec/channels/input", json = channel_settings)
        wait_for_recorder_state(host, 'RecorderStreaming')
        close_recorder(host)
    elif current_state == 'RecorderStreaming':
        print('Finishing Streaming on {:}'.format(host))
        requests.put('http://'+host+'/rest/rec/finish')
        wait_for_recorder_state(host, 'RecorderOpened')
        close_recorder(host)
    elif current_state == 'RecorderOpened':
        print('Closing Recorder on {:}'.format(host))
        requests.put('http://'+host+'/rest/rec/close')
        wait_for_recorder_state(host, 'Idle')
        close_recorder(host)
    elif current_state == 'Idle':
        print('Recorder {:} Idle'.format(host))
    else:
        raise LanXIError('Unknown State {:} on {:}'.format(current_state,host))
    return

class LanXIAcquisition(HardwareAcquisition):
    """Class defining the interface between LAN-XI acquisition and the controller
    
    This class defines the interfaces between the controller and the
    data acquisition portion of the hardware.  It is run by the Acquisition
    process, and must define how to get data from the test hardware into the
    controller."""
    def __init__(self,maximum_processes):
        """
        Constructs the LAN-XI Acquisition class and specifies values to null.
        """
        self.acquisition_map = {}
        self.output_map = {}
        self.sockets = {}
        self.processes = {}
        self.process_data_queues = {}
        self.interpretations = None
        self.master_address = None
        self.slave_addresses = set([])
        self.samples_per_read = None
        self.last_acquisition_time = None
        self.maximum_processes = maximum_processes
        self.modules_per_process = None
        self.total_processes = None
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
        # Now create a hardware map that will help us do bookkeeping
        create_harware_maps(self.acquisition_map, self.output_map, channel_data)
        # Go through the channel table and get the hardware and channel
        # information
        host_addresses = [channel.physical_device for channel in channel_data]
        host_addresses += [channel.feedback_device for channel in channel_data if (not (channel.feedback_device is None) and not (channel.feedback_device.strip() == ''))]
        self.master_address = host_addresses[0]
        self.slave_addresses = set([address for address in host_addresses if not address == self.master_address])
        self.samples_per_read = test_data.samples_per_read
        modules_per_process_floor = len(self.acquisition_map)//self.maximum_processes
        modules_per_process_remainder = len(self.acquisition_map)%self.maximum_processes
        if modules_per_process_remainder == 0:
            self.modules_per_process = modules_per_process_floor
        else:
            self.modules_per_process = modules_per_process_floor + 1
        self.total_processes = (len(self.acquisition_map) // self.modules_per_process) + (0 if len(self.acquisition_map) % self.modules_per_process == 0 else 1)
        self.acquisition_delay = (BUFFER_SIZE+2)*test_data.samples_per_write/test_data.output_oversample

    def start(self):
        """Method to start acquiring data from the hardware"""
        self.sockets = {}
        self.processes = {}
        self.process_data_queues = {}
        # Apply the trigger for multi-frame acquisition
        if len(set(self.acquisition_map)|set(self.output_map)) > 1:
            requests.put('http://'+self.master_address+'/rest/rec/apply')
        # Collect the sockets
        for acquisition_device in self.acquisition_map:
            response = requests.get('http://'+acquisition_device+'/rest/rec/destination/socket')
            port = response.json()['tcpPort']
            # Connect to the socket
            self.sockets[acquisition_device] = socket.socket(socket.AF_INET, socket.SOCK_STREAM,socket.IPPROTO_TCP)
            self.sockets[acquisition_device].connect((acquisition_device,port))
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.post('http://'+slave_address+'/rest/rec/measurements')
        if self.master_address in self.acquisition_map:
            requests.post('http://'+self.master_address+'/rest/rec/measurements')
        print('Started Measurements')
        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address,'RecorderRecording')
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address,'RecorderRecording')
            
        # Here we need to start the processes
        # Split it up into multiple processes
        socket_handles = []
        active_channels_list = []
        data_queues = []
        for acquisition_device,channel_dict in self.acquisition_map.items():
            self.process_data_queues[acquisition_device] = mp.Queue()
            active_channels = len(channel_dict)
            
            socket_handles.append(self.sockets[acquisition_device])
            active_channels_list.append(active_channels)
            data_queues.append(self.process_data_queues[acquisition_device])
            
            if len(socket_handles) % self.modules_per_process == 0:
                self.processes[acquisition_device] = mp.Process(target = lanxi_multisocket_reader, args = (socket_handles, 
                                                                                                           active_channels_list, 
                                                                                                           data_queues))
                self.processes[acquisition_device].start()
                socket_handles = []
                active_channels_list = []
                data_queues = []
        if len(socket_handles) > 0:
            self.processes[acquisition_device] = mp.Process(target = lanxi_multisocket_reader, args = (socket_handles, 
                                                                                                       active_channels_list, 
                                                                                                       data_queues))
            self.processes[acquisition_device].start()
            socket_handles = []
            active_channels_list = []
            data_queues = []
            
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
        """Method to read a frame of data from the hardware"""
        samples = 0
        full_read_data = []
        while samples < self.samples_per_read:
            read_data = []
            acquisition_indices = []
            for acquisition_device,queue_dict in self.process_data_queues.items():
                while True: # We are going to loop until we get a signal which should be every time except the first, which will pass the interpretation.
                    # Get the data from the queues
                    # print('Reading from queue')
                    data_type, data = self.process_data_queues[acquisition_device].get()
                    if data_type == 'Interpretation':
                        if self.interpretations is None:
                            self.interpretations = {}
                        self.interpretations[acquisition_device] = data # Store the interpretation
                    elif data_type == 'Signal':
                        for signal,channel_number,interpretation in zip(data,sorted(self.acquisition_map[acquisition_device]),self.interpretations[acquisition_device]):
                            acquisition_index,channel_data = self.acquisition_map[acquisition_device][channel_number]
                            array = (signal
                                     * interpretation[OpenapiMessage.Interpretation.EDescriptorType.scale_factor] # This is the scale factor
                                     + interpretation[OpenapiMessage.Interpretation.EDescriptorType.offset] # This is the offset
                                     )
                            read_data.append(array)
                            acquisition_indices.append(acquisition_index)
                        break # Exit the loop because we found the signal
            # Check if all the data are the same length
            index_map = np.empty(len(acquisition_indices),dtype=int)
            index_map[acquisition_indices] = np.arange(len(acquisition_indices))
            read_data = np.array(read_data)[index_map]
            full_read_data.append(read_data)
            samples += read_data.shape[-1]
        full_read_data = np.concatenate(full_read_data,axis=-1)
        current_time = time.time()
        if not self.last_acquisition_time is None:
            print('Took {:0.4f}s to read {:} samples'.format(current_time-self.last_acquisition_time,full_read_data.shape[-1]))
        self.last_acquisition_time = current_time
        return full_read_data
    
    def read_remaining(self):
        """Method to read the rest of the data on the acquisition from the hardware"""
        return np.zeros((sum([1 for acquisition_device,acquisition_dict in self.acquisition_map.items() for channel_number in acquisition_dict]),1))
    
    def stop(self):
        """Method to stop the acquisition"""
        if self.master_address in self.acquisition_map:
            requests.put('http://'+self.master_address+'/rest/rec/measurements/stop')
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.put('http://'+slave_address+'/rest/rec/measurements/stop')
        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address,'RecorderStreaming')
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address,'RecorderStreaming')
        # Join the processes
        for acquisition_device,process in self.processes.items():
            print('Recovering process {:}'.format(acquisition_device))
            process.join()
            print('Process {:} recovered'.format(acquisition_device))
        print('All processes recovered, ready for next acquire.')
        self.processes = {}
        self.process_data_queues = {}
        self.interpretations = None
        self.last_acquisition_time = None
    
    def close(self):
        """Method to close down the hardware"""
        if len(self.processes) > 0: # This means we are still running!
            self.stop()

    def _get_states(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            response = requests.get('http://'+host+'/rest/rec/onchange')
            state_data = response.json()
            print('Host {:}: Recorder State {:}, Input State {:}, PTP State {:}, Recording Mode'.format(
                host,state_data['moduleState'],state_data['inputStatus'],state_data['ptpStatus']))

    def _reboot_all(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            requests.put('http://'+host+'/rest/rec/reboot')
    
class LanXIOutput(HardwareOutput):
    """Abstract class defining the interface between the controller and output
    
    This class defines the interfaces between the controller and the
    output or source portion of the hardware.  It is run by the Output
    process, and must define how to get write data to the hardware from the
    control system"""
    
    def __init__(self,maximum_processes):
        """Method to start up the hardware"""
        self.sockets = {}
        self.acquisition_map = {}
        self.output_map = {}
        self.master_address = None
        self.slave_addresses = set([])
        self.oversample_factor = None
        self.output_rate = None
        self.bandwidth_string = None
        self.transfer_size = 4096*4
        self.write_size = None
        self.empty_time = 0.0
        self.generator_sample_rate = None
        self.buffer_size = 5
        self.ready_signal_factor = BUFFER_SIZE
        self.maximum_processes = maximum_processes

    def set_up_data_output_parameters_and_channels(self,
                                                   test_data : DataAcquisitionParameters,
                                                   channel_data : List[Channel]):
        # Create a hardware map that will help us do bookkeeping
        create_harware_maps(self.acquisition_map, self.output_map, channel_data)
        self.write_size = test_data.samples_per_write
        self.sample_rate = test_data.sample_rate
        # Go through the channel table and get the hardware and channel
        # information
        host_addresses = [channel.physical_device for channel in channel_data]
        host_addresses += [channel.feedback_device for channel in channel_data if (not (channel.feedback_device is None) and not (channel.feedback_device.strip() == ''))]
        self.master_address = host_addresses[0]
        self.slave_addresses = set([address for address in host_addresses if not address == self.master_address])
        print('\nInitial States:')
        self._get_states()
        
        #TODO REMOVE THIS
        # time.sleep(10)
        
        # Close all devices to start from scratch
        print('Resetting Data Acquisition System')
        self.close(reboot=False)
        
        #TODO REMOVE THIS
        # time.sleep(10)
        
        # If there are any slave addresses, need to perform PTP sync
        if len(self.slave_addresses) > 0:
            print('PTP Mode')
            master_json = {
                "synchronization": {
                    "mode": "ptp",
                    "domain": 42,
                    "preferredMaster": True
                }
            }
            requests.put('http://'+self.master_address+'/rest/rec/syncmode',json=master_json)
            slave_json = {
                "synchronization": {
                    "mode": "ptp",
                    "domain": 42,
                    "preferredMaster": False
                }
            }
            for slave_address in self.slave_addresses:
                requests.put('http://'+slave_address+'/rest/rec/syncmode',json=slave_json)
            print('Waiting for PTP Sync...')
            # Wait until PTP locks
            for slave_address in self.slave_addresses:
                wait_for_ptp_state(slave_address,'Locked')
            wait_for_ptp_state(self.master_address,'Locked')
            print('PTP Synced!')
            single_module = False
        else:
            print('Single Module Mode')
            master_json = {
                "synchronization": {
                    "mode": "stand-alone"
                }
            }
            requests.put('http://'+self.master_address+'/rest/rec/syncmode',json=master_json)
            single_module = True
        print('\nStates after synchronization')
        self._get_states()
        
        # Now we open the recorders
        open_json = {
                    	"performTransducerDetection": False, # May need to investigate this further, but for now we won't use TEDS
                    	"singleModule": single_module
                    }
        for slave_address in self.slave_addresses:
            requests.put('http://'+slave_address+'/rest/rec/open',json=open_json)
        requests.put('http://'+self.master_address+'/rest/rec/open',json=open_json)
        print('\nStates after Open')
        self._get_states()
        
        # Now get the sample rate
        for i,address in enumerate(self.acquisition_map):
            response = requests.get('http://'+address + "/rest/rec/module/info")
            module_info = response.json()
            if i == 0:
                supported_sample_rates = module_info['supportedSampleRates']
            else:
                supported_sample_rates = [v for v in supported_sample_rates if v in module_info['supportedSampleRates']]
        print('Supported Sample Rates {:}'.format(supported_sample_rates))
        bandwidth = test_data.sample_rate/2.56
        if bandwidth > 1000:
            self.bandwidth_string = '{:0.1f} kHz'.format(bandwidth/1000)
        else:
            self.bandwidth_string = str(round(bandwidth))+' Hz'
        print('Sample Rate: {:} Hz (Bandwidth {:})'.format(test_data.sample_rate,self.bandwidth_string))
        if not test_data.sample_rate in supported_sample_rates:
            raise LanXIError('Invalid Sample Rate {:}, must be one of {:}'.format(test_data.sample_rate,supported_sample_rates))
        
        # Get the Generator Sample Rate
        self.generator_sample_rate = round(np.log2(OUTPUT_RATE/test_data.sample_rate))
        if self.generator_sample_rate > 3:
            self.generator_sample_rate = 3
        elif self.generator_sample_rate < 0:
            self.generator_sample_rate = 0
        self.output_rate = OUTPUT_RATE//2**(self.generator_sample_rate)
        # Now prep the generators
        self.set_generators()
            
        # Now we need to set up the recording configuration
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                requests.put('http://'+slave_address+'/rest/rec/create')
            else:
                print('Skipping Creating Slave Address Recorder {:}, not in acquisition'.format(slave_address))
        if self.master_address in self.acquisition_map:
            requests.put('http://'+self.master_address+'/rest/rec/create')
        else:
            print('Skipping Creating Master Address Recorder {:}, not in acquisition'.format(self.master_address))
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address,'RecorderConfiguring')
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address,'RecorderConfiguring')
        print('\nStates after Recorder Create')
        self._get_states()
        
        # State is now in Recorder Configuring
        print('Recorder in Configuring State')
        # Now we have to go through and create the channels
        for acquisition_device,device_dictionary in self.acquisition_map.items():
            response = requests.get('http://'+acquisition_device+'/rest/rec/channels/input/default')
            channel_settings = response.json()
            # Go through and disable all channels
            for channel_json in channel_settings['channels']:
                channel_json['enabled'] = False
            for (channel_number,(channel_index,channel)) in device_dictionary.items():
                index,channel_json = [(i,channel_json) for i,channel_json in enumerate(channel_settings['channels']) if channel_json['channel'] == channel_number][0]
                channel_json['bandwidth'] = self.bandwidth_string
                channel_json['ccld'] = False if channel.excitation_source is None else True
                channel_json['transducer']['requiresCcld'] = channel_json['ccld']
                if channel_json['ccld']:
                    print('Device {:} channel {:} has CCLD enabled'.format(acquisition_device,channel_number))
                channel_json['destinations'] = ['socket']
                channel_json['enabled'] = True
                channel_coupling = 'DC' if channel.coupling is None else channel.coupling
                if not channel_coupling in VALID_FILTERS:
                    raise LanXIError('For LAN-XI, Coupling must be sent to one of {:}'.format(VALID_FILTERS))
                channel_json['filter'] = channel_coupling
                if not channel.maximum_value in VALID_RANGES:
                    raise LanXIError('For LAN-XI, Maximum Value must be one of {:}'.format(VALID_RANGES))
                channel_json['range'] = channel.maximum_value + ' Vpeak'
                channel_json['transducer']['sensitivity'] = float(channel.sensitivity)/1000
                channel_json['transducer']['serialNumber'] = 9999#('' if channel.serial_number is None else channel.serial_number)+('' if channel.triax_dof is None else channel.triax_dof)]
                channel_json['transducer']['type']['model'] = ('' if channel.make is None else channel.make) + ('' if channel.model is None else ' ' + channel.model)
                channel_json['transducer']['unit'] = channel.unit
            response = requests.put('http://'+acquisition_device+"/rest/rec/channels/input", json = channel_settings)
            print('Setting inputs to {:} Channels, {:} {:}'.format(acquisition_device,response.status_code,response.text))
        print('\nStates after Channel Input')
        self._get_states()
        
        # Now check for synchronization
        if len(self.slave_addresses) > 0:
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    wait_for_input_state(slave_address,'Settled')
            if self.master_address in self.acquisition_map:
                wait_for_input_state(self.master_address,'Settled')
            print('Recorder Settled, Synchronizing...')
            
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    requests.put('http://'+slave_address+'/rest/rec/synchronize')
            if self.master_address in self.acquisition_map:
                requests.put('http://'+self.master_address+'/rest/rec/synchronize')
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    wait_for_input_state(slave_address,'Synchronized')
            if self.master_address in self.acquisition_map:
                wait_for_input_state(self.master_address,'Synchronized')
            print('Recorder Synchronized, Starting Streaming...')
            
            for slave_address in self.slave_addresses:
                if slave_address in self.acquisition_map:
                    requests.put('http://'+slave_address+'/rest/rec/startstreaming')
            if self.master_address in self.acquisition_map:
                requests.put('http://'+self.master_address+'/rest/rec/startstreaming')
                
        # Wait for the module state to be recorder streaming
        for slave_address in self.slave_addresses:
            if slave_address in self.acquisition_map:
                wait_for_recorder_state(slave_address,'RecorderStreaming')
        if self.master_address in self.acquisition_map:
            wait_for_recorder_state(self.master_address,'RecorderStreaming')
        print('Recorder Streaming')
        self._get_states()
        
        print('\n\nData Acquisition Ready for Acquire')
    
    def start(self):
        """Method to start outputting data to the hardware"""
        self.empty_time += time.time()
        # Now start the generators
        master_json = None
        for generator_device,generator_channel_dict in self.output_map.items():
            json = {
                        'outputs':
                            [
                                {"number":channel_number} for channel_number in generator_channel_dict
                            ]
                    }
            if generator_device == self.master_address:
                master_json = json # Pull this out because the master should be assigned last I think.
                continue
            requests.put('http://'+generator_device+'/rest/rec/generator/start',json=json)
        if not master_json is None:
            requests.put('http://'+self.master_address+'/rest/rec/generator/start',json=master_json)
        print('States after Generator Started')
        self._get_states()
    
    def write(self,data):
        """Method to write a frame of data to the hardware"""
        for output_device,socket_dict in self.sockets.items():
            for channel_number,socket_handle in socket_dict.items():
                output_index,channel_data = self.output_map[output_device][channel_number]
                this_data = (data[output_index]/10 * 8372224).astype('int32').tobytes()
                while len(this_data) > 0:
                    sent_bytes = socket_handle.send(this_data[:self.transfer_size])
                    this_data = this_data[sent_bytes:]
        self.empty_time += self.write_size / self.output_rate
                
    def stop(self):
        """Method to stop the output"""
        master_json = None
        for generator_device,generator_channel_dict in self.output_map.items():
            json = {
                        'outputs':
                            [
                                {"number":channel_number} for channel_number in generator_channel_dict
                            ]
                    }
            if generator_device == self.master_address:
                master_json = json # Pull this out because the master should be assigned last I think.
                continue
            requests.put('http://'+generator_device+'/rest/rec/generator/stop',json=json)
        if master_json is not None:
            requests.put('http://'+self.master_address+'/rest/rec/generator/stop',json=master_json)
        self.empty_time = 0.0
        self.set_generators()
        
        
    def set_generators(self):
        if len(self.output_map) == 0:
            return
        master_json = None
        for generator_device,generator_channel_dict in self.output_map.items():
            json = {
                        'outputs':
                            [
                                {"number":channel_number} for channel_number in generator_channel_dict
                            ]
                    }
            if generator_device == self.master_address:
                master_json = json # Pull this out because the master should be assigned last I think.
                continue
            requests.put('http://'+generator_device+'/rest/rec/generator/prepare',json=json)
        if not master_json is None:
            requests.put('http://'+self.master_address+'/rest/rec/generator/prepare',json=master_json)
        print('\nStates after Generator Prepare')
        self._get_states()
        
        # Configure the generator channels
        master_json = None
        for generator_device,generator_channel_dict in self.output_map.items():
            json = {
                        'bufferSize' : self.buffer_size*self.write_size, # TODO: Re-evaluate this number
                        'outputs' : 
                            [
                                {
                                    "number" : channel_number,
                                    "floating" : False,
                                    "gain" : 1.0,
                                    "inputs" : [
                                            {
                                                'number' : 1,
                                                'signalType' : 'stream',
                                                'gain' : 1.0,
                                                'offset' : 0.0,
                                                'samplingRate' : self.generator_sample_rate,
                                            },
                                            {
                                                'number' : 2,
                                                'signalType' : 'none'
                                            }
                                        ]
                                    
                                } for channel_number in generator_channel_dict
                            ]
                    }
            if generator_device == self.master_address:
                master_json = json # Pull this out because the master should be assigned last I think.
                continue
            requests.put('http://'+generator_device+'/rest/rec/generator/output',json=json)
        if not master_json is None:
            requests.put('http://'+self.master_address+'/rest/rec/generator/output',json=master_json)
        print('\nStates after Generator Output')
        self._get_states()
        
        # Now pull the socket information for the outputs
        for generator_device,generator_dict in self.output_map.items():
            response = requests.get('http://'+generator_device+'/rest/rec/generator/output')
            output_data = response.json()
            for channel_number in generator_dict:
                output = [out for out in output_data['outputs'] if out['number'] == channel_number][0]
                if not generator_device in self.sockets:
                    self.sockets[generator_device] = {}
                self.sockets[generator_device][output['number']] = socket.socket(socket.AF_INET, socket.SOCK_STREAM,socket.IPPROTO_TCP)
                self.sockets[generator_device][output['number']].connect((generator_device,output['inputs'][0]['port']))
                print('Output Connected to Device {:} Channel {:} on Port {:}'.format(generator_device,output['number'],output['inputs'][0]['port']))
                self.oversample_factor = round(OUTPUT_RATE/(2**output['inputs'][0]['samplingRate'])/self.sample_rate)
            print('Output overampling factor: {:}x'.format(self.oversample_factor))
        
    
    def close(self,reboot=False):
        """Method to close down the hardware"""
        for device,socket_dict in self.sockets.items():
            for channel_number,socket_handle in socket_dict.items():
                socket_handle.close()
        if reboot:
            self._reboot_all()
        else:
            self._close_recorders([self.master_address] + [address for address in self.slave_addresses])
        # self._reboot_all()
        # self._close_recorder(self.master_address)
        # for slave_address in self.slave_addresses:
        #     self._close_recorder(slave_address)
    
    def ready_for_new_output(self):
        """Method that returns true if the hardware should accept a new signal"""
        # print('Time until output buffer empty {:}, time per write {:}'.format(self.empty_time - time.time(),self.write_size / self.output_rate))
        if (self.empty_time - time.time()) < (self.write_size / self.output_rate)*self.ready_signal_factor: # TODO: Might need to increase buffer
            # print('Need new output')    
            return True
        else:
            # print('No output needed')
            return False
    
    def _close_recorders(self,hosts):
        with mp.Pool(len(hosts)) as pool: # Not sure if this can be len(hosts) or if it should be self.maximum_processes
           pool.map(close_recorder,hosts)
        # host_states = {}
        # while True:
        #     for host in hosts:
        #         # print('Getting state from host {:}'.format(host))
        #         response = requests.get('http://'+host+'/rest/rec/onchange')
        #         state_data = response.json()
        #         current_state = state_data['moduleState']
        #         # print('Got state from host {:}'.format(host))
        #         if host in host_states and host_states[host] == current_state:
        #             continue
        #         host_states[host] = current_state
        #         try:
        #             operation = LANXI_STATE_SHUTDOWN[current_state]
        #         except KeyError:
        #             print('Unknown State {:} for host {:}.  Rebooting'.format(current_state,host))
        #             requests.put('http://'+host+'/rest/rec/reboot')
        #             continue
        #         if not operation is None:
        #             print('Host {:} at {:} state: {:}'.format(host,current_state,operation))
        #             requests.put('http://'+host+operation)
        #     # Check if all hosts are idle
        #     if all([v == 'Idle' for k,v in host_states.items()]):
        #         print('All hosts are idle')
        #         break
        #     time.sleep(0.2)

    def _get_states(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            response = requests.get('http://'+host+'/rest/rec/onchange')
            state_data = response.json()
            print('Host {:}: Recorder State {:}, Input State {:}, PTP State {:}, Recording Mode'.format(
                host,state_data['moduleState'],state_data['inputStatus'],state_data['ptpStatus']))

    def _reboot_all(self):
        for host in list(self.slave_addresses) + [self.master_address]:
            requests.put('http://'+host+'/rest/rec/reboot')
    