# -*- coding: utf-8 -*-
"""
This file contains a number of helper classes for the general controller.

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

import numpy as np
from qtpy import QtWidgets
import scipy.signal as sig
import multiprocessing as mp
import multiprocessing.queues as mp_queues
from enum import Enum
import time
from datetime import datetime
from typing import List,Tuple,Dict
import importlib.util
import os
    
time_reporting_threshold = 0.01

def coherence(cpsd_matrix : np.ndarray,row_column : Tuple[int] = None ):
    """Compute coherence from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex numpy array where the first index corresponds to the
        frequency line and the second and third indices correspond to the rows
        and columns of the matrix.
    row_column : Tuple[int] :
        Optional argument to compute the coherence at just a single (row,column)
        pair.  (Default value = Compute Entire Matrix)

    Returns
    -------
    coherence : np.ndarray :
        3D array of coherence values where the [i,j,k] entry corresponds to the
        coherence of the CPSD matrix for the ith frequency line, jth row, and
        kth column.
    
    """
    if row_column is None:
        diag = np.einsum('ijj->ij',cpsd_matrix)
        return np.real(np.abs(cpsd_matrix)**2/(diag[:,:,np.newaxis]*diag[:,np.newaxis,:]))
    else:
        row,column = row_column
        return np.real(np.abs(cpsd_matrix[:,row,column])**2/(cpsd_matrix[:,row,row]*cpsd_matrix[:,column,column]))

class Channel:
    """Property container for a single channel in the controller."""
    def __init__(self,node_number,node_direction,comment,
                 serial_number,triax_dof,sensitivity,unit,make,model,expiration,
                 physical_device,physical_channel,channel_type,
                 minimum_value,maximum_value,coupling,excitation_source,excitation,
                 feedback_device,feedback_channel,warning_level,abort_level):
        """Property container for a single channel in the controller.
        
        Parameters
        ----------
        node_number : str :
            Metadata specifying the node number
        node_direction : str : 
            Metadata specifying the direction at a node
        comment : str :
            Metadata specifying any additional comments on the channel
        serial_number : str :
            Metadata specifying the serial number of the instrument
        triax_dof : str :
            Metadata specifying the degree of freedom on a triaxial sensor
        sensitivity : str :
            Sensitivity value of the sensor in mV/engineering unit
        unit : str :
            The engineering unit of the sensor
        make : str :
            Metadata specifying the make of the sensor
        model : str :
            Metadata specifying the model of the sensor
        expiration : str :
            Metadata specifying the expiration date of the sensor
        physical_device : str :
            Physical hardware that the instrument is connected to
        physical_channel : str :
            Channel in the physical hardware that the instrument is connected to
        channel_type : str :
            Type of channel
        minimum_value : str :
            Minimum value of the channel in volts
        maximum_value : str :
            Maximum value of the channel in volts
        coupling : str :
            Coupling type for the channel
        excitation_source : str :
            Source for the signal conditioning for the sensor
        excitation : str :
            Level of excitation for the signal conditioning for the sensor
        feedback_device : str :
            Physical hardware that the source output teed into this channel
            originates from
        feedback_channel : str :
            Channel on the physical hardware that is teed into this channel
        warning_level : str :
            Level at which warnings will be flagged on the monitor
        abort_level : str :
            Level at which the system will shut down
        """
        self.node_number = node_number
        self.node_direction = node_direction
        self.comment = comment
        self.serial_number = serial_number
        self.triax_dof = triax_dof
        self.sensitivity = sensitivity
        self.make = make
        self.model = model
        self.expiration = expiration
        self.physical_device = physical_device
        self.physical_channel = physical_channel
        self.channel_type = channel_type
        self.unit = unit
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.coupling = coupling
        self.excitation_source = excitation_source
        self.excitation = excitation
        self.feedback_device = feedback_device
        self.feedback_channel = feedback_channel
        self.warning_level = warning_level
        self.abort_level = abort_level
    
    @classmethod
    def from_channel_table_row(cls,row : Tuple[str]):
        """Creates a Channel object from a row in the channel table

        Parameters
        ----------
        row : iterable :
            Iterable of strings from a single row of the channel table
            

        Returns
        -------
        channel : Channel
            A channel object containing the data in the given row of the
            channel table.
        
        """
        new_row = [None if val.strip() == '' else val for val in row]
        physical_device = new_row[10]
        if physical_device is None:
            return None
        node_number = new_row[0]
        node_direction = new_row[1]
        comment = new_row[2]
        serial_number = new_row[3]
        triax_dof = new_row[4]
        sensitivity = new_row[5]
        unit = new_row[6]
        make = new_row[7]
        model = new_row[8]
        expiration = new_row[9]
        physical_channel = new_row[11]
        channel_type = new_row[12]
        minimum_value = new_row[13]
        maximum_value = new_row[14]
        coupling = new_row[15]
        excitation_source = new_row[16]
        excitation = new_row[17]
        feedback_device = new_row[18]
        feedback_channel = new_row[19]
        warning_level = new_row[20]
        abort_level = new_row[21]
        return cls(node_number,node_direction,comment,
                 serial_number,triax_dof,sensitivity,unit,make,model,expiration,
                 physical_device,physical_channel,channel_type,
                 minimum_value,maximum_value,coupling,excitation_source,excitation,
                 feedback_device,feedback_channel,warning_level,abort_level)

class DataAcquisitionParameters:
    """Container to hold the global data acquisition parameters of the controller"""
    def __init__(self,channel_list : List[Channel],sample_rate,samples_per_read,
                 samples_per_write,hardware,hardware_file,environment_names,
                 environment_bools,output_oversample,maximum_acquisition_processes):
        """Container to hold the global data acquisition parameters of the controller
        
        Parameters
        ----------
        channel_list : List[Channel]:
            An iterable containing Channel objects for each channel in the
            controller.
        sample_rate : int :
            Number of samples per second that the data acquisition runs at
        samples_per_read : int :
            Number of samples the data acquisition will acquire with each read.
            Smaller numbers here will result in finer resolution for starting
            and stopping environments, but will be more computationally
            intensive to run.
        samples_per_write : int :
            Number of samples the data acquisition will output with each write.
            Smaller numbers here will result in finer resolution for starting
            and stopping environments, but will be more computationally
            intensive to run.
        hardware : int :
            Hardware index corresponding to the QCombobox selector on the 
            Channel Table tab of the main controller
        hardware_file : str :
            Path to an optional file that completes the hardware specification,
            for example, a finite element model results.
        environment_names : List[str]:
            A list of the names of environments in the controller
        environment_bools : np.ndarray :
            A 2D array specifying which channels are active in which environment.
            If the [i,j] component is True, then the ith channel is active in
            the jth environment.
        output_oversample : int
            Oversample factor of the output generator
        maximum_acquisition_processes : int
            Maximum number of processes to spin up to read data off the acquisition
        """
        self.channel_list = channel_list
        self.sample_rate = sample_rate
        self.samples_per_read = samples_per_read
        self.samples_per_write = samples_per_write
        self.hardware = hardware
        self.hardware_file = hardware_file
        self.environment_names = environment_names
        self.environment_active_channels = environment_bools
        self.output_oversample = output_oversample
        self.maximum_acquisition_processes = maximum_acquisition_processes
    
    @property
    def nyquist_frequency(self):
        """Property returning the Nyquist frequency of the data acquisition."""
        return self.sample_rate/2
    
    @property
    def output_sample_rate(self):
        """Property returning the output sample rate"""
        return self.sample_rate*self.output_oversample
        
def error_message_qt(title,message):
    """Helper class to create an error dialog.

    Parameters
    ----------
    title : str :
        Title of the window that the error message will appear in.
    message : str :
        Error message that will be displayed.
    
    """
    QtWidgets.QMessageBox.critical(None,title,message)

class VerboseMessageQueue():
    """A queue class that contains automatic logging information"""
    def __init__(self,log_queue,queue_name):
        """
        A queue class that contains automatic logging information

        Parameters
        ----------
        log_queue : mp.queues.Queue :
            A queue that a logging task will read from where the operations of
            the queue will be logged.
        queue_name : str :
            The name of the queue that will be included in the logging information

        """
        self.queue = mp.Queue()
        self.log_queue = log_queue
        self.queue_name = queue_name
        self.last_put_message = None
        self.last_put_time = -float('inf')
        self.last_get_message = None
        self.last_get_time = -float('inf')
        self.last_flush = -float('inf')
        self.time_threshold = 1.0
    
    def put(self,task_name,message_data_tuple,*args,**kwargs):
        """Puts data to a verbose queue

        Parameters
        ----------
        task_name : str
            Task name that is performing the put operation
        message_data_tuple : Tuple
            A (message,data) tuple where message is the instruction and data is
            any optional data to be passed along with the instruction.
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.put
            function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.put
            function

        """
        put_time = time.time()
        if self.last_put_message != message_data_tuple[0] or put_time - self.last_put_time > self.time_threshold:
            self.log_queue.put('{:}: {:} put {:} to {:}\n'.format(datetime.now(),task_name,message_data_tuple[0].name,self.queue_name))
            self.last_put_message = message_data_tuple[0]
            self.last_put_time = put_time
        self.queue.put(message_data_tuple,*args,**kwargs)
        
    def get(self,task_name,*args,**kwargs):
        """Gets data from a verbose queue

        Parameters
        ----------
        task_name : str :
            Name of the task that is retrieving data from the queue
        *args :
            Additional arguments that will be passed to the mp.queues.Queue.get
            function
        **kwargs :
            Additional arguments that will be passed to the mp.queues.Queue.get
            function
            

        Returns
        -------
        message_data_tuple : 
            A (message,data) tuple
        
        """
        get_time = time.time()
        message_data_tuple = self.queue.get(*args,**kwargs)
        if self.last_get_message != message_data_tuple[0] or get_time - self.last_get_time > self.time_threshold:
            self.log_queue.put('{:}: {:} got {:} from {:}\n'.format(datetime.now(),task_name,message_data_tuple[0].name,self.queue_name))
            self.last_get_message = message_data_tuple[0]
            self.last_get_time = get_time
        return message_data_tuple
    
    def flush(self,task_name):
        """Flushes a verbose queue getting all data currently in the queue
        
        After execution the queue should be empty barring race conditions.

        Parameters
        ----------
        task_name : str :
            Name of the task that is flushing the queue
            

        Returns
        -------
        data : iterable of message_data_tuples :
            A list of all (message,data) tuples currently in the queue.
        
        """
        flush_time = time.time()
        if flush_time - self.last_flush > 0.1:
            self.log_queue.put('{:}: {:} flushed {:}\n'.format(datetime.now(),task_name,self.queue_name))
            self.last_flush = flush_time
        data = []
        while True:
            try:
                data.append(self.queue.get(False))
                self.log_queue.put('{:}: {:} got {:} from {:} during flush\n'.format(datetime.now(),task_name,data[-1][0].name,self.queue_name))
            except mp.queues.Empty:
                return data
            
    def empty(self):
        """Return true if the queue is empty."""
        return self.queue.empty()

class QueueContainer:
    """A container class for the queues that the controller will manage"""
    def __init__(self,controller_communication_queue: VerboseMessageQueue,
                      acquisition_command_queue: VerboseMessageQueue,
                      output_command_queue: VerboseMessageQueue,
                      streaming_command_queue: VerboseMessageQueue,
                      log_file_queue: mp_queues.Queue,
                      input_output_sync_queue: mp_queues.Queue,
                      single_process_hardware_queue: mp_queues.Queue,
                      gui_update_queue: mp_queues.Queue,
                      environment_command_queues: Dict[str,VerboseMessageQueue],
                      environment_data_in_queues: Dict[str,mp_queues.Queue],
                      environment_data_out_queues: Dict[str,mp_queues.Queue]
                      ):
        """A container class for the queues that the controller will manage.
        
        The controller uses many queues to pass data between the various pieces.
        This class organizes those queues into one common namespace.

        Parameters
        ----------
        controller_communication_queue : VerboseMessageQueue
            Queue that is read by the controller for global controller commands
        acquisition_command_queue : VerboseMessageQueue
            Queue that is read by the acquisition subtask for acquisition commands
        output_command_queue : VerboseMessageQueue
            Queue that is read by the output subtask for output commands
        streaming_command_queue : VerboseMessageQueue
            Queue that is read by the streaming subtask for streaming commands
        log_file_queue : mp_queues.Queue
            Queue for putting logging messages that will be read by the logging
            subtask and written to a file.
        input_output_sync_queue : mp_queues.Queue
            Queue that is used to synchronize input and output signals
        single_process_hardware_queue : mp_queues.Queue
            Queue that is used to connect the acquisition and output subtasks
            for hardware implementations that cannot have acquisition and
            output in separate processes.
        gui_update_queue : mp_queues.Queue
            Queue where various subtasks put instructions for updating the
            widgets in the user interface
        environment_command_queues : Dict[str,VerboseMessageQueue]
            A dictionary where the keys are environment names and the values are
            VerboseMessageQueues that connect the main controller to the
            environment subtasks for sending instructions.
        environment_data_in_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the acquisition subtask to the
            environment subtask.  Each environment will retrieve acquired data
            from this queue.
        environment_data_out_queues : Dict[str,multiprocessing.queues.Queue]
            A dictionary where the keys are environment names and the values are
            multiprocessing queues that connect the output subtask to the
            environment subtask.  Each environment will put data that it wants
            the controller to generate in this queue.

        """
        self.controller_communication_queue = controller_communication_queue
        self.acquisition_command_queue = acquisition_command_queue
        self.output_command_queue = output_command_queue
        self.streaming_command_queue = streaming_command_queue
        self.log_file_queue = log_file_queue
        self.input_output_sync_queue = input_output_sync_queue
        self.single_process_hardware_queue = single_process_hardware_queue
        self.gui_update_queue = gui_update_queue
        self.environment_command_queues = environment_command_queues
        self.environment_data_in_queues = environment_data_in_queues
        self.environment_data_out_queues = environment_data_out_queues

def load_csv_matrix(file):
    """Loads a matrix from a CSV file

    Parameters
    ----------
    file : str :
        Path to the file that will be loaded
        

    Returns
    -------
    data : list[list[str]]
        A 2D nested list of strings containing the matrix in the CSV file.
    
    """
    with open(file,'r') as f:
        data = []
        for line in f:
            data.append([])
            for v in line.split(','):
                data[-1].append(v.strip())
    return data

def save_csv_matrix(data,file):
    """Saves 2D matrix data to a file

    Parameters
    ----------
    data : 2D iterable of str:
        A 2D nested iterable of strings that will be written to a file
    file : str :
        The path to a file where the data will be written.
    
    """
    text = '\n'.join([','.join(row) for row in data])
    with open(file,'w') as f:
        f.write(text)
        
def cpsd_to_time_history(cpsd_matrix,sample_rate,df,output_oversample = 1):
    """Generates a time history realization from a CPSD matrix

    Parameters
    ----------
    cpsd_matrix : np.ndarray :
        A 3D complex np.ndarray representing a CPSD matrix where the first
        dimension is the frequency line and the second two dimensions are the
        rows and columns of the matrix at each frequency line.
    sample_rate: float :
        The sample rate of the controller in samples per second
    df : float :
        The frequency spacing of the cpsd matrix
        

    Returns
    -------
    output : np.ndarray :
        A numpy array containing the generated signals
        
    Notes
    -----
    Uses the process described in [1]_
    
    .. [1] R. Schultz and G. Nelson, "Input signal synthesis for open-loop
       multiple-input/multiple-output testing," Proceedings of the International
       Modal Analysis Conference, 2019.
    
    """
    # Compute SVD broadcasting over all frequency lines
    [U,S,Vh] = np.linalg.svd(cpsd_matrix,full_matrices=False)
    # Reform using the sqrt of the S matrix
    Lsvd = U*np.sqrt(S[:,np.newaxis,:])@Vh
    # Compute Random Process
    W = np.sqrt(0.5)*(np.random.randn(*cpsd_matrix.shape[:-1],1)+1j*np.random.randn(*cpsd_matrix.shape[:-1],1))
    Xv = 1/np.sqrt(df) * Lsvd@W
    # Ensure that the signal is real by setting the nyquist and DC component to 0
    Xv[[0,-1],:,:] = 0
    # Compute the IFFT, using the real version makes it so you don't need negative frequencies
    zero_padding = np.zeros([((output_oversample-1)*(Xv.shape[0]-1))]+list(Xv.shape[1:]),dtype=Xv.dtype)
    xv = np.fft.irfft(np.concatenate((Xv,zero_padding),axis=0)/np.sqrt(2),axis=0)*output_oversample*sample_rate
    output = xv[:,:,0].T
    return output

def pseudorandom_signal(fmin,fmax,df,sample_rate,rms,nsignals = 1):
    fnyq = sample_rate/2
    f = np.arange(fnyq/df+1)*df
    xfft = np.zeros((nsignals,f.size),dtype='complex128')
    freq_indices = (f > fmin) & (f <= fmax)
    xfft[:,freq_indices] = np.random.randn(nsignals,freq_indices.sum()) + 1j*np.random.randn(nsignals,freq_indices.sum())
    # Make sure nyquist and dc are real
    xfft[:,0] = 0 # DC is zero
    xfft[:,-1] = xfft[:,-1].real
    x = np.fft.irfft(xfft,axis=-1)
    x /= rms_time(x,-1,True)/rms
    return x

def flush_queue(queue,timeout=None):
    """Flushes a queue by getting all the data currently in it.

    Parameters
    ----------
    queue : mp.queues.Queue or VerboseMessageQueue:
        The queue to flush
        

    Returns
    -------
    data : iterable
        A list of all data that were in the queue at flush
    
    """
    data = []
    while True:
        try:
            if isinstance(queue,VerboseMessageQueue):
                data.append(queue.get('Flush',block = False if timeout is None else True,timeout=timeout))
            else:
                data.append(queue.get(block = False if timeout is None else True,timeout=timeout))
        except mp.queues.Empty:
            return data
        
def db2scale(dB):
    """ Converts a decibel value to a scale factor

    Parameters
    ----------
    dB : float :
        Value in decibels
        

    Returns
    -------
    scale : float :
        Value in linear
    
    """
    return 10**(dB/20)

power2db = lambda power: 10*np.log10(power)

scale2db = lambda scale: 20*np.log10(scale)

def rms_time(signal,axis=None,keepdims=False):
    """Computes RMS over a time signal

    Parameters
    ----------
    signal : np.ndarray :
        Signal over which to compute the root-mean-square value
    axis : int :
        The dimension over which the mean is performed (Default value = None)
    keepdims : bool :
        Whether to keep the dimension over which mean is computed (Default value = False)

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signal
    
    """
    return np.sqrt(np.mean(signal**2,axis=axis,keepdims=keepdims))

def rms_csd(csd,df):
    """Computes RMS of a CPSD matrix

    Parameters
    ----------
    csd : np.ndarray :
        3D complex Numpy array where the first dimension is the frequency line
        and the second two dimensions are the rows and columns of the CPSD
        matrix.
    df : float :
        Frequency spacing of the CPSD matrix

    Returns
    -------
    rms : numpy scalar or numpy.ndarray
        The root-mean-square value of signals in the CPSD matrix
    
    """
    return np.sqrt(np.einsum('ijj->j',csd).real*df)

def trac(th_1,th_2=None):
    if th_2 is None:
        th_2 = th_1
    th_1_original_shape = th_1.shape
    th_1_flattened = th_1.reshape(-1,th_1.shape[-1])
    th_2_flattened = th_2.reshape(-1,th_2.shape[-1])
    trac = np.abs(np.sum(th_1_flattened*th_2_flattened.conj(),axis=-1))**2/((np.sum(th_1_flattened*th_1_flattened.conj(),axis=-1))*np.sum(th_2_flattened*th_2_flattened.conj(),axis=-1))
    return trac.reshape(th_1_original_shape[:-1])

def align_signals(measurement_buffer,specification,correlation_threshold = 0.9, perform_subsample = True):
    maximum_possible_correlation = np.sum(specification**2)
    correlation = sig.correlate(measurement_buffer,specification,mode='valid').squeeze()
    delay = np.argmax(correlation)
    print('Max Correlation: {:}'.format(np.max(correlation)/maximum_possible_correlation))
    if correlation[delay] < correlation_threshold*maximum_possible_correlation:
        return None,None,None
    # np.savez('alignment_debug.npz',measurement_buffer=measurement_buffer,
    #          specification = specification,
    #          correlation_threshold = correlation_threshold)
    specification_portion = measurement_buffer[:,delay:delay+specification.shape[-1]]
    
    if perform_subsample:
        # Compute ffts for subsample alignment
        spec_fft = np.fft.rfft(specification,axis=-1)
        spec_portion_fft = np.fft.rfft(specification_portion,axis=-1)
        
        # Compute phase angle differences for subpixel alignment
        phase_difference = np.angle(spec_portion_fft/spec_fft)
        phase_slope = phase_difference[...,1:-1] / np.arange(phase_difference.shape[-1])[1:-1]
        mean_phase_slope = np.median(phase_slope) # Use Median to discard outliers due to potentially noisy phase
        
        spec_portion_aligned_fft = spec_portion_fft*np.exp(-1j*mean_phase_slope*np.arange(spec_portion_fft.shape[-1]))
        spec_portion_aligned = np.fft.irfft(spec_portion_aligned_fft)
    else:
        spec_portion_aligned = specification_portion.copy()
        mean_phase_slope = None
    return spec_portion_aligned,delay,mean_phase_slope

def shift_signal(signal,samples_to_keep,sample_delay,phase_slope):
    # np.savez('shift_debug.npz',signal=signal,
    #          samples_to_keep = samples_to_keep,
    #          sample_delay = sample_delay,
    #          phase_slope=phase_slope)
    signal_sample_aligned = signal[...,sample_delay:sample_delay+samples_to_keep]
    sample_aligned_fft = np.fft.rfft(signal_sample_aligned,axis=-1)
    subsample_aligned_fft = sample_aligned_fft*np.exp(-1j*phase_slope*np.arange(sample_aligned_fft.shape[-1]))
    return np.fft.irfft(subsample_aligned_fft)

class GlobalCommands(Enum):
    """An enumeration that lists the commands that the controller can accept"""
    QUIT = -1
    INITIALIZE_DATA_ACQUISITION = -2
    INITIALIZE_ENVIRONMENT_PARAMETERS = -3
    RUN_HARDWARE = -4
    STOP_HARDWARE = -5
    INITIALIZE_STREAMING = -6
    STREAMING_DATA = -7
    FINALIZE_STREAMING = -8
    START_ENVIRONMENT = -9
    STOP_ENVIRONMENT = -10
    START_STREAMING = -11
    STOP_STREAMING = -12
    CREATE_NEW_STREAM = -13
    COMPLETED_SYSTEM_ID = -14
    AT_TARGET_LEVEL = -15
    UPDATE_METADATA = -16
    
class OverlapBuffer:
    """Class to hold a buffer stored in a numpy array.
    
    This buffer supports overlap; when you pull data out, it doesn't remove the
    data from the buffer."""
    def __init__(self,shape,buffer_axis=-1,starting_value=0,dtype='float64'):
        """
        Creates a buffer object

        Parameters
        ----------
        shape : tuple
            Shape of the underlying data array
        buffer_axis : int, optional
            Index corresponding to the buffer axis. The default is -1.
        starting_value : optional
            Initial value of the array.  Can be any value or array that can be
            broadcast into the shape of the array. The default is 0.
        dtype : numpy dtype, optional
            The data type of the buffer array. The default is 'float64'.
        """
        self._buffer_data = np.empty(shape,dtype)
        self._buffer_data[:] = starting_value
        self._buffer_axis = buffer_axis % self.buffer_data.ndim # Makes a positive index
        self._buffer_position = 0
        
    @property
    def buffer_position(self):
        return self._buffer_position
    
    @property
    def buffer_axis(self):
        return self._buffer_axis
    
    @property
    def buffer_data(self):
        return self._buffer_data
        
    def add_data_noshift(self,data):
        data = np.array(data)
        # Make sure the data will fit into the buffer
        data_slice = tuple([
            slice(-self.buffer_data.shape[self.buffer_axis],None)
            if i==self.buffer_axis else slice(None)
            for i in range(self.buffer_data.ndim)])
        data = data[data_slice]
        # Figure out how much we need to roll the buffer
        new_data_size = data.shape[self.buffer_axis]
        old_data_slice = tuple([
            slice(new_data_size,None) if i==self.buffer_axis else slice(None)
            for i in range(self.buffer_data.ndim)])
        self.buffer_data[:] = np.concatenate((self.buffer_data[old_data_slice],
                                           data),axis=self.buffer_axis)
        
    def add_data(self,data):
        self.add_data_noshift(data)
        self._buffer_position += data.shape[self.buffer_axis]
        if self.buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]
            
    def get_data_noshift(self,num_samples):
        data_start = -self.buffer_position
        data_end = -self.buffer_position + num_samples
        if data_end > 0:
            raise ValueError('Too many samples requested {:} > buffer position of {:}'.format(num_samples,self.buffer_position))
        data_slice = tuple([
            slice(data_start,None if data_end == 0 else data_end)
            if i==self.buffer_axis else slice(None)
            for i in range(self.buffer_data.ndim)])
        return self.buffer_data[data_slice]
    
    def get_data(self,num_samples,buffer_shift = None):
        data = self.get_data_noshift(num_samples)
        if buffer_shift is None:
            self.shift_buffer_position(-num_samples)
        else:
            self.shift_buffer_position(buffer_shift)
        return data
    
    def shift_buffer_position(self,samples):
        self._buffer_position += samples
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]
            
    def set_buffer_position(self,position = 0):
        self._buffer_position = position
        if self._buffer_position < 0:
            self._buffer_position = 0
        if self._buffer_position > self.buffer_data.shape[self.buffer_axis]:
            self._buffer_position = self.buffer_data.shape[self.buffer_axis]
            
    def __getitem__(self,key):
        return self.buffer_data[key]
    
    @property
    def shape(self):
        return self.buffer_data.shape
    

def load_python_module(module_path):
    """Loads in the Python file at the specified path as a module at runtime

    Parameters
    ----------
    module_path : str:
        Path to the module to be loaded
        

    Returns
    -------
    module : module:
        A reference to the loaded module
    """
    path,file = os.path.split(module_path)
    file,ext = os.path.splitext(file)
    spec = importlib.util.spec_from_file_location(file, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module