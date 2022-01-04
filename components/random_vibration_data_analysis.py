# -*- coding: utf-8 -*-
"""
Controller subsystem that handles the generation of new output CPSD matrices by
performing the control calculations for the Random Vibration Environment

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
import importlib
import os
from .utilities import (flush_queue,power2db,rms_csd,rms_time,
                        DataAcquisitionParameters,GlobalCommands)

from enum import Enum
from .abstract_message_process import AbstractMessageProcess
from .random_vibration_environment import RandomEnvironmentQueues,RandomVibrationParameters
from .random_vibration_signal_generation import RandomSignalGenerationMessages
from .cpsd_computation import CPSDMessages

class RandomDataAnalysisMessages(Enum):
    """Set of commands that the Random Data Analysis Process may receive."""
    INITIALIZE_DATA_ACQUISITION = 0
    INITIALIZE_TEST_PARAMETERS = 1
    RUN_TRANSFER_FUNCTION = 3
    PERFORM_CONTROL_PREDICTION = 4
    SHOW_TEST_PREDICTION = 5
    SHOW_FRF = 6
    RUN_CONTROL = 7
    STOP = 8
    
class ControlLawTypes(Enum):
    """Set of control law function types"""
    FUNCTION = 0
    GENERATOR = 1
    CLASS = 2

class RandomDataAnalysisProcess(AbstractMessageProcess):
    """Class encapsulating the data analysis process for random vibration
    
    This class performs the actual control for the Random Vibration environment.
    It accepts data from the FRF and CPSD processes and solves for an output
    CPSD that is passed to the signal generation process."""
    def __init__(self,process_name : str, queues : RandomEnvironmentQueues,environment_name : str):
        """
        Class encapsulating the data analysis process for random vibration

        Parameters
        ----------
        process_name : str
            A name to assign to the process primarily for logging purposes.
        queues : RandomEnvironmentQueues
            A list of Random Environment queues for communication with other
            parts of hte enviornment and the controller.
        environment_name : str
            The name of the environment that this process is controlling for.

        """
        super().__init__(process_name,queues.log_file_queue,queues.data_analysis_command_queue,queues.gui_update_queue)
        self.map_command(RandomDataAnalysisMessages.INITIALIZE_DATA_ACQUISITION,
                         self.initialize_data_acquisition)
        self.map_command(RandomDataAnalysisMessages.INITIALIZE_TEST_PARAMETERS,
                         self.initialize_test_parameters)
        self.map_command(RandomDataAnalysisMessages.RUN_TRANSFER_FUNCTION,
                         self.run_transfer_function)
        self.map_command(RandomDataAnalysisMessages.PERFORM_CONTROL_PREDICTION,
                         self.perform_control_prediction)
        self.map_command(RandomDataAnalysisMessages.SHOW_TEST_PREDICTION,
                         self.show_test_prediction)
        self.map_command(RandomDataAnalysisMessages.RUN_CONTROL,
                         self.run_control)
        self.map_command(RandomDataAnalysisMessages.STOP,
                         self.stop)
        self.map_command(RandomDataAnalysisMessages.SHOW_FRF,
                         self.show_frf)
        self.environment_name = environment_name
        self.queue_container = queues
        # From channel initialization
        self.num_output_channels = None
        self.num_control_channels = None
        # From sampling initialization
        self.frf_drive_voltage = None
        self.frf_update = None
        self.samples_per_read = None
        self.samples_per_frame = None
        self.sample_rate = None
        self.frf_averages = None
        self.frequency_spacing = None
        self.frequency_lines = None
        self.overlap = None
        # From spec initialization
        self.control_function = None
        self.extra_control_parameters = None
        self.control_function_type = None
        self.specification = None
        # Persistent Data
        self.frf_cpsd = None
        self.last_response_cpsd = None
        self.last_drive_cpsd = None
        self.frf = None
        self.frame_number = None
        self.response_cpsd_prediction = None
        self.drive_cpsd_prediction = None
        self.frequencies = None
        self.startup = True
        # Error indices
        self.error_indices = None
        
        
    def initialize_data_acquisition(self,data : DataAcquisitionParameters):
        """Stores global Data Acquisition Parameters to the process

        Parameters
        ----------
        data : DataAcquisitionParameters :
            The data acquisition parameters of the controller.

        """
        self.log('Initializing Data Acquisition')
        self.num_control_channels = len([channel for channel in data.channel_list if channel.control])
        self.num_output_channels = len([channel for channel in data.channel_list if not channel.feedback_device is None])
        self.sample_rate = data.sample_rate
    
    def initialize_test_parameters(self,data : RandomVibrationParameters):
        """Stores environment signal processing parameters to the process

        Parameters
        ----------
        data : RandomVibrationParameters :
            Signal processing parameters for the environment

        """
        self.frf_averages = data.system_id_averages
        self.frf_update = data.update_tf_during_control
        self.samples_per_read = data.samples_per_acquire
        self.samples_per_frame = data.samples_per_frame
        self.frequency_spacing = data.frequency_spacing
        self.frequency_lines = data.fft_lines
        self.overlap = data.overlap
        if not data.response_transformation_matrix is None:
            self.num_control_channels = data.response_transformation_matrix.shape[0]
        if not data.output_transformation_matrix is None:
            self.num_output_channels = data.output_transformation_matrix.shape[0]
        # Only set to zeros if they are none or the wrong shape
        if (self.last_response_cpsd is None
            or self.last_response_cpsd.shape != (self.frequency_lines,self.num_control_channels,self.num_control_channels)
            or self.frf_cpsd is None
            or self.frf_cpsd.shape != (self.frequency_lines,self.num_control_channels,self.num_control_channels)
            or self.last_drive_cpsd is None
            or self.last_drive_cpsd.shape != (self.frequency_lines,self.num_output_channels,self.num_output_channels)
            or self.frf is None
            or self.frf.shape != (self.frequency_lines,self.num_control_channels,self.num_output_channels)
            or self.response_cpsd_prediction is None
            or self.response_cpsd_prediction.shape != (self.frequency_lines,self.num_control_channels,self.num_control_channels)
            or self.drive_cpsd_prediction is None
            or self.drive_cpsd_prediction.shape != (self.frequency_lines,self.num_output_channels,self.num_output_channels)
            ):
            self.last_response_cpsd = np.zeros((self.frequency_lines,self.num_control_channels,self.num_control_channels),dtype='complex128')
            self.frf_cpsd = np.zeros((self.frequency_lines,self.num_control_channels,self.num_control_channels),dtype='complex128')
            self.last_drive_cpsd = np.zeros((self.frequency_lines,self.num_output_channels,self.num_output_channels),dtype='complex128')
            self.frf = np.zeros((self.frequency_lines,self.num_control_channels,self.num_output_channels),dtype='complex128')
            self.response_cpsd_prediction = np.zeros((self.frequency_lines,self.num_control_channels,self.num_control_channels),dtype='complex128')
            self.drive_cpsd_prediction = np.zeros((self.frequency_lines,self.num_output_channels,self.num_output_channels),dtype='complex128')
        self.frequencies = self.frequency_spacing * np.arange(self.frequency_lines)
        self.specification = data.specification_cpsd_matrix
        self.frf_drive_voltage = data.frf_voltage
        self.error_indices = ~np.all(self.specification == 0,axis=(-1,-2))
        # Load the control information
        path,file = os.path.split(data.control_python_script)
        file,ext = os.path.splitext(file)
        spec = importlib.util.spec_from_file_location(file, data.control_python_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.control_function_type = data.control_python_function_type
        self.extra_control_parameters = data.control_python_function_parameters
        if self.control_function_type == 1: # Generator
            # Get the generator function
            generator_function = getattr(module,data.control_python_function)()
            # Get us to the first yield statement
            next(generator_function)
            # Define the control function as the generator's send function
            self.control_function = generator_function.send
        elif self.control_function_type == 2: # Class
            self.control_function = getattr(module,data.control_python_function)(
                self.specification,self.extra_control_parameters, # Required parameters
                self.frf,self.frf_cpsd,self.last_response_cpsd,self.last_drive_cpsd) # Optional parameters
        else: # Function
            self.control_function = getattr(module,data.control_python_function)
    
    def run_transfer_function(self,data):
        """Function to handle data analysis for system identification.

        The function serves several roles, including startup, run, and shutdown
        of the data acquisition process.
        
        During startup, the function will set the number of FRF frames to zero
        
        When running, it will simply put the drive voltage the queue for the
        Signal Generation task.  For each run, it will grab all the new FRF
        data and CPSD data and tell the user interface to update the FRF plot,
        as well as store those data for later use.  
        
        During shutdown it will tell the signal generation to gracefully
        shut down the environment by ramping the data to zero.  It will also
        perform test predictions if necessary.

        Parameters
        ----------
        data : bool
            data will be True if the function is running in acquisition mode,
            which means the process will eventually quit when enough averages
            are acquired.  Otherwise, data is False in preview mode, and the 
            system will continue indefinitely until manually stopped.

        """
        if self.startup:
            self.frame_number = 0
            self.queue_container.cpsd_to_generate_queue.put(self.frf_drive_voltage)
            self.startup=False
        # Get the transfer function data
        frf_data = flush_queue(self.queue_container.updated_frf_queue)
        if len(frf_data) > 0:
            self.frf[:],frf_frames = frf_data[-1]
            self.queue_container.gui_update_queue.put((self.environment_name,
                               ('system_id_current_average_display',frf_frames)))
        else:
            frf_frames = 0
        # Get the CPSD data
        cpsd_data = flush_queue(self.queue_container.updated_cpsd_queue)
        if len(cpsd_data) > 0:
            self.last_response_cpsd[:],self.last_drive_cpsd[:],cpsd_frames = cpsd_data[-1]
            self.frf_cpsd[:] = self.last_response_cpsd[:]
        if data and frf_frames == self.frf_averages: # Data is true for acquire and false for preview
            # Stop things and do a control preview
            self.queue_container.signal_generation_command_queue.put(self.process_name,
                                 (RandomSignalGenerationMessages.START_SHUTDOWN,None))
            self.queue_container.data_analysis_command_queue.put(self.process_name,
                                 (RandomDataAnalysisMessages.PERFORM_CONTROL_PREDICTION,None))
            self.queue_container.controller_communication_queue.put(self.process_name,
                                 (GlobalCommands.COMPLETED_SYSTEM_ID,self.environment_name))
        else:
            if self.queue_container.cpsd_to_generate_queue.empty():
                self.queue_container.cpsd_to_generate_queue.put(self.frf_drive_voltage)
            self.queue_container.data_analysis_command_queue.put(self.process_name,
                         (RandomDataAnalysisMessages.RUN_TRANSFER_FUNCTION,data))
    
    def perform_control_prediction(self,data):
        """Performs control predictions based off the control law and transfer function
        
        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        if self.control_function_type == 1: # Generator
            output_cpsd = self.control_function((
                    self.frf,
                    self.specification,
                    self.frf_cpsd,
                    self.extra_control_parameters,
                    None,None))
        elif self.control_function_type == 2: # Class
            self.control_function.system_id_update(self.frf,self.frf_cpsd)
            output_cpsd = self.control_function.control(self.frf,None,None)
        else: # Function
            output_cpsd = self.control_function(
                    self.frf,
                    self.specification,
                    self.frf_cpsd,
                    self.extra_control_parameters,
                    None,None
                    )
        response_cpsd = self.frf@output_cpsd@self.frf.conjugate().transpose(0,2,1)
        rms_voltages = rms_csd(output_cpsd,self.frequency_spacing)
        response_db_error = (power2db(np.einsum('ijj->ij',response_cpsd[self.error_indices]).real)
                             - power2db(np.einsum('ijj->ij',self.specification[self.error_indices]).real))
        rms_db_error = rms_time(response_db_error,axis=0)
        self.drive_cpsd_prediction = output_cpsd
        self.last_drive_cpsd = output_cpsd
        self.response_cpsd_prediction = response_cpsd
        self.last_response_cpsd = response_cpsd
        self.queue_container.data_analysis_command_queue.put(self.process_name,(RandomDataAnalysisMessages.SHOW_TEST_PREDICTION,None))
        self.queue_container.gui_update_queue.put((self.environment_name,('excitation_voltage_list',rms_voltages)))
        self.queue_container.gui_update_queue.put((self.environment_name,('response_error_list',rms_db_error)))
        self.queue_container.cpsd_command_queue.put(self.process_name,(CPSDMessages.CLEAR_CPSD,None))
    
    def show_test_prediction(self,data):
        """Sends test prediction data to the user interface for display
        
        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        self.queue_container.gui_update_queue.put((self.environment_name,('control_predictions',(self.frequencies,
                                                                       self.drive_cpsd_prediction,
                                                                       self.response_cpsd_prediction,
                                                                       self.specification))))
    
    def show_frf(self,data):
        """
        Sends a message to show the FRF data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.gui_update_queue.put((self.environment_name,('FRF',(np.fft.rfftfreq(self.samples_per_frame,1/self.sample_rate),self.frf))))
    
    def run_control(self,data):
        """Function to handle data analysis for control.
        
        During startup, the function will initialize the number of frames to zero
        and perform an initial control.
        
        When running, it will grab all the new FRF data and CPSD data and then
        perform control with those values.  It will continue to tell the user
        interface to update the FRF plot.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        if self.startup:
            self.log('Starting Control')
            self.frame_number = 0
            # Create the new control law
            if self.control_function_type == 1: # Generator
                output_cpsd = self.control_function((
                        self.frf,
                        self.specification,
                        self.frf_cpsd,
                        self.extra_control_parameters,
                        self.last_response_cpsd,
                        self.last_drive_cpsd))
            elif self.control_function_type == 2: # Class
                output_cpsd = self.control_function.control(self.frf,
                                                        self.last_response_cpsd,
                                                        self.last_drive_cpsd)
            else: # Function
                output_cpsd = self.control_function(
                        self.frf,
                        self.specification,
                        self.frf_cpsd,
                        self.extra_control_parameters,
                        self.last_response_cpsd,
                        self.last_drive_cpsd)
            self.queue_container.cpsd_to_generate_queue.put(output_cpsd)
            self.startup = False
        # Get the transfer function data
        frf_data = flush_queue(self.queue_container.updated_frf_queue)
        do_control = False
        if len(frf_data) > 0 and self.frf_update:
            self.frf[:],frf_frames = frf_data[-1]
            do_control = True
        # Get the CPSD data
        cpsd_data = flush_queue(self.queue_container.updated_cpsd_queue)
        if len(cpsd_data) > 0:
            self.last_response_cpsd[:],self.last_drive_cpsd[:],cpsd_frames = cpsd_data[-1]
            self.queue_container.gui_update_queue.put((self.environment_name,
                                       ('update_control_response',(self.frequencies,self.last_response_cpsd))))
            response_db_error = (power2db(np.einsum('ijj->ij',self.last_response_cpsd[self.error_indices]).real)
                                 - power2db(np.einsum('ijj->ij',self.specification[self.error_indices]).real))
            rms_db_error = rms_time(response_db_error,axis=0)
            self.queue_container.gui_update_queue.put(
                (self.environment_name,('test_response_error_list',rms_db_error)))
            do_control = True
        if do_control:
            self.log('Controlling')
            # Create the new control law
            if self.control_function_type == 1:
                output_cpsd = self.control_function((
                        self.frf,
                        self.specification,
                        self.frf_cpsd,
                        self.extra_control_parameters,
                        self.last_response_cpsd,
                        self.last_drive_cpsd))
            elif self.control_function_type == 2: # Class
                output_cpsd = self.control_function.control(self.frf,
                                                        self.last_response_cpsd,
                                                        self.last_drive_cpsd)
            else: # Function
                output_cpsd = self.control_function(
                        self.frf,
                        self.specification,
                        self.frf_cpsd,
                        self.extra_control_parameters,
                        self.last_response_cpsd,
                        self.last_drive_cpsd)
            self.queue_container.cpsd_to_generate_queue.put(output_cpsd)
            self.log('Finished Controlling')
            rms_voltages = rms_csd(output_cpsd,self.frequency_spacing)
            self.queue_container.gui_update_queue.put((self.environment_name,('test_output_voltage_list',rms_voltages)))
        self.queue_container.data_analysis_command_queue.put(self.process_name,(RandomDataAnalysisMessages.RUN_CONTROL,None))
    
    def stop(self,data):
        """stops the data analysis process from analyzing data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        # Remove any run_transfer_function or run_control from the queue
        instructions = self.queue_container.data_analysis_command_queue.flush(self.process_name)
        for instruction in instructions:
            if not instruction[0] in [RandomDataAnalysisMessages.RUN_TRANSFER_FUNCTION,RandomDataAnalysisMessages.RUN_CONTROL]:
                self.queue_container.data_analysis_command_queue.put(self.process_name,instruction)
        flush_queue(self.queue_container.cpsd_to_generate_queue)
        self.startup = True

def random_data_analysis_process(environment_name : str,queue_container: RandomEnvironmentQueues):
    """Random vibration data analysis process function called by multiprocessing
    
    This function defines the Random Vibration Data Analysis process that
    gets run by the multiprocessing module when it creates a new process.  It
    creates a RandomDataAnalysisProcess object and runs it.

    Parameters
    ----------
    environment_name : str :
        Name of the environment associated with this signal generation process
        
    queue_container: RandomEnvironmentQueues :
        A container of queues that allows communcation with the random vibration
        environment as well as the rest of the controller.
    """

    data_analysis_instance = RandomDataAnalysisProcess(environment_name + ' Data Analysis',queue_container,environment_name)
    
    data_analysis_instance.run()