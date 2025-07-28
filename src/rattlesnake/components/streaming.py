# -*- coding: utf-8 -*-
"""
Controller subsystem that handles streaming data and metadata to NetCDF4 files
on the disk.

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

from .utilities import QueueContainer,GlobalCommands,DataAcquisitionParameters,flush_queue
from .abstract_message_process import AbstractMessageProcess
from .abstract_environment import AbstractMetadata
import numpy as np
import netCDF4 as nc
from typing import Dict

class StreamingProcess(AbstractMessageProcess):
    """Class containing the functionality to stream data to disk.
    
    This class will handle receiving data from the acquisition and saving it
    to a netCDF file."""
    def __init__(self,process_name : str, queue_container : QueueContainer):
        """Constructor for the StreamingProcess class
        
        Sets up the ``command_map`` and initializes all data members.

        Parameters
        ----------
        process_name : str
            The name of the process.
        queue_container : QueueContainer
            A container containing the queues used to communicate between
            controller processes

        """
        super().__init__(process_name,queue_container.log_file_queue,queue_container.streaming_command_queue,queue_container.gui_update_queue)
        self.map_command(GlobalCommands.INITIALIZE_STREAMING,self.initialize)
        self.map_command(GlobalCommands.STREAMING_DATA,self.write_data)
        self.map_command(GlobalCommands.FINALIZE_STREAMING,self.finalize)
        self.map_command(GlobalCommands.CREATE_NEW_STREAM,self.create_new_stream)
        self.netcdf_handle = None
        # Track the variable we are streaming data to
        self.stream_variable = 'time_data'
        self.stream_dimension = 'time_samples'
        self.stream_index = 0
        
    def initialize(self,data):
        """Creates a file with all metadata from the controller
        
        Creates a netCDF4 dataset and stores all the global data acquisition
        parameters as well as the parameters from each environment.

        Parameters
        ----------
        data : tuple
            Tuple containing a string filename, global DataAcquisitionParameters
            defining the controller settings, and a dictionary containing the
            environment names as keys and the environment metadata (inheriting
            from AbstractMetadata) as values for each environment.

        """
        filename : str
        global_data_parameters : DataAcquisitionParameters
        environment_metadata : Dict[str,AbstractMetadata]
        filename,global_data_parameters,environment_metadata = data
        self.stream_variable = 'time_data'
        self.stream_dimension = 'time_samples'
        self.stream_index = 0
        self.netcdf_handle = nc.Dataset(filename,'w',format='NETCDF4',clobber=True)
        # Create dimensions
        self.netcdf_handle.createDimension('response_channels',len(global_data_parameters.channel_list))
        self.netcdf_handle.createDimension('output_channels',len([channel for channel in global_data_parameters.channel_list if not channel.feedback_device is None]))
        self.netcdf_handle.createDimension(self.stream_dimension,None)
        self.netcdf_handle.createDimension('num_environments',len(global_data_parameters.environment_names))
        # Create attributes
        self.netcdf_handle.file_version = '3.0.0'
        self.netcdf_handle.sample_rate = global_data_parameters.sample_rate
        self.netcdf_handle.time_per_write = global_data_parameters.samples_per_write/global_data_parameters.output_sample_rate
        self.netcdf_handle.time_per_read = global_data_parameters.samples_per_read/global_data_parameters.sample_rate
        self.netcdf_handle.hardware = global_data_parameters.hardware
        self.netcdf_handle.hardware_file = 'None' if global_data_parameters.hardware_file is None else global_data_parameters.hardware_file
        self.netcdf_handle.maximum_acquisition_processes = global_data_parameters.maximum_acquisition_processes
        self.netcdf_handle.output_oversample = global_data_parameters.output_oversample
        # Create Variables
        self.netcdf_handle.createVariable(self.stream_variable,'f8',('response_channels',self.stream_dimension))
        var = self.netcdf_handle.createVariable('environment_names',str,('num_environments',))
        for i,name in enumerate(global_data_parameters.environment_names):
            var[i] = name
        var = self.netcdf_handle.createVariable('environment_active_channels','i1',('response_channels','num_environments'))
        var[...] = global_data_parameters.environment_active_channels.astype('int8')
        # Create channel table variables
        labels = [['node_number',str],
                  ['node_direction',str],
                  ['comment',str],
                  ['serial_number',str],
                  ['triax_dof',str],
                  ['sensitivity',str],
                  ['unit',str],
                  ['make',str],
                  ['model',str],
                  ['expiration',str],
                  ['physical_device',str],
                  ['physical_channel',str],
                  ['channel_type',str],
                  ['minimum_value',str],
                  ['maximum_value',str],
                  ['coupling',str],
                  ['excitation_source',str],
                  ['excitation',str],
                  ['feedback_device',str],
                  ['feedback_channel',str],
                  ['warning_level',str],
                  ['abort_level',str],
                  ]
        for (label,netcdf_datatype) in labels:
            var = self.netcdf_handle.createVariable('/channels/'+label,netcdf_datatype,('response_channels',))
            channel_data = [getattr(channel,label) for channel in global_data_parameters.channel_list]
            if netcdf_datatype == 'i1':
                channel_data = np.array([1 if val else 0 for val in channel_data])
            else:
                channel_data = ['' if val is None else val for val in channel_data]
            for i,cd in enumerate(channel_data):
                var[i] = cd
        # Now write all the environment data to the netCDF file
        for environment_name,metadata in environment_metadata.items():
            group_handle = self.netcdf_handle.createGroup(environment_name)
            metadata.store_to_netcdf(group_handle)

    def write_data(self,data):
        """Writes data to an initialized netCDF file

        Parameters
        ----------
        data : np.ndarray
            Data to be written to the netCDF file


        """
        if self.netcdf_handle is None:
            return
        test_data = data
        timesteps = slice(self.netcdf_handle.dimensions[self.stream_dimension].size,None,None)
        self.netcdf_handle.variables[self.stream_variable][:,timesteps] = test_data
    
    def create_new_stream(self,data):
        if self.netcdf_handle is None:
            return
        self.stream_index += 1
        self.stream_variable = 'time_data_{:}'.format(self.stream_index)
        self.stream_dimension = 'time_samples_{:}'.format(self.stream_index)
        self.netcdf_handle.createDimension(self.stream_dimension,None)
        self.netcdf_handle.createVariable(self.stream_variable,'f8',('response_channels',self.stream_dimension))
    
    def finalize(self,data):
        """Closes the netCDF file when data writing is complete

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        if not self.netcdf_handle is None:
            self.netcdf_handle.close()
            self.netcdf_handle = None
            
    def quit(self,data):
        """Stops the process.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``
        """
        self.finalize(None)
        return True

def streaming_process(queue_container:QueueContainer):
    """Function passed to multiprocessing as the streaming process
    
    This process creates the ``StreamingProcess`` object and calls the ``run``
    command.

    Parameters
    ----------
    queue_container : QueueContainer
        A container containing the queues used to communicate between
        controller processes

    """

    streaming_instance = StreamingProcess('Streaming',queue_container)
    
    streaming_instance.run()