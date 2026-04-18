"""
Abstract System Identification Function Mocks

This module provides dummy implementations of abstract system identification
classes for testing purposes.
"""

from rattlesnake.environment.abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdMetadata,
)
from rattlesnake.user_interface.abstract_sys_id_user_interface import AbstractSysIdUI
import openpyxl


class DummyAbstractSysIdMetadata(AbstractSysIdMetadata):
    """
    Dummy implementation of AbstractSysIdMetadata for testing.
    """

    def __init__(self):
        """
        Initialize the DummyAbstractSysIdMetadata.
        """
        super().__init__()

    @property
    def number_of_channels(self):
        """
        Number of channels.
        """
        pass

    @property
    def response_channel_indices(self):
        """
        Indices of the response channels.
        """
        pass

    @property
    def reference_channel_indices(self):
        """
        Indices of the reference channels.
        """
        pass

    @property
    def response_transformation_matrix(self):
        """
        Transformation matrix for response channels.
        """
        pass

    @property
    def reference_transformation_matrix(self):
        """
        Transformation matrix for reference channels.
        """
        pass

    @property
    def sample_rate(self):
        """
        Sample rate.
        """
        pass

    def store_to_netcdf(self, netcdf_group_handle):
        """
        Store metadata to a NetCDF group.

        Args:
            netcdf_group_handle: Handle to the NetCDF group.

        Returns:
            The result of the superclass store_to_netcdf method.
        """
        return super().store_to_netcdf(netcdf_group_handle)


class DummyAbstractSysIdUI(AbstractSysIdUI):
    """
    Dummy implementation of AbstractSysIdUI for testing.
    """

    def __init__(
        self,
        environment_name,
        environment_command_queue,
        controller_communication_queue,
        log_file_queue,
        system_id_tabwidget,
    ):
        """
        Initialize the DummyAbstractSysIdUI.

        Args:
            environment_name: Name of the environment.
            environment_command_queue: Queue for environment commands.
            controller_communication_queue: Queue for controller communication.
            log_file_queue: Queue for logging.
            system_id_tabwidget: Tab widget for system identification.
        """
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
            system_id_tabwidget,
        )

    def initialize_data_acquisition(self, data_acquisition_parameters):
        """
        Initialize data acquisition.

        Args:
            data_acquisition_parameters: Parameters for data acquisition.

        Returns:
            The result of the superclass initialize_data_acquisition method.
        """
        return super().initialize_data_acquisition(data_acquisition_parameters)

    def collect_environment_definition_parameters(self):
        """
        Collect environment definition parameters.

        Returns:
            The result of the superclass collect_environment_definition_parameters method.
        """
        return super().collect_environment_definition_parameters()

    @property
    def initialized_control_names(self):
        """
        Initialized control names.
        """
        pass

    @property
    def initialized_output_names(self):
        """
        Initialized output names.
        """
        pass

    def initialize_environment(self):
        """
        Initialize the environment.

        Returns:
            The result of the superclass initialize_environment method.
        """
        return super().initialize_environment()

    def retrieve_metadata(self, netcdf_handle):
        """
        Retrieve metadata.

        Args:
            netcdf_handle: Handle to the NetCDF file.

        Returns:
            The result of the superclass retrieve_metadata method.
        """
        return super().retrieve_metadata(netcdf_handle)

    def update_gui(self, queue_data):
        """
        Update the GUI.

        Args:
            queue_data: Data from the queue to update the GUI.

        Returns:
            The result of the superclass update_gui method.
        """
        return super().update_gui(queue_data)

    def create_environment_template(
        environment_name: str, workbook: openpyxl.workbook.workbook.Workbook
    ):
        """
        Create an environment template.

        Args:
            environment_name: Name of the environment.
            workbook: Excel workbook to add the template to.
        """
        pass

    def set_parameters_from_template(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        """
        Set parameters from a template.

        Args:
            worksheet: Excel worksheet containing the parameters.
        """
        pass

    def start_control(self):
        """
        Start control.

        Returns:
            The result of the superclass start_control method.
        """
        return super().start_control()

    def stop_control(self):
        """
        Stop control.

        Returns:
            The result of the superclass stop_control method.
        """
        return super().stop_control()


class DummyAbstractSysIdEnvironment(AbstractSysIdEnvironment):
    """
    Dummy implementation of AbstractSysIdEnvironment for testing.
    """

    def __init__(
        self,
        environment_name,
        command_queue,
        gui_update_queue,
        controller_communication_queue,
        log_file_queue,
        collector_command_queue,
        signal_generator_command_queue,
        spectral_processing_command_queue,
        data_analysis_command_queue,
        data_in_queue,
        data_out_queue,
        acquisition_active,
        output_active,
    ):
        """
        Initialize the DummyAbstractSysIdEnvironment.

        Args:
            environment_name: Name of the environment.
            command_queue: Queue for environment commands.
            gui_update_queue: Queue for GUI updates.
            controller_communication_queue: Queue for controller communication.
            log_file_queue: Queue for logging.
            collector_command_queue: Queue for collector commands.
            signal_generator_command_queue: Queue for signal generator commands.
            spectral_processing_command_queue: Queue for spectral processing commands.
            data_analysis_command_queue: Queue for data analysis commands.
            data_in_queue: Queue for incoming data.
            data_out_queue: Queue for outgoing data.
            acquisition_active: Value indicating if acquisition is active.
            output_active: Value indicating if output is active.
        """
        super().__init__(
            environment_name,
            command_queue,
            gui_update_queue,
            controller_communication_queue,
            log_file_queue,
            collector_command_queue,
            signal_generator_command_queue,
            spectral_processing_command_queue,
            data_analysis_command_queue,
            data_in_queue,
            data_out_queue,
            acquisition_active,
            output_active,
        )

    def stop_environment(self, data):
        """
        Stop the environment.

        Args:
            data: Data associated with the stop command.

        Returns:
            The result of the superclass stop_environment method.
        """
        return super().stop_environment(data)
