from rattlesnake.environment.abstract_sysid_environment import (
    AbstractSysIdEnvironment,
    AbstractSysIdMetadata,
)
from rattlesnake.user_interface.abstract_sys_id_user_interface import AbstractSysIdUI
import openpyxl


class DummyAbstractSysIdMetadata(AbstractSysIdMetadata):
    def __init__(self):
        super().__init__()

    @property
    def number_of_channels(self):
        pass

    @property
    def response_channel_indices(self):
        pass

    @property
    def reference_channel_indices(self):
        pass

    @property
    def response_transformation_matrix(self):
        pass

    @property
    def reference_transformation_matrix(self):
        pass

    @property
    def sample_rate(self):
        pass

    def store_to_netcdf(self, netcdf_group_handle):
        return super().store_to_netcdf(netcdf_group_handle)


class DummyAbstractSysIdUI(AbstractSysIdUI):
    def __init__(
        self,
        environment_name,
        environment_command_queue,
        controller_communication_queue,
        log_file_queue,
        system_id_tabwidget,
    ):
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
            system_id_tabwidget,
        )

    def initialize_data_acquisition(self, data_acquisition_parameters):
        return super().initialize_data_acquisition(data_acquisition_parameters)

    def collect_environment_definition_parameters(self):
        return super().collect_environment_definition_parameters()

    @property
    def initialized_control_names(self):
        pass

    @property
    def initialized_output_names(self):
        pass

    def initialize_environment(self):
        return super().initialize_environment()

    def retrieve_metadata(self, netcdf_handle):
        return super().retrieve_metadata(netcdf_handle)

    def update_gui(self, queue_data):
        return super().update_gui(queue_data)

    def create_environment_template(
        environment_name: str, workbook: openpyxl.workbook.workbook.Workbook
    ):
        pass

    def set_parameters_from_template(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        pass

    def start_control(self):
        return super().start_control()

    def stop_control(self):
        return super().stop_control()


class DummyAbstractSysIdEnvironment(AbstractSysIdEnvironment):
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
        return super().stop_environment(data)
