from rattlesnake.environment.abstract_environment import (
    EnvironmentMetadata,
    EnvironmentInstructions,
    Environment,
)
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.environment_registry import (
    UNIMPLEMENTED_ENVIRONMENT,
    ENVIRONMENT_CLASS,
    ENVIRONMENT_METADATA,
    SYSID_ENVIRONMENTS,
)
from rattlesnake.environment.time_environment import TimeEnvironment, TimeMetadata
from .mock_utilities import mock_channel_list_bools
from unittest import mock
from enum import Enum

# IMPLEMENTED_ENVIRONMENT = [
#     environment
#     for environment in EnvironmentType
#     if environment not in UNIMPLEMENTED_ENVIRONMENT
# ]
IMPLEMENTED_ENVIRONMENT = [EnvironmentType.NONE, EnvironmentType.TIME]


def environment_metadata_dict():
    environment_metadata_dict = ENVIRONMENT_METADATA
    environment_metadata_dict[EnvironmentType.NONE] = MockEnvironmentMetadata
    return environment_metadata_dict


def environment_dict():
    environment_dict = ENVIRONMENT_CLASS
    environment_dict[EnvironmentType.NONE] = MockEnvironment
    return environment_dict


def build_environment(environment_type, queue_container, event_container):
    environment_lookup = environment_dict()
    new_environment = environment_lookup[environment_type]
    if environment_type in SYSID_ENVIRONMENTS:
        environment = new_environment(
            "Environment Name",
            "Queue Name",
            queue_container,
            event_container.acquisition_active_event,
            event_container.output_active_event,
            event_container.environment_active_events["Environment 0"],
            event_container.environment_ready_events["Environment 0"],
            event_container.environment_sysid_active_events["Environment 0"],
            event_container.environment_sysid_stored_events["Environment 0"],
        )
    else:
        environment = new_environment(
            "Environment Name",
            "Queue Name",
            queue_container,
            event_container.acquisition_active_event,
            event_container.output_active_event,
            event_container.environment_active_events["Environment 0"],
            event_container.environment_ready_events["Environment 0"],
        )
    return environment


# region Type
class MockEnvironmentType(Enum):
    ENVIRONMENT = 0


# region Metadata
class MockEnvironmentMetadata(EnvironmentMetadata):
    def __init__(self):
        super().__init__(
            environment_type=MockEnvironmentType.ENVIRONMENT,
            environment_name="Mock Environment",
            channel_list_bools=mock_channel_list_bools(),
            sample_rate=1000,
        )
        self.queue_name = "Environment 0"

    def validate(self, hardware_metadata):
        return super().validate(hardware_metadata)

    @classmethod
    def create_blank_worksheet_template(cls, worksheet):
        return super().create_blank_worksheet_template(worksheet)

    @classmethod
    def load_metadata_from_netcdf(
        cls, netcdf_handle, environment_name, channel_list_bools, hardware_metadata
    ):
        return super().load_metadata_from_netcdf(
            netcdf_handle, environment_name, channel_list_bools, hardware_metadata
        )

    def save_metadata_to_netcdf(self, netcdf_group_handle):
        return super().save_metadata_to_netcdf(netcdf_group_handle)

    @classmethod
    def load_metadata_from_worksheet(
        cls, worksheet, environment_name, channel_list_bools, hardware_metadata
    ):
        return super().load_metadata_from_worksheet(
            worksheet, environment_name, channel_list_bools, hardware_metadata
        )

    def save_metadata_to_worksheet(self, worksheet):
        return super().save_metadata_to_worksheet(worksheet)


# region Instructions
class MockEnvironmentInstructions(EnvironmentInstructions):
    def __init__(self):
        super().__init__(MockEnvironmentType.ENVIRONMENT, "Environment 0")

    def validate(self):
        return super().validate()


# region Environment
class MockEnvironment(Environment):
    def __init__(
        self,
        environment_name,
        queue_name,
        queue_container,
        acquisition_active_event,
        output_active_event,
        active_event,
        ready_event,
    ):
        super().__init__(
            environment_name,
            queue_name,
            queue_container.environment_command_queues["Environment 0"],
            queue_container.gui_update_queue,
            queue_container.controller_command_queue,
            queue_container.log_file_queue,
            queue_container.environment_data_in_queues["Environment 0"],
            queue_container.environment_data_out_queues["Environment 0"],
            acquisition_active_event,
            output_active_event,
            active_event,
            ready_event,
        )

        self.set_ready()

    def initialize_hardware(self, hardware_metadata):
        super().initialize_hardware(hardware_metadata)
        self.set_ready()

    def initialize_environment(self, environment_metadata):
        super().initialize_environment(environment_metadata)
        self.set_ready()
        return None

    def stop_environment(self, data):
        super().stop_environment(data)
        self.clear_active()
