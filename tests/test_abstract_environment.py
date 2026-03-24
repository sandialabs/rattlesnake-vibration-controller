import multiprocessing as mp
from unittest import mock

import pytest
from functions.common_functions import fake_time

# from PyQt5 import QtWidgets  # comment out unused import

from rattlesnake.environment.abstract_environment import AbstractEnvironment, AbstractMetadata
from rattlesnake.user_interface.abstract_user_interface import AbstractUI
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# Initialize log_file_queue that verbose queues will use
@pytest.fixture()
def log_file_queue():
    return mp.Queue()


# Create environment_command_queue for AbstractUI/AbstractEnvironment
@pytest.fixture()
def environment_command_queue(log_file_queue):
    return VerboseMessageQueue(log_file_queue, "Environment Command Queue")


# Create controller_command_queue for AbstractUI/AbstractEnvironment
@pytest.fixture()
def controller_command_queue(log_file_queue):
    return VerboseMessageQueue(log_file_queue, "Controller Communication Queue")


# Create gui_update_queue for AbstractEnvironment
@pytest.fixture()
def gui_update_queue():
    return mp.Queue()


# Create data_in_queue for AbstractEnvironment
@pytest.fixture()
def data_in_queue():
    return mp.Queue()


# Create data_out_queue for AbstractEnvironment
@pytest.fixture()
def data_out_queue():
    return mp.Queue()


# Initialize the acquisition_active multiprocessing integer
@pytest.fixture
def acquisition_active():
    return mp.Value("i", 0)


# Initialize the output_active multiprocessing integer
@pytest.fixture
def output_active():
    return mp.Value("i", 0)


# Create a dummy AbstractMetadata class
class DummyAbstractMetadata(AbstractMetadata):
    def __init__(self):
        pass

    def store_to_netcdf(self, netcdf_group_handle):
        return super().store_to_netcdf(netcdf_group_handle)


# Create a dummy AbstractUI class
class DummyAbstractUI(AbstractUI):
    def __init__(self, log_file_queue, environment_command_queue, controller_command_queue):
        super().__init__(
            "Environment Name",
            environment_command_queue,
            controller_command_queue,
            log_file_queue,
        )

    def start_control(self):
        return super().start_control()

    def stop_control(self):
        return super().stop_control()

    def collect_environment_definition_parameters(self):
        return super().collect_environment_definition_parameters()

    def initialize_data_acquisition(self, data_acquisition_parameters):
        return super().initialize_data_acquisition(data_acquisition_parameters)

    def initialize_environment(self):
        return super().initialize_environment()

    def retrieve_metadata(self, netcdf_handle):
        return super().retrieve_metadata(netcdf_handle)

    def update_gui(self, queue_data):
        return super().update_gui(queue_data)

    @staticmethod
    def create_environment_template(environment_name, workbook):
        return super().create_environment_template(environment_name, workbook)

    def set_parameters_from_template(self, worksheet):
        return super().set_parameters_from_template(worksheet)


# Create a dummy AbstractEnvironment class
class DummyAbstractEnvironment(AbstractEnvironment):
    def __init__(
        self,
        log_file_queue,
        environment_command_queue,
        controller_command_queue,
        gui_update_queue,
        data_in_queue,
        data_out_queue,
    ):
        super().__init__(
            "Environment Name",
            environment_command_queue,
            gui_update_queue,
            controller_command_queue,
            log_file_queue,
            data_in_queue,
            data_out_queue,
            mp.Value("i", 0),
            mp.Value("i", 0),
        )

    def initialize_data_acquisition_parameters(self, data_acquisition_parameters):
        return super().initialize_data_acquisition_parameters(data_acquisition_parameters)

    def initialize_environment_test_parameters(self, environment_parameters):
        return super().initialize_environment_test_parameters(environment_parameters)

    def stop_environment(self, data):
        return super().stop_environment(data)


# Initialize the AbstractUI class
@pytest.fixture
def abstract_ui(log_file_queue, environment_command_queue, controller_command_queue):
    return DummyAbstractUI(log_file_queue, environment_command_queue, controller_command_queue)


# Initialize the AbstractEnvironment class
@pytest.fixture
def abstract_environment(
    log_file_queue,
    environment_command_queue,
    controller_command_queue,
    gui_update_queue,
    data_in_queue,
    data_out_queue,
):
    return DummyAbstractEnvironment(
        log_file_queue,
        environment_command_queue,
        controller_command_queue,
        gui_update_queue,
        data_in_queue,
        data_out_queue,
    )


# Test the AbstractMetadata class init
def test_abstract_metadata_init():
    abstract_metadata = DummyAbstractMetadata()

    assert isinstance(abstract_metadata, DummyAbstractMetadata)


# # Test if the AbstractUI class initializes
# def test_abstract_ui_init(log_file_queue, environment_command_queue, controller_command_queue):
#     abstract_ui = DummyAbstractUI(
#         log_file_queue, environment_command_queue, controller_command_queue)

#     # Test if the class was made
#     assert isinstance(abstract_ui, DummyAbstractUI)
#     # Test the command_map property
#     assert abstract_ui.command_map == {'Start Control': abstract_ui.start_control,
#                                        'Stop Control': abstract_ui.stop_control}
#     # Test the log_file_queue property
#     assert abstract_ui.log_file_queue == log_file_queue
#     # Test the environment_command_queue property
#     assert abstract_ui.environment_command_queue == environment_command_queue
#     # Test the controller_command_queue property
#     assert abstract_ui.controller_communication_queue == controller_command_queue
#     # Test the environment_name property
#     assert abstract_ui.environment_name == "Environment Name"
#     # Test the log_name property
#     assert abstract_ui.log_name == "Environment Name UI"


# Test the AbstractUI log function
# Prevent from writing to the log_file_queue
@mock.patch("rattlesnake.environment.abstract_environment.Queue.put")
# Replace the date and time with a string
@mock.patch("rattlesnake.user_interface.abstract_user_interface.datetime")
def test_abstract_ui_log(mock_time, mock_put, abstract_ui):
    message = "Test Message"
    mock_time.now = fake_time

    abstract_ui.log(message)

    # Test if the correct string was writtin to log_file_queue
    mock_put.assert_called_with(
        "{:}: {:} -- {:}\n".format("Datetime", "Environment Name UI", message)
    )


# Test the AbstractEnvironment class init
def test_abstract_environment_init(
    log_file_queue,
    environment_command_queue,
    controller_command_queue,
    gui_update_queue,
    data_in_queue,
    data_out_queue,
):
    abstract_environment = DummyAbstractEnvironment(
        log_file_queue,
        environment_command_queue,
        controller_command_queue,
        gui_update_queue,
        data_in_queue,
        data_out_queue,
    )

    # Test if the class was made
    assert isinstance(abstract_environment, DummyAbstractEnvironment)
    # Test the acquisition_active property
    assert abstract_environment.acquisition_active is False
    # Test the output_active property
    assert abstract_environment.output_active is False
    # Test the environment_command_queue property
    assert abstract_environment.environment_command_queue == environment_command_queue
    # Test the controller_communication_queue property
    assert abstract_environment.controller_communication_queue == controller_command_queue
    # Test the log_file_queue property
    assert abstract_environment.log_file_queue == log_file_queue
    # Test the gui_update_queue property
    assert abstract_environment.gui_update_queue == gui_update_queue
    # Test the data_in_queue property
    assert abstract_environment.data_in_queue == data_in_queue
    # Test the data_out_queue property
    assert abstract_environment.data_out_queue == data_out_queue
    # Test the environment_name property
    assert abstract_environment.environment_name == "Environment Name"
    # Test the command_map property
    assert abstract_environment.command_map == {
        GlobalCommands.QUIT: abstract_environment.quit,
        GlobalCommands.INITIALIZE_DATA_ACQUISITION: abstract_environment.initialize_data_acquisition_parameters,
        GlobalCommands.INITIALIZE_ENVIRONMENT_PARAMETERS: abstract_environment.initialize_environment_test_parameters,
        GlobalCommands.STOP_ENVIRONMENT: abstract_environment.stop_environment,
    }


# Test the AbstractEnvironment log function
# Prevent from writing to the log_file_queue
@mock.patch("rattlesnake.environment.abstract_environment.Queue.put")
# Replace the date and time with a string
@mock.patch("rattlesnake.environment.abstract_environment.datetime")
def test_abstract_environment_log(mock_time, mock_put, abstract_environment):
    message = "Test Message"
    mock_time.now = fake_time

    abstract_environment.log(message)

    # Test if the correct string was writtin to log_file_queue
    mock_put.assert_called_with("{:}: {:} -- {:}\n".format("Datetime", "Environment Name", message))


# Test the AbstractEnvironment quit function
def test_abstract_environment_quit(abstract_environment):
    data = abstract_environment.quit(None)

    # Test that the quit function returns True
    assert data is True


# Test adding commands to AbstractEnvironment
def test_abstract_environment_map_command(abstract_environment):
    key = "Test Key"

    def function():
        return "Test Function"

    abstract_environment.map_command(key, function)

    # Test that the key maps to the function
    data = abstract_environment.command_map[key]
    assert data == function


# Test the AbstractEnvironment run function
# Loop through different given keys and functions
@pytest.mark.parametrize(
    "mock_function, mock_key",
    [
        (mock.MagicMock(return_value=False), "Test Key"),
        (mock.MagicMock(side_effect=KeyError), "Test Key"),
        (mock.MagicMock(return_value=False), "Not a key"),
    ],
)
# Force get command to return values
@mock.patch("rattlesnake.utilities.VerboseMessageQueue.get")
# Prevent from writing to log_file_queue
@mock.patch("rattlesnake.environment.abstract_environment.AbstractEnvironment.log")
def test_abstract_environment_run(
    mock_log, mock_get, mock_function, mock_key, abstract_environment
):
    # Add the key function and quit function to the command map
    abstract_environment._command_map = {
        mock_key: mock_function,
        "Quit Key": abstract_environment.quit,
    }

    # Make the get command return "Test Key", then "Quit Key"
    mock_get.side_effect = [("Test Key", None), ("Quit Key", None)]

    abstract_environment.run()

    # Test that the function was called if the key exists
    if mock_key == "Test Key":
        mock_function.assert_called()
    # Test that the quit command was ran
    mock_log.assert_called_with("Stopping Process")


if __name__ == "__main__":
    log_file_queue = mp.Queue()
    test_abstract_environment_init(log_file_queue)
    abstract_environment = DummyAbstractEnvironment(log_file_queue)
    test_abstract_environment_run(
        mock_key="Test Key",
        mock_function=mock.MagicMock(return_value=False),
        abstract_environment=abstract_environment,
    )
