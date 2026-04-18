"""
Tests for Abstract Environment

This module contains tests for the AbstractEnvironment and AbstractUI classes,
using dummy implementations to verify their basic behavior and communication.
"""

import multiprocessing as mp
from unittest import mock

import pytest

from functions.common_functions import fake_time
from rattlesnake.environment.abstract_environment import AbstractEnvironment, AbstractMetadata
from rattlesnake.user_interface.abstract_user_interface import AbstractUI
from rattlesnake.utilities import GlobalCommands, VerboseMessageQueue


# Initialize log_file_queue that verbose queues will use
@pytest.fixture()
def log_file_queue():
    """
    Fixture for a log file queue.
    """
    return mp.Queue()


# Create environment_command_queue for AbstractUI/AbstractEnvironment
@pytest.fixture()
def environment_command_queue(log_file_queue):
    """
    Fixture for an environment command queue.
    """
    return VerboseMessageQueue(log_file_queue, "Environment Command Queue")


# Create controller_command_queue for AbstractUI/AbstractEnvironment
@pytest.fixture()
def controller_command_queue(log_file_queue):
    """
    Fixture for a controller communication queue.
    """
    return VerboseMessageQueue(log_file_queue, "Controller Communication Queue")


# Create gui_update_queue for AbstractEnvironment
@pytest.fixture()
def gui_update_queue():
    """
    Fixture for a GUI update queue.
    """
    return mp.Queue()


# Create data_in_queue for AbstractEnvironment
@pytest.fixture()
def data_in_queue():
    """
    Fixture for an incoming data queue.
    """
    return mp.Queue()


# Create data_out_queue for AbstractEnvironment
@pytest.fixture()
def data_out_queue():
    """
    Fixture for an outgoing data queue.
    """
    return mp.Queue()


# Initialize the acquisition_active multiprocessing integer
@pytest.fixture
def acquisition_active():
    """
    Fixture for an acquisition active status value.
    """
    return mp.Value("i", 0)


# Initialize the output_active multiprocessing integer
@pytest.fixture
def output_active():
    """
    Fixture for an output active status value.
    """
    return mp.Value("i", 0)


# Create a dummy AbstractMetadata class
class DummyAbstractMetadata(AbstractMetadata):
    """
    Dummy implementation of AbstractMetadata for testing.
    """

    def __init__(self):
        """
        Initialize the DummyAbstractMetadata.
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


# Create a dummy AbstractUI class
class DummyAbstractUI(AbstractUI):
    """
    Dummy implementation of AbstractUI for testing.
    """

    def __init__(self, log_file_queue, environment_command_queue, controller_command_queue):
        """
        Initialize the DummyAbstractUI.

        Args:
            log_file_queue: Queue for logging.
            environment_command_queue: Queue for environment commands.
            controller_command_queue: Queue for controller communication.
        """
        super().__init__(
            "Environment Name",
            environment_command_queue,
            controller_command_queue,
            log_file_queue,
        )

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

    def collect_environment_definition_parameters(self):
        """
        Collect environment definition parameters.

        Returns:
            The result of the superclass collect_environment_definition_parameters method.
        """
        return super().collect_environment_definition_parameters()

    def initialize_data_acquisition(self, data_acquisition_parameters):
        """
        Initialize data acquisition.

        Args:
            data_acquisition_parameters: Parameters for data acquisition.

        Returns:
            The result of the superclass initialize_data_acquisition method.
        """
        return super().initialize_data_acquisition(data_acquisition_parameters)

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

    @staticmethod
    def create_environment_template(environment_name, workbook):
        """
        Create an environment template.

        Args:
            environment_name: Name of the environment.
            workbook: Excel workbook to add the template to.

        Returns:
            The result of the superclass create_environment_template method.
        """
        return super().create_environment_template(environment_name, workbook)

    def set_parameters_from_template(self, worksheet):
        """
        Set parameters from a template.

        Args:
            worksheet: Excel worksheet containing the parameters.

        Returns:
            The result of the superclass set_parameters_from_template method.
        """
        return super().set_parameters_from_template(worksheet)


# Create a dummy AbstractEnvironment class
class DummyAbstractEnvironment(AbstractEnvironment):
    """
    Dummy implementation of AbstractEnvironment for testing.
    """

    def __init__(
        self,
        log_file_queue,
        environment_command_queue,
        controller_command_queue,
        gui_update_queue,
        data_in_queue,
        data_out_queue,
    ):
        """
        Initialize the DummyAbstractEnvironment.

        Args:
            log_file_queue: Queue for logging.
            environment_command_queue: Queue for environment commands.
            controller_command_queue: Queue for controller communication.
            gui_update_queue: Queue for GUI updates.
            data_in_queue: Queue for incoming data.
            data_out_queue: Queue for outgoing data.
        """
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
        """
        Initialize data acquisition parameters.

        Args:
            data_acquisition_parameters: Parameters for data acquisition.

        Returns:
            The result of the superclass initialize_data_acquisition_parameters method.
        """
        return super().initialize_data_acquisition_parameters(data_acquisition_parameters)

    def initialize_environment_test_parameters(self, environment_parameters):
        """
        Initialize environment test parameters.

        Args:
            environment_parameters: Parameters for the environment test.

        Returns:
            The result of the superclass initialize_environment_test_parameters method.
        """
        return super().initialize_environment_test_parameters(environment_parameters)

    def stop_environment(self, data):
        """
        Stop the environment.

        Args:
            data: Data associated with the stop command.

        Returns:
            The result of the superclass stop_environment method.
        """
        return super().stop_environment(data)


# Initialize the AbstractUI class
@pytest.fixture
def abstract_ui(log_file_queue, environment_command_queue, controller_command_queue):
    """
    Fixture for a DummyAbstractUI instance.
    """
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
    """
    Fixture for a DummyAbstractEnvironment instance.
    """
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
    """
    Test the initialization of the AbstractMetadata class via DummyAbstractMetadata.
    """
    abstract_metadata = DummyAbstractMetadata()

    assert isinstance(abstract_metadata, DummyAbstractMetadata)


# Test the AbstractUI log function
# Prevent from writing to the log_file_queue
@mock.patch("rattlesnake.environment.abstract_environment.Queue.put")
# Replace the date and time with a string
@mock.patch("rattlesnake.user_interface.abstract_user_interface.datetime")
def test_abstract_ui_log(mock_time, mock_put, abstract_ui):
    """
    Test the logging functionality of the AbstractUI class.
    """
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
    """
    Test the initialization of the AbstractEnvironment class via DummyAbstractEnvironment.
    """
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
    """
    Test the logging functionality of the AbstractEnvironment class.
    """
    message = "Test Message"
    mock_time.now = fake_time

    abstract_environment.log(message)

    # Test if the correct string was writtin to log_file_queue
    mock_put.assert_called_with("{:}: {:} -- {:}\n".format("Datetime", "Environment Name", message))


# Test the AbstractEnvironment quit function
def test_abstract_environment_quit(abstract_environment):
    """
    Test the quit functionality of the AbstractEnvironment class.
    """
    data = abstract_environment.quit(None)

    # Test that the quit function returns True
    assert data is True


# Test adding commands to AbstractEnvironment
def test_abstract_environment_map_command(abstract_environment):
    """
    Test the mapping of custom commands to the AbstractEnvironment.
    """
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
    """
    Test the run loop of the AbstractEnvironment class.
    """
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
