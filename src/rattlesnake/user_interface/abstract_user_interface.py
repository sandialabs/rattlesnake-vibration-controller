from rattlesnake.utilities import VerboseMessageQueue, DataAcquisitionParameters
from rattlesnake.environment.abstract_environment import AbstractMetadata
from abc import ABC, abstractmethod
from multiprocessing.queues import Queue
import netCDF4 as nc4
from datetime import datetime
import openpyxl


# region: User Interface
class AbstractUI(ABC):
    """Abstract User Interface class defining the interface with the controller

    This class is used to define the interface between the User Interface of a
    environment in the controller and the main controller."""

    @abstractmethod
    def __init__(
        self,
        environment_name: str,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
    ):
        """
        Stores data required by the controller to interact with the UI

        This class stores data required by the controller to interact with the
        user interface for a given environment.  This includes the environment
        name and queues to pass information between the controller and
        environment.  It additionally initializes the ``command_map`` which is
        used by the Test Profile functionality to map profile instructions to
        operations on the user interface.


        Parameters
        ----------
        environment_name : str
            The name of the environment
        environment_command_queue : VerboseMessageQueue
            A queue that will provide instructions to the corresponding
            environment
        controller_communication_queue : VerboseMessageQueue
            The queue that relays global communication messages to the controller
        log_file_queue : Queue
            The queue that will be used to put messages to the log file.


        """
        self._environment_name = environment_name
        self._log_name = environment_name + " UI"
        self._log_file_queue = log_file_queue
        self._environment_command_queue = environment_command_queue
        self._controller_communication_queue = controller_communication_queue
        self._command_map = {
            "Start Control": self.start_control,
            "Stop Control": self.stop_control,
        }

    @property
    def command_map(self) -> dict:
        """Dictionary mapping profile instructions to functions of the UI that
        are called when the instruction is executed."""
        return self._command_map

    @abstractmethod
    def start_control(self):
        """Runs the corresponding environment in the controller"""

    @abstractmethod
    def stop_control(self):
        """Stops the corresponding environment in the controller"""

    @abstractmethod
    def collect_environment_definition_parameters(self) -> AbstractMetadata:
        """
        Collect the parameters from the user interface defining the environment

        Returns
        -------
        AbstractMetadata
            A metadata or parameters object containing the parameters defining
            the corresponding environment.

        """

    @abstractmethod
    def initialize_data_acquisition(self, data_acquisition_parameters: DataAcquisitionParameters):
        """Update the user interface with data acquisition parameters

        This function is called when the Data Acquisition parameters are
        initialized.  This function should set up the environment user interface
        accordingly.

        Parameters
        ----------
        data_acquisition_parameters : DataAcquisitionParameters :
            Container containing the data acquisition parameters, including
            channel table and sampling information.

        """

    @abstractmethod
    def initialize_environment(self) -> AbstractMetadata:
        """
        Update the user interface with environment parameters

        This function is called when the Environment parameters are initialized.
        This function should set up the user interface accordingly.  It must
        return the parameters class of the environment that inherits from
        AbstractMetadata.

        Returns
        -------
        AbstractMetadata
            An AbstractMetadata-inheriting object that contains the parameters
            defining the environment.

        """

    @abstractmethod
    def retrieve_metadata(
        self, netcdf_handle: nc4._netCDF4.Dataset
    ):  # pylint: disable=c-extension-no-member
        """Collects environment parameters from a netCDF dataset.

        This function retrieves parameters from a netCDF dataset that was written
        by the controller during streaming.  It must populate the widgets
        in the user interface with the proper information.

        This function is the "read" counterpart to the store_to_netcdf
        function in the AbstractMetadata class, which will write parameters to
        the netCDF file to document the metadata.

        Note that the entire dataset is passed to this function, so the function
        should collect parameters pertaining to the environment from a Group
        in the dataset sharing the environment's name, e.g.

        ``group = netcdf_handle.groups[self.environment_name]``
        ``self.definition_widget.parameter_selector.setValue(group.parameter)``

        Parameters
        ----------
        netcdf_handle : nc4._netCDF4.Dataset :
            The netCDF dataset from which the data will be read.  It should have
            a group name with the enviroment's name.

        """

    @abstractmethod
    def update_gui(self, queue_data: tuple):
        """Update the environment's graphical user interface

        This function will receive data from the gui_update_queue that
        specifies how the user interface should be updated.  Data will usually
        be received as ``(instruction,data)`` pairs, where the ``instruction`` notes
        what operation should be taken or which widget should be modified, and
        the ``data`` notes what data should be used in the update.

        Parameters
        ----------
        queue_data : tuple
            A tuple containing ``(instruction,data)`` pairs where ``instruction``
            defines and operation or widget to be modified and ``data`` contains
            the data used to perform the operation.
        """

    @property
    def log_file_queue(self) -> Queue:
        """A property containing a reference to the queue accepting messages
        that will be written to the log file"""
        return self._log_file_queue

    @property
    def environment_command_queue(self) -> VerboseMessageQueue:
        """A property containing a reference to the queue accepting commands
        that will be delivered to the environment"""
        return self._environment_command_queue

    @property
    def controller_communication_queue(self) -> VerboseMessageQueue:
        """A property containing a reference to the queue accepting global
        commands that will be delivered to the controller"""
        return self._controller_communication_queue

    @property
    def environment_name(self):
        """A property containing the environment's name"""
        return self._environment_name

    @property
    def log_name(self):
        """A property containing the name that the UI will be referenced by in
        the log file, which will typically be ``self.environment_name + ' UI'``"""
        return self._log_name

    def log(self, message: str):
        """Write a message to the log file

        This function puts a message onto the ``log_file_queue`` so it will
        eventually be written to the log file.

        When written to the log file, the message will include the date and
        time that the message was queued, the name that the UI uses in the log
        file (``self.log_file``), and then the message itself.

        Parameters
        ----------
        message : str :
            A message that will be written to the log file.

        """
        self.log_file_queue.put(f"{datetime.now()}: {self.log_name} -- {message}\n")

    @staticmethod
    @abstractmethod
    def create_environment_template(
        environment_name: str, workbook: openpyxl.workbook.workbook.Workbook
    ):
        """Creates a template worksheet in an Excel workbook defining the
        environment.

        This function creates a template worksheet in an Excel workbook that
        when filled out could be read by the controller to re-create the
        environment.

        This function is the "write" counterpart to the
        ``set_parameters_from_template`` function in the ``AbstractUI`` class,
        which reads the values from the template file to populate the user
        interface.

        Parameters
        ----------
        environment_name : str :
            The name of the environment that will specify the worksheet's name
        workbook : openpyxl.workbook.workbook.Workbook :
            A reference to an ``openpyxl`` workbook.

        """

    @abstractmethod
    def set_parameters_from_template(self, worksheet: openpyxl.worksheet.worksheet.Worksheet):
        """
        Collects parameters for the user interface from the Excel template file

        This function reads a filled out template worksheet to create an
        environment.  Cells on this worksheet contain parameters needed to
        specify the environment, so this function should read those cells and
        update the UI widgets with those parameters.

        This function is the "read" counterpart to the
        ``create_environment_template`` function in the ``AbstractUI`` class,
        which writes a template file that can be filled out by a user.


        Parameters
        ----------
        worksheet : openpyxl.worksheet.worksheet.Worksheet
            An openpyxl worksheet that contains the environment template.
            Cells on this worksheet should contain the parameters needed for the
            user interface.

        """
