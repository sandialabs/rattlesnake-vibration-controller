"""
Defines data analysis performed for environments that use system identification

Abstract environment that can be used to create new environment control strategies
in the controller that use system identification.

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

import multiprocessing as mp
from enum import Enum

from rattlesnake.process.abstract_message_process import AbstractMessageProcess
from rattlesnake.utilities import VerboseMessageQueue, flush_queue


class SysIDDataAnalysisCommands(Enum):
    """Valid commands to send to the data analysis process of an environment using system id"""

    INITIALIZE_PARAMETERS = 0
    RUN_NOISE = 1
    RUN_TRANSFER_FUNCTION = 2
    START_SHUTDOWN_AND_RUN_SYSID = 3
    START_SHUTDOWN = 4
    STOP_SYSTEM_ID = 5
    SHUTDOWN_ACHIEVED = 6
    SYSTEM_ID_COMPLETE = 7
    LOAD_TRANSFER_FUNCTION = 8
    LOAD_NOISE = 9


class SysIDDataAnalysisUICommands(Enum):

    NOISE_UPDATE = 0
    SYS_ID_UPDATE = 1


class AbstractSysIDAnalysisProcess(AbstractMessageProcess):
    """Process to perform data analysis and control calculations in an environment
    using system id"""

    def __init__(
        self,
        process_name: str,
        command_queue: VerboseMessageQueue,
        data_in_queue: mp.queues.Queue,
        data_out_queue: mp.queues.Queue,
        environment_command_queue: VerboseMessageQueue,
        log_file_queue: mp.queues.Queue,
        gui_update_queue: mp.queues.Queue,
        environment_name: str,
    ):
        """Initialize the environment process

        Parameters
        ----------
        process_name : str
            The name of the process
        command_queue : VerboseMessageQueue
            A queue used to send commands to this process
        data_in_queue : mp.queues.Queue
            A queue receiving frames of data from the data collector
        data_out_queue : mp.queues.Queue
            A queue to put the next output or analysis results for the environment to use
        environment_command_queue : VerboseMessageQueue
            A queue used to send commands to the main environment process
        log_file_queue : mp.queues.Queue
            A queue used to send log file strings
        gui_update_queue : mp.queues.Queue
            A queue used to send updates back to the graphical user interface
        environment_name : str
            The name of the environment owning this process
        """
        super().__init__(process_name, log_file_queue, command_queue, gui_update_queue)
        self.map_command(
            SysIDDataAnalysisCommands.INITIALIZE_PARAMETERS,
            self.initialize_sysid_parameters,
        )
        self.map_command(SysIDDataAnalysisCommands.RUN_NOISE, self.run_sysid_noise)
        self.map_command(
            SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,
            self.run_sysid_transfer_function,
        )
        self.map_command(SysIDDataAnalysisCommands.STOP_SYSTEM_ID, self.stop_sysid)
        self.map_command(SysIDDataAnalysisCommands.LOAD_NOISE, self.load_sysid_noise)
        self.map_command(
            SysIDDataAnalysisCommands.LOAD_TRANSFER_FUNCTION,
            self.load_sysid_transfer_function,
        )
        self.environment_name = environment_name
        self.environment_command_queue = environment_command_queue
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.parameters = None
        self.frames = None
        self.frequencies = None
        self.sysid_frf = None
        self.sysid_coherence = None
        self.sysid_response_cpsd = None
        self.sysid_reference_cpsd = None
        self.sysid_response_noise = None
        self.sysid_reference_noise = None
        self.sysid_condition = None
        self.startup = True

    def initialize_sysid_parameters(self, data):
        """Stores parameters describing the system identification into the object

        Parameters
        ----------
        data : AbstractSysIdMetadata
            A metadata object containing the parameters to define the system identification
        """
        self.parameters = data

    def load_sysid_noise(self, spectral_data):
        """Loads noise data from a previous system identification

        Parameters
        ----------
        spectral_data : tuple
            A tuple containing frames, frequencies, system id FRFs, coherence, response cpsd,
            reference_cpsd and condition number
        """
        self.log("Obtained Spectral Data")
        (
            self.frames,
            self.frequencies,
            _,
            _,
            self.sysid_response_noise,
            self.sysid_reference_noise,
            _,
        ) = spectral_data

    def load_sysid_transfer_function(self, spectral_data, skip_sysid=True):
        """Loads system ID data from a previous system identification

        Parameters
        ----------
        spectral_data : tuple
            A tuple containing frames, frequencies, system id FRFs, coherence, response cpsd,
            reference_cpsd and condition number
        skip_sysid : bool, optional
            If True, send the system identification complete flag to the controller. By default True
        """
        self.log("Obtained Spectral Data")
        (
            self.frames,
            self.frequencies,
            self.sysid_frf,
            self.sysid_coherence,
            self.sysid_response_cpsd,
            self.sysid_reference_cpsd,
            self.sysid_condition,
        ) = spectral_data
        if skip_sysid:
            self.environment_command_queue.put(
                self.process_name,
                (
                    SysIDDataAnalysisCommands.SYSTEM_ID_COMPLETE,
                    (
                        self.frames,
                        0,
                        self.frequencies,
                        self.sysid_frf,
                        self.sysid_coherence,
                        self.sysid_response_cpsd,
                        self.sysid_reference_cpsd,
                        self.sysid_condition,
                        self.sysid_response_noise,
                        self.sysid_reference_noise,
                    ),
                ),
            )

    def run_sysid_noise(self, auto_shutdown):
        """Starts and runs the system identification noise phase.

        Parameters
        ----------
        auto_shutdown : bool
            If True, the environment will automatically shut down when the requested number of
            frames is reached.  If False, the noise characterization will run until manually
            stopped.
        """
        if self.startup:
            self.startup = False
            self.frames = 0
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.load_sysid_noise(spectral_data[-1])
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        SysIDDataAnalysisUICommands.NOISE_UPDATE,
                        (
                            self.frames,
                            self.parameters.sysid_noise_averages,
                            self.frequencies,
                            self.sysid_response_noise,
                            self.sysid_reference_noise,
                        ),
                    ),
                )
            )
        if auto_shutdown and self.parameters.sysid_noise_averages == self.frames:
            self.environment_command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.START_SHUTDOWN_AND_RUN_SYSID, None),
            )
            self.stop_sysid(None)
        else:
            self.command_queue.put(
                self.process_name, (SysIDDataAnalysisCommands.RUN_NOISE, auto_shutdown)
            )

    def run_sysid_transfer_function(self, auto_shutdown):
        """Starts and runs the system identification

        Parameters
        ----------
        auto_shutdown : bool
            If True, the system identification will stop automatically upon reaching the requested
            number of measurement frames.  If False, it will run indefinitely until manually
            stopped.
        """
        if self.startup:
            self.startup = False
            self.frames = 0
        spectral_data = flush_queue(self.data_in_queue)
        if len(spectral_data) > 0:
            self.load_sysid_transfer_function(spectral_data[-1], skip_sysid=False)
            self.gui_update_queue.put(
                (
                    self.environment_name,
                    (
                        SysIDDataAnalysisUICommands.SYS_ID_UPDATE,
                        (
                            self.frames,
                            self.parameters.sysid_averages,
                            self.frequencies,
                            self.sysid_frf,
                            self.sysid_coherence,
                            self.sysid_response_cpsd,
                            self.sysid_reference_cpsd,
                            self.sysid_condition,
                        ),
                    ),
                )
            )
        if auto_shutdown and self.parameters.sysid_averages == self.frames:
            self.environment_command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.START_SHUTDOWN, (False, True)),
            )
            self.stop_sysid(None)
            self.environment_command_queue.put(
                self.process_name,
                (
                    SysIDDataAnalysisCommands.SYSTEM_ID_COMPLETE,
                    (
                        self.frames,
                        self.parameters.sysid_averages,
                        self.frequencies,
                        self.sysid_frf,
                        self.sysid_coherence,
                        self.sysid_response_cpsd,
                        self.sysid_reference_cpsd,
                        self.sysid_condition,
                        self.sysid_response_noise,
                        self.sysid_reference_noise,
                    ),
                ),
            )
        else:
            self.command_queue.put(
                self.process_name,
                (SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION, auto_shutdown),
            )

    def stop_sysid(self, data):  # pylint: disable=unused-argument
        """Stops the currently running system identification phase

        Parameters
        ----------
        data : ignored
            This argument is not used, but is required by the calling signature of functions
            that get called via the command map.
        """
        # Remove any run_transfer_function or run_control from the queue
        instructions = self.command_queue.flush(self.process_name)
        for instruction in instructions:
            if not instruction[0] in [
                SysIDDataAnalysisCommands.RUN_NOISE,
                SysIDDataAnalysisCommands.RUN_TRANSFER_FUNCTION,
            ]:
                self.command_queue.put(self.process_name, instruction)
        flush_queue(self.data_out_queue)
        self.startup = True
        self.environment_command_queue.put(
            self.process_name, (SysIDDataAnalysisCommands.SHUTDOWN_ACHIEVED, None)
        )


def sysid_data_analysis_process(
    environment_name: str,
    command_queue: VerboseMessageQueue,
    data_in_queue: mp.queues.Queue,
    data_out_queue: mp.queues.Queue,
    environment_command_queue: VerboseMessageQueue,
    gui_update_queue: mp.queues.Queue,
    log_file_queue: mp.queues.Queue,
    process_name=None,
):
    """An function called by multiprocessing to start up the system identification analysis
    process.

    Some environments may override the AbstractSysIDAnalysisProcess class and therefore should
    redefine this function to call that class.

    Parameters
    ----------
    environment_name : str
        The name of the environment
    command_queue : VerboseMessageQueue
        A queue used to send commands to this process
    data_in_queue : mp.queues.Queue
        A queue used to send frames of data and spectral quantities to the data analysis process
    data_out_queue : mp.queues.Queue
        A queue used to send control and analysis results back to the environment
    environment_command_queue : VerboseMessageQueue
        A queue used to send commands to the environment
    gui_update_queue : mp.queues.Queue
        A queue used to send updates to the graphical user interface
    log_file_queue : mp.queues.Queue
        A queue used to send log file messages
    process_name : _type_, optional
        A name for the process.  If not specified, it will be the environment name appended with
        Data Analysis.
    """
    data_analysis_instance = AbstractSysIDAnalysisProcess(
        environment_name + " Data Analysis" if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        log_file_queue,
        gui_update_queue,
        environment_name,
    )

    data_analysis_instance.run()
