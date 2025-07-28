# -*- coding: utf-8 -*-
"""
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

import datetime
import multiprocessing as mp
import sys

# import numpy as np  # unused import
from qtpy import QtWidgets

from components import (
    ControlSelect,
    ControlTypes,
    EnvironmentSelect,
    QueueContainer,
    Ui,
    VerboseMessageQueue,
    acquisition_process,
    all_environment_processes,
    output_process,
    streaming_process,
)


def log_file_task(queue: mp.queues.Queue):
    """A multiprocessing function that collects logging data and writes to file

    Parameters
    ----------
    queue : mp.queues.Queue
        The multiprocessing queue to collect logging messages from


    """
    with open("Rattlesnake.log", "w") as f:
        while True:
            output = queue.get()
            if output == "quit":
                f.write("Program quitting, logging terminated.")
                break
            num_newlines = output.count("\n")
            if num_newlines > 1:
                output = output.replace("\n", "////", num_newlines - 1)
            f.write(output)
            f.flush()


if __name__ == "__main__":
    mp.freeze_support()  # Required to compile into an executable
    print("Loading Rattlesnake...")
    # Create the user interface application
    app = QtWidgets.QApplication(sys.argv)

    # Check to see if the arguments have specified the control strategy
    upper_args = [arg.upper() for arg in sys.argv]
    control_type = None
    for ct in ControlTypes:
        if ct.name in upper_args:
            control_type = ct
            print("Using Control Type {:} from command line".format(ct.name))
            break
    if control_type is None:
        control_type, close_flag = ControlSelect.select_control()
        if close_flag == False:
            sys.exit()

    loaded_profile = None
    if control_type == ControlTypes.COMBINED:
        environment_select_results = EnvironmentSelect.select_environment()
        if environment_select_results[0] == 0:
            sys.exit()
        environments = environment_select_results[1]
        if environment_select_results[0] == -1:
            loaded_profile = environment_select_results[2]
    else:
        environments = [[control_type, control_type.name.title()]]

    # Create the processes
    # Set up the log file process
    log_file_queue = mp.Queue()
    log_file_process = mp.Process(target=log_file_task, args=(log_file_queue,))
    log_file_process.start()

    # Set up the other command queues
    acquisition_command_queue = VerboseMessageQueue(
        log_file_queue, "Acquisition Command Queue"
    )
    output_command_queue = VerboseMessageQueue(log_file_queue, "Output Command Queue")
    streaming_command_queue = VerboseMessageQueue(
        log_file_queue, "Streaming Command Queue"
    )

    # Set up synchronization queues
    input_output_sync_queue = mp.Queue()
    #    environment_sync_queue = VerboseMessageQueue(log_file_queue,'Environment Sync Queue')
    single_process_hardware_queue = mp.Queue()

    # Set up shared memory to know when acquisition and output are running
    acquisition_active = mp.Value("i", 0)
    output_active = mp.Value("i", 0)

    # Create Queues for communication back to the GUI
    gui_update_queue = mp.Queue()

    # Set up a global communication queue so the subprocesses can talk back to the controller
    controller_communication_queue = VerboseMessageQueue(
        log_file_queue, "Controller Communication Queue"
    )

    # Set up the individual environment processes and queues
    environment_processes = {}
    environment_command_queues = {}
    environment_data_in_queues = {}
    environment_data_out_queues = {}
    for environment_type, environment_name in environments:
        # Create the queue
        environment_command_queues[environment_name] = VerboseMessageQueue(
            log_file_queue, environment_name + " Command Queue"
        )
        environment_data_in_queues[environment_name] = mp.Queue()
        environment_data_out_queues[environment_name] = mp.Queue()
        # Select the right process function
        process_fn = all_environment_processes[environment_type]
        # Start the process
        environment_processes[environment_name] = mp.Process(
            target=process_fn,
            args=(
                environment_name,
                environment_command_queues[environment_name],
                gui_update_queue,
                controller_communication_queue,
                log_file_queue,
                environment_data_in_queues[environment_name],
                environment_data_out_queues[environment_name],
                acquisition_active,
                output_active,
            ),
        )
        environment_processes[environment_name].start()

    # Put all the queues into one nice package
    queue_container = QueueContainer(
        controller_communication_queue,
        acquisition_command_queue,
        output_command_queue,
        streaming_command_queue,
        log_file_queue,
        input_output_sync_queue,
        #                                     environment_sync_queue,
        single_process_hardware_queue,
        gui_update_queue,
        environment_command_queues,
        environment_data_in_queues,
        environment_data_out_queues,
    )

    # Set up the processes
    acquisition_proc = mp.Process(
        target=acquisition_process,
        args=(queue_container, environments, acquisition_active),
    )
    acquisition_proc.start()

    output_proc = mp.Process(
        target=output_process, args=(queue_container, environments, output_active)
    )
    output_proc.start()

    streaming_proc = mp.Process(target=streaming_process, args=(queue_container,))
    streaming_proc.start()

    window = Ui(environments, queue_container, loaded_profile)

    # Run the program
    app.exec_()

    # Rejoin all proceseses
    log_file_queue.put(
        "{:}: Joining Acquisition Process\n".format(datetime.datetime.now())
    )
    acquisition_proc.join()
    log_file_queue.put("{:}: Joining Output Process\n".format(datetime.datetime.now()))
    output_proc.join()
    log_file_queue.put(
        "{:}: Joining Streaming Process\n".format(datetime.datetime.now())
    )
    streaming_proc.join()
    for environment_name, environment_process in environment_processes.items():
        log_file_queue.put(
            "{:}: Joining {:} Process\n".format(
                datetime.datetime.now(), environment_name
            )
        )
        environment_process.join()
    log_file_queue.put(
        "{:}: Joining Log File Process\n".format(datetime.datetime.now())
    )
    log_file_queue.put("quit")
    log_file_process.join()
