"""
Rattlesnake Queue Test Functions

This code contains the functions used in test_queues.py. These functions
clear the queues and mock a datetime.now function.
"""

from rattlesnake.utilities import VerboseMessageQueue, GlobalCommands
from functions.common_functions import fake_time
import multiprocessing as mp
from unittest import mock
from datetime import datetime


# Clear verbose queue
def clear_verbose_queue(q, task_name, verbose_array):
    # Mock the datetime and message_id objects used during the log message in the VerboseQueue.get function
    with (
        mock.patch("rattlesnake.utilities.datetime") as mock_time,
        mock.patch("rattlesnake.utilities.VerboseMessageQueue.generate_message_id") as mock_id,
    ):
        mock_time.now = fake_time
        mock_id.return_value = "1"

        # Clear the queue and store data to verbose_array
        idx = 0
        while not q.empty():
            output_value = q.get(task_name)
            verbose_array[idx] = output_value[1]
            idx += 1


# Clear the log_file_queue
def clear_log_queue(q, log_string):
    # Get string from queue and store it to the log_string bstring
    while not q.empty():
        output_string = q.get()
        output_string = output_string.encode("utf-8")
        log_string.value = log_string.value + output_string


if __name__ == "__main__":
    time = datetime.now()
    pass
