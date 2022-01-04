# -*- coding: utf-8 -*-
"""
Defines abstract processes that can be used as subprocesses in the controller.
Uses the message,data producer and consumer paradigm.

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

from abc import ABC
from .utilities import VerboseMessageQueue,GlobalCommands
from datetime import datetime
import traceback
import os
from multiprocessing.queues import Queue

class AbstractMessageProcess(ABC):
    """Abstract class for a subprocess of an environment.
    
    This class operates similarly to an AbstractEnvironment class but is
    designed to be a sub-process of the environment
    
    """
    def __init__(self,process_name : str, log_file_queue : Queue, command_queue : VerboseMessageQueue, gui_update_queue : Queue):
        """
        Constructor for the AbstractMessageProcess class.
        
        Sets up private data members for the properties.  Initializes the
        ``command_map`` with the ``GlobalCommands.QUIT`` message

        Parameters
        ----------
        process_name : str
            Name of the process
        log_file_queue : Queue
            Queue to which log file messages should be written.
        command_queue : VerboseMessageQueue
            Queue from which instructions for the process will be pulled
        gui_update_queue : Queue
            Queue to which GUI update instructions will be written.

        """
        self._process_name = process_name
        self._log_file_queue = log_file_queue
        self._gui_update_queue = gui_update_queue
        self._command_queue = command_queue
        self._command_map = {GlobalCommands.QUIT:self.quit}
    
    def log(self,message):
        """Write a message to the log file
        
        This function puts a message onto the ``log_file_queue`` so it will
        eventually be written to the log file.
        
        When written to the log file, the message will include the date and
        time that the message was queued, the name of the environment, and
        then the message itself.

        Parameters
        ----------
        message : str :
            A message that will be written to the log file.

        """
        self.log_file_queue.put('{:}: {:} -- {:}\n'.format(datetime.now(),self.process_name,message))
        
    @property
    def process_name(self) -> str:
        """Property containing the name of the process used when writing log messages"""
        return self._process_name

    @property
    def command_map(self) -> dict:
        """Dictionary mapping instructions to functions of the class"""
        return self._command_map
    
    @property
    def gui_update_queue(self) -> Queue:
        """Queue to which GUI update instructions will be written."""
        return self._gui_update_queue
    
    def map_command(self,key,function):
        """Maps commands to instructions
        
        Maps the instruction ``key`` to the function ``function`` so when
        ``(key,data)`` pairs are pulled from the ``command_queue``, the function
        ``function`` is called with argument ``data``.

        Parameters
        ----------
        key :
            Instruction pulled from the command queue
            
        function :
            Function to be called when the given ``key`` is pulled from the
            ``command_queue``

        """
        self._command_map[key] = function
    
    @property
    def command_queue(self) -> VerboseMessageQueue:
        """Queue from which instructions for the process will be pulled"""
        return self._command_queue
        
    @property
    def log_file_queue(self) -> VerboseMessageQueue:
        """Queue to which log file messages should be written."""
        return self._log_file_queue
    
    def run(self):
        """The main function that is run by the process
        
        A function that is called by the process function that
        sits in a while loop waiting for instructions on the command queue.
        
        When the instructions are recieved, they are separated into
        ``(message,data)`` pairs.  The ``message`` is used in conjuction with
        the ``command_map`` to identify which function should be called, and
        the ``data`` is passed to that function as the argument.  If the
        function returns a truthy value, it signals to the ``run`` function
        that it is time to stop the loop and exit.


        """
        self.log('Starting Process with PID {:}'.format(os.getpid()))
        while True:
            # Get the message from the queue
            message,data = self.command_queue.get(self.process_name)
            # Call the function corresponding to that message with the data as argument
            try:
                function = self.command_map[message]
            except KeyError:
                self.log('Undefined Message {:}, acceptable messages are {:}'.format(message,[key for key in self.command_map]))
                continue
            try:
                halt_flag = function(data)
            except Exception:
                tb = traceback.format_exc()
                self.log('ERROR\n\n {:}'.format(tb))
                self.gui_update_queue.put(('error',('{:} Error'.format(self.process_name),'!!!UNKNOWN ERROR!!!\n\n{:}'.format(tb))))
                halt_flag = False
            # If we get a true value, stop.
            if halt_flag:
                self.log('Stopping Process')
                break
            
    def quit(self,data):
        """Returns True to stop the ``run`` while loop and exit the process

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        Returns
        -------
        True :
            This function returns True to signal to the ``run`` while loop
            that it is time to close down the environment.

        """
        return True