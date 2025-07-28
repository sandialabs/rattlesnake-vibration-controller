# -*- coding: utf-8 -*-
"""
Controller subsystem that handles computation of FRFs, CPSDs, and other spectral
quantities of interest

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
from .utilities import (flush_queue,DataAcquisitionParameters,VerboseMessageQueue,GlobalCommands)
import scipy.signal as sig
import numpy as np
from enum import Enum
from .abstract_message_process import AbstractMessageProcess
import time

WAIT_TIME = 0.05

class SpectralProcessingCommands(Enum):
    """Collection of instructions that the FRF Computation Process might get"""
    INITIALIZE_PARAMETERS = 0
    RUN_SPECTRAL_PROCESSING = 1
    CLEAR_SPECTRAL_PROCESSING = 2
    STOP_SPECTRAL_PROCESSING = 3
    SENT_SPECTRAL_DATA = 4
    SHUTDOWN_ACHIEVED = 5

class AveragingTypes(Enum):
    LINEAR = 0
    EXPONENTIAL = 1

class Estimator(Enum):
    H1 = 0
    H2 = 1
    H3 = 2
    HV = 3

class SpectralProcessingMetadata():
    def __init__(self,
                 averaging_type, averages,
                 exponential_averaging_coefficient,
                 frf_estimator,
                 num_response_channels,
                 num_reference_channels,
                 frequency_spacing,
                 sample_rate,
                 num_frequency_lines,
                 compute_cpsd = True,
                 compute_frf = True,
                 compute_coherence = True,
                 compute_apsd = True):
        self.averaging_type = averaging_type
        self.averages = averages
        self.exponential_averaging_coefficient = exponential_averaging_coefficient
        self.frf_estimator = frf_estimator
        self.num_response_channels = num_response_channels
        self.num_reference_channels = num_reference_channels
        self.frequency_spacing = frequency_spacing
        self.sample_rate = sample_rate
        self.num_frequency_lines = num_frequency_lines
        self.compute_cpsd = compute_cpsd
        self.compute_frf = compute_frf
        self.compute_coherence = compute_coherence
        self.compute_apsd = compute_apsd

    def __eq__(self,other):
        try:
            return np.all([np.all(self.__dict__[field] == other.__dict__[field]) for field in self.__dict__])
        except (AttributeError,KeyError):
            return False
        
    @property
    def requires_full_spectral_response(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.H2, Estimator.H3])
            or self.compute_cpsd):
            return True
        else:
            return False
        
    @property
    def requires_diagonal_spectral_response(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.HV])
            or self.compute_apsd or self.compute_coherence):
            return True
        else:
            return False
        
    @property
    def requires_full_spectral_reference(self):
        if ((self.compute_frf and self.frf_estimator in [Estimator.H1, Estimator.H3, Estimator.HV])
            or self.compute_cpsd or self.compute_coherence):
            return True
        else:
            return False
    
    @property
    def requires_diagonal_spectral_reference(self):
        if self.compute_apsd:
            return True
        else:
            return False
    
    @property
    def requires_spectral_reference_response(self):
        if self.compute_frf or self.compute_coherence:
            return True
        else:
            return False

class SpectralProcessingProcess(AbstractMessageProcess):
    """Class defining a subprocess that computes a FRF from a time history."""
    def __init__(self,process_name : str, 
                 command_queue : VerboseMessageQueue,
                 data_in_queue : mp.queues.Queue,
                 data_out_queue : mp.queues.Queue,
                 environment_command_queue : VerboseMessageQueue,
                 gui_update_queue : mp.queues.Queue,
                 log_file_queue : mp.queues.Queue,
                 environment_name : str):
        """
        Constructor for the FRF Computation Process
        
        Sets up the ``command_map`` and initializes internal data

        Parameters
        ----------
        process_name : str
            Name for the process that will be used in the Log file.
        command_queue : VerboseMessageQueue :
            The queue containing instructions for the FRF process
        data_for_frf_queue : mp.queues.Queue :
            Queue containing input data for the FRF computation
        updated_frf_queue : mp.queues.Queue :
            Queue where frf process will put computed frfs
        gui_update_queue : mp.queues.Queue :
            Queue for gui updates
        log_file_queue : mp.queues.Queue :
            Queue for writing to the log file
        environment_name : str
            Name of the environment that controls this subprocess.

        """
        super().__init__(process_name,log_file_queue,command_queue,gui_update_queue)
        self.map_command(SpectralProcessingCommands.INITIALIZE_PARAMETERS,self.initialize_parameters)
        self.map_command(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,self.run_spectral_processing)
        self.map_command(SpectralProcessingCommands.CLEAR_SPECTRAL_PROCESSING,self.clear_spectral_processing)
        self.map_command(SpectralProcessingCommands.STOP_SPECTRAL_PROCESSING,self.stop_spectral_processing)
        self.environment_name = environment_name
        self.data_in_queue = data_in_queue
        self.data_out_queue = data_out_queue
        self.environment_command_queue = environment_command_queue
        self.response_spectral_matrix = None
        self.reference_spectral_matrix = None
        self.response_reference_spectral_matrix = None
        self.reference_diagonal_matrix = None
        self.response_diagonal_matrix = None
        self.response_fft = None
        self.reference_fft = None
        self.spectral_processing_parameters = None
        self.frames_computed = 0
           
    def initialize_parameters(self,data : SpectralProcessingMetadata):
        """Initializes the signal processing parameters from the environment.

        Parameters
        ----------
        data :
            Container containing the setting specific to the environment.

        """
        if self.spectral_processing_parameters is None:
            reshape_arrays = True
        elif (self.spectral_processing_parameters.num_frequency_lines !=
              data.num_frequency_lines
              or
              self.spectral_processing_parameters.num_response_channels !=
              data.num_response_channels
              or
              self.spectral_processing_parameters.num_reference_channels !=
              data.num_reference_channels
              or
              self.spectral_processing_parameters.averages != 
              data.averages
              or self.spectral_processing_parameters.averaging_type !=
              data.averaging_type):
            reshape_arrays = True
        else:
            reshape_arrays = False
        self.spectral_processing_parameters = data
        if reshape_arrays:
            self.log('Initializing Empty Arrays')
            self.frames_computed = 0
            self.response_spectral_matrix = None
            self.reference_spectral_matrix = None
            self.reference_response_spectral_matrix = None
            self.reference_diagonal_matrix = None
            self.response_diagonal_matrix = None
            if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
                self.response_fft = np.nan*np.ones((
                    self.spectral_processing_parameters.averages,
                    self.spectral_processing_parameters.num_response_channels,
                    self.spectral_processing_parameters.num_frequency_lines),dtype=complex)
                self.reference_fft = np.nan*np.ones((
                    self.spectral_processing_parameters.averages,
                    self.spectral_processing_parameters.num_reference_channels,
                    self.spectral_processing_parameters.num_frequency_lines),dtype=complex)
                # print(self.response_fft.shape)
            else:
                self.response_fft = None
                self.reference_fft = None
    
    def run_spectral_processing(self,data):
        """Continuously compute FRFs from time histories.
        
        This function accepts data from the ``data_for_frf_queue`` and computes
        FRF matrices from the time data.  It uses a rolling buffer to append
        data.  The oldest data is pushed out of the buffer by the newest data.
        The test level is also passed with the response data and output
        data.  The test level is used to ensure that no frame uses
        discontinuous data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        data = flush_queue(self.data_in_queue,timeout = WAIT_TIME)
        if len(data) == 0:
            time.sleep(WAIT_TIME)
            self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
            return
        frames_received = len(data)
        self.log('Received {:} Frames'.format(frames_received))
        if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
            response_fft,reference_fft = [value for value in zip(*data)]
            self.response_fft = np.concatenate((self.response_fft[frames_received:],response_fft[-self.response_fft.shape[0]:]),axis=0)
            self.reference_fft = np.concatenate((self.reference_fft[frames_received:],reference_fft[-self.reference_fft.shape[0]:]),axis=0)
            self.log('Buffered Frames (Resp Shape: {:}, Ref Shape: {:})'.format(self.response_fft.shape,self.reference_fft.shape))
            # Exclude any with NaNs
            exclude_averages = np.any(np.isnan(self.response_fft),axis=(-1,-2))
            self.log('Computed Number Averages {:}'.format((~exclude_averages).sum()))
            # Return if there is actually no data
            if np.all(exclude_averages):
                self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
                return
            self.log('Mean FFT Value Over Averaged Frames: \n  {:}'.format(np.mean(np.abs(self.reference_fft[~exclude_averages]),axis=(-1,-2))))
            # Now we compute the spectral matrices depending on what is required.
            # Compute the response power spectra
            response_spectral_time = time.time()
            if self.spectral_processing_parameters.requires_full_spectral_response:
                self.log('Computing Full Spectral Response Matrix')
                self.response_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.response_fft[~exclude_averages],
                    np.conj(self.response_fft[~exclude_averages])
                    )/self.response_fft[~exclude_averages].shape[0]
                # Get the diagonal matrix as well
                self.response_diagonal_matrix = np.einsum(
                    'fii->fi',
                    self.response_spectral_matrix)
            elif self.spectral_processing_parameters.requires_diagonal_spectral_response:
                self.log('Computing Diagonal of Spectral Response Matrix')
                # self.response_diagonal_matrix = np.einsum(
                #     'aif,aif->fi',
                #     self.response_fft[~exclude_averages],
                #     np.conj(self.response_fft[~exclude_averages])
                #     )/self.response_fft[~exclude_averages].shape[0]
                self.response_diagonal_matrix = np.mean(self.response_fft[~exclude_averages]*np.conj(self.response_fft[~exclude_averages]),axis=0).T
            if (self.spectral_processing_parameters.requires_full_spectral_response
                or self.spectral_processing_parameters.requires_diagonal_spectral_response):
                self.log('Computed Response Spectral Matrix in {:0.2f} seconds'.format(time.time()-response_spectral_time))
                
            # Compute the reference power spectra
            reference_spectral_time = time.time()
            if self.spectral_processing_parameters.requires_full_spectral_reference:
                self.log('Computing Full Spectral Reference Matrix')
                self.reference_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.reference_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.reference_fft[~exclude_averages].shape[0]
                # Get the diagonal matrix as well
                self.reference_diagonal_matrix = np.einsum(
                    'fii->fi',
                    self.reference_spectral_matrix)
            elif self.spectral_processing_parameters.requires_diagonal_spectral_reference:
                self.log('Computing Diagonal of Spectral Reference Matrix')
                self.reference_diagonal_matrix = np.einsum(
                    'aif,aif->fi',
                    self.reference_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.reference_fft[~exclude_averages].shape[0]
            if (self.spectral_processing_parameters.requires_full_spectral_reference
                or self.spectral_processing_parameters.requires_diagonal_spectral_reference):
                self.log('Computed Reference Spectral Matrix in {:0.2f} seconds'.format(time.time()-reference_spectral_time))
                    
            # Compute cross spectra between reference and response
            if self.spectral_processing_parameters.requires_spectral_reference_response:
                cross_spectral_time = time.time()
                self.log('Computing Full Cross Spectral Response/Reference Matrix')
                self.response_reference_spectral_matrix = np.einsum(
                    'aif,ajf->fij',
                    self.response_fft[~exclude_averages],
                    np.conj(self.reference_fft[~exclude_averages])
                    )/self.response_fft[~exclude_averages].shape[0]
                self.log('Computed Crossspectral Matrix in {:0.2f} seconds'.format(time.time()-cross_spectral_time))
            frames = self.spectral_processing_parameters.averages - np.sum(exclude_averages)
                
        else: # For exponential averaging
            for frame in data:
                response_fft, reference_fft = frame
                
                # Compute response spectra
                response_spectral_time = time.time()
                if self.spectral_processing_parameters.requires_full_spectral_response:
                    self.log('Computing Full Spectral Response Matrix')
                    if self.response_spectral_matrix is None:
                        self.response_spectral_matrix = np.einsum('if,jf->fij',response_fft,np.conj(response_fft))
                    else:
                        self.response_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',response_fft,np.conj(response_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_spectral_matrix
                            )
                    # Get the diagonal matrix as well
                    self.response_diagonal_matrix = np.einsum(
                        'fii->fi',
                        self.response_spectral_matrix)
                elif self.spectral_processing_parameters.requires_diagonal_spectral_response:
                    self.log('Computing Diagonal of Spectral Response Matrix')
                    if self.response_diagonal_matrix is None:
                        self.response_diagonal_matrix = np.einsum('if,if->fi',response_fft,np.conj(response_fft))
                    else:
                        self.response_diagonal_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,if->fi',response_fft,np.conj(response_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_diagonal_matrix
                            )
                if (self.spectral_processing_parameters.requires_full_spectral_response
                    or self.spectral_processing_parameters.requires_diagonal_spectral_response):
                    self.log('Computed Response Spectral Matrix in {:0.2f} seconds'.format(time.time()-response_spectral_time))
                        
                # Compute the reference spectra
                reference_spectral_time = time.time()
                if self.spectral_processing_parameters.requires_full_spectral_reference:
                    self.log('Computing Full Spectral Reference Matrix')
                    if self.reference_spectral_matrix is None:
                        self.reference_spectral_matrix = np.einsum('if,jf->fij',reference_fft,np.conj(reference_fft))
                    else:
                        self.reference_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',reference_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.reference_spectral_matrix
                            )
                    # Get the diagonal matrix as well
                    self.reference_diagonal_matrix = np.einsum(
                        'fii->fi',
                        self.reference_spectral_matrix)
                elif self.spectral_processing_parameters.requires_diagonal_spectral_reference:
                    self.log('Computing Diagonal of Spectral Reference Matrix')
                    if self.reference_diagonal_matrix is None:
                        self.reference_diagonal_matrix = np.einsum('if,if->fi',reference_fft,np.conj(reference_fft))
                    else:
                        self.reference_diagonal_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,if->fi',reference_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.reference_diagonal_matrix
                            )
                if (self.spectral_processing_parameters.requires_full_spectral_reference
                    or self.spectral_processing_parameters.requires_diagonal_spectral_reference):
                    self.log('Computed Reference Spectral Matrix in {:0.2f} seconds'.format(time.time()-reference_spectral_time))
                        
                # Compute reference and response cross spectra
                if self.spectral_processing_parameters.requires_spectral_reference_response:
                    cross_spectral_time = time.time()
                    self.log('Computing Full Cross Spectral Response/Reference Matrix')
                    if self.response_reference_spectral_matrix is None:
                        self.response_reference_spectral_matrix = np.einsum('if,jf->fij',response_fft,np.conj(reference_fft))
                    else:
                        self.response_reference_spectral_matrix = (
                            self.spectral_processing_parameters.exponential_averaging_coefficient
                            *np.einsum('if,jf->fij',response_fft,np.conj(reference_fft))
                            +
                            (1-self.spectral_processing_parameters.exponential_averaging_coefficient)
                            *self.response_reference_spectral_matrix
                            )
                    self.log('Computed Crossspectral Matrix in {:0.2f} seconds'.format(time.time()-cross_spectral_time))
                self.frames_computed += 1
                
            frames = self.frames_computed
        self.log('Computed Spectral Matrices for {:} frames in {:0.2f} seconds'.format(frames, time.time() - response_spectral_time))
        Gffpinv = None
        Gfxpinv = None
        if self.spectral_processing_parameters.compute_frf:
            frf_time = time.time()
            if self.spectral_processing_parameters.frf_estimator == Estimator.H1:
                if Gffpinv is None:
                    Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
                frf = self.response_reference_spectral_matrix@Gffpinv
            elif self.spectral_processing_parameters.frf_estimator == Estimator.H2:
                Gfx = self.response_reference_spectral_matrix.conj().transpose(0,2,1)
                Gfxpinv = np.linalg.pinv(Gfx,rcond=1e-12,hermitian=True)
                frf = self.response_spectral_matrix@Gfxpinv
            elif self.spectral_processing_parameters.frf_estimator == Estimator.H3:
                if Gffpinv is None:
                    Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
                Gfx = self.response_reference_spectral_matrix.conj().transpose(0,2,1)
                if Gfxpinv is None:
                    Gfxpinv = np.linalg.pinv(Gfx,rcond=1e-12,hermitian=True)
                frf = (self.response_spectral_matrix@Gfxpinv + self.response_reference_spectral_matrix@Gffpinv)/2
            elif self.spectral_processing_parameters.frf_estimator == Estimator.HV:
                Gxx = self.response_diagonal_matrix.T[...,np.newaxis,np.newaxis]
                Gxf = np.einsum('fij->ifj',self.response_reference_spectral_matrix)[...,np.newaxis,:]
                Gff = self.reference_spectral_matrix
                Gff = np.broadcast_to(Gff,Gxx.shape[:-2]+Gff.shape[-2:])
                Gffx = np.block([[Gff, np.conj(np.moveaxis(Gxf, -2, -1))],
                                 [Gxf, Gxx]])
                # Compute eigenvalues
                lam, evect = np.linalg.eigh(np.moveaxis(Gffx, -2, -1))
                # Get the evect corresponding to the minimum eigenvalue
                evect = evect[..., 0]  # Assumes evals are sorted ascending
                frf = np.moveaxis(-evect[..., :-1] / evect[..., -1:],  # Scale so last value is -1
                                  -3, -2)
            self.log('Computed FRF in {:0.2f} seconds'.format(time.time()-frf_time))
            cond_time = time.time()
            frf_condition = np.linalg.cond(frf)
            self.log('Computed FRF Condition Number in {:0.2f} seconds'.format(time.time()-cond_time))
        else:
            frf = None
            frf_condition = None
        if self.spectral_processing_parameters.compute_coherence:
            coh_time = time.time()
            if Gffpinv is None:
                Gffpinv = np.linalg.pinv(self.reference_spectral_matrix,rcond=1e-12,hermitian=True)
            coherence = (np.einsum('fij,fjk,fik->fi',
                                   self.response_reference_spectral_matrix,
                                   Gffpinv,
                                   self.response_reference_spectral_matrix.conj()) / 
                         self.response_diagonal_matrix).real
            self.log('Computed Coherence in {:0.2f} seconds'.format(time.time()-coh_time))
        else:
            coherence = None
        if self.spectral_processing_parameters.compute_cpsd:
            cpsd_time = time.time()
            reference_spectral_matrix = self.reference_spectral_matrix.copy()
            response_spectral_matrix = self.response_spectral_matrix.copy()
            # Normalize
            response_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/ # Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            response_spectral_matrix[1:-1] *= 2
            reference_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/# Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            reference_spectral_matrix[1:-1] *= 2
            self.log('Computed CPSDs in {:0.2f} seconds'.format(time.time()-cpsd_time))
        elif self.spectral_processing_parameters.compute_apsd:
            apsd_time = time.time()
            reference_spectral_matrix = self.reference_diagonal_matrix.copy()
            response_spectral_matrix = self.response_diagonal_matrix.copy()
            # Normalize
            response_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/ # Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            response_spectral_matrix[1:-1] *= 2
            reference_spectral_matrix *= (self.spectral_processing_parameters.frequency_spacing/# Window correction was done in the data collector
                                         self.spectral_processing_parameters.sample_rate**2)
            reference_spectral_matrix[1:-1] *= 2
            self.log('Computed APSDs in {:0.2f} seconds'.format(time.time()-apsd_time))
        else:
            response_spectral_matrix = None
            reference_spectral_matrix = None
        frequencies = np.arange(self.spectral_processing_parameters.num_frequency_lines)*self.spectral_processing_parameters.frequency_spacing
        self.log('Sending Updated Spectral Data')
        self.data_out_queue.put((frames,frequencies,frf,coherence,
                                 response_spectral_matrix,
                                 reference_spectral_matrix,frf_condition))
        # Keep running
        self.command_queue.put(self.process_name,(SpectralProcessingCommands.RUN_SPECTRAL_PROCESSING,None))
        
    def clear_spectral_processing(self,data):
        """Clears all data in the buffer so the FRF starts fresh from new data

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        self.frames_computed = 0
        self.response_spectral_matrix = None
        self.reference_spectral_matrix = None
        self.response_reference_spectral_matrix = None
        if self.spectral_processing_parameters.averaging_type == AveragingTypes.LINEAR:
            self.response_fft[:] = np.nan
            self.reference_fft[:] = np.nan
        else:
            self.response_fft = None
            self.reference_fft = None
    
    def stop_spectral_processing(self,data):
        """Stops computing FRFs from time data.

        Parameters
        ----------
        data : Ignored
            This parameter is not used by the function but must be present
            due to the calling signature of functions called through the
            ``command_map``

        """
        time.sleep(WAIT_TIME)
        flushed_data = self.command_queue.flush(self.process_name)
        # Put back any quit message that may have been pulled off
        for message,data in flushed_data:
            if message == GlobalCommands.QUIT:
                self.command_queue.put(self.process_name,(message,data))
        flush_queue(self.data_out_queue)
        self.environment_command_queue.put(self.process_name,(SpectralProcessingCommands.SHUTDOWN_ACHIEVED,None))

def spectral_processing_process(environment_name : str,
                               command_queue : VerboseMessageQueue,
                               data_in_queue : mp.queues.Queue,
                               data_out_queue : mp.queues.Queue,
                               environment_command_queue : VerboseMessageQueue,
                               gui_update_queue : mp.queues.Queue,
                               log_file_queue : mp.queues.Queue,
                               process_name = None
                               ):
    """Function passed to multiprocessing as the FRF computation process
    
    This process creates the ``FRFComputationProcess`` object and calls the
    ``run`` function.


    Parameters
    ----------
    environment_name : str :
        Name of the environment that this subprocess belongs to.
    command_queue : VerboseMessageQueue :
        The queue containing instructions for the FRF process
    data_for_frf_queue : mp.queues.Queue :
        Queue containing input data for the FRF computation
    updated_frf_queue : mp.queues.Queue :
        Queue where frf process will put computed frfs
    gui_update_queue : mp.queues.Queue :
        Queue for gui updates
    log_file_queue : mp.queues.Queue :
        Queue for writing to the log file

    """
    
    spectral_processing_instance = SpectralProcessingProcess(
        environment_name + ' Spectral Processing Computation'
        if process_name is None else process_name,
        command_queue,
        data_in_queue,
        data_out_queue,
        environment_command_queue,
        gui_update_queue,
        log_file_queue, environment_name)
    spectral_processing_instance.run()
