import numpy as np
import os
import inspect
import traceback

from qtpy import QtWidgets, QtCore
from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawUI,
    AbstractControlLawComputation,
)
from rattlesnake.utilities import VerboseMessageQueue, rms_csd

def cpsd_coherence(cpsd):
    num = np.abs(cpsd)**2
    den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
           cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
    den[den==0.0] = 1 # Set to 1
    return np.real(num/
                   den)

def cpsd_phase(cpsd):
    return np.angle(cpsd)

def cpsd_from_coh_phs(asd,coh,phs):
    return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])

def cpsd_autospectra(cpsd):
    return np.einsum('ijj->ij',cpsd)

def match_coherence_phase(cpsd_original,cpsd_to_match):
    coh = cpsd_coherence(cpsd_to_match)
    phs = cpsd_phase(cpsd_to_match)
    asd = cpsd_autospectra(cpsd_original)
    return cpsd_from_coh_phs(asd,coh,phs)

def trace(cpsd):
    return np.einsum('ijj->i',cpsd)

def pseudoinverse_control(specification, # Specifications
                          warning_levels, # Warning levels
                          abort_levels, # Abort Levels
                          transfer_function,  # Transfer Functions
                          noise_response_cpsd,  # Noise levels and correlation 
                          noise_reference_cpsd, # from the system identification
                          sysid_response_cpsd,  # Response levels and correlation
                          sysid_reference_cpsd, # from the system identification
                          multiple_coherence, # Coherence from the system identification
                          frames, # Number of frames in the CPSD and FRF matrices
                          total_frames, # Total frames that could be in the CPSD and FRF matrices
                          extra_parameters = '', # Extra parameters for the control law
                          last_response_cpsd = None, # Last Control Response for Error Correction
                          last_output_cpsd = None, # Last Control Excitation for Drive-based control
                          ):
    """
    A control law that simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the specification by that inverse via
    the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    return output

def match_trace_pseudoinverse(specification, # Specifications
                              warning_levels, # Warning levels
                              abort_levels, # Abort Levels
                              transfer_function,  # Transfer Functions
                              noise_response_cpsd,  # Noise levels and correlation 
                              noise_reference_cpsd, # from the system identification
                              sysid_response_cpsd,  # Response levels and correlation
                              sysid_reference_cpsd, # from the system identification
                              multiple_coherence, # Coherence from the system identification
                              frames, # Number of frames in the CPSD and FRF matrices
                              total_frames, # Total frames that could be in the CPSD and FRF matrices
                              extra_parameters = '', # Extra parameters for the control law
                              last_response_cpsd = None, # Last Control Response for Error Correction
                              last_output_cpsd = None, # Last Control Excitation for Drive-based control
                              ):
    """
    A control law that initially performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.  On subsequent iterations, it will scale
    the output at each frequency line up or down depending on if the frequency line
    is on average higher or low.  This is equivalent to matching the "trace" (sum of
    the diagonal) of the CPSD specification in a closed-loop fashion.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # If it's the first time through, do the actual control
    if last_output_cpsd is None:
        # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function,rcond)
        # Return the least squares solution for the new output CPSD
        output = tf_pinv@specification@tf_pinv.conjugate().transpose(0,2,1)
    else:
        # Scale the last output cpsd by the trace ratio between spec and last response
        trace_ratio = trace(specification)/trace(last_response_cpsd)
        trace_ratio[np.isnan(trace_ratio)] = 0
        output =  last_output_cpsd*trace_ratio[:,np.newaxis,np.newaxis]
    return output

def buzz_control(specification, # Specifications
                 warning_levels, # Warning levels
                 abort_levels, # Abort Levels
                 transfer_function,  # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 extra_parameters = '', # Extra parameters for the control law
                 last_response_cpsd = None, # Last Control Response for Error Correction
                 last_output_cpsd = None, # Last Control Excitation for Drive-based control
                 ):
    """
    A control law that updates the coherence and phase of the specification
    with the coherence and phase derived from the system identification phase.
    It then simply performs a pseudoinverse on the transfer function
    matrix and pre- and post-multiplies the updated specification by that inverse
    via the formula Gvv = H^+ Gxx (H^*)^+.
    
    Parameters
    ----------
    specification : np.ndarray
        The response specification that the control law will attempt to achieve.
        Shape is (num_frequencies x num_control_channels x num_control_channels).
    warning_levels : np.ndarray
        The warning levels provided by the specification where the control will
        notify the user if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    abort_levels : np.ndarray
        The abort levels provided by the specification where the control will
        shut down if reached. Shape is (2 x num_frequencies x
        num_control_channels), where the [0] index is the upper limit and the
        [1] index is the lower limit on the first dimension.  This will be
        NaN if no limit is specified at a given frequency line or channel.
    transfer_function : np.ndarray
        The system transfer function between the excitation voltage and the
        control channel responses.  Shape is (num_frequencies x 
        num_control_channels x num_excitation_channels)
    noise_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the response coordinates.  Shape
        is (num_frequencies x num_control_channels x num_control_channels).
    noise_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the noise floor
        analysis that occurs during the system identification.  Can be used
        to identify signal to noise ratio in the reference coordinates.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    sysid_response_cpsd : np.ndarray
        The CPSD measured from the control channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        response coordinates for the transfer function calculation.  Can also
        be used to provide "preferred" relationships between the responses for
        uncorrelated inputs.  Shape is (num_frequencies x num_control_channels
        x num_control_channels).
    sysid_reference_cpsd : np.ndarray
        The CPSD measured from the excitation channels during the system
        identification.  Can be used to identify signal to noise ratio in the
        reference coordinates for the transfer function calculation.  Shape
        is (num_frequencies x num_excitation_channels x num_excitation_channels).
    multiple_coherence : np.ndarray
        Multiple coherence function which shows how the measured responses are
        related to the measured excitation signals.  Multiple coherence will be
        1 if the measured responses are completely due to the input signals and
        0 if the measured responses are not related to the input signals at all.
        Can be used to determine which frequency lines are most accurately
        computed in the transfer function.
    frames : int
        Specifies the number of measurement frames used to compute the current
        system identification estimates.
    total_frames : int
        Specifies the number of frames specified to be used in the system
        identification estimate.
    extra_parameters : str, optional
        A string containing any optional parameters the control law may need to
        use. It is up to the control law to parse this string to extract the
        required information that it needs.  The default is ''.
    last_response_cpsd : np.ndarray, optional
        The CPSD measured from the control channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        response coordinates or to provide error-based control by comparing the
        achieved responses against the desired specification.  Shape is 
        (num_frequencies x num_control_channels x num_control_channels).
        If it is the first time through the control, and there is no previously
        measured response, this will be None.
    last_output_cpsd : np.ndarray, optional
        The CPSD measured from the excitation channels during the vibration
        control.  Can be used to identify signal to noise ratio in the
        reference coordinates or to provide drive-based control.  Shape is 
        (num_frequencies x num_excitation_channels x num_excitation_channels).
        If it is the first time through the control, and there is no previously
        measured excitation, this will be None.
    
    Returns
    -------
    np.ndarray
        The output CPSD matrix with shape
        (num_frequencies x num_excitation_channels x num_excitation_channels)
    
    """
    try:
        rcond = float(extra_parameters)
    except ValueError:
        rcond = 1e-15
    # Create a new specification using the autospectra from the original and
    # phase and coherence of the buzz_cpsd
    modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
    # Invert the transfer function using the pseudoinverse
    tf_pinv = np.linalg.pinv(transfer_function,rcond)
    # Return the least squares solution for the new output CPSD
    return tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)

def buzz_control_generator():
    output_cpsd = None
    modified_spec = None
    while True:
        (specification, # Specifications
         warning_levels, # Warning levels
         abort_levels, # Abort Levels
         transfer_function,  # Transfer Functions
         noise_response_cpsd,  # Noise levels and correlation 
         noise_reference_cpsd, # from the system identification
         sysid_response_cpsd,  # Response levels and correlation
         sysid_reference_cpsd, # from the system identification
         multiple_coherence, # Coherence from the system identification
         frames, # Number of frames in the CPSD and FRF matrices
         total_frames, # Total frames that could be in the CPSD and FRF matrices
         extra_parameters, # Extra parameters for the control law
         last_response_cpsd, # Last Control Response for Error Correction
         last_output_cpsd, # Last Control Excitation for Drive-based control
            ) = yield output_cpsd
        # Only comput the modified spec if it hasn't been yet.
        if modified_spec is None:
            modified_spec = match_coherence_phase(specification,sysid_response_cpsd)
         # Invert the transfer function using the pseudoinverse
        tf_pinv = np.linalg.pinv(transfer_function)
        # Assign the output_cpsd so it is yielded next time through the loop
        output_cpsd = tf_pinv@modified_spec@tf_pinv.conjugate().transpose(0,2,1)

class buzz_control_class:
    def __init__(self,
                 specification : np.ndarray, # Specifications
                 warning_levels  : np.ndarray, # Warning levels
                 abort_levels  : np.ndarray, # Abort Levels
                 extra_parameters : str, # Extra parameters for the control law
                 transfer_function : np.ndarray = None,  # Transfer Functions
                 noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                 noise_reference_cpsd : np.ndarray = None, # from the system identification
                 sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                 sysid_reference_cpsd : np.ndarray = None, # from the system identification
                 multiple_coherence : np.ndarray = None, # Coherence from the system identification
                 frames = None, # Number of frames in the CPSD and FRF matrices
                 total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                 last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                 last_output_cpsd : np.ndarray = None, # Last Control Excitation for Drive-based control
                 ):
        # Store the specification to the class
        if sysid_response_cpsd is None: # If it's the first time through we won't have a buzz test yet
            self.specification = specification
        else: # Otherwise we can compute the modified spec right away
            self.specification = self.match_coherence_phase(specification, sysid_response_cpsd)
            
    def system_id_update(self,
                         transfer_function : np.ndarray = None,  # Transfer Functions
                         noise_response_cpsd : np.ndarray = None,  # Noise levels and correlation 
                         noise_reference_cpsd : np.ndarray = None, # from the system identification
                         sysid_response_cpsd : np.ndarray = None,  # Response levels and correlation
                         sysid_reference_cpsd : np.ndarray = None, # from the system identification
                         multiple_coherence : np.ndarray = None, # Coherence from the system identification
                         frames = None, # Number of frames in the CPSD and FRF matrices
                         total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Update the specification with the buzz_cpsd
        self.specification = self.match_coherence_phase(self.specification,sysid_response_cpsd)

    def control(self,
                transfer_function : np.ndarray = None,  # Transfer Functions
                multiple_coherence : np.ndarray = None, # Coherence from the system identification
                frames = None, # Number of frames in the CPSD and FRF matrices
                total_frames = None, # Total frames that could be in the CPSD and FRF matrices
                last_response_cpsd : np.ndarray = None, # Last Control Response for Error Correction
                last_output_cpsd : np.ndarray = None) -> np.ndarray:
        # Perform the control
        tf_pinv = np.linalg.pinv(transfer_function)
        return tf_pinv @ self.specification @ tf_pinv.conjugate().transpose(0,2,1)
        
    def cpsd_coherence(self,cpsd):
        num = np.abs(cpsd)**2
        den = (cpsd[:,np.newaxis,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2])]*
               cpsd[:,np.arange(cpsd.shape[1]),np.arange(cpsd.shape[2]),np.newaxis])
        den[den==0.0] = 1 # Set to 1
        return np.real(num/
                       den)
    
    def cpsd_phase(self,cpsd):
        return np.angle(cpsd)
    
    def cpsd_from_coh_phs(self,asd,coh,phs):
        return np.exp(phs*1j)*np.sqrt(coh*asd[:,:,np.newaxis]*asd[:,np.newaxis,:])
    
    def cpsd_autospectra(self,cpsd):
        return np.einsum('ijj->ij',cpsd)
    
    def match_coherence_phase(self,cpsd_original,cpsd_to_match):
        coh = self.cpsd_coherence(cpsd_to_match)
        phs = self.cpsd_phase(cpsd_to_match)
        asd = self.cpsd_autospectra(cpsd_original)
        return self.cpsd_from_coh_phs(asd,coh,phs)


# ---------------------------------------------------------------------------
# Interactive Sandbox Random-Vibration Control Law
#
# This is a demonstration / experimentation tool. It allows the user to:
#   1. switch between built-in control laws already defined in this file
#   2. enter arbitrary Python code that computes self.output
#
# SECURITY NOTE:
#   This is intentionally NOT sandboxed against malicious code. It is meant only
#   for trusted users in a demonstration/development setting.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper functions for the sandbox UI / execution
# ---------------------------------------------------------------------------


def _describe_object(obj):
    """
    Return a short human-readable description of an object for the sandbox UI.
    """
    if obj is None:
        return "type: NoneType\nrepr: None"

    lines = []
    lines.append(f"type: {type(obj).__name__}")

    shape = getattr(obj, "shape", None)
    if shape is not None:
        lines.append(f"shape: {shape}")

    dtype = getattr(obj, "dtype", None)
    if dtype is not None:
        lines.append(f"dtype: {dtype}")

    if callable(obj):
        try:
            sig = inspect.signature(obj)
            lines.append(f"signature: {sig}")
        except Exception:
            pass

    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        first_line = doc.strip().splitlines()[0]
        lines.append(f"doc: {first_line}")

    try:
        rep = repr(obj)
        if len(rep) > 500:
            rep = rep[:500] + "..."
        lines.append(f"repr: {rep}")
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI Class
# ---------------------------------------------------------------------------


class SandboxUI(AbstractControlLawUI):
    """
    UI for interactively selecting or authoring a random-vibration control law.

    The parameter dictionary sent to the environment contains only:
        - mode
        - custom_code

    Persistence between control calls can be implemented by the user by assigning
    values to `self` inside the custom code, e.g.:

        if not hasattr(self, "cached_pinv"):
            self.cached_pinv = np.linalg.pinv(transfer_function, rcond=1e-12)

    The control output CPSD must be assigned to:
        self.output
    """

    DEFAULT_CODE = """# Available variables include:
#   specification
#   warning_levels
#   abort_levels
#   transfer_function
#   noise_response_cpsd
#   noise_reference_cpsd
#   sysid_response_cpsd
#   sysid_reference_cpsd
#   multiple_coherence
#   frames
#   total_frames
#   last_response_cpsd
#   last_output_cpsd
#   np
#   trace
#   cpsd_coherence
#   cpsd_phase
#   cpsd_from_coh_phs
#   cpsd_autospectra
#   match_coherence_phase
#   self
#
# Store the control output CPSD in self.output.
#
# Example: plain pseudoinverse control

tf_pinv = np.linalg.pinv(transfer_function, rcond=1e-15)
self.output = tf_pinv @ specification @ tf_pinv.conjugate().transpose(0, 2, 1)
"""

    def __init__(
        self,
        process_name,
        send_parameters_queue,
        window,
        parent_ui_class,
        data_acquisition_parameters,
        environment_parameters,
    ):
        super().__init__(process_name, send_parameters_queue, window, parent_ui_class)

        self.data_acquisition_parameters = data_acquisition_parameters
        self.environment_parameters = environment_parameters

        self.available_symbols = {}

        self.runtime_symbols = {
            "transfer_function": None,
            "noise_response_cpsd": None,
            "noise_reference_cpsd": None,
            "sysid_response_cpsd": None,
            "sysid_reference_cpsd": None,
            "multiple_coherence": None,
            "frames": None,
            "total_frames": None,
            "last_response_cpsd": None,
            "last_output_cpsd": None,
        }

        self._control_symbols_initialized = False

        self._build_ui()
        self._populate_mode_selector()
        self._refresh_symbol_list()
        self._populate_symbol_list_once()
        if self.symbol_list.count() > 0:
            self.symbol_list.setCurrentRow(0)
        self._connect_callbacks()

        self.window.setWindowTitle("Interactive Control Sandbox")
        self.window.resize(900, 700)
        self.window.show()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self.window)

        # Top controls
        top_row = QtWidgets.QHBoxLayout()
        layout.addLayout(top_row)

        top_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_selector = QtWidgets.QComboBox()
        top_row.addWidget(self.mode_selector)

        self.send_button = QtWidgets.QPushButton("Send Parameters / Update Code")
        top_row.addWidget(self.send_button)

        top_row.addStretch()

        # Splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: available variables
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        self.symbol_list_label = QtWidgets.QLabel("Available Variables / Helpers")
        left_layout.addWidget(self.symbol_list_label)

        self.symbol_list = QtWidgets.QListWidget()
        left_layout.addWidget(self.symbol_list)

        self.symbol_details_label = QtWidgets.QLabel("Selected Symbol Details")
        left_layout.addWidget(self.symbol_details_label)

        self.symbol_details = QtWidgets.QPlainTextEdit()
        self.symbol_details.setReadOnly(True)
        left_layout.addWidget(self.symbol_details)

        splitter.addWidget(left_widget)

        # Right: custom code editor + status
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self.code_label = QtWidgets.QLabel("Custom Control Code")
        right_layout.addWidget(self.code_label)

        self.code_edit = QtWidgets.QPlainTextEdit()
        self.code_edit.setPlainText(self.DEFAULT_CODE)
        right_layout.addWidget(self.code_edit)

        self.code_help_label = QtWidgets.QLabel(
            "Instructions: assign the output CPSD to `self.output`.\n"
            "Persistent state may be stored on `self`."
        )
        right_layout.addWidget(self.code_help_label)

        self.status_label = QtWidgets.QLabel("Status / Last Error")
        right_layout.addWidget(self.status_label)

        self.status_text = QtWidgets.QPlainTextEdit()
        self.status_text.setReadOnly(True)
        right_layout.addWidget(self.status_text)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 600])

    def _set_custom_mode_enabled(self, enabled: bool):
        widgets = [
            self.symbol_list_label,
            self.symbol_list,
            self.symbol_details_label,
            self.symbol_details,
            self.code_label,
            self.code_edit,
            self.code_help_label,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def _populate_mode_selector(self):
        self.mode_selector.clear()
        self.mode_selector.addItem("pseudoinverse_control", "pseudoinverse_control")
        self.mode_selector.addItem("match_trace_pseudoinverse", "match_trace_pseudoinverse")
        self.mode_selector.addItem("buzz_control", "buzz_control")
        self.mode_selector.addItem("custom_code", "custom_code")
        self.mode_selector.setCurrentIndex(0)
        self._update_mode_ui()

    def _populate_symbol_list_once(self):
        self.symbol_list.clear()
        for name in self.available_symbols:
            self.symbol_list.addItem(name)

    def _refresh_current_symbol_details(self):
        current_item = self.symbol_list.currentItem()
        if current_item is None:
            self.symbol_details.setPlainText("")
            return
        name = current_item.text()
        obj = self.available_symbols.get(name, None)
        self.symbol_details.setPlainText(_describe_object(obj))

    def _refresh_symbol_list(self):
        env = self.environment_parameters

        self.available_symbols = {
            "np": np,
            "trace": trace,
            "cpsd_coherence": cpsd_coherence,
            "cpsd_phase": cpsd_phase,
            "cpsd_from_coh_phs": cpsd_from_coh_phs,
            "cpsd_autospectra": cpsd_autospectra,
            "match_coherence_phase": match_coherence_phase,
            "specification": getattr(env, "specification_cpsd_matrix", None),
            "warning_levels": getattr(env, "specification_warning_matrix", None),
            "abort_levels": getattr(env, "specification_abort_matrix", None),
            "transfer_function": self.runtime_symbols["transfer_function"],
            "noise_response_cpsd": self.runtime_symbols["noise_response_cpsd"],
            "noise_reference_cpsd": self.runtime_symbols["noise_reference_cpsd"],
            "sysid_response_cpsd": self.runtime_symbols["sysid_response_cpsd"],
            "sysid_reference_cpsd": self.runtime_symbols["sysid_reference_cpsd"],
            "multiple_coherence": self.runtime_symbols["multiple_coherence"],
            "frames": self.runtime_symbols["frames"],
            "total_frames": self.runtime_symbols["total_frames"],
            "last_response_cpsd": self.runtime_symbols["last_response_cpsd"],
            "last_output_cpsd": self.runtime_symbols["last_output_cpsd"],
            "self": "<control computation object>",
        }

        # Do not rebuild the QListWidget every time; only refresh the details
        self._refresh_current_symbol_details()

    def _connect_callbacks(self):
        self.mode_selector.currentIndexChanged.connect(self._update_mode_ui)
        self.send_button.clicked.connect(self.send_parameters)
        self.symbol_list.currentTextChanged.connect(self._update_symbol_details)

    def _update_mode_ui(self):
        mode = self.mode_selector.currentData()
        custom = mode == "custom_code"
        self._set_custom_mode_enabled(custom)
        self.send_button.setText("Apply Code" if custom else "Apply Mode")

    def _update_symbol_details(self, name):
        if name not in self.available_symbols:
            self.symbol_details.setPlainText("")
            return
        obj = self.available_symbols[name]
        self.symbol_details.setPlainText(_describe_object(obj))

    def collect_parameters(self):
        return {
            "mode": self.mode_selector.currentData(),
            "custom_code": self.code_edit.toPlainText(),
        }

    def update_ui_control(self, control_results: dict):
        if not self._control_symbols_initialized:
            if "last_response_cpsd" in control_results:
                self.runtime_symbols["last_response_cpsd"] = control_results["last_response_cpsd"]
            if "last_output_cpsd" in control_results:
                self.runtime_symbols["last_output_cpsd"] = control_results["last_output_cpsd"]
            if "frames" in control_results:
                self.runtime_symbols["frames"] = control_results["frames"]
            if "total_frames" in control_results:
                self.runtime_symbols["total_frames"] = control_results["total_frames"]

            self._refresh_symbol_list()
            self._control_symbols_initialized = True

        status_lines = []

        mode = control_results.get("mode", None)
        if mode is not None:
            status_lines.append(f"Mode: {mode}")

        rms_input_level = control_results.get("rms_input_level", None)
        if rms_input_level is not None:
            status_lines.append(
                "RMS input level: " + ", ".join(f"{v:0.4g}" for v in np.atleast_1d(rms_input_level))
            )

        error_text = control_results.get("last_error", "")
        if error_text:
            status_lines.append("")
            status_lines.append("Last Error:")
            status_lines.append(error_text)

        self.status_text.setPlainText("\n".join(status_lines))

    def update_ui_sysid(
        self,
        sysid_frf,
        sysid_response_noise,
        sysid_reference_noise,
        sysid_response_cpsd,
        sysid_reference_cpsd,
        multiple_coherence,
    ):

        self._control_symbols_initialized = False

        self.runtime_symbols["transfer_function"] = sysid_frf
        self.runtime_symbols["noise_response_cpsd"] = sysid_response_noise
        self.runtime_symbols["noise_reference_cpsd"] = sysid_reference_noise
        self.runtime_symbols["sysid_response_cpsd"] = sysid_response_cpsd
        self.runtime_symbols["sysid_reference_cpsd"] = sysid_reference_cpsd
        self.runtime_symbols["multiple_coherence"] = multiple_coherence
        self._refresh_symbol_list()

        self.status_text.setPlainText(
            "System identification data updated and available to custom code."
        )


# ---------------------------------------------------------------------------
# Computation Class
# ---------------------------------------------------------------------------


class SandboxInteractiveControl(AbstractControlLawComputation):
    """
    Interactive computation object for selecting built-in random-vibration control
    laws or executing custom code.

    Persistent user state can be stored by the executed code on `self`.
    The executed custom code must assign the output CPSD to `self.output`.
    """

    def __init__(
        self,
        environment_name: str,
        gui_update_queue: VerboseMessageQueue,
        specification: np.ndarray,
        warning_levels: np.ndarray,
        abort_levels: np.ndarray,
        extra_parameters: str,
        transfer_function: np.ndarray = None,
        noise_response_cpsd: np.ndarray = None,
        noise_reference_cpsd: np.ndarray = None,
        sysid_response_cpsd: np.ndarray = None,
        sysid_reference_cpsd: np.ndarray = None,
        multiple_coherence: np.ndarray = None,
        frames=None,
        total_frames=None,
        last_response_cpsd: np.ndarray = None,
        last_output_cpsd: np.ndarray = None,
    ):
        super().__init__(environment_name, gui_update_queue)

        # Runtime data/state
        self.specification = specification
        self.warning_levels = warning_levels
        self.abort_levels = abort_levels
        self.transfer_function = transfer_function
        self.noise_response_cpsd = noise_response_cpsd
        self.noise_reference_cpsd = noise_reference_cpsd
        self.sysid_response_cpsd = sysid_response_cpsd
        self.sysid_reference_cpsd = sysid_reference_cpsd
        self.multiple_coherence = multiple_coherence
        self.frames = frames
        self.total_frames = total_frames
        self.last_response_cpsd = last_response_cpsd
        self.last_output_cpsd = last_output_cpsd

        # Interactive parameters
        self.mode = "pseudoinverse_control"
        self.custom_code = SandboxUI.DEFAULT_CODE

        # Output / diagnostics
        self.output = None
        self.last_error = ""

    def system_id_update(
        self,
        transfer_function: np.ndarray = None,
        noise_response_cpsd: np.ndarray = None,
        noise_reference_cpsd: np.ndarray = None,
        sysid_response_cpsd: np.ndarray = None,
        sysid_reference_cpsd: np.ndarray = None,
        multiple_coherence: np.ndarray = None,
        frames=None,
        total_frames=None,
    ):
        self.transfer_function = transfer_function
        self.noise_response_cpsd = noise_response_cpsd
        self.noise_reference_cpsd = noise_reference_cpsd
        self.sysid_response_cpsd = sysid_response_cpsd
        self.sysid_reference_cpsd = sysid_reference_cpsd
        self.multiple_coherence = multiple_coherence
        self.frames = frames
        self.total_frames = total_frames

    def update_parameters(self, parameters=dict):
        self.mode = parameters.get("mode", self.mode)
        self.custom_code = parameters.get("custom_code", self.custom_code)

    def control(
        self,
        transfer_function: np.ndarray = None,
        multiple_coherence: np.ndarray = None,
        frames=None,
        total_frames=None,
        last_response_cpsd: np.ndarray = None,
        last_output_cpsd: np.ndarray = None,
    ) -> np.ndarray:
        if transfer_function is not None:
            self.transfer_function = transfer_function
        if multiple_coherence is not None:
            self.multiple_coherence = multiple_coherence
        if frames is not None:
            self.frames = frames
        if total_frames is not None:
            self.total_frames = total_frames

        self.last_response_cpsd = last_response_cpsd
        self.last_output_cpsd = last_output_cpsd

        previous_output = None if self.output is None else np.array(self.output, copy=True)

        try:
            if self.mode == "pseudoinverse_control":
                self.output = pseudoinverse_control(
                    self.specification,
                    self.warning_levels,
                    self.abort_levels,
                    self.transfer_function,
                    self.noise_response_cpsd,
                    self.noise_reference_cpsd,
                    self.sysid_response_cpsd,
                    self.sysid_reference_cpsd,
                    self.multiple_coherence,
                    self.frames,
                    self.total_frames,
                    "",
                    self.last_response_cpsd,
                    self.last_output_cpsd,
                )

            elif self.mode == "match_trace_pseudoinverse":
                self.output = match_trace_pseudoinverse(
                    self.specification,
                    self.warning_levels,
                    self.abort_levels,
                    self.transfer_function,
                    self.noise_response_cpsd,
                    self.noise_reference_cpsd,
                    self.sysid_response_cpsd,
                    self.sysid_reference_cpsd,
                    self.multiple_coherence,
                    self.frames,
                    self.total_frames,
                    "",
                    self.last_response_cpsd,
                    self.last_output_cpsd,
                )

            elif self.mode == "buzz_control":
                self.output = buzz_control(
                    self.specification,
                    self.warning_levels,
                    self.abort_levels,
                    self.transfer_function,
                    self.noise_response_cpsd,
                    self.noise_reference_cpsd,
                    self.sysid_response_cpsd,
                    self.sysid_reference_cpsd,
                    self.multiple_coherence,
                    self.frames,
                    self.total_frames,
                    "",
                    self.last_response_cpsd,
                    self.last_output_cpsd,
                )

            elif self.mode == "custom_code":
                self._run_custom_code()

            else:
                raise ValueError(f"Unknown control mode: {self.mode}")

            self._validate_output()
            self.last_error = ""

        except Exception:
            self.last_error = traceback.format_exc()
            # Keep the previous valid output if custom/built-in execution fails
            if previous_output is not None:
                self.output = previous_output

        self.send_results()
        return self.output

    def _validate_output(self):
        if self.output is None:
            raise ValueError("Control law did not produce self.output.")

        if not isinstance(self.output, np.ndarray):
            raise TypeError(
                f"self.output must be a numpy.ndarray, got {type(self.output).__name__}"
            )

        if self.transfer_function is None:
            raise ValueError(
                "Cannot validate self.output because transfer_function is not available."
            )

        if self.transfer_function.ndim != 3:
            raise ValueError(
                f"transfer_function must be a 3D array, got ndim={self.transfer_function.ndim}"
            )

        n_f, _, n_o = self.transfer_function.shape
        expected_shape = (n_f, n_o, n_o)

        if self.output.ndim != 3:
            raise ValueError(f"self.output must be a 3D CPSD matrix, got ndim={self.output.ndim}")

        if self.output.shape != expected_shape:
            raise ValueError(
                f"self.output shape {self.output.shape} does not match expected output "
                f"CPSD shape {expected_shape} derived from transfer_function.shape="
                f"{self.transfer_function.shape}"
            )

        if not np.iscomplexobj(self.output):
            self.output = self.output.astype(np.complex128)

        if not np.all(np.isfinite(self.output.real)) or not np.all(np.isfinite(self.output.imag)):
            raise ValueError("self.output contains non-finite values.")

    def _run_custom_code(self):
        """
        Execute user-provided code.

        The code must assign the output CPSD to self.output.
        Users may persist state across calls by assigning attributes to self.
        """
        exec_locals = {
            "np": np,
            "trace": trace,
            "cpsd_coherence": cpsd_coherence,
            "cpsd_phase": cpsd_phase,
            "cpsd_from_coh_phs": cpsd_from_coh_phs,
            "cpsd_autospectra": cpsd_autospectra,
            "match_coherence_phase": match_coherence_phase,
            "specification": self.specification,
            "warning_levels": self.warning_levels,
            "abort_levels": self.abort_levels,
            "transfer_function": self.transfer_function,
            "noise_response_cpsd": self.noise_response_cpsd,
            "noise_reference_cpsd": self.noise_reference_cpsd,
            "sysid_response_cpsd": self.sysid_response_cpsd,
            "sysid_reference_cpsd": self.sysid_reference_cpsd,
            "multiple_coherence": self.multiple_coherence,
            "frames": self.frames,
            "total_frames": self.total_frames,
            "last_response_cpsd": self.last_response_cpsd,
            "last_output_cpsd": self.last_output_cpsd,
            "self": self,
        }

        exec(self.custom_code, {}, exec_locals)

        if self.output is None:
            raise ValueError("Custom code did not assign a value to self.output.")

    def collect_results(self):
        rms_input_level = None
        if self.output is not None:
            df = 1.0
            try:
                df = getattr(self, "df", df)
            except Exception:
                pass
            rms_input_level = rms_csd(self.output, df)

        return {
            "mode": self.mode,
            "rms_input_level": rms_input_level,
            "last_error": self.last_error,
            "frames": self.frames,
            "total_frames": self.total_frames,
            "last_response_cpsd": self.last_response_cpsd,
            "last_output_cpsd": self.last_output_cpsd,
        }

    @staticmethod
    def get_ui_class():
        return SandboxUI