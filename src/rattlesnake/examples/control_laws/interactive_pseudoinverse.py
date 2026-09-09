# -*- coding: utf-8 -*-
"""
This is an example problem showing how the Interactive Control works.  It
performs a simple pseudoinverse control and allows you to interactively adjust
the RCOND parameter.

Created on Mon Dec 16 07:18:19 2024

@author: dprohe
"""

import os

from rattlesnake.environment.abstract_interactive_control_law import (
    AbstractControlLawUI,
    AbstractControlLawComputation,
)
from rattlesnake.utilities import VerboseMessageQueue, rms_csd
from rattlesnake.environment.random_vibration_sys_id_environment import RandomVibrationMetadata

import numpy as np

from qtpy import uic


class PseudoinverseUI(AbstractControlLawUI):
    def __init__(
        self,
        log_name,
        environment_command_queue,
        interactive_control_law_window,
        parent_ui_class,
        data_acquisition_parameters,
        environment_parameters: RandomVibrationMetadata,
    ):
        super().__init__(
            log_name, environment_command_queue, interactive_control_law_window, parent_ui_class
        )
        uic.loadUi(
            os.path.join(os.path.split(__file__)[0], "interactive_pseudoinverse.ui"), self.window
        )
        self.window.setWindowTitle("The ULTIMATE Control Law")
        self.rcond_map = np.logspace(-9, 0, 1001)
        try:
            rcond = float(environment_parameters.control_python_function_parameters)
            index = np.argmin(abs(rcond - self.rcond_map))
        except ValueError:
            index = 0
        self.window.rcond_slider.setValue(index)
        self.slider_callback()
        self.window.rcond_slider.valueChanged.connect(self.slider_callback)
        self.window.update_control_button.clicked.connect(self.send_parameters)
        self.data_acquisition_parameters = data_acquisition_parameters
        self.environment_parameters = environment_parameters

    def slider_callback(self, value=None):
        self.window.rcond_label.setText(f"rcond = {self.rcond:.3e}")
        if self.window.auto_update_control_checkbox.isChecked():
            self.send_parameters()

    @property
    def rcond(self):
        index = self.window.rcond_slider.value()
        return self.rcond_map[index]

    def collect_parameters(self):
        return {"rcond": self.rcond, "df": self.environment_parameters.frequency_spacing}

    def update_ui_control(self, control_results: dict):
        rms_input_level = control_results["rms_input_level"]
        self.window.levels_label.setText(
            "Levels = " + ", ".join(f"{v:.3e}" for v in rms_input_level)
        )

    def update_ui_sysid(
        self,
        sysid_frf,
        sysid_response_noise,
        sysid_reference_noise,
        sysid_response_cpsd,
        sysid_reference_cpsd,
        sysid_coherence,
    ):
        self.window.levels_label.setText("Got System ID Data!  I'm not going to use it though.")


class PseudoinverseComputation(AbstractControlLawComputation):
    def __init__(
        self,
        environment_name: str,
        gui_update_queue: VerboseMessageQueue,
        specification: np.ndarray,  # Specifications
        warning_levels: np.ndarray,  # Warning levels
        abort_levels: np.ndarray,  # Abort Levels
        extra_parameters: str,  # Extra parameters for the control law
        transfer_function: np.ndarray = None,  # Transfer Functions
        noise_response_cpsd: np.ndarray = None,  # Noise levels and correlation
        noise_reference_cpsd: np.ndarray = None,  # from the system identification
        sysid_response_cpsd: np.ndarray = None,  # Response levels and correlation
        sysid_reference_cpsd: np.ndarray = None,  # from the system identification
        multiple_coherence: np.ndarray = None,  # Coherence from the system identification
        frames=None,  # Number of frames in the CPSD and FRF matrices
        total_frames=None,  # Total frames that could be in the CPSD and FRF matrices
        last_response_cpsd: np.ndarray = None,  # Last Control Response for Error Correction
        last_output_cpsd: np.ndarray = None,  # Last Control Excitation for Drive-based control
    ):
        super().__init__(environment_name, gui_update_queue)
        self.transfer_function = transfer_function
        self.rcond = None
        self.df = None
        self.specification = specification
        self.output = None

    def system_id_update(
        self,
        transfer_function: np.ndarray = None,  # Transfer Functions
        noise_response_cpsd: np.ndarray = None,  # Noise levels and correlation
        noise_reference_cpsd: np.ndarray = None,  # from the system identification
        sysid_response_cpsd: np.ndarray = None,  # Response levels and correlation
        sysid_reference_cpsd: np.ndarray = None,  # from the system identification
        multiple_coherence: np.ndarray = None,  # Coherence from the system identification
        frames=None,  # Number of frames in the CPSD and FRF matrices
        total_frames=None,  # Total frames that could be in the CPSD and FRF matrices
    ):
        # Update the specification with the buzz_cpsd
        self.transfer_function = transfer_function

    def control(
        self,
        transfer_function: np.ndarray = None,  # Transfer Functions
        multiple_coherence: np.ndarray = None,  # Coherence from the system identification
        frames=None,  # Number of frames in the CPSD and FRF matrices
        total_frames=None,  # Total frames that could be in the CPSD and FRF matrices
        last_response_cpsd: np.ndarray = None,  # Last Control Response for Error Correction
        last_output_cpsd: np.ndarray = None,
    ) -> np.ndarray:
        # Perform the control
        if transfer_function is not None:
            self.transfer_function = transfer_function
        tf_pinv = np.linalg.pinv(self.transfer_function, rcond=self.rcond)
        self.output = tf_pinv @ self.specification @ tf_pinv.conjugate().transpose(0, 2, 1)
        self.send_results()
        return self.output

    def update_parameters(self, parameters=dict):
        self.rcond = parameters["rcond"]
        self.df = parameters["df"]

    @staticmethod
    def get_ui_class():
        return PseudoinverseUI

    def collect_results(self):
        return {"rms_input_level": rms_csd(self.output, self.df)}
