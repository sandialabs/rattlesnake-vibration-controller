from rattlesnake.user_interface.abstract_sys_id_user_interface import AbstractSysIdUI
from rattlesnake.utilities import VerboseMessageQueue
from rattlesnake.environment.environment_utilities import ControlTypes
from rattlesnake.user_interface.ui_utilities import (
    environment_definition_ui_paths,
    environment_prediction_ui_paths,
    environment_run_ui_paths,
)
from qtpy import QtWidgets, uic
from multiprocessing.queues import Queue

control_type = ControlTypes.Skeleton  # noqa pylint: disable=no-member


class SkeletonUI(AbstractSysIdUI):
    def __init__(
        self,
        environment_name: str,
        definition_tabwidget: QtWidgets.QTabWidget,
        system_id_tabwidget: QtWidgets.QTabWidget,
        test_predictions_tabwidget: QtWidgets.QTabWidget,
        run_tabwidget: QtWidgets.QTabWidget,
        environment_command_queue: VerboseMessageQueue,
        controller_communication_queue: VerboseMessageQueue,
        log_file_queue: Queue,
    ):
        super().__init__(
            environment_name,
            environment_command_queue,
            controller_communication_queue,
            log_file_queue,
            system_id_tabwidget,
        )
        # Add the page to the control definition tabwidget
        self.definition_widget = QtWidgets.QWidget()
        uic.loadUi(environment_definition_ui_paths[control_type], self.definition_widget)
        definition_tabwidget.addTab(self.definition_widget, self.environment_name)
        # Add the page to the control prediction tabwidget
        self.prediction_widget = QtWidgets.QWidget()
        uic.loadUi(environment_prediction_ui_paths[control_type], self.prediction_widget)
        test_predictions_tabwidget.addTab(self.prediction_widget, self.environment_name)
        # Add the page to the run tabwidget
        self.run_widget = QtWidgets.QWidget()
        uic.loadUi(environment_run_ui_paths[control_type], self.run_widget)
        run_tabwidget.addTab(self.run_widget, self.environment_name)

    def collect_environment_definition_parameters(self):
        pass

    def create_environment_template(self, environment_name, workbook):
        pass

    def initialize_data_acquisition(self, data_acquisition_parameters):
        pass

    def initialize_environment(self):
        pass

    @property
    def initialized_control_names(self):
        pass

    @property
    def initialized_output_names(self):
        pass

    def retrieve_metadata(self, netcdf_handle):
        pass

    def set_parameters_from_template(self, worksheet):
        pass

    def start_control(self):
        pass

    def stop_control(self):
        pass

    def update_gui(self, queue_data):
        if super().update_gui(queue_data):
            return
