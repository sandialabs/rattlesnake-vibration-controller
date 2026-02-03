import numpy as np
from qtpy import uic, QtWidgets

from .environments import sds_prediction_table_ui_path
from .sds_sys_id_utilities import DecayedSineTable
from .sds_sys_id_metadata import SRSParameters, SpecParameters
from .sds_sys_id_utilities import SDSCommands
from .utilities import VerboseMessageQueue


class SDSPredictionTable:

    def __init__(
        self,
        parent_widget: QtWidgets.QWidget,
        environment_command_queue: VerboseMessageQueue,
        log_name: str,
        sds_table: None | DecayedSineTable = None,
        drive_names: None | np.ndarray = None,
        response_names: None | np.ndarray = None,
        srs_parameters: None | SRSParameters = None,
        spec_parameters: None | SpecParameters = None,
    ):
        uic.loadUi(sds_prediction_table_ui_path, parent_widget)
        # Processing data
        self.parent_widget = parent_widget
        self.environment_command_queue = environment_command_queue
        self.log_name = log_name
        self.sds_table = sds_table
        self.drive_names = drive_names
        self.response_names = response_names
        self.srs_parameters = srs_parameters
        self.spec_parameters = spec_parameters
        # Keep track of tables and tabs
        self.sds_table_widgets = []
        # Persistent calculated data
        self.response_time_history = None
        self.response_srs = None

        # Connect callbacks
        self.parent_widget.excitation_selector.currentIndexChanged.connect(
            self.update_drive_plot_ui
        )
        self.parent_widget.response_selector.currentIndexChanged.connect(
            self.update_response_plot_ui
        )
        self.parent_widget.response_error_list.itemClicked.connect(self.update_response_selector)
        self.parent_widget.excitation_voltage_list.itemClicked.connect(
            self.update_excitation_selector
        )

        # Update information
        self.update_ui()

    def update_names(
        self, drive_names: None | np.ndarray = None, response_names: None | np.ndarray = None
    ):
        self.drive_names = drive_names
        self.response_names = response_names
        self.update_names_ui()

    def update_srs_parameters(self, srs_parameters: None | SRSParameters = None):
        self.srs_parameters = srs_parameters

    def update_spec_parameters(
        self,
        spec_parameters: None | SpecParameters = None,
    ):
        self.spec_parameters = spec_parameters

    def update_prediction_information(
        self, response_time_history: np.ndarray, response_srs: np.ndarray
    ):
        self.response_time_history = response_time_history
        self.response_srs = response_srs

    def perform_prediction(self):
        self.environment_command_queue.put(
            self.log_name, (SDSCommands.SDS_TABLE_PREDICTION, self.sds_table)
        )

    def synchronize_sds_table(self):
        pass

    def lock_table(
        self, frequencies=None, amplitudes=None, delays=None, decays=None, all_data=None
    ):
        pass

    def update_ui(self):
        self.update_names_ui()
        self.update_table_ui()
        self.update_response_plot_ui()
        self.update_drive_plot_ui()

    def update_names_ui(self):
        # Update the drives if there are names
        if self.drive_names is not None:
            self.sds_table_widgets = []
            self.parent_widget.sds_tab_widget.clear()
            self.parent_widget.excitation_selector.clear()
            for name in self.drive_names:
                self.add_tab(name)
                self.parent_widget.excitation_selector.addItem(name)
        if self.response_names is not None:
            self.parent_widget.response_selector.clear()
            for name in self.response_names:
                self.parent_widget.response_selector.addItem(name)

    def add_tab(self, name: str):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setRowCount(0)
        table.setHorizontalHeaderLabels(["Frequency", "Amplitude", "Delay", "Decay"])
        layout.addWidget(table)
        self.sds_table_widgets.append(table)
        self.parent_widget.sds_tab_widget.addTab(tab, name)

    def update_table_ui(self):
        pass

    def update_response_plot_ui(self):
        pass

    def update_drive_plot_ui(self):
        pass

    def update_response_selector(self, item):
        index = self.parent_widget.response_error_list.row(item)
        self.parent_widget.response_selector.setCurrentIndex(index)

    def update_excitation_selector(self, item):
        index = self.parent_widget.excitation_voltage_list.row(item)
        self.parent_widget.excitation_selector.setCurrentIndex(index)
