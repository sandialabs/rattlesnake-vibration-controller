import sys
import os
from qtpy import QtWidgets, QtCore, uic
from rattlesnake.user_interface.sds_sys_id_prediction_table import SDSPredictionTable
from rattlesnake.engine import RattlesnakeController
from rattlesnake.utilities import DIRECTORY


class SDSRunTableDialog(QtWidgets.QDialog):

    def __init__(
        self,
        rattlesnake: RattlesnakeController,
        environment_name,
        prediction_mode,
        parent=None,
        other_voltage_lists=None,
    ):
        super().__init__(parent)
        uic.loadUi(
            os.path.join(DIRECTORY, "user_interface", "ui_files", "srs_sds_run_table.ui"),
            self,
        )
        self.setWindowTitle("SDS Run Table")
        self.run_table = SDSPredictionTable(
            self.prediction_table_placeholder,
            rattlesnake,
            environment_name,
            prediction_mode=prediction_mode,
            other_voltage_lists=other_voltage_lists,
        )

        self.load_table_button.clicked.connect(self.load_sds_table)
        self.allow_manual_updates_checkbox.stateChanged.connect(self.set_table_lock)
        self.allow_automatic_updates_checkbox.stateChanged.connect(self.set_automatic_updates)

    def set_run_table(self, run_table):
        self.run_table = run_table

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def load_sds_table(self):
        pass

    def set_table_lock(self):
        if self.allow_manual_updates_checkbox.isChecked():
            self.run_table.lock_table(all_data=False, frequencies=True)
        else:
            self.run_table.lock_table(all_data=True)

    def set_automatic_updates(self):
        pass
