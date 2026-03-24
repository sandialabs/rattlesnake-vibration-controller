from rattlesnake.utilities import Channel, DataAcquisitionParameters
from qtpy import QtWidgets
import pyqtgraph as pg
import numpy as np


def fake_time():
    return "Datetime"


def create_hardware_dict_acquisition():
    hardware_dict = {
        0: "rattlesnake.hardware.nidaqmx_hardware_multitask.NIDAQmxAcquisition",
        1: "rattlesnake.hardware.lanxi_hardware_multiprocessing.LanXIAcquisition",
        2: "rattlesnake.hardware.data_physics_hardware.DataPhysicsAcquisition",
        4: "rattlesnake.hardware.exodus_modal_solution_hardware.ExodusAcquisition",
        5: "rattlesnake.hardware.state_space_virtual_hardware.StateSpaceAcquisition",
        6: "rattlesnake.hardware.sdynpy_system_virtual_hardware.SDynPySystemAcquisition",
    }
    return hardware_dict


def create_hardware_dict_output():
    hardware_dict = {
        0: "rattlesnake.hardware.nidaqmx_hardware_multitask.NIDAQmxOutput",
        1: "rattlesnake.hardware.lanxi_hardware_multiprocessing.LanXIOutput",
        2: "rattlesnake.hardware.data_physics_hardware.DataPhysicsOutput",
        4: "rattlesnake.hardware.exodus_modal_solution_hardware.ExodusOutput",
        5: "rattlesnake.hardware.state_space_virtual_hardware.StateSpaceOutput",
        6: "rattlesnake.hardware.sdynpy_system_virtual_hardware.SDynPySystemOutput",
    }
    return hardware_dict


def create_data_acquisition_parameters():
    channel_list = [
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Accel",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        ),
        Channel.from_channel_table_row(
            [
                "221",
                "Y+",
                "",
                "19644",
                "X+",
                "",
                "",
                "",
                "",
                "",
                "Virtual",
                "",
                "Force",
                "",
                "",
                "",
                "",
                "",
                "Phys_dev",
                "",
                "5",
                "10",
            ]
        ),
    ]
    sample_rate = 2000
    time_per_read = 0.25
    time_per_write = 0.25
    output_oversample = 2
    hardware_selector_idx = 6
    hardware_file = "ExampleFile.nc4"
    environments = ["Modal"]
    environment_booleans = np.array([[True]])
    acquisition_processes = 1
    task_trigger = 0
    task_trigger_output_channel = ""
    data_acquisition_parameters = DataAcquisitionParameters(
        channel_list,
        sample_rate,
        round(sample_rate * time_per_read),
        round(sample_rate * time_per_write * output_oversample),
        hardware_selector_idx,
        hardware_file,
        environments,
        environment_booleans,
        output_oversample,
        maximum_acquisition_processes=acquisition_processes,
        task_trigger=task_trigger,
        task_trigger_output_channel=task_trigger_output_channel,
    )

    return data_acquisition_parameters


class DummyMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.definition_tabwidget = QtWidgets.QTabWidget()
        self.system_id_tabwidget = QtWidgets.QTabWidget()
        self.test_predictions_tabwidget = QtWidgets.QTabWidget()
        self.run_tabwidget = QtWidgets.QTabWidget()
        self.plot_widget = pg.PlotWidget()
