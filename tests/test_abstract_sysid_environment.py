import multiprocessing as mp

# from unittest import mock  # unused import

import numpy as np
import pytest
from functions.abstract_sysid_functions import (
    DummyAbstractSysIdEnvironment,
    DummyAbstractSysIdMetadata,
    # DummyAbstractSysIdUI,  # usused import
)
from functions.common_functions import DummyMainWindow

# from PyQt5 import QtWidgets  # unused import

from rattlesnake.environment.abstract_sysid_environment import SystemIdCommands
from rattlesnake.utilities import VerboseMessageQueue


@pytest.fixture()
def log_file_queue():
    return mp.Queue


@pytest.fixture
def app(qtbot):
    return qtbot


@pytest.fixture
def main_window(app):
    return DummyMainWindow()


@pytest.fixture
def abstract_sysid_metadata():
    return DummyAbstractSysIdMetadata()


@pytest.mark.parametrize("sysid_idx", [0, 1, 2, 3, 4])
def test_system_id_commands(sysid_idx):
    sysid_command = SystemIdCommands(sysid_idx)

    assert isinstance(sysid_command, SystemIdCommands)


def test_abstract_sysid_metadata_init():
    abstract_sysid_metadata = DummyAbstractSysIdMetadata()

    assert isinstance(abstract_sysid_metadata, DummyAbstractSysIdMetadata)


# def test_abstract_sysid_ui_init(log_file_queue, main_window):
#     abstract_sysid_ui = DummyAbstractSysIdUI("Environment Name",
#                                              VerboseMessageQueue(log_file_queue, 'Environment Command Queue'),
#                                              VerboseMessageQueue(log_file_queue, 'Controller Communication Queue'),
#                                              log_file_queue, main_window.system_id_tabwidget)

#     assert isinstance(abstract_sysid_ui, DummyAbstractSysIdUI)


def test_abstract_sysid_environment(log_file_queue):
    abstract_sysid_environment = DummyAbstractSysIdEnvironment(
        "Environment Name",
        VerboseMessageQueue(log_file_queue, "Environment Command Queue"),
        mp.Queue(),
        VerboseMessageQueue(log_file_queue, "Controller Communication Queue"),
        log_file_queue,
        VerboseMessageQueue(log_file_queue, "Collector Command Queue"),
        VerboseMessageQueue(log_file_queue, "Signal Generator Command Queue"),
        VerboseMessageQueue(log_file_queue, "Spectral Processing Command Queue"),
        VerboseMessageQueue(log_file_queue, "Data Analysis Command Queue"),
        mp.Queue(),
        mp.Queue(),
        mp.Value("i", 0),
        mp.Value("i", 0),
    )

    assert isinstance(abstract_sysid_environment, DummyAbstractSysIdEnvironment)


if __name__ == "__main__":
    # test_abstract_sysid_metadata_init(transform_matrix=None)  # function has no parameter
    test_abstract_sysid_metadata_init()
