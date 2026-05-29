"""
Tests for Abstract System Identification Environment

This module contains tests for the AbstractSysIdEnvironment and AbstractSysIdMetadata
classes, verifying their initialization and basic functionality.
"""

import multiprocessing as mp

import pytest

from functions.abstract_sysid_functions import (
    DummyAbstractSysIdEnvironment,
    DummyAbstractSysIdMetadata,
)
from functions.common_functions import DummyMainWindow
from rattlesnake.environment.abstract_sysid_environment import SystemIdCommands
from rattlesnake.utilities import VerboseMessageQueue

# import numpy as np  # unused import


@pytest.fixture()
def log_file_queue():
    """
    Fixture for a log file queue.
    """
    return mp.Queue


@pytest.fixture
def app(qtbot):
    """
    Fixture for the Qt application bot.
    """
    return qtbot


@pytest.fixture
def main_window(app):
    """
    Fixture for a DummyMainWindow instance.
    """
    return DummyMainWindow()


@pytest.fixture
def abstract_sysid_metadata():
    """
    Fixture for a DummyAbstractSysIdMetadata instance.
    """
    return DummyAbstractSysIdMetadata()


@pytest.mark.parametrize("sysid_idx", [0, 1, 2, 3, 4])
def test_system_id_commands(sysid_idx):
    """
    Test the initialization of SystemIdCommands with various indices.
    """
    sysid_command = SystemIdCommands(sysid_idx)

    assert isinstance(sysid_command, SystemIdCommands)


def test_abstract_sysid_metadata_init():
    """
    Test the initialization of AbstractSysIdMetadata via DummyAbstractSysIdMetadata.
    """
    abstract_sysid_metadata = DummyAbstractSysIdMetadata()

    assert isinstance(abstract_sysid_metadata, DummyAbstractSysIdMetadata)


def test_abstract_sysid_environment(log_file_queue):
    """
    Test the initialization of AbstractSysIdEnvironment via DummyAbstractSysIdEnvironment.
    """
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
    test_abstract_sysid_metadata_init()
