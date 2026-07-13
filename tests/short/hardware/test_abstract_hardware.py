import netCDF4 as nc4
import numpy as np
import openpyxl
import pytest

from rattlesnake.hardware.abstract_hardware import (
    HardwareAcquisition,
    HardwareMetadata,
    HardwareOutput,
)
from rattlesnake.hardware.hardware_utilities import Channel, HardwareType
from rattlesnake.testing.mock_utilities import mock_channel_list
from rattlesnake.hardware.skeleton_hardware import (
    SkeletonHardwareMetadata,
    SkeletonHardwareAcquisition,
    SkeletonHardwareOutput,
)
from rattlesnake.user_interface.ui_utilities import HardwareAssistModules
from rattlesnake.utilities import RattlesnakeError


# region Fixtures
@pytest.fixture
def channel_list():
    return mock_channel_list()


@pytest.fixture
def hardware_metadata(channel_list):
    return SkeletonHardwareMetadata(
        channel_list=channel_list,
        sample_rate=1024,
        time_per_read=0.25,
        time_per_write=0.125,
    )


# endregion


# region Hardware Metadata
def test_hardware_metadata_init(hardware_metadata):
    """
    Verifies that mock hardware metadata initializes required attributes
    and is an instance of ``HardwareMetadata``.
    """
    assert isinstance(hardware_metadata, HardwareMetadata)

    assert hasattr(hardware_metadata, "hardware_type")
    assert hasattr(hardware_metadata, "channel_list")
    assert hasattr(hardware_metadata, "sample_rate")
    assert hasattr(hardware_metadata, "time_per_read")
    assert hasattr(hardware_metadata, "time_per_write")
    assert hasattr(hardware_metadata, "output_oversample")

    assert hardware_metadata.hardware_type == HardwareType.SKELETON
    assert hardware_metadata.sample_rate == 1024
    assert hardware_metadata.time_per_read == 0.25
    assert hardware_metadata.time_per_write == 0.125
    assert hardware_metadata.output_oversample == 1


def test_hardware_metadata_channel_list_property(channel_list):
    """
    Verifies that the channel list property returns and stores the configured
    hardware channel list.
    """
    metadata = SkeletonHardwareMetadata(
        channel_list=[], sample_rate=1024, time_per_read=0.25, time_per_write=0.125
    )

    assert metadata.channel_list == []

    metadata.channel_list = channel_list

    assert metadata.channel_list is channel_list


def test_hardware_metadata_samples_per_read(hardware_metadata):
    """
    Verifies that ``samples_per_read`` returns the expected number of samples
    per acquisition frame.
    """
    hardware_metadata.sample_rate = 1000
    hardware_metadata.time_per_read = 0.25

    assert hardware_metadata.samples_per_read == 250


def test_hardware_metadata_samples_per_write(hardware_metadata):
    """
    Verifies that ``samples_per_write`` includes the output oversampling
    factor.
    """
    hardware_metadata.sample_rate = 1000
    hardware_metadata.time_per_write = 0.3
    hardware_metadata.output_oversample = 10

    assert hardware_metadata.samples_per_write == 3000


def test_hardware_metadata_nyquist_frequency(hardware_metadata):
    """
    Verifies that the Nyquist frequency is half the acquisition sample rate.
    """
    hardware_metadata.sample_rate = 1000

    assert hardware_metadata.nyquist_frequency == 500


def test_hardware_metadata_output_sample_rate(hardware_metadata):
    """
    Verifies that output sample rate is the acquisition sample rate multiplied
    by the output oversampling factor.
    """
    hardware_metadata.sample_rate = 1000
    hardware_metadata.output_oversample = 10

    assert hardware_metadata.output_sample_rate == 10000


def test_hardware_metadata_validate_truth(hardware_metadata):
    """
    Verifies that a valid channel list passes base metadata validation.
    """
    hardware_metadata.validate()


def test_hardware_metadata_validate_duplicate_channels(
    hardware_metadata,
    channel_list,
):
    """
    Verifies that duplicate channels raise ``RattlesnakeError``.
    """
    hardware_metadata.channel_list = channel_list + [channel_list[0]]

    with pytest.raises(RattlesnakeError):
        hardware_metadata.validate()


def test_hardware_metadata_valid_channel_dict(hardware_metadata, channel_list):
    """
    Verifies that the base valid-channel dictionary contains all channel
    attributes and maps them to lists.
    """
    valid_channel_dict = hardware_metadata.valid_channel_dict(channel_list[0])

    assert set(valid_channel_dict) == set(Channel().channel_attr_list)
    assert all(isinstance(value, list) for value in valid_channel_dict.values())


def test_hardware_metadata_assist_mode_modules(hardware_metadata):
    """
    Verifies that the base assist-mode module dictionary contains all channel
    attributes and defaults to ``HardwareAssistModules.NONE``.
    """
    assist_mode_modules = hardware_metadata.assist_mode_modules

    assert set(assist_mode_modules) == set(Channel().channel_attr_list)
    assert all(
        value == HardwareAssistModules.NONE for value in assist_mode_modules.values()
    )


def test_hardware_metadata_save_channel_table_to_workbook(channel_list):
    """
    Verifies that a hardware channel table is written to an Excel workbook.
    """
    workbook = openpyxl.Workbook()

    SkeletonHardwareMetadata.save_channel_table_to_workbook(channel_list, workbook)

    worksheet = workbook.active

    assert worksheet.title == "Channel Table"
    assert worksheet.cell(row=1, column=2).value == "Test Article Definition"
    assert worksheet.cell(row=1, column=5).value == "Instrument Definition"
    assert worksheet.cell(row=1, column=12).value == "Channel Definition"
    assert worksheet.cell(row=1, column=20).value == "Output Feedback"
    assert worksheet.cell(row=1, column=22).value == "Limits"

    assert worksheet.cell(row=2, column=1).value == "Channel Index"
    assert worksheet.cell(row=2, column=2).value == "Node Number"
    assert worksheet.cell(row=3, column=1).value == 0
    assert worksheet.cell(row=4, column=1).value == 1


def test_hardware_metadata_load_channel_table_from_workbook(channel_list):
    """
    Verifies that a hardware channel table can be reconstructed from an Excel
    workbook.
    """
    workbook = openpyxl.Workbook()
    SkeletonHardwareMetadata.save_channel_table_to_workbook(channel_list, workbook)

    loaded_channel_list = SkeletonHardwareMetadata.load_channel_table_from_workbook(
        workbook
    )

    assert len(loaded_channel_list) == len(channel_list)

    for loaded_channel, expected_channel in zip(loaded_channel_list, channel_list):
        for attr in Channel().channel_attr_list:
            expected_value = getattr(expected_channel, attr)
            loaded_value = getattr(loaded_channel, attr)

            if expected_value is None:
                assert loaded_value in (None, "")
            else:
                assert str(loaded_value) == str(expected_value)


def test_hardware_metadata_load_channel_table_from_workbook_multiple_channel_sheets():
    """
    Verifies that multiple candidate channel table worksheets raise
    ``RattlesnakeError``.
    """
    workbook = openpyxl.Workbook()
    workbook.active.title = "Channel Table"
    workbook.create_sheet("Backup Channel Table")

    with pytest.raises(RattlesnakeError):
        SkeletonHardwareMetadata.load_channel_table_from_workbook(workbook)


def test_hardware_metadata_save_blank_hardware_to_workbook():
    """
    Verifies that a blank hardware worksheet is created with expected labels.
    """
    workbook = openpyxl.Workbook()

    SkeletonHardwareMetadata.save_blank_hardware_to_workbook(workbook)

    assert "Hardware" in workbook.sheetnames

    worksheet = workbook["Hardware"]

    assert worksheet.cell(1, 1).value == "Hardware Type"
    assert worksheet.cell(2, 1).value == "Hardware File"
    assert worksheet.cell(3, 1).value == "Sample Rate"
    assert worksheet.cell(4, 1).value == "Time Per Read"
    assert worksheet.cell(5, 1).value == "Time Per Write"
    assert worksheet.cell(6, 1).value == "Maximum Acquisition Processes"
    assert worksheet.cell(7, 1).value == "Integration Oversampling"
    assert worksheet.cell(8, 1).value == "Task Trigger"
    assert worksheet.cell(9, 1).value == "Task Trigger Output Channel"
    assert worksheet.cell(10, 1).value == "Damping Ratio"


def test_hardware_metadata_save_and_load_metadata_from_workbook(hardware_metadata):
    """
    Verifies that hardware metadata saved to an Excel workbook can be loaded
    into a metadata object.
    """
    workbook = openpyxl.Workbook()

    SkeletonHardwareMetadata.save_blank_hardware_to_workbook(workbook)
    hardware_metadata.save_metadata_to_workbook(workbook)

    loaded_metadata = SkeletonHardwareMetadata.load_metadata_from_workbook(workbook)

    assert isinstance(loaded_metadata, SkeletonHardwareMetadata)
    assert loaded_metadata.hardware_type == hardware_metadata.hardware_type
    assert loaded_metadata.sample_rate == hardware_metadata.sample_rate
    assert loaded_metadata.time_per_read == hardware_metadata.time_per_read
    assert loaded_metadata.time_per_write == hardware_metadata.time_per_write
    assert loaded_metadata.output_oversample == hardware_metadata.output_oversample
    assert len(loaded_metadata.channel_list) == len(hardware_metadata.channel_list)


def test_hardware_metadata_save_metadata_to_workbook(hardware_metadata):
    """
    Verifies that saving metadata to a workbook writes hardware fields and
    channel table information.
    """
    workbook = openpyxl.Workbook()

    SkeletonHardwareMetadata.save_blank_hardware_to_workbook(workbook)
    hardware_metadata.save_metadata_to_workbook(workbook)

    assert "Channel Table" in workbook.sheetnames
    assert "Hardware" in workbook.sheetnames

    hardware_worksheet = workbook["Hardware"]

    assert hardware_worksheet.cell(1, 2).value == str(
        hardware_metadata.hardware_type.value
    )
    assert hardware_worksheet.cell(3, 2).value == str(hardware_metadata.sample_rate)
    assert hardware_worksheet.cell(4, 2).value == str(hardware_metadata.time_per_read)
    assert hardware_worksheet.cell(5, 2).value == str(hardware_metadata.time_per_write)


def test_hardware_metadata_save_and_load_metadata_from_netcdf(
    hardware_metadata,
    tmp_path,
):
    """
    Verifies that hardware metadata saved to netCDF can be loaded into a
    metadata object.
    """
    path = tmp_path / "hardware_metadata.nc4"

    with nc4.Dataset(path, "w", format="NETCDF4") as dataset:
        hardware_metadata.save_metadata_to_netcdf(dataset)

    with nc4.Dataset(path, "r") as dataset:
        loaded_metadata = SkeletonHardwareMetadata.load_metadata_from_netcdf(dataset)

    assert isinstance(loaded_metadata, SkeletonHardwareMetadata)
    assert loaded_metadata.hardware_type == hardware_metadata.hardware_type
    assert loaded_metadata.sample_rate == hardware_metadata.sample_rate
    assert loaded_metadata.time_per_read == pytest.approx(
        hardware_metadata.samples_per_read / hardware_metadata.sample_rate
    )
    assert loaded_metadata.time_per_write == pytest.approx(
        hardware_metadata.samples_per_write / hardware_metadata.output_sample_rate
    )
    assert loaded_metadata.output_oversample == hardware_metadata.output_oversample
    assert len(loaded_metadata.channel_list) == len(hardware_metadata.channel_list)


def test_hardware_metadata_save_metadata_to_netcdf_contents(
    hardware_metadata,
    tmp_path,
):
    """
    Verifies that saving hardware metadata to netCDF creates expected
    dimensions, attributes, and variables.
    """
    path = tmp_path / "hardware_metadata.nc4"

    with nc4.Dataset(path, "w", format="NETCDF4") as dataset:
        hardware_metadata.save_metadata_to_netcdf(dataset)

    with nc4.Dataset(path, "r") as dataset:
        assert "response_channels" in dataset.dimensions
        assert "output_channels" in dataset.dimensions
        assert "time_samples" in dataset.dimensions
        assert "time_data" in dataset.variables
        assert "channels" in dataset.groups

        assert dataset.file_version == "3.0.0"
        assert dataset.sample_rate == hardware_metadata.sample_rate
        assert dataset.hardware == hardware_metadata.hardware_type.value
        assert dataset.output_oversample == hardware_metadata.output_oversample

        assert dataset.dimensions["response_channels"].size == len(
            hardware_metadata.channel_list
        )


def test_hardware_metadata_load_channel_table_from_netcdf(
    hardware_metadata,
    tmp_path,
):
    """
    Verifies that a channel table can be reconstructed from a netCDF dataset.
    """
    path = tmp_path / "hardware_metadata.nc4"

    with nc4.Dataset(path, "w", format="NETCDF4") as dataset:
        hardware_metadata.save_metadata_to_netcdf(dataset)

    with nc4.Dataset(path, "r") as dataset:
        loaded_channel_list = SkeletonHardwareMetadata.load_channel_table_from_netcdf(
            dataset
        )

    assert len(loaded_channel_list) == len(hardware_metadata.channel_list)

    for loaded_channel, expected_channel in zip(
        loaded_channel_list,
        hardware_metadata.channel_list,
    ):
        for attr in Channel().channel_attr_list:
            expected_value = getattr(expected_channel, attr)
            loaded_value = getattr(loaded_channel, attr)

            if expected_value is None:
                assert loaded_value in (None, "")
            else:
                assert str(loaded_value) == str(expected_value)


# endregion


# region Hardware Acquisition
def test_hardware_acquisition_init():
    """
    Verifies that the mock acquisition class is an instance of
    ``HardwareAcquisition``.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    assert isinstance(hardware_acquisition, HardwareAcquisition)


def test_hardware_acquisition_initialize_hardware(hardware_metadata):
    """
    Verifies that acquisition hardware stores supplied metadata during
    initialization.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    hardware_acquisition.initialize_hardware(hardware_metadata)

    assert hardware_acquisition.metadata is hardware_metadata


def test_hardware_acquisition_start():
    """
    Verifies that acquisition hardware can be started.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    hardware_acquisition.start()

    assert hardware_acquisition.started is True


def test_hardware_acquisition_read():
    """
    Verifies that reading from acquisition hardware returns a NumPy array.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    data = hardware_acquisition.read()

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 10)


def test_hardware_acquisition_read_remaining():
    """
    Verifies that reading remaining acquisition data returns a NumPy array.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    data = hardware_acquisition.read_remaining()

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)


def test_hardware_acquisition_stop():
    """
    Verifies that acquisition hardware can be stopped.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    hardware_acquisition.stop()

    assert hardware_acquisition.stopped is True


def test_hardware_acquisition_close():
    """
    Verifies that acquisition hardware can be closed.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    hardware_acquisition.close()

    assert hardware_acquisition.closed is True


def test_hardware_acquisition_get_acquisition_delay():
    """
    Verifies that acquisition delay is returned as an integer.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    acquisition_delay = hardware_acquisition.get_acquisition_delay()

    assert isinstance(acquisition_delay, int)
    assert acquisition_delay == 0


def test_hardware_acquisition_functions(hardware_metadata):
    """
    Calls all acquisition interface methods on a mock implementation and
    verifies expected return types.
    """
    hardware_acquisition = SkeletonHardwareAcquisition()

    hardware_acquisition.initialize_hardware(hardware_metadata)
    hardware_acquisition.start()
    read_data = hardware_acquisition.read()
    remaining_data = hardware_acquisition.read_remaining()
    hardware_acquisition.stop()
    hardware_acquisition.close()
    acquisition_delay = hardware_acquisition.get_acquisition_delay()

    assert hardware_acquisition.metadata is hardware_metadata
    assert hardware_acquisition.started is True
    assert hardware_acquisition.stopped is True
    assert hardware_acquisition.closed is True
    assert isinstance(read_data, np.ndarray)
    assert isinstance(remaining_data, np.ndarray)
    assert isinstance(acquisition_delay, int)


# endregion


# region Hardware Output
def test_hardware_output_init():
    """
    Verifies that the mock output class is an instance of ``HardwareOutput``.
    """
    hardware_output = SkeletonHardwareOutput()

    assert isinstance(hardware_output, HardwareOutput)


def test_hardware_output_initialize_hardware(hardware_metadata):
    """
    Verifies that output hardware stores supplied metadata during
    initialization.
    """
    hardware_output = SkeletonHardwareOutput()

    hardware_output.initialize_hardware(hardware_metadata)

    assert hardware_output.metadata is hardware_metadata


def test_hardware_output_start():
    """
    Verifies that output hardware can be started.
    """
    hardware_output = SkeletonHardwareOutput()

    hardware_output.start()

    assert hardware_output.started is True


def test_hardware_output_write():
    """
    Verifies that output hardware accepts and stores output data.
    """
    hardware_output = SkeletonHardwareOutput()
    output_data = np.zeros((1, 100))

    hardware_output.write(output_data)

    assert hardware_output.last_write is output_data


def test_hardware_output_stop():
    """
    Verifies that output hardware can be stopped.
    """
    hardware_output = SkeletonHardwareOutput()

    hardware_output.stop()

    assert hardware_output.stopped is True


def test_hardware_output_close():
    """
    Verifies that output hardware can be closed.
    """
    hardware_output = SkeletonHardwareOutput()

    hardware_output.close()

    assert hardware_output.closed is True


def test_hardware_output_ready_for_new_output():
    """
    Verifies that output hardware reports readiness for new output.
    """
    hardware_output = SkeletonHardwareOutput()

    assert hardware_output.ready_for_new_output() is True


def test_hardware_output_functions(hardware_metadata):
    """
    Calls all output interface methods on a mock implementation.
    """
    hardware_output = SkeletonHardwareOutput()
    output_data = np.zeros((1, 100))

    hardware_output.initialize_hardware(hardware_metadata)
    hardware_output.start()
    hardware_output.write(output_data)
    hardware_output.stop()
    hardware_output.close()
    ready = hardware_output.ready_for_new_output()

    assert hardware_output.metadata is hardware_metadata
    assert hardware_output.started is True
    assert hardware_output.last_write is output_data
    assert hardware_output.stopped is True
    assert hardware_output.closed is True
    assert ready is True


# endregion
