import openpyxl
import netCDF4 as nc4

import rattlesnake.examples.defaults as defaults

from rattlesnake.process.abstract_sysid_data_analysis import (
    SysIdMetadata,
    SysIdDataPackage,
)


def netcdf_sysid_data_package():
    netcdf_dir = defaults.DIRECTORY + "/environment/sysid/sysid_data.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups["sysid"]
    data_package = SysIdDataPackage.load_package_from_netcdf(netcdf_group)

    return data_package


def netcdf_sysid_metadata(hardware_metadata):
    netcdf_dir = defaults.DIRECTORY + "/environment/sysid/sysid.nc4"
    netcdf_dataset = nc4.Dataset(netcdf_dir)
    netcdf_group = netcdf_dataset.groups["sysid"]
    metadata = SysIdMetadata.load_metadata_from_netcdf(netcdf_group, hardware_metadata)

    return metadata


def worksheet_sysid_metadata(hardware_metadata):
    worksheet_dir = defaults.DIRECTORY + "/environment/sysid/sysid.xlsx"
    workbook = openpyxl.load_workbook(worksheet_dir, read_only=True)
    worksheet = workbook["sysid"]

    metadata = SysIdMetadata.load_metadata_from_worksheet(
        worksheet, hardware_metadata, start_row=2
    )

    return metadata


def manual_sysid_metadata(hardware_metadata):
    sample_rate = hardware_metadata.sample_rate
    sysid_frame_size = hardware_metadata.sample_rate
    sysid_averaging_type = "Linear"
    sysid_noise_averages = 20
    sysid_averages = 20
    sysid_exponential_averaging_coefficient = 0.01
    sysid_estimator = "H1"
    sysid_level = 0.01
    sysid_level_ramp_time = 0.5
    sysid_signal_type = "Random"
    sysid_window = "Hann"
    sysid_overlap = 0.5
    sysid_burst_on = 0.5
    sysid_pretrigger = 0.05
    sysid_burst_ramp_fraction = 0.05
    sysid_low_frequency_cutoff = 0
    sysid_high_frequency_cutoff = int(sample_rate / 2)
    stream_file = None
    auto_shutdown = False
    return SysIdMetadata(
        sample_rate=sample_rate,
        sysid_frame_size=sysid_frame_size,
        sysid_averaging_type=sysid_averaging_type,
        sysid_noise_averages=sysid_noise_averages,
        sysid_averages=sysid_averages,
        sysid_exponential_averaging_coefficient=sysid_exponential_averaging_coefficient,
        sysid_estimator=sysid_estimator,
        sysid_level=sysid_level,
        sysid_level_ramp_time=sysid_level_ramp_time,
        sysid_signal_type=sysid_signal_type,
        sysid_window=sysid_window,
        sysid_overlap=sysid_overlap,
        sysid_burst_on=sysid_burst_on,
        sysid_pretrigger=sysid_pretrigger,
        sysid_burst_ramp_fraction=sysid_burst_ramp_fraction,
        sysid_low_frequency_cutoff=sysid_low_frequency_cutoff,
        sysid_high_frequency_cutoff=sysid_high_frequency_cutoff,
        stream_file=stream_file,
        auto_shutdown=auto_shutdown,
    )
