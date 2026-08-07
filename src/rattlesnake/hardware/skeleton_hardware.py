# -*- coding: utf-8 -*-
"""
This file defines a skeleton of a hardware backend. This file should be
modified to construct a full hardware implementation.

Rattlesnake Vibration Control Software
Copyright (C) 2021  National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.hardware.abstract_hardware import (
    HardwareMetadata,
    HardwareAcquisition,
    HardwareOutput,
)


class SkeletonHardwareMetadata(HardwareMetadata):
    def __init__(
        self,
        channel_list,
        sample_rate,
        time_per_read,
        time_per_write,
        output_oversample=1,
    ):
        super().__init__(
            HardwareType.SKELETON,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample=output_oversample,
        )

    def validate(self):
        return super().validate()

    def valid_channel_dict(self, channel):
        return super().valid_channel_dict(channel)

    @property
    def assist_mode_modules(self):
        return super().assist_mode_modules

    @classmethod
    def load_metadata_from_workbook(cls, workbook):
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_workbook(workbook)

        return cls(
            channel_list=channel_list,
            sample_rate=sample_rate,
            time_per_read=time_per_read,
            time_per_write=time_per_write,
            output_oversample=output_oversample,
        )

    def save_metadata_to_netcdf(self, netcdf_dataset):
        return super().save_metadata_to_netcdf(netcdf_dataset)

    @classmethod
    def load_metadata_from_netcdf(cls, netcdf_dataset):
        (
            hardware_type,
            channel_list,
            sample_rate,
            time_per_read,
            time_per_write,
            output_oversample,
        ) = super().load_metadata_from_netcdf(netcdf_dataset)

        return cls(
            channel_list=channel_list,
            sample_rate=sample_rate,
            time_per_read=time_per_read,
            time_per_write=time_per_write,
            output_oversample=output_oversample,
        )

    def save_metadata_to_workbook(self, workbook):
        return super().save_metadata_to_workbook(workbook)


class SkeletonHardwareAcquisition(HardwareAcquisition):
    def __init__(self):
        self.metadata = None
        self.started = False
        self.closed = False
        self.stopped = False

    def initialize_hardware(self, metadata):
        self.metadata = metadata

    def start(self):
        self.started = True

    def read(self):
        return np.zeros((2, 10))

    def read_remaining(self):
        return np.zeros((2, 3))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def get_acquisition_delay(self):
        return 0


class SkeletonHardwareOutput(HardwareOutput):
    def __init__(self):
        self.metadata = None
        self.started = False
        self.closed = False
        self.stopped = False
        self.last_write = None

    def initialize_hardware(self, metadata):
        self.metadata = metadata

    def start(self):
        self.started = True

    def write(self, data):
        self.last_write = data

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def ready_for_new_output(self):
        return True
