from rattlesnake.environment.abstract_control_law import AbstractControlClass
from unittest import mock
import numpy as np
import pytest


class DummyAbstractControlLaw(AbstractControlClass):
    def __init__(
        self,
        specification,
        warning_levels,
        abort_levels,
        extra_control_parameters,
        transfer_function=None,
        buzz_cpsd=None,
        last_response_cpsd=None,
        last_output_cpsd=None,
    ):
        super().__init__(
            specification,
            warning_levels,
            abort_levels,
            extra_control_parameters,
            transfer_function,
            buzz_cpsd,
            last_response_cpsd,
            last_output_cpsd,
        )

    def system_id_update(self, transfer_function, buzz_cpsd):
        return super().system_id_update(transfer_function, buzz_cpsd)

    def control(self, transfer_function, last_response_cpsd=None, last_output_cpsd=None):
        return super().control(transfer_function, last_response_cpsd, last_output_cpsd)


def test_abstract_control_init():
    zero_array = np.zeros((0, 1))
    abstract_control_class = DummyAbstractControlLaw(
        zero_array, zero_array, zero_array, "Parameters"
    )

    assert isinstance(abstract_control_class, AbstractControlClass)
