"""
Tests for Abstract Control Law

This module contains tests for the AbstractControlClass, using a dummy
implementation to verify initialization and basic functionality.
"""

import numpy as np

from rattlesnake.environment.abstract_control_law import AbstractControlClass


class DummyAbstractControlLaw(AbstractControlClass):
    """
    Dummy implementation of AbstractControlClass for testing.
    """

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
        """
        Initialize the DummyAbstractControlLaw.

        Args:
            specification: Control specification.
            warning_levels: Warning levels for control.
            abort_levels: Abort levels for control.
            extra_control_parameters: Additional parameters for control.
            transfer_function: Initial transfer function.
            buzz_cpsd: Initial buzz CPSD.
            last_response_cpsd: Last response CPSD.
            last_output_cpsd: Last output CPSD.
        """
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
        """
        Update system identification parameters.

        Args:
            transfer_function: New transfer function.
            buzz_cpsd: New buzz CPSD.

        Returns:
            The result of the superclass system_id_update method.
        """
        return super().system_id_update(transfer_function, buzz_cpsd)

    def control(self, transfer_function, last_response_cpsd=None, last_output_cpsd=None):
        """
        Perform control calculation.

        Args:
            transfer_function: Current transfer function.
            last_response_cpsd: Last response CPSD.
            last_output_cpsd: Last output CPSD.

        Returns:
            The result of the superclass control method.
        """
        return super().control(transfer_function, last_response_cpsd, last_output_cpsd)


def test_abstract_control_init():
    """
    Test the initialization of the AbstractControlClass via DummyAbstractControlLaw.
    """
    zero_array = np.zeros((0, 1))
    abstract_control_class = DummyAbstractControlLaw(
        zero_array, zero_array, zero_array, "Parameters"
    )

    assert isinstance(abstract_control_class, AbstractControlClass)
