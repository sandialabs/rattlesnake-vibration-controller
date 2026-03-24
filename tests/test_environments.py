from rattlesnake.environment.environment_utilities import ControlTypes
from unittest import mock
import pytest


# Test Control Type initialization
@pytest.mark.parametrize("control_idx", [0, 1, 2, 4, 6])
def test_control_type(control_idx):
    control_type = ControlTypes(control_idx)

    # Test if control type variable is a ControlType initilization
    assert isinstance(control_type, ControlTypes)
