import pytest

from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment import environment_registry

IMPLEMENTED_ENVIRONMENT = [
    environment
    for environment in EnvironmentType
    if environment not in environment_registry.UNIMPLEMENTED_ENVIRONMENT
]


@pytest.mark.parametrize("environment_type", IMPLEMENTED_ENVIRONMENT)
def test_implemented_environment_types_are_registered(environment_type):
    """
    Every EnvironmentType that is not explicitly unimplemented should be
    present in all environment registry dictionaries.
    """

    assert environment_type in environment_registry.ENVIRONMENT_COMMANDS
    assert environment_type in environment_registry.ENVIRONMENT_METADATA
    assert environment_type in environment_registry.ENVIRONMENT_INSTRUCTION
    assert environment_type in environment_registry.ENVIRONMENT_CLASS
    assert environment_type in environment_registry.ENVIRONMENT_PROCESS
