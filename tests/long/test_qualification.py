import time

import pytest

from rattlesnake.testing.mock_user_interface import (
    launch_temporary_rattlesnake_ui_environment,
    launch_temporary_rattlesnake_ui_profile,
)

from rattlesnake.examples.headless_example import build_rattlesnake_object
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType


@pytest.mark.parametrize("threaded", [False])
@pytest.mark.parametrize("import_method", ["manual"])
@pytest.mark.parametrize("hardware_type", [HardwareType.SDYNPY_SYSTEM])
@pytest.mark.parametrize(
    "environment_type",
    [
        EnvironmentType.TIME,
        EnvironmentType.MODAL,
        EnvironmentType.SINE,
        EnvironmentType.RANDOM,
        EnvironmentType.TRANSIENT,
    ],
)
# @pytest.mark.parametrize("test_type", ["environment", "profile"])
@pytest.mark.parametrize("test_type", ["environment"])
def test_rattlesnake_qualification(
    threaded, import_method, hardware_type, environment_type, test_type
):
    run_profile = False
    start_environment = False
    if test_type == "environment":
        start_environment = True
    elif test_type == "profile":
        run_profile = True

    rattlesnake = build_rattlesnake_object(
        threaded=threaded,
        import_method=import_method,
        hardware_type=hardware_type,
        environment_type=environment_type,
        stream_type=StreamType.NO_STREAM,
        load_sysid=False,
        run_sysid=True,
        start_hardware=True,
        start_environment=start_environment,
        run_profile=run_profile,
    )
    time.sleep(10)
    rattlesnake.shutdown()

    assert True


@pytest.mark.parametrize("threaded", [True, False])
@pytest.mark.parametrize("import_method", ["manual", "netcdf", "worksheet"])
@pytest.mark.parametrize(
    "hardware_type",
    [
        HardwareType.SDYNPY_SYSTEM,
        HardwareType.EXODUS,
        HardwareType.SDYNPY_FRF,
        HardwareType.STATE_SPACE,
    ],
)
@pytest.mark.parametrize(
    "environment_type",
    [
        EnvironmentType.TIME,
        EnvironmentType.MODAL,
        EnvironmentType.SINE,
        EnvironmentType.RANDOM,
        EnvironmentType.TRANSIENT,
    ],
)
def not_test_rattlesnake_ui_profile_qualification(
    threaded, import_method, hardware_type, environment_type
):
    rattlesnake = build_rattlesnake_object(
        threaded=threaded,
        import_method=import_method,
        hardware_type=hardware_type,
        environment_type=environment_type,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=False,
        run_profile=False,
    )

    launch_temporary_rattlesnake_ui_profile(rattlesnake, 60)


@pytest.mark.parametrize("threaded", [True, False])
@pytest.mark.parametrize("import_method", ["manual", "netcdf", "worksheet"])
@pytest.mark.parametrize(
    "hardware_type",
    [
        HardwareType.SDYNPY_SYSTEM,
        HardwareType.EXODUS,
        HardwareType.SDYNPY_FRF,
        HardwareType.STATE_SPACE,
    ],
)
@pytest.mark.parametrize(
    "environment_type",
    [
        EnvironmentType.TIME,
        EnvironmentType.MODAL,
        EnvironmentType.SINE,
        EnvironmentType.RANDOM,
        EnvironmentType.TRANSIENT,
    ],
)
def not_test_rattlesnake_ui_start_environment_qualification(
    threaded, import_method, hardware_type, environment_type
):
    rattlesnake = build_rattlesnake_object(
        threaded=threaded,
        import_method=import_method,
        hardware_type=hardware_type,
        environment_type=environment_type,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=True,
        run_profile=False,
    )

    launch_temporary_rattlesnake_ui_environment(rattlesnake, 60)


@pytest.mark.parametrize("threaded", [False])
@pytest.mark.parametrize("import_method", ["manual"])
@pytest.mark.parametrize(
    "hardware_type",
    [
        HardwareType.SDYNPY_SYSTEM,
    ],
)
@pytest.mark.parametrize(
    "environment_type",
    [
        EnvironmentType.TIME,
        EnvironmentType.MODAL,
        EnvironmentType.SINE,
        EnvironmentType.RANDOM,
        EnvironmentType.TRANSIENT,
    ],
)
def not_test_minimal_qualification(
    threaded, import_method, hardware_type, environment_type
):
    rattlesnake = build_rattlesnake_object(
        threaded=threaded,
        import_method=import_method,
        hardware_type=hardware_type,
        environment_type=environment_type,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=False,
        run_profile=False,
    )

    launch_temporary_rattlesnake_ui_profile(rattlesnake, 60)
