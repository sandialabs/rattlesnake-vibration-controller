from rattlesnake.examples.headless_example import build_rattlesnake_object
from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType


def test_rattlesnake_qualification():
    rattlesnake = build_rattlesnake_object(
        threaded=True,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.TIME,
        stream_type=StreamType.NO_STREAM,
        load_sysid=False,
        run_sysid=False,
        start_hardware=True,
        start_environment=False,
        run_profile=True,
    )
    rattlesnake.shutdown()

    assert True
