import time

from rattlesnake.hardware.hardware_utilities import HardwareType
from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.process.streaming import StreamType
from rattlesnake.examples.headless_example import build_rattlesnake_object


def transient_zero_trac():
    rattlesnake = build_rattlesnake_object(
        threaded=False,
        timeout=20,
        import_method="manual",
        hardware_type=HardwareType.SDYNPY_SYSTEM,
        environment_type=EnvironmentType.TRANSIENT,
        stream_type=StreamType.NO_STREAM,
        load_sysid=True,
        run_sysid=False,
        start_hardware=True,
        start_environment=True,
        run_profile=False,
    )
    time.sleep(10)
    rattlesnake.stop_environment("Transient 0")
    rattlesnake.shutdown()


if __name__ == "__main__":
    transient_zero_trac()
