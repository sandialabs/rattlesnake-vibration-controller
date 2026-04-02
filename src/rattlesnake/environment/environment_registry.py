from rattlesnake.environment.environment_utilities import EnvironmentType
from rattlesnake.environment.transient_sys_id_environment import (
    TransientCommands,
    TransientMetadata,
    TransientEnvironment,
    transient_process,
)
from rattlesnake.environment.sine_sys_id_environment import (
    SineCommands,
    SineMetadata,
    SineEnvironment,
    sine_process,
)
from rattlesnake.environment.modal_environment import (
    ModalCommands,
    ModalMetadata,
    ModalEnvironment,
    modal_process,
)
from rattlesnake.environment.random_vibration_sys_id_environment import (
    RandomVibrationCommands,
    RandomVibrationMetadata,
    RandomVibrationEnvironment,
    random_vibration_process,
)
from rattlesnake.environment.time_environment import (
    TimeParameters,
    TimeEnvironment,
    time_process,
)

ENVIRONMENT_COMMANDS = {}
ENVIRONMENT_METADATA = {}
ENVIRONMENT_CLASS = {}
ENVIRONMENT_PROCESS = {}

# Time Environment
ENVIRONMENT_METADATA[EnvironmentType.TIME] = TimeParameters
ENVIRONMENT_CLASS[EnvironmentType.TIME] = TimeEnvironment
ENVIRONMENT_PROCESS[EnvironmentType.TIME] = time_process

# Modal Environment
ENVIRONMENT_COMMANDS[EnvironmentType.MODAL] = ModalCommands
ENVIRONMENT_METADATA[EnvironmentType.MODAL] = ModalMetadata
ENVIRONMENT_CLASS[EnvironmentType.MODAL] = ModalEnvironment
ENVIRONMENT_PROCESS[EnvironmentType.MODAL] = modal_process

# Sine Environment
ENVIRONMENT_COMMANDS[EnvironmentType.SINE] = SineCommands
ENVIRONMENT_METADATA[EnvironmentType.SINE] = SineMetadata
ENVIRONMENT_CLASS[EnvironmentType.SINE] = SineEnvironment
ENVIRONMENT_PROCESS[EnvironmentType.SINE] = sine_process

# Transient Environment
ENVIRONMENT_COMMANDS[EnvironmentType.TRANSIENT] = TransientCommands
ENVIRONMENT_METADATA[EnvironmentType.TRANSIENT] = TransientMetadata
ENVIRONMENT_CLASS[EnvironmentType.TRANSIENT] = TransientEnvironment
ENVIRONMENT_PROCESS[EnvironmentType.TRANSIENT] = transient_process

# Random Environment
ENVIRONMENT_COMMANDS[EnvironmentType.RANDOM] = RandomVibrationCommands
ENVIRONMENT_METADATA[EnvironmentType.RANDOM] = RandomVibrationMetadata
ENVIRONMENT_CLASS[EnvironmentType.RANDOM] = RandomVibrationEnvironment
ENVIRONMENT_PROCESS[EnvironmentType.RANDOM] = random_vibration_process
