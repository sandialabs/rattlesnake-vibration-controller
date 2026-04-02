from rattlesnake.environment.environment_utilities import ControlTypes
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
ENVIRONMENT_METADATA[ControlTypes.TIME] = TimeParameters
ENVIRONMENT_CLASS[ControlTypes.TIME] = TimeEnvironment
ENVIRONMENT_PROCESS[ControlTypes.TIME] = time_process

# Modal Environment
ENVIRONMENT_COMMANDS[ControlTypes.MODAL] = ModalCommands
ENVIRONMENT_METADATA[ControlTypes.MODAL] = ModalMetadata
ENVIRONMENT_CLASS[ControlTypes.MODAL] = ModalEnvironment
ENVIRONMENT_PROCESS[ControlTypes.MODAL] = modal_process

# Sine Environment
ENVIRONMENT_COMMANDS[ControlTypes.SINE] = SineCommands
ENVIRONMENT_METADATA[ControlTypes.SINE] = SineMetadata
ENVIRONMENT_CLASS[ControlTypes.SINE] = SineEnvironment
ENVIRONMENT_PROCESS[ControlTypes.SINE] = sine_process

# Transient Environment
ENVIRONMENT_COMMANDS[ControlTypes.TRANSIENT] = TransientCommands
ENVIRONMENT_METADATA[ControlTypes.TRANSIENT] = TransientMetadata
ENVIRONMENT_CLASS[ControlTypes.TRANSIENT] = TransientEnvironment
ENVIRONMENT_PROCESS[ControlTypes.TRANSIENT] = transient_process

# Random Environment
ENVIRONMENT_COMMANDS[ControlTypes.RANDOM] = RandomVibrationCommands
ENVIRONMENT_METADATA[ControlTypes.RANDOM] = RandomVibrationMetadata
ENVIRONMENT_CLASS[ControlTypes.RANDOM] = RandomVibrationEnvironment
ENVIRONMENT_PROCESS[ControlTypes.RANDOM] = random_vibration_process
