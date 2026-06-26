from rattlesnake.environment.environment_utilities import EnvironmentType

UNIMPLEMENTED_ENVIRONMENT = [EnvironmentType.NONE]

ENVIRONMENT_COMMANDS = {}
ENVIRONMENT_METADATA = {}
ENVIRONMENT_CLASS = {}
ENVIRONMENT_PROCESS = {}
SYSID_ENVIRONMENTS = []

for environment_type in EnvironmentType:
    if environment_type in UNIMPLEMENTED_ENVIRONMENT:
        continue

    match environment_type:
        case EnvironmentType.TIME:
            from rattlesnake.environment.time_environment import (
                TimeCommands,
                TimeMetadata,
                TimeEnvironment,
                time_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.TIME] = TimeCommands
            ENVIRONMENT_METADATA[EnvironmentType.TIME] = TimeMetadata
            ENVIRONMENT_CLASS[EnvironmentType.TIME] = TimeEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.TIME] = time_process
        case EnvironmentType.MODAL:
            from rattlesnake.environment.modal_environment import (
                ModalCommands,
                ModalMetadata,
                ModalEnvironment,
                modal_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.MODAL] = ModalCommands
            ENVIRONMENT_METADATA[EnvironmentType.MODAL] = ModalMetadata
            ENVIRONMENT_CLASS[EnvironmentType.MODAL] = ModalEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.MODAL] = modal_process
        case EnvironmentType.SINE:
            from rattlesnake.environment.sine_sys_id_environment import (
                SineCommands,
                SineMetadata,
                SineEnvironment,
                sine_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.SINE] = SineCommands
            ENVIRONMENT_METADATA[EnvironmentType.SINE] = SineMetadata
            ENVIRONMENT_CLASS[EnvironmentType.SINE] = SineEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.SINE] = sine_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.SINE)
        case EnvironmentType.TRANSIENT:
            from rattlesnake.environment.transient_sys_id_environment import (
                TransientCommands,
                TransientMetadata,
                TransientEnvironment,
                transient_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.TRANSIENT] = TransientCommands
            ENVIRONMENT_METADATA[EnvironmentType.TRANSIENT] = TransientMetadata
            ENVIRONMENT_CLASS[EnvironmentType.TRANSIENT] = TransientEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.TRANSIENT] = transient_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.TRANSIENT)
        case EnvironmentType.RANDOM:
            from rattlesnake.environment.random_vibration_sys_id_environment import (
                RandomVibrationCommands,
                RandomVibrationMetadata,
                RandomVibrationEnvironment,
                random_vibration_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.RANDOM] = RandomVibrationCommands
            ENVIRONMENT_METADATA[EnvironmentType.RANDOM] = RandomVibrationMetadata
            ENVIRONMENT_CLASS[EnvironmentType.RANDOM] = RandomVibrationEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.RANDOM] = random_vibration_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.RANDOM)
