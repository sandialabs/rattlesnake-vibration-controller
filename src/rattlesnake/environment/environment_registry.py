from rattlesnake.environment.environment_utilities import EnvironmentType

UNIMPLEMENTED_ENVIRONMENT = [
    EnvironmentType.NONE,
    EnvironmentType.SYSID_SKELETON,
]

ENVIRONMENT_COMMANDS = {}
ENVIRONMENT_METADATA = {}
ENVIRONMENT_INSTRUCTION = {}
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
                TimeInstructions,
                TimeEnvironment,
                time_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.TIME] = TimeCommands
            ENVIRONMENT_METADATA[EnvironmentType.TIME] = TimeMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.TIME] = TimeInstructions
            ENVIRONMENT_CLASS[EnvironmentType.TIME] = TimeEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.TIME] = time_process
        case EnvironmentType.MODAL:
            from rattlesnake.environment.modal_environment import (
                ModalCommands,
                ModalMetadata,
                ModalInstructions,
                ModalEnvironment,
                modal_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.MODAL] = ModalCommands
            ENVIRONMENT_METADATA[EnvironmentType.MODAL] = ModalMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.MODAL] = ModalInstructions
            ENVIRONMENT_CLASS[EnvironmentType.MODAL] = ModalEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.MODAL] = modal_process
        case EnvironmentType.SINE:
            from rattlesnake.environment.sine_sys_id_environment import (
                SineCommands,
                SineMetadata,
                SineInstructions,
                SineEnvironment,
                sine_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.SINE] = SineCommands
            ENVIRONMENT_METADATA[EnvironmentType.SINE] = SineMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.SINE] = SineInstructions
            ENVIRONMENT_CLASS[EnvironmentType.SINE] = SineEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.SINE] = sine_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.SINE)
        case EnvironmentType.TRANSIENT:
            from rattlesnake.environment.transient_sys_id_environment import (
                TransientCommands,
                TransientMetadata,
                TransientInstructions,
                TransientEnvironment,
                transient_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.TRANSIENT] = TransientCommands
            ENVIRONMENT_METADATA[EnvironmentType.TRANSIENT] = TransientMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.TRANSIENT] = TransientInstructions
            ENVIRONMENT_CLASS[EnvironmentType.TRANSIENT] = TransientEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.TRANSIENT] = transient_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.TRANSIENT)
        case EnvironmentType.RANDOM:
            from rattlesnake.environment.random_vibration_sys_id_environment import (
                RandomVibrationCommands,
                RandomVibrationMetadata,
                RandomVibrationInstructions,
                RandomVibrationEnvironment,
                random_vibration_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.RANDOM] = RandomVibrationCommands
            ENVIRONMENT_METADATA[EnvironmentType.RANDOM] = RandomVibrationMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.RANDOM] = (
                RandomVibrationInstructions
            )
            ENVIRONMENT_CLASS[EnvironmentType.RANDOM] = RandomVibrationEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.RANDOM] = random_vibration_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.RANDOM)

        case EnvironmentType.SKELETON:
            from rattlesnake.environment.skeleton_environment import (
                SkeletonCommands,
                SkeletonMetadata,
                SkeletonInstructions,
                SkeletonEnvironment,
                skeleton_process,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.SKELETON] = SkeletonCommands
            ENVIRONMENT_METADATA[EnvironmentType.SKELETON] = SkeletonMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.SKELETON] = SkeletonInstructions
            ENVIRONMENT_CLASS[EnvironmentType.SKELETON] = SkeletonEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.SKELETON] = skeleton_process
        case EnvironmentType.SYSID_SKELETON:
            from rattlesnake.environment.skeleton_sys_id_environment import (
                SkeletonCommands as SysIdSkeletonCommands,
                SkeletonMetadata as SysIdSkeletonMetadata,
                SkeletonEnvironment as SysIdSkeletonEnvironment,
                skeleton_process as sysid_skeleton_process,
            )
            from rattlesnake.environment.skeleton_environment import (
                SkeletonInstructions,
            )

            ENVIRONMENT_COMMANDS[EnvironmentType.SYSID_SKELETON] = SysIdSkeletonCommands
            ENVIRONMENT_METADATA[EnvironmentType.SYSID_SKELETON] = SysIdSkeletonMetadata
            ENVIRONMENT_INSTRUCTION[EnvironmentType.SYSID_SKELETON] = (
                SkeletonInstructions
            )
            ENVIRONMENT_CLASS[EnvironmentType.SYSID_SKELETON] = SysIdSkeletonEnvironment
            ENVIRONMENT_PROCESS[EnvironmentType.SYSID_SKELETON] = sysid_skeleton_process
            SYSID_ENVIRONMENTS.append(EnvironmentType.SYSID_SKELETON)
