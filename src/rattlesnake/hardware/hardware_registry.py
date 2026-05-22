from rattlesnake.hardware.hardware_utilities import HardwareType

UNIMPLEMENTED_HARDWARE = [
    HardwareType.NONE,
    HardwareType.DP_QUATTRO,
    HardwareType.DP_900,
    HardwareType.EXODUS,
    HardwareType.SDYNPY_FRF,
]


HARDWARE_METADATA = {}
HARDWARE_ACQUISITION = {}
HARDWARE_OUTPUT = {}

for hardware_type in HardwareType:
    if hardware_type in UNIMPLEMENTED_HARDWARE:
        continue

    match hardware_type:
        case HardwareType.NI_DAQMX:
            from rattlesnake.hardware.nidaqmx_hardware_multitask import (
                NIDAQmxMetadata,
                NIDAQmxAcquisition,
                NIDAQmxOutput,
            )
            HARDWARE_METADATA[HardwareType.NI_DAQMX] = NIDAQmxMetadata
            HARDWARE_ACQUISITION[HardwareType.NI_DAQMX] = NIDAQmxAcquisition
            HARDWARE_OUTPUT[HardwareType.NI_DAQMX] = NIDAQmxOutput
        case HardwareType.LAN_XI:
            from rattlesnake.hardware.lanxi_hardware_multiprocessing import (
                LanXIMetadata,
                LanXIAcquisition,
                LanXIOutput,
            )
            HARDWARE_METADATA[HardwareType.LAN_XI] = LanXIMetadata
            HARDWARE_ACQUISITION[HardwareType.LAN_XI] = LanXIAcquisition
            HARDWARE_OUTPUT[HardwareType.LAN_XI] = LanXIOutput
        case HardwareType.DP_QUATTRO:
            from rattlesnake.hardware.data_physics_hardware import (
                DataPhysicsAcquisition,
                DataPhysicsOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.DP_QUATTRO] = DataPhysicsAcquisition
            HARDWARE_OUTPUT[HardwareType.DP_QUATTRO] = DataPhysicsOutput
        case HardwareType.DP_900:
            from rattlesnake.hardware.data_physics_dp900_hardware import (
                DataPhysicsDP900Acquisition,
                DataPhysicsDP900Output,
            )

            HARDWARE_ACQUISITION[HardwareType.DP_900] = DataPhysicsDP900Acquisition
            HARDWARE_OUTPUT[HardwareType.DP_900] = DataPhysicsDP900Output
        case HardwareType.EXODUS:
            from rattlesnake.hardware.exodus_modal_solution_hardware import (
                ExodusAcquisition,
                ExodusOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.EXODUS] = ExodusAcquisition
            HARDWARE_OUTPUT[HardwareType.EXODUS] = ExodusOutput
        case HardwareType.STATE_SPACE:
            from rattlesnake.hardware.state_space_virtual_hardware import (
                StateSpaceMetadata,
                StateSpaceAcquisition,
                StateSpaceOutput,
            )
            HARDWARE_METADATA[HardwareType.STATE_SPACE] = StateSpaceMetadata
            HARDWARE_ACQUISITION[HardwareType.STATE_SPACE] = StateSpaceAcquisition
            HARDWARE_OUTPUT[HardwareType.STATE_SPACE] = StateSpaceOutput
        case HardwareType.SDYNPY_SYSTEM:
            from rattlesnake.hardware.sdynpy_system_virtual_hardware import (
                SDynPySystemMetadata,
                SDynPySystemAcquisition,
                SDynPySystemOutput,
            )

            HARDWARE_METADATA[HardwareType.SDYNPY_SYSTEM] = SDynPySystemMetadata
            HARDWARE_ACQUISITION[HardwareType.SDYNPY_SYSTEM] = SDynPySystemAcquisition
            HARDWARE_OUTPUT[HardwareType.SDYNPY_SYSTEM] = SDynPySystemOutput
        case HardwareType.SDYNPY_FRF:
            from rattlesnake.hardware.sdynpy_frf_virtual_hardware import (
                SDynPyFRFAcquisition,
                SDynPyFRFOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.SDYNPY_FRF] = SDynPyFRFAcquisition
            HARDWARE_OUTPUT[HardwareType.SDYNPY_FRF] = SDynPyFRFOutput
