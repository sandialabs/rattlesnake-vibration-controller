from rattlesnake.hardware.hardware_utilities import HardwareType

UNIMPLEMENTED_HARDWARE = [
    HardwareType.NONE,
    HardwareType.NI_DAQMX,
    HardwareType.LAN_XI,
    HardwareType.DP_QUATTRO,
    HardwareType.DP_900,
    HardwareType.EXODUS,
    HardwareType.STATE_SPACE,
    HardwareType.SDYNPY_FRF,
]


HARDWARE_METADATA = {}
HARDWARE_ACQUISITION = {}
HARDWARE_OUTPUT = {}
UI_HARDWARE_OPTIONS = {"Select Hardware": HardwareType.NONE}
# See user_interface.py => self.hardware_widgets for options
UI_HARDWARE_WIDGETS = {HardwareType.NONE: []}
UI_ASK_FOR_FILE = []

for hardware_type in HardwareType:
    if hardware_type in UNIMPLEMENTED_HARDWARE:
        continue

    match hardware_type:
        case HardwareType.NI_DAQMX:
            from rattlesnake.hardware.nidaqmx_hardware_multitask import (
                NIDAQmxAcquisition,
                NIDAQmxOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.NI_DAQMX] = NIDAQmxAcquisition
            HARDWARE_OUTPUT[HardwareType.NI_DAQMX] = NIDAQmxOutput
            UI_HARDWARE_OPTIONS["NI DAQmx"] = HardwareType.NI_DAQMX
            UI_HARDWARE_WIDGETS[HardwareType.NI_DAQMX] = [
                "sample_rate",
                "buffer_size",
                "task_trigger",
                "trigger_output",
            ]
        case HardwareType.LAN_XI:
            from rattlesnake.hardware.lanxi_hardware_multiprocessing import (
                LanXIAcquisition,
                LanXIOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.LAN_XI] = LanXIAcquisition
            HARDWARE_OUTPUT[HardwareType.LAN_XI] = LanXIOutput
            UI_HARDWARE_OPTIONS["HBK LAN-XI"] = HardwareType.LAN_XI
            UI_HARDWARE_WIDGETS[HardwareType.LAN_XI] = [
                "lanxi_sample_rate",
                "buffer_size",
                "lanxi_processes",
                "lanxi_ip",
            ]
        case HardwareType.DP_QUATTRO:
            from rattlesnake.hardware.data_physics_hardware import (
                DataPhysicsAcquisition,
                DataPhysicsOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.DP_QUATTRO] = DataPhysicsAcquisition
            HARDWARE_OUTPUT[HardwareType.DP_QUATTRO] = DataPhysicsOutput
            UI_HARDWARE_OPTIONS["Data Physics Quattro"] = HardwareType.DP_QUATTRO
            UI_HARDWARE_WIDGETS[HardwareType.DP_QUATTRO] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.DP_QUATTRO)
        case HardwareType.DP_900:
            from rattlesnake.hardware.data_physics_dp900_hardware import (
                DataPhysicsDP900Acquisition,
                DataPhysicsDP900Output,
            )

            HARDWARE_ACQUISITION[HardwareType.DP_900] = DataPhysicsDP900Acquisition
            HARDWARE_OUTPUT[HardwareType.DP_900] = DataPhysicsDP900Output
            UI_HARDWARE_OPTIONS["Data Physics 900 Series"] = HardwareType.DP_900
            UI_HARDWARE_WIDGETS[HardwareType.DP_900] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.DP_900)
        case HardwareType.EXODUS:
            from rattlesnake.hardware.exodus_modal_solution_hardware import (
                ExodusAcquisition,
                ExodusOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.EXODUS] = ExodusAcquisition
            HARDWARE_OUTPUT[HardwareType.EXODUS] = ExodusOutput
            UI_HARDWARE_OPTIONS["Exodus Modal Solution..."] = HardwareType.EXODUS
            UI_HARDWARE_WIDGETS[HardwareType.EXODUS] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "damping_ratio",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.EXODUS)
        case HardwareType.STATE_SPACE:
            from rattlesnake.hardware.state_space_virtual_hardware import (
                StateSpaceAcquisition,
                StateSpaceOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.STATE_SPACE] = StateSpaceAcquisition
            HARDWARE_OUTPUT[HardwareType.STATE_SPACE] = StateSpaceOutput
            UI_HARDWARE_OPTIONS["State Space Integration..."] = HardwareType.STATE_SPACE
            UI_HARDWARE_WIDGETS[HardwareType.STATE_SPACE] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.STATE_SPACE)
        case HardwareType.SDYNPY_SYSTEM:
            from rattlesnake.hardware.sdynpy_system_virtual_hardware import (
                SDynPySystemMetadata,
                SDynPySystemAcquisition,
                SDynPySystemOutput,
            )

            HARDWARE_METADATA[HardwareType.SDYNPY_SYSTEM] = SDynPySystemMetadata
            HARDWARE_ACQUISITION[HardwareType.SDYNPY_SYSTEM] = SDynPySystemAcquisition
            HARDWARE_OUTPUT[HardwareType.SDYNPY_SYSTEM] = SDynPySystemOutput
            UI_HARDWARE_OPTIONS["SDynPy System Integration..."] = (
                HardwareType.SDYNPY_SYSTEM
            )
            UI_HARDWARE_WIDGETS[HardwareType.SDYNPY_SYSTEM] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.SDYNPY_SYSTEM)
        case HardwareType.SDYNPY_FRF:
            from rattlesnake.hardware.sdynpy_frf_virtual_hardware import (
                SDynPyFRFAcquisition,
                SDynPyFRFOutput,
            )

            HARDWARE_ACQUISITION[HardwareType.SDYNPY_FRF] = SDynPyFRFAcquisition
            HARDWARE_OUTPUT[HardwareType.SDYNPY_FRF] = SDynPyFRFOutput
            UI_HARDWARE_OPTIONS["SDynPy FRF Convolution..."] = HardwareType.SDYNPY_FRF
            UI_HARDWARE_WIDGETS[HardwareType.SDYNPY_FRF] = [
                "sample_rate",
                "buffer_size",
                "integration_oversample",
                "select_file",
            ]
            UI_ASK_FOR_FILE.append(HardwareType.SDYNPY_FRF)
