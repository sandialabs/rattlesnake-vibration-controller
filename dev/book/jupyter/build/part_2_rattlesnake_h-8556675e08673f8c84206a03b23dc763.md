(sec:rattlesnake_hardware)=
# Part 2: Rattlesnake Hardware Devices

Designed for flexibility, Rattlesnake can be used with multiple hardware devices and even perform virtual control using a synthetic data acquisition system.  This Part will cover the hardware-specific implementation details that must be considered when running Rattlesnake with each hardware device.

Rattlesnake is designed so there is minimal differences in software workflow when using different hardware devices.  Nonetheless, there are some slight differences in how channels and devices must be specified, and these differences are primarily found on the `Data Acquisition Setup` tab in the Rattlesnake software.

The Chapters in this Part document the hardware devices that are able to be used by Rattlesnake, as well as the virtual devices that can be used to simulate control.  @sec:nidaq_hardware describes the implementation and utilization of NI-DAQmx devices.  @sec:lanxi_hardware describes the HBK LAN-XI hardware.  @sec:dp_quattro_hardware and @sec:dp_900_hardware describe the Data Physics Quattro and 900-series hardware devices, respectively.  @sec:state_space_hardware, @sec:exodus_hardware, and @sec:sdynpy_hardware describe virtual or synthetic hardware devices that are defined using state space matrices, eigensolution results stored in an exodus file, or a SDynPy System object, respectively.

If a user is interested in implementing a new hardware device, @sec:new_hardware describes some of the things to be aware of.  Implementation of new hardware devices will require a good amount of knowledge of the Rattlesnake architecture.