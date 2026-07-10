---
numbering:
  heading_1:
    enumerator: 1.%s
    template: "Section %s"
---
# Document Overview

(sec:doc_overview)=
# Document Overview

This document provides information about the functionality in the Rattlesnake software, as well as instructions for how to use that functionality.  The document is divided into Parts targeting different aspects of the software.

* @sec:rattlesnake_overview provides an overview of the software as well as instructions for how to acquire and run the software.  @sec:acquiring_and_running_rattlesnake includes instructions for setting up the Python ecosystem required to run the software, if necessary.  @sec:using_rattlesnake describes the Rattlesnake user interface (UI).  Each main portion of the Rattlesnake interface is described along with the parameters that should be defined within that interface.
* @sec:rattlesnake_hardware describes the hardware devices available to the Rattlesnake software, as well as the hardware-specific considerations in the controller.  Each Chapter in this Part is dedicated to a specific hardware device.  Synthetic or virtual control is also discussed in this part, as well as instructions to extend Rattlesnake to additional hardware devices.
* @sec:rattlesnake_environments describes the various control environments contained within the Rattlesnake software.  The environments defined within Rattlesnake are where the next output data that will be sent to the exciters are computed based off the previously acquired data.  Each chapter in this Part is dedicated to a environment type within Rattlesnake.  This chapter also provides instructions to combine environments as well as extend Rattlesnake to additional environments.
* The @sec:examples contain several example problems.  Users with a reasonable understanding of the Rattlesnake workflow can use these chapters as a kind of "Quick Start" guide to the software.  @sec:example_nidaqmx demonstrates a series of tests on a simple beam using a low-cost NI cDAQ hardware device.  @sec:example_sdynpy and @sec:example_state_space demonstrate virtual control problems using SDynPy System and State Space models, respectively, which only require a computer to run.