![Rattlesnake Logo](/logo/Rattlesnake_Logo_Banner.png)

## Quality Metrics

[![Pylint](./badges/pylint.svg)](https://sandialabs.github.io/rattlesnake-vibration-controller/reports/pylint/)

* GitHub Pages Home 🏠: [https://sandialabs.github.io/rattlesnake-vibration-controller/](https://sandialabs.github.io/rattlesnake-vibration-controller/)
* Lint 🧹: [https://sandialabs.github.io/rattlesnake-vibration-controller/reports/pylint/](https://sandialabs.github.io/rattlesnake-vibration-controller/reports/pylint/)

## Work in Progress

![Coverage](./badges/coverage.svg)

[coverage report](https://sandialabs.github.io/rattlesnake-vibration-controller/reports/coverage/)

* Coverage 🛡️: [https://sandialabs.github.io/rattlesnake-vibration-controller/reports/coverage/](https://sandialabs.github.io/rattlesnake-vibration-controller/reports/coverage/)

## Overview

This project aims to develop a Combined Environments, Multiple-Input/Multiple-Output (MIMO) vibration
controller that can better simulate dynamic environments than a traditional single-shaker test.

The controller is nicknamed "Rattlesnake," which blends together snakes (as it is written in Python programming language), 
vibration (rattlesnakes are famous for shaking their tails to create sound),
and New Mexico (the location of the main Sandia National Laboratories campus, where rattlesnakes can commonly be found).

Rattlesnake can be run as a Python script using the code from this repository, or an executable can be downloaded from the [Releases](https://github.com/sandialabs/rattlesnake-vibration-controller/releases) page.

See the [User's Manual](https://github.com/sandialabs/rattlesnake-vibration-controller/releases/download/v3.0.0/Rattlesnake.pdf) for more information.

### Flexible

The controller can currently run using National Instruments hardware using the [NI-DAQmx](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z000000P8baSAC&l=en-US)
interface or [B+K LAN-XI](https://www.bksv.com/en/instruments/daq-data-acquisition/lan-xi-daq-system) hardware using their OpenAPI.
It can also run synthetic control problems using finite element results or State Space Matrices.  Advanced users can implement new hardware devices in Rattlesnake.

### Capable

Rattlesnake has been run with over 50 control channels and 12 shaker drives while streaming over 250 channels to disk.

### Research Focus

To facilitate MIMO vibration research and development, users can program their own control laws to load into the controller at runtime.
See [Control Laws](https://github.com/sandialabs/rattlesnake-vibration-controller/tree/main/control_laws) for examples.
