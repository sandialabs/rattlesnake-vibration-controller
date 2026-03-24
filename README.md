![Rattlesnake Logo](/logo/Rattlesnake_Logo_Banner.png)

# Rattlesnake Vibration Controller

<!--[![CI](https://github.com/sandialabs/rattlesnake-vibration-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/sandialabs/rattlesnake-vibration-controller/actions/workflows/ci.yml)
-->
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![GitHub license](https://img.shields.io/badge/license-GNU-blue.svg)](https://github.com/sandialabs/rattlesnake-vibration-controller/blob/main/LICENSE)

[![GitHub Pages badge](https://img.shields.io/badge/GitHub%20Pages-blueviolet?logo=github)](https://sandialabs.github.io/rattlesnake-vibration-controller/)
<!--
[![Pylint Report badge](https://img.shields.io/badge/Pylint%20Report-blue)](https://sandialabs.github.io/rattlesnake-vibration-controller/main/reports/pylint/) [![Coverage Report badge](https://img.shields.io/badge/Coverage%20Report-blue)](https://sandialabs.github.io/rattlesnake-vibration-controller/main/reports/coverage/)
-->

## Overview

This project aims to develop a Combined Environments, Multiple-Input/Multiple-Output (MIMO) vibration
controller that can better simulate dynamic environments than a traditional single-shaker test.

The controller is nicknamed "Rattlesnake," which blends together snakes (as it is written in Python programming language), 
vibration (rattlesnakes are famous for shaking their tails to create sound),
and New Mexico (the location of the main Sandia National Laboratories campus, where rattlesnakes can commonly be found).

Rattlesnake can be run as a Python script using the code from this repository, or an executable can be downloaded from the [Releases](https://github.com/sandialabs/rattlesnake-vibration-controller/releases) page.

See the documentation for more information:

* Rattlesnake User's Manual
  * (Work in progress) [Released](https://sandialabs.github.io/rattlesnake-vibration-controller/main/book/jupyter/index.html) ![branch-main](https://img.shields.io/badge/branch-main-blue)
  * [Development](https://sandialabs.github.io/rattlesnake-vibration-controller/dev/book/jupyter/index.html) ![branch-dev](https://img.shields.io/badge/branch-dev-orange)
* Rattlesnake User's Manual, Version 3, SAND2024-14378, Oct 2024. [Download](https://github.com/sandialabs/rattlesnake-vibration-controller/releases/download/v3.0.0/Rattlesnake.pdf) (78 MB)

### Flexible

The controller can currently run using National Instruments hardware using the [NI-DAQmx](https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z000000P8baSAC&l=en-US)
interface or [B+K LAN-XI](https://www.bksv.com/en/instruments/daq-data-acquisition/lan-xi-daq-system) hardware using their OpenAPI.
It can also run synthetic control problems using finite element results or State Space Matrices.  Advanced users can implement new hardware devices in Rattlesnake.

### Capable

Rattlesnake has been run with over 50 control channels and 12 shaker drives while streaming over 250 channels to disk.

### Research Focus

To facilitate MIMO vibration research and development, users can program their own control laws to load into the controller at runtime.
See [Control Laws](https://github.com/sandialabs/rattlesnake-vibration-controller/tree/main/control_laws) for examples.
