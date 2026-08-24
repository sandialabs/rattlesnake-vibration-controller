---
title: Rattlesnake Vibration Controller
authors:
  - name: Daniel Rohe
    affiliation: Sandia National Laboratories
    email: dprohe@sandia.gov
  - name: Ryan Schultz
    affiliation: Sandia National Laboratories
    email: rschult@sandia.gov
  - name: Norman Hunter
    affiliation: Sandia National Laboratories
    email: nfhunte@sandia.gov
  - name: Cody Langston
    affiliation: Sandia National Laboratories/University of Georgia
  - name: Chad Hovey
    affiliation: Sandia National Laboratories
numbering:
  heading_1:
    template: "Chapter %s"
    start: 1
  figure:
    enumerator: 1.%s
  table:
    enumerator: 1.%s
  equation:
    enumerator: 1.%s
  code:
    enumerator: 1.%s
---

:::{warning} Warning: Research and Development Software
:class: dropdown
Rattlesnake is, at its heart, software designed to do research and development into MIMO Vibration Control problems without dealing with the restrictions and closed-off nature of proprietary software.  Users of Rattlesnake must be aware that various portions of the software are active research topics, so they should be used with caution.

As with any capability in active research, there is the chance that the capability does not behave as expected.  Vibration test equipment can vary in cost from hundreds of dollars to millions of dollars, and test articles can range from a few dollars of stock material for a simple beam to almost priceless if they are one-of-a-kind.  Vibration control software that does not behave as expected could therefore cause damage to such equipment, which could impart catastrophic costs to the user of the software.

**By using the Rattlesnake software, users accept this risk to their equipment and test articles.  There is no warranty expressed or implied by using this software.  If you do not accept this risk, do not use this software.**
:::

Rattlesnake is a combined-environments, multiple input/multiple output control system for
dynamic excitation of structures under test.  It provides capabilities to control multiple
responses on the part using multiple exciters using various control strategies.  Rattlesnake
is written in the Python programming language to facilitate multiple input/multiple output
vibration research by allowing users to prescribe custom control laws to the controller.
Rattlesnake can target multiple hardware devices, or even perform synthetic control to simulate
a test virtually.  Rattlesnake has been used to execute control problems with up to 200
response channels and 24 shaker drives.  This document describes the functionality,
architecture, and usage of the Rattlesnake controller to perform combined environments testing.

(sec:introduction)=
# Introduction

The field of multiple input/multiple output (MIMO) vibration testing has grown substantially in the past few years.  MIMO vibration testing provides the capability to match a field environment more accurately and at more locations on the test article than traditional single-axis vibration testing.  Unfortunately, many existing vibration control systems are proprietary, which makes it difficult to implement new MIMO techniques.  

Currently, MIMO vibration practitioners must either develop a control system from scratch to implement their ideas, or alternatively convince an MIMO vibration software vendor to implement their ideas into existing devices, and neither of these approaches are conducive to the rapid and iterative nature of research.

The Rattlesnake framework was developed to overcome these limitations and facilitate MIMO vibration research.  Rattlesnake is a MIMO control system that provides the user the ability to overcome testing challenges by providing a flexible framework that can be extended and modified to meet testing demands.  Rattlesnake can run multiple environments simultaneously, providing a combined-environments capability that does not yet exist in commercial software packages.  It can target multiple hardware devices, or even perform control virtually using a state space model, finite element model (FEM) results, or a [SDynPy](https://github.com/sandialabs/sdynpy) System.

:::{figure} figures/Rattlesnake_Logo_Banner.png
:label: fig:rattlesnake_logo
:align: center

Rattlesnake Logo
:::