---
numbering:
  heading_2:
    start: 14
  figure:
    enumerator: 14.%s
  table:
    enumerator: 14.%s
  equation:
    enumerator: 14.%s
  code:
    enumerator: 14.%s
---
# Multiple Input/Multiple Output Sine Control

(sec:mimo_sine)=
# Multiple Input/Multiple Output Sine Control

The MIMO Sine Environment aims to control the vibration response of a component to a sweeping sinusoidal response.  The sinusoidal response is generally specified to have certain amplitude and phase relationships at certain frequencies.  A linear or logarithmic sweep rate is given, which tells the controller how fast to vary the frequency.  The goal of the MIMO Sine Environment is to modulate the drive signal frequencies, amplitudes, and phases to match the desired response frequencies, amplitudes, and phases.

