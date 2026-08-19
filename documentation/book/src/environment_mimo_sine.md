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

The MIMO Sine environment in Rattlesnake is used to generate deterministic sinusoidal excitations at one or more drive channels so that the measured response at one or more control channels matches a prescribed specification. Unlike the Random Vibration environment, which operates on power spectral density matrices, the Sine environment operates on explicitly defined amplitude, phase, frequency, and sweep-rate information over time.

At a high level, the MIMO Sine environment does three things:

1. defines one or more sine sweeps or dwells as a specification,
2. uses a system identification phase to estimate the transfer functions between drive and response channels,
3. computes drive signals that should reproduce the specified response amplitudes and phases, and then updates those drives during a run based on measured tracking data.

The environment supports:

- multiple control channels,
- multiple excitation channels,
- transformed control and excitation coordinates,
- multiple simultaneous tones in one test,
- drive and response prediction,
- and several tracking/filtering methods for extracting amplitude and phase from the measured data.

:::{warning} Closed-loop Responsiveness
Because Rattlesnake originated as a random vibration controller, it's acquisition and output strategies operate on blocks of data rather than individual samples.  Therefore, users may find the MIMO Sine environment to be less responsive to control error than equivalent commercial solutions.  This time delay may also result in certain closed-loop implementations going unstable.  Users may approach sample-by-sample control by reducing the buffer size of the controller (decreasing the time between reads and writes).  However, this can be dangerous; if the output process runs out of samples to generate, it will perform a hard stop, which may damage test equipment or test articles.  For important tests, always ensure computational resources are sufficient the specified settings in a low-risk environment, perhaps with test levels set to some small value.
:::

## Governing Equations

At any instant in a sine test, the desired response of the structure may be represented as a complex response vector

\begin{equation}
\mathbf{x}(\omega) = \mathbf{a}_x(\omega)e^{i\mathbf{\phi}_x(\omega)}
\end{equation}

where $\mathbf{a}_x$ is the vector of desired response amplitudes and $\mathbf{\phi}_x$ is the vector of desired response phases.

The measured structural dynamics are represented by a frequency response matrix

\begin{equation}
\mathbf{H}_{xv}(\omega)
\end{equation}

relating excitation voltages $\mathbf{v}(\omega)$ to control responses $\mathbf{x}(\omega)$:

\begin{equation}
\mathbf{x}(\omega) = \mathbf{H}_{xv}(\omega)\mathbf{v}(\omega)
\end{equation}

The basic control problem is therefore to determine a drive vector $\mathbf{v}(\omega)$ such that the desired response is achieved. In a least-squares sense, a first estimate can be obtained using the pseudoinverse

\begin{equation}
\mathbf{v}(\omega) = \mathbf{H}_{xv}^{+}(\omega)\mathbf{x}(\omega)
\end{equation}

where $(\cdot)^+$ denotes the pseudoinverse.

Rattlesnake computes these complex drive values across the active tones and then reconstructs real time-domain signals for the output channels. During the run, measured response amplitudes and phases are extracted from the acquired time histories, and the control law computes corrections to the drive amplitudes and phases.

Because the Sine environment operates on deterministic sinusoidal content rather than stationary random spectra, a great deal of the implementation revolves around:

- constructing sinusoidal specifications over time,
- predicting the resulting deterministic response,
- and tracking amplitude and phase accurately during the run.

## Specification Definition

The Sine environment specification is defined as one or more **sine tones**, each consisting of a set of frequency breakpoints and associated amplitude/phase information for each control channel.

Each tone may be a linear sweep or a logarithmic sweep depending on how the sweep type and sweep rate are specified between breakpoints.  Currently, there is no "dwell" capability to pause at a specific frequency for a certain amount of time; however this can be approximated by using a slow sweep rate between two very close frequency breakpoints.

In Rattlesnake, the specifications are entered on the `Environment Definition` tab using one tab per sine tone. Each tab corresponds to a `SineSpecification` object internally.

A specification tone is defined by:

- a **name**, allowing individual sine tones to be given physical meanings or labels
- a **start time**, allowing the various tones playing in an environment to be staggered in time
- a list of **frequency breakpoints**, defining the frequencies at which the specification is defined and interpolated between
- a set of **amplitude breakpoints** for each control channel, defining the desired amplitude of the sine tone at that frequency
- a set of **phase breakpoints** for each control channel, defining the desired phasing between control channels at that frequency
- **sweep type** between adjacent breakpoints, which is either linear or logarithmic
- **sweep rate** between adjacent breakpoints, which defines how fast the frequency varies over time
- optional **warning** and **abort** amplitude limits, which can be used by the UI to flag poor control or by the control law to adjust control.

The environment can contain multiple such tones, which are combined in time to form the overall specification.

Specification information can be entered manually into the breakpoint table (see @sec:sine_specification).  Alternatively, the sine tone information may be loaded from a file.

### Specification File Format

The MIMO Sine environment accepts specification files in either NumPy archive format (`*.npz`) or MATLAB format (`*.mat`).  These files define one sine tone at a time. A specification file loaded into one sine-tone tab of the environment must contain the information needed to construct a Sine Specification.  If multiple sine tones are desired for a given environment, multiple files should be prepared and loaded individually in separate sine tone tabs in the `Environment Definition` tab (see @sec:sine_specification).

Both NumPy and MATLAB files are structured identically, with the same field names and shapes associated with each field.  Note that for 1D arrays, MATLAB can either specify $n \times 1$ or $1 \times n$ arrays; Rattlesnake will squeeze out the extra dimension.

For a specification where $n_f$ is the number of frequency breakpoints and $n_c$ is the number of control channels, the fields are:

* **frequency** A $n_f$ array containing the frequency value in hertz of each breakpoint.
* **amplitude** A $n_c \times n_f$ array containing the amplitude values for each control channel at each breakpoint.  The units are assumed to be the engineering unit specified in the channel table for this channel.
* **phase** A $n_c \times n_f$ array containing the phase values for each control channel at each frequency breakpoint in degrees.  If not defined, all phases will be assumed to be zero.
* **sweep_type** A $n_f - 1$ array containing the sweep type.  The $i$th value in this array is the sweep type between the $i$th and $(i+1)$th frequency breakpoint.  This array should have a value of `0` for linear or `1` for logarithmic sweeps.  If not defined, all sweeps will be assumed to be linear.
* **sweep_rate** A $n_f - 1$ array containing the sweep rate.  The $i$th value in this array is the sweep rate between the $i$th and $(i+1)$th frequency breakpoint.  For linear sweeps, this value is in Hz/s.  For logarithmic sweeps, this value is in octaves per minute.
* **warning** A $2 \times 2 \times n_c \times n_f$ array of warning specification levels.  The first dimension specifies lower (index `0`) or upper (index `1`) warning limits.  The second dimension specifies left (index `0`) or right (index `1`) breakpoint limits.  This enables a user to specify a different warning level to the left of a breakpoint vs. to the right of that same breakpoint.  If a warning is not desired at a given channel or breakpoint, a value of `NaN` can be specified in the array.  If no warning levels are desired, this field can be omitted.  Warning levels are defined in the same units as the `amplitude` field.
* **abort** A $2 \times 2 \times n_c \times n_f$ array of abort specification levels.  The first dimension specifies lower (index `0`) or upper (index `1`) abort limits.  The second dimension specifies left (index `0`) or right (index `1`) breakpoint limits.  This enables a user to specify a different abort level to the left of a breakpoint vs. to the right of that same breakpoint.  If a abort is not desired at a given channel or breakpoint, a value of `NaN` can be specified in the array.  If no abort levels are desired, this field can be omitted.  Abort levels are defined in the same units as the `amplitude` field.
* **start_time** A scalar value used to define the starting time (in seconds) of this sine tone.  This can be used to stagger the starts of the various sine tones in the test.  If omitted, a start time of 0 will be specified.
* **name** A scalar string used to define the name of the sine tone in the UI.  If omitted, a default name will be given to the sine tone, such as `Sine 1`, `Sine 2`, etc.

For any array with dimension of size $n_c$, the ordering of this dimension must be identical to the ordering of the control degrees of freedom in the environment loading the file.  No bookkeeping or reordering of specification data to match the channel data occurs in the Sine environment.  If a transformation matrix is used, then the ordering of this dimension must be identical to the rows of the transformation matrix.

## Defining the MIMO Sine Environment in Rattlesnake
In addition to the specification, there are a number of sampling and signal processing parameters that are used by the MIMO Sine environment.  These, along with the specification, are defined on the `Environment Definition` tab in the Rattlesnake controller, on a sub-tab corresponding to a MIMO Sine environment.  @fig:sine_environment_definition

:::{figure} figures/sine_environment_definition.png
:label: fig:sine_environment_definition
:align: center
UI used to define a MIMO Random Vibration environment.
:::

### Sampling Parameters
The @fig:sine_definition:sampling_parameters_groupbox section of the MIMO Sine definition sub-tab consists of the following parameters:

```{embed} #sec:sine_definition:sampling_parameters_groupbox
```

### Signal Generation Parameters
The @fig:sine_definition:signal_generation_groupbox section of the MIMO Sine definition sub-tab consists of the following parameters:

```{embed} #sec:sine_definition:signal_generation_groupbox
```

### Tracking Filter Parameters
The @fig:sine_definition:tracking_filter_groupbox section of the MIMO Sine definition sub-tab consists of properties used to specify the tracking filter, which is used to extract sine-tone information from the measured time histories.  Depending on which filter type is selected, different parameters will be made available to the user.

With the digital tracking filter selected, the following parameters are available:

```{embed} #sec:sine_definition__tracking_filter:tracking_filter_groupbox
```

With the Vold-Kalman filter selected, the following parameters are available:

```{embed} #sec:sine_definition__vold_kalman_filter:tracking_filter_groupbox
```

If the user is unsure as to which filter they should use, they can use the filter explorer to investigate filter performance, see @sec:sine_filter_explorer.

```{embed} #sec:sine_definition:tracking_filter_groupbox
```

### Control Parameters

The @fig:sine_definition:control_parameters_groupbox section of the MIMO Sine definition sub-tab consists of properties used to specify the control settings used by the controller.  Like other environment types, MIMO Sine allows users to load Python scripts containing custom control laws.  Unfortunately, MIMO Sine control laws are by their nature complex, so simple implementations like functions or generators are not available; only class definitions are available.  See @sec:sine_custom_control_law for more information on defining a custom control law.

```{embed} #sec:sine_definition:control_parameters_groupbox
```

### Control Channels

The @fig:sine_definition:control_channels_groupbox section of the MIMO Sine definition sub-tab allows users to select the channels in the environment that will be used for control.

```{embed} #sec:sine_definition:control_channels_groupbox
```

### Control and Drive Transforms

The @fig:sine_definition:transformation_matrices_groupbox section of the MIMO Sine definition sub-tab consists of properties used to specify the response and drive transformations, allowing virtual control degrees of freedom or drive signals to be constructed from physical channels.

```{embed} #sec:sine_definition:transformation_matrices_groupbox
```

Note that if transformation matrices are defined, the number of control channels ends up being the number of rows of the `Response Transformation Matrix`, rather than the number of physical control channels.  The number of physical control channels will be equal to the number of columns of the transformation matrix.  The number of control channels in the specification should be equal to the number of rows in the transformation.

See @sec:rattlesnake_environments_transformation_matrices for more information on specifying transformation matrices.

(sec:sine_specification)=
### Test Specification
The @fig:sine_definition:test_specification_groupbox section of the MIMO Sine definition sub-tab consists of properties used to specify the test specification.  The main tabs of this section denote the individual sine tones present in this environment.  Clicking on the tab labelled `+` will create a new sine tone.  This section also includes four plots to display the specification data in different formats.

```{embed} #sec:sine_definition:test_specification_groupbox
```

Within each sine tone, there are three tables behind a sub-tab interface.  The `Breakpoint Table` tab shows the frequency breakpoints, as well as various buttons to add or remove breakpoints.

```{embed} #sec:sine_definition__sine_table_breakpoint_tab:auto
```

The `Warning Table` contains amplitude limits that will trigger warnings.

```{embed} #sec:sine_definition__sine_table_warning_tab:auto
```

The `Abort Table` contains amplitude limits that will trigger aborts.

```{embed} #sec:sine_definition__sine_table_abort_tab:auto
```

Both warning and abort limit tables allow for specifying the amplitude on the left and right side of the breakpoint independently.  This can allow for, for example, looser tolerances during segments of the test that have higher sweep rates.  Upper and lower tolerances can also be specified.  The warning and abort limits will be shown in the plots along with the breakpoints themselves.

## System Identification for the MIMO Sine Environment

When all environments are defined aht the `Initialize Environments` button is pressed, Rattlesnake will proceed to the next phase of the test, which is defined on the `System Identification` tab.

Before the Sine environment can predict or control the response, it must estimate the transfer function matrix between the drive channels and control channels. This is done through the shared system identification workflow used by the system-ID-based environments.

A typical system identification UI for the sine environment is shown in @fig:mimo_sine_system_id.

:::{figure} figures/sine_system_identification.png
:label: fig:mimo_sine_system_id
:align: center
System identification UI used by the MIMO Random Vibration environment.
:::

Rattlesnake's system identification phase will start with a noise floor check, where the data acquisition records data on all the channels without specifying an output signal.  After the noise floor is computed, the system identification phase will play out the specified signals to the excitation devices, and transfer functions will be computed using the responses of the control channels to those excitation signals.  @sec:using_rattlesnake_system_identification describes the System Identification tab and its various parameters and capabilities.

The Sine environment then uses those transfer function to compute initial drive amplitudes and phases.

## Test Predictions for the MIMO Sine Environment

Once the system identification is available, the Sine environment can use that information in conjunction with the defined control law to compute initial excitation signals and resulting control responses. The control responses are computed by convolving the excitation signals with the impulse response (the inverse fast Fourier transform of the transfer functions).  The specified tracking filters are then used to extract the sine tone amplitudes and phases from the signals.  These predictions appear on the `Test Predictions` tab, shown conceptually in @fig:mimo_sine_prediction.

:::{figure} figures/sine_prediction.png
:label: fig:mimo_sine_prediction
:align: center

Prediction UI used by the MIMO Sine environment.
:::

The prediction UI allows the user to inspect:

- predicted excitation time histories, amplitudes, or phases over time or frequency,
- predicted filter success in extracting amplitude and phase information from response time histories,
- predicted response time histories, amplitudes, or phases over time or frequency,
- predicted peak drive voltages,
- predicted response errors,
- warning and abort threshold comparisons.

The predicted response amplitudes and phases are compared directly against the specification. If time histories are compared, they are compared against time histories synthesized from the specification.  Channels that are predicted to violate warning or abort limits are highlighted in the response error matrix.

### Excitation Prediction

The excitation display on the top portion of the window contains the following displays:

```{embed} #sec:sine_prediction:excitation_voltage_groupbox
```
```{embed} #sec:sine_prediction:auto_1
```

Users can plot drive signals, amplitude, and phase quantities over time or frequency.  Certain quantities must be plotted on a tone-by-tone basis, while others can be plotted as a superposition of all tones.

### Response Prediction

Similarly, the response prediction display on the bottom of the window contains the following displays:

```{embed} #sec:sine_prediction:response_error_groupbox
```
```{embed} #sec:sine_prediction:auto_2
```

## Running the MIMO Sine Environment

The `Run Test` tab is used to actually run the environment after system identification and prediction are complete.

A typical run page is shown in @fig:mimo_sine_run.

:::{figure} figures/sine_run.png
:label: fig:mimo_sine_run
:align: center

Run GUI used by the MIMO Sine environment.
:::

The run page allows the user to:

- select test level,
- optionally run a partial environment with certain sine tones or certain portions of the sweep time,
- monitor achieved response amplitudes and phases,
- observe drive updates from the control law,
- open plots for individual tones and channels,
- and save control data.

### Drive Updates

The @fig:sine_run:control_updates_groupbox shows the updates that the controller is applying to the open-loop drive signal based on error correction.  It is plotted as a complex amplitude, so the phase is encoded as the angle to the $x$-axis, and the amplitude is the distance from the origin.  A history of previous drive updates is plotted with lightening colors, so users can identify if the drive updates are diverging.

```{embed} #sec:sine_run:control_updates_groupbox
```

### Environment Control

The @fig:sine_run:environment_control_groupbox allows the user to define how the Sine environment will be run.  It contains the following controls:

```{embed} #sec:sine_run:environment_control_groupbox
```



### Response Amplitude and Phase

The @fig:sine_run:amplitude_groupbox and @fig:sine_run:phase_groupbox portions of the controller show the amplitude and phase extracted from the channel (column) and sine tone (row) selected in the Response Signal Selector.

```{embed} #sec:sine_run:amplitude_groupbox
```
```{embed} #sec:sine_run:phase_groupbox
```

### Response and Drive Channel Selection

The @fig:sine_run:channel_selector_groupbox and @fig:sine_run:groupBox allow the user to visualize the response amplitude error (in dB) and the drive update amplitude (in dB) respectively.  Clicking on a cell in either table will display the data for that sine tone (row) and control or drive channel (column).

```{embed} #sec:sine_run:channel_selector_groupbox
```
```{embed} #sec:sine_run:groupBox
```

### Individual Tone/Channel Displays

The @fig:sine_run:data_display_groupbox portion of the Run Test tab allows the user to create new windows showing specific sine tones and channels.

```{embed} #sec:sine_run:data_display_groupbox
```

An example window is shown in @fig:sine_channel_window.

:::{figure} figures/sine_channel_window.png
:label: fig:sine_channel_window
:align: center

A window displaying the amplitude and phase response compared to the specification.
:::

## Tracking Amplitude and Phase

A key challenge in sine control is accurately extracting the instantaneous amplitude and phase of the response which can then be compared to the specification.

Rattlesnake currently supports two main tracking approaches, the Digital Tracking Filter and the Vold-Kalman filter.

### Digital Tracking Filter (DTF)

The digital tracking filter multiplies the signal by reference sine/cosine signals and low-pass filters the result to extract:

- instantaneous in-phase content,
- instantaneous quadrature content,
- amplitude,
- phase.

This is useful for many swept-sine applications and is relatively lightweight.

### Vold-Kalman Filter (VK)

The Vold-Kalman filter provides a more selective order-tracking approach that can be especially useful when sinusoidal components are close together or when more refined tracking is required.

The tradeoff is increased computational cost and additional configuration parameters such as:

- filter order,
- bandwidth,
- block size,
- overlap.

Rattlesnake provides a filter explorer dialog to visualize the effects of the filter settings on representative specification data.

## Prediction and Control Signal Construction

After system identification, the Sine control law computes a set of complex excitation values over time/frequency that should reproduce the desired response.

The initial drive is often referred to as a **preshaped drive**.

Internally, the environment computes:

- predicted drive amplitudes,
- predicted drive phases,
- predicted response amplitudes,
- predicted response phases,
- and reconstructed time-domain drive signals.

During control, measured response amplitudes and phases are compared against the target values, and a correction is applied to the complex drive signals. The current implementation maintains complex drive corrections over tones and channels and updates future signal blocks accordingly.

This allows closed-loop correction without needing to reconstruct the entire test from scratch at every timestep.


## Saving Control Data

The Sine environment allows saving current control data from the run tab.

This typically includes:

- achieved response signals,
- achieved amplitudes,
- achieved phases,
- drive modifications,
- frequencies and arguments over time,
- target amplitudes and phases.

This data is useful for:
- debugging,
- offline analysis,
- reporting,
- or future control-law development.

## Using Transformation Matrices
Transformation matrices in the Sine environment behave identically to the the Random Vibration environment.  See @sec:rattlesnake_environments_transformation_matrices for more information.

## Writing a Custom Sine Control Law

The Sine environment supports custom control laws, typically as Python classes. These classes are expected to expose methods such as:

- `system_id_update(...)`
- `initialize_control(...)`
- `update_control(...)`
- `generate_signal(...)`
- `finalize_control(...)`

The control law is provided with:

- the sine specification,
- the transfer functions from system identification,
- the measured response amplitudes and phases,
- and any extra control parameters.

Because the Sine environment is deterministic and time-evolving, custom control laws are typically stateful and class-based rather than simple one-shot functions.

The default sine control law is implemented in `sine_sys_id_utilities.py` and is a good starting point for understanding the expected behavior and data flow.

## Output Files and Saved Data

The Sine environment uses the same broader Rattlesnake save/load infrastructure as the other environments:

- hardware and environment metadata may be written to workbook templates or netCDF files,
- system identification data may be saved and reused,
- control data may be saved from the run tab.

Unlike Random, the Sine environment is not primarily defined by CPSD matrices on the run side; instead it emphasizes:

- tone definitions,
- amplitude and phase targets,
- time-varying drive and response signals.