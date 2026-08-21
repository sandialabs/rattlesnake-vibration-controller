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

1. defines one or more sine sweeps as a specification,
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

:::{warning} Capability Under Active Development
The Sine environment and its default control law have been used on several large-scale tests successfully.  However, it is one of the least tested environments in Rattlesnake, and should be considered under active research and development.  Rattlesnake, at its heart, is research software, and its various environments have been developed to allow users to perform research into those types of environment.  The Sine environment is certainly one of those capabilities that is under active research and development; therefore, users should not expect the Sine environment to be as "polished" as other environments.  Before using the Sine environment on any test of high consequence, ensure that you thoroughly understand how the environment works and have tested it out on less consequential tests.
:::

## Governing Equations

At any instant in a sine test, the desired response of the structure may be represented as a complex response vector

\begin{equation}
\mathbf{x}(\omega) = \mathbf{a}_x(\omega)e^{i\boldsymbol{\phi}_x(\omega)}
\end{equation}

where $\mathbf{a}_x$ is the vector of desired response amplitudes and $\boldsymbol{\phi}_x$ is the vector of desired response phases.

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
In addition to the specification, there are a number of sampling and signal processing parameters that are used by the MIMO Sine environment.  These, along with the specification, are defined on the `Environment Definition` tab in the Rattlesnake controller, on a sub-tab corresponding to a MIMO Sine environment, as shown in @fig:sine_environment_definition.

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

(sec:sine_tracking_filter_parameters)=
### Tracking Filter Parameters
The @fig:sine_definition:tracking_filter_groupbox section of the MIMO Sine definition sub-tab consists of properties used to specify the tracking filter, which is used to extract sine-tone information from the measured time histories.  Depending on which filter type is selected, different parameters will be made available to the user.

```{embed} #sec:sine_definition:tracking_filter_groupbox
```

With the digital tracking filter selected, the following parameters are available:

```{embed} #sec:sine_definition__tracking_filter:tracking_filter_groupbox
```

With the Vold-Kalman filter selected, the following parameters are available:

```{embed} #sec:sine_definition__vold_kalman_filter:tracking_filter_groupbox
```

If the user is unsure as to which filter they should use, they can use the filter explorer to investigate filter performance, see @sec:sine_filter_explorer.

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

```{embed} #sec:sine_definition__sine_table_breakpoint_tab:auto_1
```

The `Warning Table` contains amplitude limits that will trigger warnings.

```{embed} #sec:sine_definition__sine_table_warning_tab:auto_1
```

The `Abort Table` contains amplitude limits that will trigger aborts.

```{embed} #sec:sine_definition__sine_table_abort_tab:auto_1
```

Both warning and abort limit tables allow for specifying the amplitude on the left and right side of the breakpoint independently.  This can allow for, for example, looser tolerances during segments of the test that have higher sweep rates.  Upper and lower tolerances can also be specified.  The warning and abort limits will be shown in the plots along with the breakpoints themselves.

## System Identification for the MIMO Sine Environment

When all environments are defined and the `Initialize Environments` button is pressed, Rattlesnake will proceed to the next phase of the test, which is defined on the `System Identification` tab.

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

A central task in the MIMO Sine environment is the extraction of the instantaneous amplitude and instantaneous phase of the measured response for each control channel and each active sine tone. These extracted quantities are used for:

- comparing the achieved response to the specification,
- computing warning and abort status,
- plotting the achieved response in the Run Test tab,
- and updating the complex drive signals during closed-loop control.

Because the Sine environment is deterministic rather than stationary, Rattlesnake does not primarily control to averaged random quantities such as CPSD matrices during a run. Instead, it estimates the response of each active tone directly in the time domain and expresses that response as an amplitude and phase evolving over time.

At a conceptual level, each tracked response component is modeled as

\begin{equation}
y(t) \approx A(t)\cos\!\bigl(\theta(t)+\phi(t)\bigr)
\end{equation}

where:

- $A(t)$ is the instantaneous amplitude,
- $\theta(t)$ is the known sinusoidal argument from the frequency breakpoint specification,
- $\phi(t)$ is the phase correction required to match the measured response.

The primary filtering problem is therefore to recover $A(t)$ and $\phi(t)$ from the measured response $y(t)$, given the known excitation or response reference phase history.

Rattlesnake currently supports two approaches for this task:

1. a Digital Tracking Filter (DTF), and
2. an Overlapped Vold-Kalman (VK) Filter.

The DTF is computationally faster and relatively intuitive. The VK filter is more mathematically sophisticated and can provide better discrimination between nearby or crossing sine tones, at the cost of additional computational expense.

### Digital Tracking Filter

The digital tracking filter operates by demodulating the signal against a known sinusoidal reference and then low-pass filtering the demodulated components.

Suppose the measured signal contains a component of interest whose known instantaneous argument is $\theta(t)$. Then the measured signal may be projected onto in-phase and quadrature reference signals:

\begin{equation}
y_0(t) = y(t)\cos\theta(t)
\end{equation}

\begin{equation}
y_{90}(t) = -y(t)\sin\theta(t)
\end{equation}

These two signals contain a slowly varying baseband component associated with the desired sine tone, along with higher-frequency content that is rejected by low-pass filtering.

After low-pass filtering, Rattlesnake obtains filtered in-phase and quadrature estimates, which may be denoted $x_0(t)$ and $x_{90}(t)$. From these, the instantaneous amplitude and phase are reconstructed as

\begin{equation}
A(t) = 2\sqrt{x_0(t)^2 + x_{90}(t)^2}
\end{equation}

\begin{equation}
\phi(t) = \operatorname{atan2}\!\bigl(x_{90}(t), x_0(t)\bigr)
\end{equation}

This is exactly the logic implemented in the digital tracking filter generator used by the Sine environment.

The DTF therefore behaves like a classical synchronous detector followed by a low-pass filter:

- the multiplication by sine/cosine translates the tone of interest to baseband,
- the low-pass filter removes the higher-frequency terms,
- and the baseband complex envelope is converted back to amplitude and phase.

#### Tracking Filter Cutoff

The Tracking Filter Cutoff parameter sets the low-pass cutoff as a fraction of the instantaneous tone frequency.

A higher cutoff:

- allows faster changes in amplitude and phase to be tracked
- rejects noise and neighboring tones less effectively

A lower cutoff:

- provides smoother amplitude and phase estimates
- responds more slowly to rapid signal changes

#### Tracking Filter Order

The Tracking Filter Order parameter sets the order of the Butterworth low-pass filter used after demodulation. Higher-order filters roll off more sharply, which can improve rejection of unwanted content, but may introduce a slower or more oscillatory transient response.

#### Digital Tracking Filter Summary

The DTF is generally a good choice when:

- tones are well separated,
- computational cost must be low,
- and rapid online execution is important.

### Vold-Kalman Filter

The Vold-Kalman Filter [@vold1995_high_resolution_order_tracking_extreme_slew_rates_using_kalman_tracking_filters],[@blough2007_understanding_kalman_vold_kalman] is a more advanced order tracking method, and includes the method to track multiple sine tones simultaneously.  The basic formulation of the VK filter is as follows.

For a given discretely-sampled signal $y(n)$ containing primarily a harmonic component, the equation for $y(n)$ can be written as

\begin{equation}
\label{eq:vk_data_equation}
y(n) = x(n)\exp(j\Theta(n)) + \eta(n)
\end{equation}

Here $x(n)$ is the instantaneous complex amplitude of the sine tone, and $\Theta(n)$ is the instantaneous phase or argument of the sine tone.  $\eta(n)$ represents an error term.  To use the VK filter, we therefore need to know the argument $\Theta(n)$ of the sine tone over time; because we can analytically construct the sine sweep frequencies from the breakpoint table in the specification, we can solve for this quantity as the integral of frequency over time.  The solution of the VK filter is therefore concerned with the identification of the complex amplitude $x(n)$ at each sample for the sine tone, as this will contain instantaneous amplitude and phase of that sine tone, which will be compared to the specified amplitudes and phases to judge test accuracy and perform closed-loop control.

In addition to the data equation @eq:vk_data_equation, the VK filter also contains structural equations which describes the mathematical characteristics of the sine tone to be extracted.  The goal in this case is that the complex envelope varies slowly over time, at least in comparison to the frequency of the sine tones themselves.  Therefore, we can write the structural equation in terms of finite difference operations on the envelope.  For first-order, second-order, and third-order filters, these equations are:

\begin{equation}
\label{eq:vk_first_order}
\nabla x(n) = x(n) - x(n+1) = \varepsilon(n)
\end{equation}
\begin{equation}
\label{eq:vk_second_order}
\nabla^2 x(n) = x(n) - 2x(n+1) + x(n+2) = \varepsilon(n)
\end{equation}
\begin{equation}
\label{eq:vk_third_order}
\nabla^3 x(n) = x(n) - 3x(n+1) + 3x(n+2) - x(n+3) = \varepsilon(n)
\end{equation}

In these equations the value $\varepsilon(n)$ represents a small term that allows for the slow modulation of the envelope.

All discrete variables from these equations can be arranged into vectors of length $N$ where $N$ is the total number of samples.

\begin{equation}
\mathbf{y} = \left[y(1), y(2), \dots, y(N)\right]^T
\end{equation}
\begin{equation}
\mathbf{x} = \left[x(1), x(2), \dots, x(N)\right]^T
\end{equation}
\begin{equation}
\boldsymbol{\eta} = \left[\eta(1), \eta(2), \dots, \eta(N)\right]^T
\end{equation}
\begin{equation}
\boldsymbol{\varepsilon} = \left[\varepsilon(1), \varepsilon(2), \dots, \varepsilon(N)\right]^T
\end{equation}

We can then set up matrix forms for the structural equations @eq:vk_structural_matrix and data equations @eq:vk_data_matrix.

\begin{equation}
\label{eq:vk_structural_matrix}
\mathbf{A}\mathbf{x} = \boldsymbol{\varepsilon}
\end{equation}

\begin{equation}
\label{eq:vk_data_matrix}
\mathbf{y}-\mathbf{C}\mathbf{x} = \boldsymbol{\eta}
\end{equation}

where the coefficient matrix $\mathbf{A}$ is

\begin{equation}
\mathbf{A} = \begin{bmatrix}
1      & -1     &  0     & \cdots & 0 & 0 \\
0      & 1      & -1     & \cdots & 0 & 0  \\
0      & 0      &  1     & \cdots & 0 & 0  \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0      & 0      &      0 & \cdots & 1 & -1
\end{bmatrix}
\end{equation}

for the first-order filter or 

\begin{equation}
\mathbf{A} = \begin{bmatrix}
1      & -2     &  1     & \cdots & 0 & 0 & 0 \\
0      & 1      & -2     & \cdots & 0 & 0 & 0 \\
0      & 0      &  1     & \cdots & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots\\
0      & 0      &      0 & \cdots & 1 & -2 & 1
\end{bmatrix}
\end{equation}

for the second-order filter, etc.  And, the coefficient matrix $\mathbf{C}$ is a diagonal matrix consisting of the complex argument phasors at each sample in time.

\begin{equation}
\mathbf{C} = \begin{bmatrix}
\exp(j\Theta(1)) & 0 & 0 & \cdots & 0 \\
0 & \exp(j\Theta(2)) & 0 & \cdots & 0 \\
0 & 0 & \exp(j\Theta(3)) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \exp(j\Theta(N)) \\
\end{bmatrix}
\end{equation}

The goal is to minimize the values $\boldsymbol{\varepsilon}$ and $\boldsymbol{\eta}$, so we can represent the vectors as a scalar product.

\begin{equation}
\boldsymbol{\varepsilon}^T\boldsymbol{\varepsilon} = \mathbf{x}^T\mathbf{A}^T\mathbf{A}\mathbf{x}
\end{equation}

\begin{equation}
\boldsymbol{\eta}^H\boldsymbol{\eta} = \left(\mathbf{y}^T - \mathbf{x}^H\mathbf{C}^H\right)\left(\mathbf{y} - \mathbf{x}\mathbf{C}\right)
\end{equation}

The weighted sum of these parameters form the loss function

\begin{equation}
\label{eq:vk_loss_function}
J = r^2\boldsymbol{\varepsilon}^T\boldsymbol{\varepsilon} + \boldsymbol{\eta}^H\boldsymbol{\eta}
\end{equation}

where $r$ is a weighting parameter that determines how heavily the structural equations are weighted compared to the data equations.

The derivative of this function with respect to $x$ set to zero gives the minimum of this function.

\begin{equation}
\frac{\partial J}{\partial x} = 2r^2\mathbf{A}^T\mathbf{A}\mathbf{x} + 2\left(\mathbf{x}-\mathbf{C}^H\mathbf{y}\right) = \mathbf{0}
\end{equation}

The solution is then

\begin{equation}
\mathbf{x} = \left(r^2\mathbf{A}^T\mathbf{A}+\mathbf{E}\right)^{-1}\mathbf{C}^H\mathbf{y}
\end{equation}

The above equations are satisfactory for a single tone.  However, when multiple tones are simultaneously present in a given signal, we can solve for all tones simultaneously.  In this case, the equation for $y(n)$ becomes

\begin{equation}
\label{eq:vk_data_equation_multitone}
y(n) = \sum_{k=1}^{P} x_k(n)\exp(j\Theta_k(n)) + \eta(n)
\end{equation}

where now there is a separate complex amplitude $x_k(n)$ and argument $\Theta_k(n)$ for each of the $P$ sine tones in the signal.

Following the same logic as above, the function to minimize becomes

\begin{equation}
J = \sum_{k=1}^{P} r^2{\boldsymbol{\varepsilon}_k}^T\boldsymbol{\varepsilon}_k + \boldsymbol{\eta}^H\boldsymbol{\eta} \\
  = \sum_{k=1}^{P} r^2{\mathbf{x}_k}^H\mathbf{A}^T\mathbf{A}\mathbf{x}_k + \left( \mathbf{y}^T - \sum_{k=1}^{P} {\mathbf{x}_k}^H{\mathbf{C}_k}^H \right)\left( \mathbf{y} - \sum_{k=1}^{P} {\mathbf{C}_k}{\mathbf{x}_k} \right)
\end{equation}

The solution is then the $\mathbf{x}_k$ for each of the $P$ sine tones that minimize this equation.  Setting the derivatives to zero gives:

\begin{equation}
\frac{\partial J}{\partial \mathbf{x}_i^H} = \mathbf{B}_i \mathbf{x}_i + \mathbf{C}_i^H + \sum_{k=1, k\ne i}^P \mathbf{C}_k\mathbf{x}_k - \mathbf{C}_i^H\mathbf{y}=\mathbf{0}
\end{equation}

where $\mathbf{B}_i = r^2\mathbf{A}^T\mathbf{A} + \mathbf{I}$ and $\mathbf{I}$ is the identity matrix.  We also note that $\mathbf{C}_i^H\mathbf{C}_i=\mathbf{I}$ due to the phases of the Hermetian diagonal phasor matrices cancelling out when multiplied.

We can assemble this into a large system of equations of the form $\mathbf{B} \mathbf{x} = \mathbf{b}$ with

\begin{equation}
\mathbf{B} = \begin{bmatrix}
\mathbf{B}_1 & \mathbf{C}_1^H\mathbf{C}_2 & \cdots & \mathbf{C}_1^H\mathbf{C}_P \\
\mathbf{C}_2^H\mathbf{C}_1 & \mathbf{B}_2 & \cdots & \mathbf{C}_2^H\mathbf{C}_P \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{C}_P^H\mathbf{C}_1 & \mathbf{C}_P^H\mathbf{C}_2 & \cdots & \mathbf{B}_P \\
\end{bmatrix}
\end{equation}

\begin{equation}
\mathbf{x} = \begin{bmatrix}
\mathbf{x}_1 \\ \mathbf{x}_2 \\ \vdots \\ \mathbf{x}_P
\end{bmatrix}
\end{equation}

\begin{equation}
\mathbf{b} = \begin{bmatrix}
\mathbf{C}_1^H\mathbf{y} \\
\mathbf{C}_2^H\mathbf{y} \\
\vdots \\ 
\mathbf{C}_P^H\mathbf{y} \\
\end{bmatrix}
\end{equation}

As the $\mathbf{B}$ matrix has a sparse, banded form, the equation is solved using the sparse solvers in SciPy's `linalg` package.  Once the envelope $\mathbf{x}$ is found, it can be split out into the individual complex amplitudes for each sine tone.  From these complex amplitudes, the amplitude and phase of the individual sine tones can be derived.

The complex envelope contains the information of interest:

\begin{equation}
A(n) = |x(n)|
\end{equation}

\begin{equation}
\phi(n) = \angle x(n)
\end{equation}

There are a few parameters that can be selected when setting up the VK filter.

#### Filter order in the Vold-Kalman method

The **Vold-Kalman Filter Order** parameter controls the order of the finite-difference smoothness constraint.

Conceptually:

- **1st order** penalizes first differences and favors slowly changing envelopes, using @eq:vk_first_order for the structural equation,
- **2nd order** penalizes curvature and often gives smoother practical behavior, using @eq:vk_second_order for the structural equation,
- **3rd order** penalizes higher-order variation and can provide still stronger smoothing, using @eq:vk_third_order for the structural equation.

Higher order generally increases filter selectivity and mathematical smoothness of the solution at the expense of additional computation.

#### Bandwidth in the Vold-Kalman method

The **Vold-Kalman Filter Bandwidth** controls how narrowly the filter tracks the desired tone.

A smaller bandwidth gives a more selective filter and improves rejection of neighboring tones and broadband noise, but it can make the response slower and may distort rapidly changing amplitude modulation.  A larger bandwidth allows faster tracking of amplitude or phase changes but reduces selectivity and can admit contamination from other content.  The bandwidth parameter is closely related to the weighting parameter $r$ in @eq:vk_loss_function, as it controls the tradeoff between smoothness of the solution and matching of the data.

#### Multiple simultaneous tones

One of the major advantages of the Vold-Kalman formulation is that multiple sinusoidal components can be solved simultaneously. This is especially important when multiple sine tones exist in the signal and is absolutely necessary if the sine tones cross in frequency.

In these cases, a simple tracking filter may struggle to separate the contributions of the different tones, while the VK formulation can solve for them together as a coupled estimation problem.  This is one of the main reasons the Sine environment offers the VK filter as an alternative to the DTF.

#### Online and Overlapped Vold-Kalman Filtering

The classical Vold-Kalman filter is most naturally formulated over a full signal block. However, Rattlesnake must use it in an online control environment, where the signal is still being acquired and the control law must update in a timely way.

To make this possible, Rattlesnake implements the VK filter in an overlapped blockwise form.

A blockwise VK solution can exhibit startup and ending transients on each analysis block. If consecutive blocks were simply stitched together end-to-end, these edge effects would become visible in the extracted amplitudes and phases.

To mitigate this, Rattlesnake:

1. divides the signal into overlapping blocks,
2. solves the VK problem on each block,
3. rejects the responses in the overlapped regions, keeping only the "center" portion of each block where the transients are minimal

The **Vold-Kalman Filter Block Size** sets the number of samples used in each VK solve.

A larger block size generally gives the filter more information, which can remove the effects of startup and ending transients.  However, this increases computational cost and decreases controller responsiveness, as the control must acquire more data prior to making a control decision.

The **Vold-Kalman Filter Block Overlap** sets the overlap fraction between consecutive blocks.

This overlap is used to reduce discontinuities between blocks and to suppress the effect of the VK startup/end transients. In practice, too little overlap can leave visible stitching artifacts due to the startup and ending transients.  Increasing overlap improves continuity between blocks; however, it increases redundant computation.

### Practical Filter Tradeoffs

From a practical standpoint, the DTF and VK filters represent a tradeoff between:

- computational cost,
- noise rejection,
- separation of nearby tones,
- transient fidelity,
- and robustness for large MIMO tests.

Digital Tracking Filter is often preferable when:

- only one tone or well-separated tones are active,
- computational budget is limited,
- the controller must remain very responsive,
- and modest noise rejection is sufficient.

Vold-Kalman Filter is often preferable when:

- tones overlap or cross,
- better selectivity is required,
- the measured response contains significant other-content or noise,
- or the user needs more reliable extraction of individual components from a crowded signal.

The DTF is relatively cheap computationally.  The VK filter can become expensive, especially because in the Sine environment the filter is applied separately for each control channel. Thus, the total computational burden scales with:

- number of control channels,
- number of simultaneously active tones,
- selected VK block size,
- selected overlap,
- selected filter order.

For that reason, a filter configuration that is acceptable for a single-channel example may become too expensive for a large multi-channel test.

This is why the filter explorer described in @sec:sine_filter_explorer is important: it gives the user the ability to investigate not only the quality of the extracted amplitude and phase, but also the practical computational behavior of the chosen filter.

(sec:sine_filter_explorer)=
### Using the Filter Explorer

The Filter Explorer is accessed by clicking the [**Explore...**](#fig:sine_definition:explore_filter_button) button on the `Environment Definition` tab.  This brings up the Filter Explorer dialog box, which is shown in @fig:sine_filter_explorer.  The Filter Explorer dialog will take the defined specification and generate a time signal that exactly matches the amplitudes and phases defined for each sine tone in the specification for the selected channel.  Noise can also be added to this perfect specification realization to represent realistic test conditions that the filter should reject.  The user can then select from filter parameters mirroring those found on the `Environment Definition` tab described in @sec:sine_tracking_filter_parameters.  The filter explorer will then attempt to filter the known signal from the specification, extract amplitude and phase information, and present those data in comparison to the known data from the specification.  This allows the user to investigate filter choices prior to running a test.

:::{figure} figures/sine_filter_explorer.png
:label: fig:sine_filter_explorer
The Filter Explorer dialog box that allows users to understand the effect of their tracking filter choices.
:::

Descriptions of the filter selection parameters are found in @sec:sine_tracking_filter_parameters.  Additional widgets and displays on the Filter Explorer dialog box include:

```{embed} #sec:sine_definition__filter_explorer_dialog:auto_1
```

## Output NetCDF File Structure

Like the other environments in Rattlesnake, the MIMO Sine environment stores its metadata in a netCDF group whose name matches the environment name. This group contains the parameters needed to reconstruct the environment definition, including the sine specifications, the control law configuration, the tracking filter settings, the shared system identification settings, and any transformation matrices.

The root netCDF dataset also contains the global hardware metadata and channel table information described is @sec:using_rattlesnake_output_files. The material in this section focuses only on the additional data stored inside the Sine environment’s group.

Because the Sine environment derives from the shared system-identification environment infrastructure, its netCDF group contains both sine-specific fields and shared system-identification fields.

### NetCDF Dimensions

The Sine environment creates the following dimensions in its netCDF group.

- **control_channels** — the number of physical control channels used by the environment.
- **response_transformation_rows** — the number of rows in the response transformation matrix, if one is defined.
- **response_transformation_cols** — the number of columns in the response transformation matrix, if one is defined.
- **reference_transformation_rows** — the number of rows in the excitation/output transformation matrix, if one is defined.
- **reference_transformation_cols** — the number of columns in the excitation/output transformation matrix, if one is defined.

### NetCDF Attributes

The following attributes are stored directly on the Sine environment’s netCDF group.

- **sysid_sample_rate** — the sample rate used during system identification.
- **sysid_frame_size** — the number of samples per frame used during system identification.
- **sysid_averaging_type** — the averaging scheme used in the system identification (`Linear` or `Exponential`).
- **sysid_noise_averages** — the number of frames used in the noise-floor characterization.
- **sysid_averages** — the number of frames used in the transfer-function measurement.
- **sysid_exponential_averaging_coefficient** — the exponential averaging coefficient when exponential averaging is selected.
- **sysid_estimator** — the estimator used for FRF computation, such as H1, H2, H3, or Hv.
- **sysid_level** — the excitation level used during system identification.
- **sysid_level_ramp_time** — the ramp time used to transition into and out of the system identification level.
- **sysid_signal_type** — the signal type used during system identification.
- **sysid_window** — the window applied to the time frames during system identification.
- **sysid_overlap** — the overlap fraction used during system identification.
- **sysid_burst_on** — the fraction of the burst-random frame that is “on,” if burst random is used.
- **sysid_pretrigger** — the fraction of the frame used as pretrigger for burst-random system identification.
- **sysid_burst_ramp_fraction** — the fraction of the burst-random “on” interval used to ramp the burst up and down.
- **sysid_low_frequency_cutoff** — the low-frequency cutoff used during system identification.
- **sysid_high_frequency_cutoff** — the high-frequency cutoff used during system identification.
- **sample_rate** — the sample rate of the environment in samples per second.
- **samples_per_frame** — the number of samples acquired per frame during the run.
- **ramp_time** — the time in seconds used to ramp the environment up from zero or back down to zero.
- **number_of_channels** — the total number of channels in the environment.
- **update_drives_after_environment** — 1 if the open-loop drives should be updated after the environment finishes, 0 otherwise.
- **phase_fit** — 1 if phase fitting is enabled, 0 otherwise.
- **control_convergence** — the scale factor used for the on-line control correction.
- **allow_automatic_aborts** — 1 if the environment is allowed to stop automatically on abort, 0 otherwise.
- **control_python_script** — the path to the Python script containing the custom control law, if one is used.
- **control_python_class** — the class name in the Python script used for the control law, if one is used.
- **control_python_parameters** — additional text parameters passed to the custom control law.
- **tracking_filter_type** — the selected tracking filter type (0 for digital tracking filter, 1 for Vold-Kalman filter).
- **tracking_filter_cutoff** — the cutoff ratio used by the digital tracking filter.
- **tracking_filter_order** — the order of the digital tracking filter.
- **vk_filter_order** — the order of the Vold-Kalman filter.
- **vk_filter_bandwidth** — the bandwidth parameter of the Vold-Kalman filter.
- **vk_filter_blocksize** — the block size used by the blockwise Vold-Kalman implementation.
- **vk_filter_overlap** — the overlap fraction used in the blockwise Vold-Kalman implementation.
- **buffer_blocks** — the number of signal-generation blocks to keep buffered during control.


### NetCDF Variables

The following variables are stored directly on the Sine environment’s netCDF group.

- **control_channel_indices** — the indices of the physical control channels in the environment.  These indices correspond to the physical channels that define the control degrees of freedom before any response transformation is applied.  Type: 32-bit integer; Dimensions: `control_channels`
- **response_transformation_matrix** — the response transformation matrix applied to the physical control channels. This variable is only present if a response transformation matrix is defined.  Type: 64-bit float; Dimensions: `response_transformation_rows` × `response_transformation_cols`
- **reference_transformation_matrix** — the output/excitation transformation matrix applied to the physical drive channels.  This variable is only present if an excitation/output transformation matrix is defined.  Type: 64-bit float; Dimensions: `reference_transformation_rows` × `reference_transformation_cols`

### The `specifications` Group

A subgroup named `specifications` is created inside the environment group. Within this group, one subgroup is created for each sine tone in the environment. The subgroup name is the name of the tone as shown in the UI.

For each tone subgroup, the following dimensions are defined:

- **num_breakpoints** — the number of frequency breakpoints in that sine tone.
- **specification_channels** — the number of control channels represented in that tone specification.
- **two** — a helper dimension of size 2 used for warning and abort matrices.

For each sine tone subgroup, the following attribute is stored:

- **start_time** — the start time of that tone in seconds relative to the overall environment.

Each tone subgroup also stores the following variables:

- **spec_frequency** — the frequency breakpoints for the tone.  Type: 64-bit float;  Dimensions: `num_breakpoints`
- **spec_amplitude** — the specified amplitude at each breakpoint for each control channel.  Type: 64-bit float; Dimensions: `num_breakpoints` × `specification_channels`
- **spec_phase** — the specified phase at each breakpoint for each control channel.  These phase values are stored in **radians** in the netCDF file, even though the external `.npz` or `.mat` sine specification files store phase in degrees.  Type: 64-bit float; Dimensions: `num_breakpoints` × `specification_channels`
- **spec_sweep_type** — the sweep type between adjacent breakpoints.  The values correspond to the internal sweep-type representation used by the Sine environment, with `0` representing a linear sweep and `1` representing a logarithmic sweep.  The last breakpoint’s stored sweep type is kept for consistency with the full breakpoint table, even though there is no following segment beyond the final breakpoint; this value is effectively ignored.  Type: 8-bit integer;  Dimensions: `num_breakpoints`
- **spec_sweep_rate** — the sweep rate associated with each breakpoint segment.  As with the sweep type, the final row is retained so the saved data matches the internal breakpoint table representation; the value is effectively ignored.  Sweep rates are stored in Hz/s for linear sweeps and octaves per minute for logarithmic sweeps.  Type: 64-bit float;  Dimensions: `num_breakpoints`
- **spec_warning** — the warning thresholds defined at each breakpoint for each control channel.  Type: 64-bit float; Dimensions: `num_breakpoints` × `two` × `two` × `specification_channels`
- **spec_abort** — the abort thresholds defined at each breakpoint for each control channel.  Type: 64-bit float; Dimensions: `num_breakpoints` × `two` × `two` × `specification_channels`

For the warning and abort limits, the meanings of the dimensions are:

- first **two** index:
  - `0` = lower warning limit
  - `1` = upper warning limit

- second **two** index:
  - `0` = left side of the breakpoint
  - `1` = right side of the breakpoint


### Interpretation of Stored Specification Data

The Sine environment stores the specification in the netCDF4 file in the same logical form as used internally during the run:

- frequency-major arrays,
- amplitude and phase per control channel,
- sweep metadata,
- warning and abort breakpoint information,
- one subgroup per sine tone.

This is slightly different from the external `.mat` and `.npz` sine specification files used when loading individual tones, which are designed to be convenient for exchange with MATLAB or NumPy workflows. In particular:

- external specification files store phase in **degrees**,
- internal/netCDF storage uses **radians**,
- and the warning/abort breakpoint arrays are transposed into the controller’s preferred breakpoint-major form.

## Saving Control Data

The Sine environment allows saving current control data from the run tab.  This occurs when the user clicks the [**Save Control Data**](#fig:sine_run:save_control_data_button) button.

Data is stored to a NumPy Archive (`*.npz`) file.

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

(sec:sine_custom_control_law)=
## Writing a Custom Sine Control Law

Unlike the Random Vibration environment, where relatively simple function-based control laws can often be sufficient, the MIMO Sine environment is inherently more stateful.  The control logic must manage:

- the prescribed response specification as a function of time,
- the system identification transfer functions,
- the initial open-loop drive synthesis,
- the online correction of those drives,
- the generation of drive signals in finite blocks,
- and optional post-run updates to the drive signal.

For this reason, the Sine environment is built around **class-based** custom control laws rather than one-shot function calls.  A custom Sine control law must therefore be implemented as a Python class with the methods described below.  The easiest way to understand the intent of these methods is to compare them to the current default control law, implemented in `rattlesnake/environment/sine_sys_id_utilities.py` as `DefaultSineControlLaw`.

A MIMO sine test is not simply a static inversion problem at one frequency.  The environment must:

- define one or more tones over time,
- compute an initial open-loop drive for those tones,
- track the achieved response while the test is running,
- update the excitation in a stable way,
- and generate the output in chunks that match the controller’s block-based acquisition and output architecture.

The control law therefore needs to maintain internal state such as:

- the current target amplitudes and phases,
- the current preshaped drive amplitudes and phases,
- the accumulated drive correction,
- write and analysis indices,
- and previously sent or measured portions of the signal.

This is why the Sine environment expects a control-law **class** rather than a simple stateless function.

### Required Control-Law Lifecycle Methods

A custom Sine control law class is expected to expose the following methods:

- `__init__(...)`
- `system_id_update(...)`
- `initialize_control(...)`
- `update_control(...)`
- `generate_signal(...)`
- `finalize_control(...)`

The environment calls these methods at different stages of the test.

#### `__init__(...)`

The constructor is called when the environment metadata are initialized and the control law is created.  It should store all persistent information required by the control law, including:

- specification information,
- sample rate,
- output oversample,
- ramp time,
- convergence factor,
- buffer sizes,
- extra user parameters,
- and any system-ID data already available.

The constructor is also a reasonable place to precompute signal representations derived from the specification, such as:

- combined time-domain target signals,
- per-tone target signals,
- breakpoint-derived instantaneous frequencies,
- target amplitudes,
- target phases.

The current default control law does exactly this.  In particular, it uses `SineSpecification.create_combined_signals(...)` to build:

- the total specified response signal,
- each tone’s response signal,
- the instantaneous frequencies,
- the cosine arguments,
- the target amplitudes,
- and the target phases.

These are stored as persistent arrays so they do not need to be recomputed repeatedly during the run.

#### `system_id_update(...)`

This method is called when the system identification data have been obtained or updated.  Its purpose is to convert the system-ID results into a usable open-loop drive estimate.

A custom implementation should use the transfer functions to compute some initial estimate of the drive amplitudes and phases required to reproduce the specified response.

In the default implementation, this method performs the following major steps:

1. stores the transfer function matrix and related system-ID quantities,
2. interpolates the FRF pseudoinverse onto the instantaneous frequencies of the specification,
3. computes the desired complex response
   $$
   \mathbf{x}(t) = \mathbf{a}_x(t)e^{i\boldsymbol{\phi}_x(t)}
   $$
4. maps that desired response into a complex drive estimate
   $$
   \mathbf{v}(t) = \mathbf{H}_{xv}^{+}(t)\mathbf{x}(t)
   $$
5. extracts the resulting drive amplitudes and phases,
6. reconstructs the corresponding time-domain preshaped drive signals.

This stage is where the default control law solves for the open-loop drive signals that would produce the best estimate of the desired response, assuming the transfer functions are measured exactly.

Any replacement control law must do something equivalent, even if the exact mathematical method differs.

#### `initialize_control(...)`

This method is called at the start of a particular run.  It is responsible for initializing the runtime state needed for online control.

Typical responsibilities include:

- selecting which tones are active in this run,
- selecting the time region of interest,
- initializing write and analysis indices,
- computing ramp-up and ramp-down portions of the signal,
- preparing the first signal block to send to output,
- initializing storage for future drive corrections,
- resetting any history buffers.

The current default control law uses this method to:

- select the controlled tones and time interval,
- define ramped portions of both the target and the preshaped drive,
- prepare the first portion of the drive signal,
- initialize `control_drive_correction`,
- initialize arrays that store sent and achieved signal data.

This method returns the first excitation signal block that will be sent to the output process.

#### `update_control(...)`

This is the key online feedback method.  It is called when a new portion of the measured response has been analyzed and reduced to:

- tracked time histories,
- amplitudes,
- phases,
- frequencies,
- and a time-delay estimate.

A custom control law should compare the achieved response to the target response over the current block and update its internal drive-correction state.

The default implementation does this by:

1. storing the achieved response data,
2. identifying the corresponding time block within the full specification,
3. reconstructing the target complex response over that block,
4. reconstructing the achieved complex response over that block,
5. computing the complex error,
6. projecting that error back through the transfer function pseudoinverse,
7. scaling the result by a convergence factor,
8. accumulating the resulting correction into `self.control_drive_correction`.

Conceptually, the default implementation is applying a correction of the form

\begin{equation}
\Delta \mathbf{v}(t) \propto \mathbf{H}_{xv}^{+}(t)\left(\mathbf{x}_{\mathrm{target}}(t)-\mathbf{x}_{\mathrm{achieved}}(t)\right)
\end{equation}

and accumulating this correction across blocks.

Any custom control law that performs closed-loop sine control must implement an analogous update rule, even if the exact form of the correction differs.

#### `generate_signal(...)`

This method is called whenever the signal generation process needs another block of output samples.

It should:

- determine the next signal block time range,
- retrieve the nominal preshaped excitation over that block,
- apply any accumulated corrections,
- convert the complex drive representation back into a real time-domain signal,
- enforce output limits if needed,
- return the next signal block,
- and indicate whether the signal generation is complete; there is no more data to generate.

In the default implementation, this method forms a complex excitation signal of the form

\begin{equation}
\mathbf{v}_{\mathrm{block}}(t)
=
\mathbf{a}_v(t)e^{i\boldsymbol{\phi}_v(t)}
+
\Delta\mathbf{v}
\end{equation}

and then converts it into a real time signal using cosine reconstruction.

It also applies an optional drive-voltage limit by clipping the magnitude of the complex excitation vector.

This is the method that turns the control law’s internal state into actual samples to be output by the hardware.

#### `finalize_control(...)`

This method is called after the run has completed.  It allows the control law to perform any final processing or to prepare updated preshaped drive information for later use.

In the current default implementation, this method primarily returns the current drive signals and associated metadata:

- preshaped drive signals,
- frequencies,
- arguments,
- amplitudes,
- phases,
- ramp sample count.

A more advanced custom implementation could use this stage to:

- update future open-loop drives,
- compress or archive state,
- compute statistics for the UI,
- or refine the starting signal for the next run.

### What the Default Control Law Does

The current `DefaultSineControlLaw` in `sine_sys_id_utilities.py` can be understood as having the following stages.

#### Stage 1: Parse and store the specification

In `__init__`, the default control law:

- stores the sine specifications,
- computes the combined target signals,
- computes instantaneous frequency, argument, amplitude, and phase trajectories,
- and determines the active tone slices.

This establishes the full target trajectory that the environment will attempt to realize.

#### Stage 2: Ingest the system identification results

In `system_id_update`, the default law:

- stores FRFs and related noise/coherence information,
- computes the pseudoinverse of the FRF matrix,
- interpolates that pseudoinverse onto the tone trajectories,
- and computes preshaped drive amplitudes and phases.

This stage generates the initial open-loop estimate of the excitation.

#### Stage 3: Build preshaped drive signals

Still within `system_id_update`, the default implementation reconstructs the corresponding time-domain preshaped drives by applying the amplitude and phase trajectories to the tone arguments.

These preshaped drives are what the environment uses as the baseline excitation before any online correction is applied.  This is the signal used to develop the test predictions shown on the `Test Predictions` tab.

#### Stage 4: Initialize runtime control state

In `initialize_control`, the default control law:

- selects the active tones,
- defines the run interval,
- sets up ramp-up and ramp-down data,
- seeds the write and analysis indices,
- prepares the first excitation signal block,
- and initializes the correction state.

This transitions the control law from an open-loop prediction object into a live online controller.

#### Stage 5: Update based on measured response

In `update_control`, the default implementation:

- stores the measured response data,
- constructs the target complex response for the current block,
- constructs the achieved complex response for the same block,
- computes the blockwise complex error,
- maps that error back through the interpolated FRF pseudoinverse,
- scales the update by the convergence factor,
- and accumulates the correction.

This is the core online feedback stage.

#### Stage 6: Generate the next output block

In `generate_signal`, the default control law:

- takes the next preshaped excitation block,
- adds the accumulated complex correction,
- clips amplitudes if a maximum drive limit is configured,
- converts the result to a real time-domain drive block,
- and returns it.

This continues until all requested signal data have been generated.

#### Stage 7: Finalize

In `finalize_control`, the default law returns the final drive predictions and associated arrays so they can be reused or displayed after the run.

### What a Custom Sine Control Law Must Do

A replacement control law should preserve the same broad responsibilities, even if the mathematics differ.

At minimum, a custom implementation should be able to:

1. accept and store specification information,
2. accept updated system identification results,
3. compute an initial open-loop drive estimate,
4. initialize per-run state,
5. compare achieved response to target response during the run,
6. compute and accumulate drive corrections,
7. generate output signal blocks,
8. optionally finalize or update future drive predictions after the run.

In other words, even if a custom control law uses a completely different control strategy, it still needs to participate correctly in the same lifecycle as the default implementation.

### Practical Guidance for Implementing a Replacement

#### Preserve state explicitly

The Sine control problem is inherently stateful. A custom class should explicitly track:

- target signal arrays,
- preshaped drive arrays,
- current write and analysis indices,
- accumulated corrections,
- and any other information needed between method calls.

#### Be careful about coordinate systems

The control law may be operating on:

- physical control channels,
- transformed response coordinates,
- physical drive channels,
- transformed drive coordinates.

A custom implementation should ensure that the transfer functions, specification, and drive outputs are all interpreted in compatible coordinates.

#### Respect blockwise execution

The environment acquires and outputs data in finite blocks. A custom control law must therefore be designed to work incrementally, not just as a single full-signal solve.

#### Use tracked data, not raw data

By the time `update_control(...)` is called, the environment has already extracted:

- amplitudes,
- phases,
- frequencies,
- and aligned time-domain information.

Most custom control laws should work with those reduced quantities rather than trying to reinterpret the raw time histories from scratch.

#### Think carefully about output limits

The default implementation optionally clips the magnitude of the complex excitation to a maximum drive voltage. A replacement control law should decide explicitly:

- how actuator saturation is handled,
- whether clipping is acceptable,
- and what effect that clipping has on stability or convergence.

### Summary

A custom Sine control law in Rattlesnake is best understood as a **stateful object managing the full sine-test lifecycle**, not merely as a function that computes one correction.

The default implementation provides a working reference for:

- how to convert system ID into initial drives,
- how to track achieved response,
- how to compute blockwise corrections,
- and how to generate real output signals.

When implementing a replacement control law, the most important thing is not to duplicate the exact mathematics of the default implementation, but rather to preserve the required lifecycle:

- initialize,
- update from system ID,
- initialize the run,
- update from measured response,
- generate output,
- finalize.

Any custom control law that satisfies that lifecycle correctly can be integrated into the Sine environment.