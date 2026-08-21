---
numbering:
  heading_2:
    start: 15
  figure:
    enumerator: 15.%s
  table:
    enumerator: 15.%s
  equation:
    enumerator: 15.%s
  code:
    enumerator: 15.%s
---
# Multiple Input/Multiple Output Shock Control

(sec:mimo_shock)=
# Multiple Input/Multiple Output Shock Control

The MIMO Shock environment in Rattlesnake is currently implemented as a sum-of-decayed-sines (SDS) shock environment. Its purpose is to generate one or more drive signals whose resulting measured responses match a specified shock response spectrum (SRS) at one or more control channels.

The environment supports:

- multiple control channels,
- multiple excitation channels,
- transformed control and excitation coordinates,
- user-specified or synthesized decayed-sine tables,
- prediction of response time histories and SRS,
- run-time hit counting and history tracking,
- and optional automatic between-hit updates to the SDS table,

:::{warning} Capability Under Active Development
The MIMO Shock / SDS environment is under active research and development. The infrastructure for environment definition, prediction, run-time hit execution, shock history tracking, and control-data saving is in place, but users should still consider this capability to be evolving. As with the Sine environment, users should validate settings carefully in low-risk conditions before applying them to important hardware.
:::

## Governing Equations

The SDS environment represents a shock signal as a sum of exponentially decaying sinusoids. For one drive channel, the excitation may be written as

\begin{equation}
v(t) = \sum_{k=1}^{N} A_k e^{-\zeta_k \omega_k (t-\tau_k)} \sin\!\bigl(\omega_k (t-\tau_k)\bigr)
\end{equation}

for $t \ge \tau_k$, where:

- $A_k$ is the amplitude of the $k$-th decayed sine,
- $\zeta_k$ is its decay parameter,
- $\omega_k = 2\pi f_k$ is its angular frequency,
- $\tau_k$ is its delay.

For multiple drive channels, one such sum is constructed per drive channel.

The resulting measured control response $\mathbf{x}(t)$ is related to the drives through the structural dynamics of the system. In the frequency domain this is still governed by the transfer function matrix

\begin{equation}
\mathbf{X}(\omega) = \mathbf{H}_{xv}(\omega)\mathbf{V}(\omega)
\end{equation}

where:

- $\mathbf{H}_{xv}(\omega)$ is the FRF matrix from drive voltages to control responses,
- $\mathbf{V}(\omega)$ is the complex drive vector at frequency $\omega$,
- $\mathbf{X}(\omega)$ is the complex response vector at frequency $\omega$.

Instead of controlling directly to the transient $\mathbf{x}(t)$ or its frequency response $\mathbf{X}(\omega)$, the goal is to generate a transient $\mathbf{x}(t)$ whose SRS matches the specified SRS at the control channels.

If a measured or predicted response time history is denoted $x_i(t)$ for control channel $i$, then the environment computes its SRS

\begin{equation}
S_i(f)
\end{equation}

using a bank of single-degree-of-freedom oscillators with specified damping and response type.

The SDS control problem therefore becomes:

1. choose the sine-tone frequencies $f_k$,
2. choose drive amplitudes $A_k$,
3. choose drive decays $\zeta_k$,
4. choose drive delays $\tau_k$,

such that the measured response SRS approximately matches the specification at each control channel.

The current default control law begins from the MIMO inverse idea:

\begin{equation}
\mathbf{V}(\omega) = \mathbf{H}_{xv}^{+}(\omega)\mathbf{X}_{\text{target}}(\omega)
\end{equation}

where $\mathbf{X}_{\text{target}}(\omega)$ is chosen to have the desired response amplitudes together with optimized response phases. These complex drive values are then converted into decayed-sine amplitudes and delays.

Because an SRS is not itself a complex spectrum, this requires an additional synthesis step: a response SRS target is first converted into a family of decayed sinusoids whose resulting transient approximately reproduces that SRS. The transfer functions are then used to solve for the corresponding drive signals that produce those desired decayed sine response signals.

## Specification Definition

The SDS environment specification is defined in terms of a **shock response spectrum** (SRS) rather than a CPSD or deterministic transient signal.

The specification consists of:

- a list of SRS frequencies,
- an SRS amplitude at each frequency for each control channel,
- optional lower SRS limits,
- optional upper SRS limits,
- and a target number of hits.

Internally this is represented by a `SpecParameters` object, which stores:

- `frequencies`
- `srs_spec`
- `srs_lower_limit`
- `srs_upper_limit`
- `num_hits`

These data define the required response at the control channels.

### Specification File Format

Rather than entering the breakpoint table manually, which for large channel-count and tone-count tests would be tedious, the user can load the specification from an external file.  The SDS environment currently loads specification data from a NumPy archive (`*.npz`) file or a MATLAB (`*.mat`) file.

Both NumPy and MATLAB files are structured identically, with the same field names and shapes associated with each field.  Note that for 1D arrays, MATLAB can either specify $n \times 1$ or $1 \times n$ arrays; Rattlesnake will squeeze out the extra dimension.

Suppose the specification contains $n_f$ SRS frequencies and $n_c$ control channels, then the specification file should contain the following fields:

- **f** — a one-dimensional array of length $n_f$ containing the SRS frequencies in Hz
- **srs** — a two-dimensional array of shape $n_f \times n_c$ containing the desired SRS amplitudes
- **lower_limit** — a two-dimensional array of shape $n_f \times n_c$ containing the lower SRS limits
- **upper_limit** — a two-dimensional array of shape $n_f \times n_c$ containing the upper SRS limits
- **num_hits** — a scalar integer specifying the requested number of hits at the current selected level

Any frequency/channel combination for which no control or no limit is desired may be represented with `NaN`.

For any array with dimension of size $n_c$, the ordering of this dimension must be identical to the ordering of the control degrees of freedom in the environment loading the file.  No bookkeeping or reordering of specification data to match the channel data occurs in the SDS environment.  If a transformation matrix is used, then the ordering of this dimension must be identical to the rows of the transformation matrix.

This is the same general rule used elsewhere in Rattlesnake: specifications are always defined in the actual control coordinates of the environment.

## Defining the MIMO Shock Environment in Rattlesnake

The SDS environment definition page allows the user to define:

- sampling and block parameters,
- how the sine-tone frequencies are chosen,
- whether and how a compensation pulse is used,
- how the sine-tone decays are defined,
- the SRS computation settings,
- the SDS synthesis iteration settings,
- the control law definition,
- the control channels,
- transformation matrices,
- and the SRS specification itself.

A representative definition page is shown in @fig:mimo_sds_definition.

:::{figure} figures/srs_sds_definition.png
:label: fig:mimo_sds_definition
:align: center

UI used to define the MIMO Shock / SDS environment.
:::

### Sampling Parameters

The @fig:srs_sds_definition:sampling_groupbox section determines the sample rate and the transient block size used for one SDS hit.

```{embed} #sec:srs_sds_definition:sampling_groupbox
```

The block size is important because it defines the duration available for the decayed-sine signal, and the sample rate is important to sufficiently extract the maximum response in the for the SRS computation.  Various documentation, such as MIL-STD-810 [@MIL-STD-810H], suggest a sample rate of 10x the largest frequency of the shock response spectrum in order to adequately resolve the maximum response.

### Sine Tone Definition

The SDS environment supports multiple strategies for defining the frequencies used in the decayed-sine table, in the @fig:srs_sds_definition:tones_groupbox portion of the window.  This portion also defines if a compensation pulse is used.  The compensation pulse is represented internally as an additional decayed-sine row with its own frequency and decay.  This is useful in some SDS synthesis workflows to improve the shape or baseline behavior of the transient, often to create a signal with zero net velocity or displacement, so there is not a "snap-back" of the shaker system when the signal ends.

```{embed} #sec:srs_sds_definition:tones_groupbox
```

The default option is to specify the @fig:srs_sds_definition__from_spec_tones:sine_tone_groupbox simply using the frequency breakpoints from the specification SRS as the sine tones in the SDS.  In this case, the sine tone table is only for display purposes.

```{embed} #sec:srs_sds_definition__from_spec_tones:sine_tone_groupbox
```

A second option is to specify the the @fig:srs_sds_definition__octave_tones:tones_groupbox by octave.  This will involve specifying the minimum and maximum frequency as well as the number of sine tones per octave.

```{embed} #sec:srs_sds_definition__octave_tones:tones_groupbox
```

The final approach is to manually specify the @fig:srs_sds_definition__manual_tones:sine_tone_groupbox.  In this case, the sine tone table will be editable, and options to add or remove tones will become available.

```{embed} #sec:srs_sds_definition__manual_tones:sine_tone_groupbox
```

### Decay Definition

The environment supports several equivalent ways to define decay:

- damping ($\zeta$ in some literature),
- time constant ($\tau$ in some literature),
- number of time constants in the block.

Internally these are converted into the damping-style decay values used during synthesis.

The environment supports either one common decay for all tones, or one decay value per tone.

Decay values are specified in the @fig:srs_sds_definition:decay_groupbox portion of the window.

```{embed} #sec:srs_sds_definition:decay_groupbox
```

If a common decay is used, the @fig:srs_sds_definition__common_decay:decay_groupbox portion will have the additional widget

```{embed} #sec:srs_sds_definition__common_decay:decay_groupbox
```

If decays are specified per sine tone, then they can be entered in the @fig:srs_sds_definition__per_tone_decay:sine_tone_groupbox portion of the window.

```{embed} #sec:srs_sds_definition__per_tone_decay:sine_tone_groupbox
```

### SRS Parameters

The SRS section defines how the response SRS is computed. This includes:

- SRS type,
- absolute vs. relative displacement convention,
- damping.

These parameters are used both for:

- prediction, and
- measured-hit postprocessing.

### SDS Synthesis Parameters

The SDS synthesis settings control the iterative decayed-sine synthesis algorithm. These include:

- number of iterations,
- convergence,
- scale factor,
- error tolerance.

These are used when generating a decayed-sine representation that approximates the desired SRS.

### Control Law Definition

Like other advanced environments, SDS supports loading a custom Python control law. The definition page allows the user to specify:

- the Python module,
- the control object (typically a function),
- the control-law type,
- and any additional control parameters.

The default control law uses a MIMO inverse-based approach and is described later in the custom-control-law section.

### Control Channels

The control-channel list defines which channels are used to evaluate and control the SRS response. The ordering of these channels determines the ordering of the specification columns when no transformation is used.

### Transformation Matrices

The SDS environment supports both response and excitation transformations. These work the same way conceptually as in the other environments. A response transformation maps physical control channels into virtual control channels, and an excitation transformation maps physical drive channels into virtual drive signals.

See @sec:rattlesnake_environments_transformation_matrices for the shared transformation-matrix workflow.

### Test Specification

The SRS specification itself is entered or loaded in the Test Specification section of the definition page. The user can:

- add or remove breakpoints,
- edit the required SRS values,
- edit lower and upper limits,
- set the target number of hits.

The lower and upper limit tables are presented on separate tabs and therefore are documented through page states.

## System Identification for the MIMO Shock Environment

Like the Random, Sine, and Transient environments, the SDS environment uses a system identification phase to estimate the transfer functions between the drive channels and control channels.

These transfer functions are needed because the environment must map desired response behavior back into the corresponding drive signals.

A representative system identification UI is shared with the other system-ID environments and is described in detail in @sec:using_rattlesnake_system_identification.

The SDS environment uses the system ID results to populate a `SysIdDataPackage`, which includes:

- FRFs,
- coherence,
- response CPSDs,
- reference CPSDs,
- noise-floor spectra,
- and associated frequencies.

The default SDS control law uses the FRFs directly when solving the MIMO inverse problem for the drive amplitudes and phases.

## Test Prediction for the MIMO Shock Environment

Once system identification is complete, the SDS environment can compute a prediction of the drive signals and the resulting response.

A representative prediction page is shown in @fig:mimo_sds_prediction.

:::{figure} figures/srs_sds_prediction.png
:label: fig:mimo_sds_prediction
:align: center

Prediction UI used by the MIMO Shock / SDS environment.
:::

The prediction page and associated run-table dialog allow the user to inspect:

- the current SDS table,
- synthesized drive time histories,
- predicted response time histories,
- predicted response SRS,
- peak drive voltages,
- peak response errors relative to the SRS specification.

### Prediction table

The SDS prediction UI is centered around a table of decayed sine terms. For each excitation channel, the table stores:

- frequency
- amplitude
- decay
- delay

These parameters fully define the synthesized drive transient for that drive channel.

### Prediction plots

The prediction page displays:

- synthesized drive time histories,
- response time histories,
- predicted response SRS overlaid against the specification and any limits.

These plots allow the user to inspect whether the current open-loop or updated SDS table is likely to meet the specification.

## Running the MIMO Shock Environment

The `Run Test` tab of the SDS environment is where actual shock hits are executed and tracked.

A representative run page is shown in @fig:mimo_sds_run.

:::{figure} figures/srs_sds_run.png
:label: fig:mimo_sds_run
:align: center

Run GUI used by the MIMO Shock / SDS environment.
:::

The SDS run workflow differs from the Random and Sine environments because the natural unit of operation is **a hit**, not a continuously running stationary control loop.

The SDS run mode supports:

- manual single-hit execution,
- automatic repeated hits,
- hit counting,
- hit history,
- run-time SDS table use,
- optional automatic updates of the SDS table,
- and post-hit response visualization.

### Hit-based operation

A single SDS hit consists of:

1. constructing a transient drive waveform from the current SDS table,
2. playing that transient through the outputs,
3. measuring the responses,
4. aligning the measured drive and response to the expected transient,
5. computing the response SRS,
6. optionally updating the SDS table through the control law.

This is fundamentally different from the continuously updating loop used by Random Vibration.

### Manual mode

In manual mode, each press of **Start Environment** performs exactly one hit and then returns to idle.

This is useful when dialing in the transient carefully or when the operator wants explicit control over each impact.

### Automatic mode

In automatic mode, one press of **Start Environment** begins a sequence of repeated hits separated by the requested interval. The sequence continues until:

- the requested number of hits at the selected test level has been reached, or
- the operator presses **Stop Environment**.

If post-hit computations take longer than the requested interval, the next hit is simply launched as soon as the computations finish.

### Hit counters and history

The SDS run page tracks:

- total number of hits,
- number of hits at the currently selected test level,
- progress toward the requested target hit count,
- a full shock history dialog.

Unlike a hardcoded “0 dB only” notion of target-level hits, the SDS environment now interprets “hits at level” relative to the currently selected run test level. Thus, if the test level is set to $-3$ dB, then the displayed hit count and automatic stop logic both operate on the number of historical hits performed at $-3$ dB.

## Shock History

The Shock History dialog provides an overview of what has been done to the test article, including:

- total hits,
- hits at the selected level,
- number of distinct test levels used,
- a histogram of hits by level,
- a chronology plot of hit level versus hit number,
- an optional detailed table of every hit.

This is especially useful when many lower-level “dial-in” hits are performed before full-level hits.

## Run-Time SDS Table

The SDS Run Table dialog allows the user to:

- inspect the current run SDS table,
- manually edit or load SDS tables,
- allow or disallow manual updates,
- allow or disallow automatic control-law updates,
- visualize predicted and measured response quantities.

This dialog is particularly important because the SDS environment is table-driven: the current SDS table defines the transient that will be played on the next hit.

## Tracking Response Quality

The SDS run page presents several useful response metrics, including:

- measured response SRS,
- measured response time history,
- peak output voltage by drive channel,
- worst-case dB response error per control channel,
- a global response plot showing all measured control-channel SRS curves.

The response error list is compared directly against the specification and warning/abort limits. When a measured SRS violates a limit, the corresponding entry is highlighted.

## Writing a Custom SDS Control Law

The SDS environment supports custom control laws through a Python function or class, but the current default implementation is a useful reference because it demonstrates the full chain from:

- specification SRS
- to response target construction
- to MIMO inverse
- to drive amplitudes, phases, and delays.

A custom SDS control law is expected to produce updated SDS table quantities:

- amplitudes,
- decays,
- delays

for each decayed-sine term and drive channel.

### What the default SDS control law does

The current default control law, implemented in `sds_sys_id_control_law.py`, proceeds in several stages.

#### Stage 1: Build the target response SRS

The control law begins from the target SRS stored in the environment metadata. This target is defined for each control channel at the SDS frequencies.

#### Stage 2: Generate a decayed-sine representation of the target response

For each control channel, the helper routines synthesize a decayed-sine signal whose resulting SRS approximates the target SRS. This produces:

- sine frequencies,
- provisional response amplitudes,
- decays,
- delays.

#### Stage 3: Interpolate the FRF matrix

The measured system-identification FRFs are interpolated onto the SDS frequencies.

#### Stage 4: Solve a MIMO inverse with optimized response phases

At each SDS frequency, the control law solves a MIMO inverse problem using the transfer function matrix and an optimized set of response phases. This produces a complex drive vector for that frequency.

This is done using a pseudoinverse-based solve with phase-target optimization to balance:

- response accuracy,
- and drive effort.

#### Stage 5: Convert complex drives into amplitudes and delays

The complex drive values are converted into:

- drive amplitudes,
- drive phases,

and the phases are converted to delays via

\begin{equation}
\tau = -\frac{\phi}{2\pi f}
\end{equation}

#### Stage 6: Optionally include a compensation pulse row

If a compensation pulse is enabled, the returned SDS table includes an additional row for it.

### What a replacement SDS control law must do

A custom SDS control law should, at minimum, be able to:

1. accept environment metadata and system-ID data,
2. use the SRS specification as the response target,
3. map that target into drive quantities through the FRFs,
4. return amplitudes, decays, and delays in the table format expected by the environment.

During a run, if automatic table updates are enabled, the environment may call the control law after a completed hit using:

- the measured response SRS,
- the measured response time history,
- the previous SDS table,
- and the measured drive history.

Thus a replacement control law can be either:

- a largely open-loop constructor of an SDS table, or
- an iterative between-hit updater.

## Output NetCDF File Structure

Like the other environments in Rattlesnake, the SDS environment stores its metadata in a netCDF group whose name matches the environment name.

Because the SDS environment derives from the shared system-identification infrastructure, its netCDF group contains:

- SDS-specific metadata,
- shared system-ID metadata,
- and, when saving control data, the current SDS table, most recent hit data, and hit history.

### NetCDF Dimensions

The SDS environment defines dimensions associated with:

- the control channels,
- the transformed control channels,
- the SDS frequencies,
- the drive channels,
- the specification channels,
- and the hit-history length if control data are saved.

More specifically, dimensions may include:

- **control_channels** — the number of physical control channels.
- **specification_channels** — the number of specification/control channels after transformation.
- **tone_data_size** — the number of tone-definition values used when tones are specified explicitly.
- **num_decays** — the number of explicitly stored decay values if decays are not common across tones.
- **num_frequencies** — the number of SRS frequency lines in the specification.
- **num_spec_signals** — the number of control/specification channels in the SRS specification.
- **sds_frequencies** — the number of frequencies in the run SDS table when saving control data.
- **sds_drive_channels** — the number of drive channels in the run SDS table when saving control data.
- **hit_history_length** — the number of historical hits when saving control data.
- plus transformation-matrix dimensions if transformations are present.

### NetCDF Attributes

The SDS environment group stores both shared system-ID attributes and SDS-specific attributes.

#### Shared system-identification attributes

The group stores the standard system-ID metadata such as:

- **sysid_sample_rate**
- **sysid_frame_size**
- **sysid_averaging_type**
- **sysid_noise_averages**
- **sysid_averages**
- **sysid_exponential_averaging_coefficient**
- **sysid_estimator**
- **sysid_level**
- **sysid_level_ramp_time**
- **sysid_signal_type**
- **sysid_window**
- **sysid_overlap**
- **sysid_burst_on**
- **sysid_pretrigger**
- **sysid_burst_ramp_fraction**
- **sysid_low_frequency_cutoff**
- **sysid_high_frequency_cutoff**

These have the same meanings as in the other system-ID-capable environments.

#### SDS-specific attributes

The SDS environment additionally stores:

- **block_size** — the number of samples in one SDS hit block.
- compensation-pulse settings through the compensation-pulse subgroup,
- SRS settings through the SRS subgroup,
- decay settings through the decay subgroup,
- control-law settings through the control subgroup,
- synthesis iteration settings through the SDS subgroup.

When saving current control data after or during a run, additional scalar attributes are also stored, including:

- **current_test_level_db** — the run test level in dB,
- **current_test_level_scale** — the corresponding linear scale factor,
- **total_hits** — cumulative number of hits,
- **hits_at_target** — cumulative number of hits at the currently selected test level,
- **allow_automatic_updates** — whether the run table is being automatically updated.

### NetCDF Variables and Subgroups

The SDS metadata are organized into several logical subgroups.

#### Tone parameters subgroup

The subgroup **tone_parameters** stores:

- **strategy** (attribute) — the tone-generation strategy:
  - from specification,
  - octave spacing,
  - or manual tones.
- **tone_data** (variable, optional) — the associated tone-definition values.

#### Compensation pulse subgroup

The subgroup **compensation_pulse_parameters** stores:

- **use_compensation_pulse** — whether a compensation pulse is enabled,
- **compensation_frequency** — compensation frequency if specified explicitly,
- **compensation_decay** — compensation decay value.

#### Decay parameters subgroup

The subgroup **decay_parameters** stores:

- **decay_strategy**An unknown exception has occurred.