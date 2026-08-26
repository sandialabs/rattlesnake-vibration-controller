---
numbering:
  heading_2:
    start: 13
  figure:
    enumerator: 13.%s
  table:
    enumerator: 13.%s
  equation:
    enumerator: 13.%s
  code:
    enumerator: 13.%s
---
# Multiple Input/Multiple Output Transient Control

(sec:mimo_transient)=
# Multiple Input/Multiple Output Transient Control

The MIMO Transient environment aims to control the vibration response of a component to a specific time history by creating output signals with the correct levels and phasing.  The governing equation for MIMO Transient control is 
    
\begin{equation}\label{eq:forward_mimo_transient}
\mathbf{X} = {\mathbf{H}_{xv}}\mathbf{V}
\end{equation}

where the frequency spectrum matrix of the responses $\mathbf{X}$ result from some signals $\mathbf{V}$ exciting the structure represented by transfer function matrices $\mathbf{H}_{xv}$.  In a typical transient control problem, the control system tries to compute the a signal with spectrum matrix $\mathbf{V}$ that best reproduces the desired response with frequency spectrum $\mathbf{X}$.

## Signal Definition
The first step in defining a transient control problem is the definition of the response signal that is desired.  Rattlesnake accepts the specification in the form of a 2D array consisting of a time response of one or more channels in the test.  Signals can be loaded from Numpy `*.npy` or `*.npz` files or Matlab `*.mat` files.  Both Matlab `*.mat` and Numpy `*.npz` files should contain the following data members:


* **signal** A $n_c \times n_s$ array containing the time signal the controller will try to reproduce on the test article.  For `*.npy` files which do not have fields, the signal array is stored directly to the file.
* **t** A $n_s$ array of times corresponding to columns of the `signal` field.  If not specified (either by not including a `t` field in a `*.mat` or `*.npz` file or by using an `*.npy` file), the `signal` will be assumed to be at the sample rate defined in the controller.  If `t` is specified but is not at the sample rate of the controller, the signal will be linearly interpolated to be at the sample rate of the controller.

  
The ordering of the rows of the array defining the signal is the same order as the control channels in the Channel Table on the `Data Acquisition Setup` tab that are selected as control channels on the `Environment Definition` tab.  The specification is defined in the engineering units specified by the `Engineering Unit` column of the channel table for the control channels.

## Defining the MIMO Transient Environment in Rattlesnake

In addition to the specification, there are a number of signal processing parameters that are used by the MIMO Transient environment.  These, along with the specification, are defined on the `Environment Definition` tab in the Rattlesnake controller on a sub-tab corresponding to a MIMO Transient environment.  @fig:transientenvironmentdefinition shows a MIMO Transient sub-tab.  The following subsections describe the parameters that can be specified, as well as their effects on the analysis.

:::{figure} figures/transient_environment_definition.png
:label: fig:transientenvironmentdefinition
:align: center
GUI used to define a MIMO Transient environment.
:::

### Signal Parameters
The `Signal Parameters Parameters` section of the MIMO Transient definition sub-tab consists of the following parameters:

* **Sample Rate** Sample rate of the data acquisition system.  This is for display only; the sample rate is set on the `Data Acquisition Setup` tab
* **Signal Time** The time in seconds that the signal will take to generate
* **Signal Samples** The number of samples in the loaded signal
* **Ramp Time** If the environment is cancelled manually, this is the time taken to reduce the signal to zero to prevent a hard stop of the excitation system, which could damage test equipment or articles.

### Control Parameters
The `Control Parameters` section of the MIMO Transient definition sub-tab consists of the following parameters:

* **Input Channels** A display showing the total number of physical channels this environment is measuring, including excitation channels and control channels.
* **Output Channels** A display showing the total number of physical channels this environment is outputting to excitation devices such as vibration shakers.
* **Control Channels** A display showing the total number of physical channels this environment is controlling to.

The `Control Parameters` section of the MIMO Transient definition sub-tab also includes functionality for loading in custom control laws.  See @sec:rattlesnake_environments_custom_control_law_transient for information on defining a custom transient control law.
    
* **Control Python Script** Python script used to specify the control law.
* **Load** Opens a file dialog to load in a Python script containing the control law.
* **Control Python Function** Selects the function, generator function, or class in the Python script to use as the control law.
* **Control Type** Select if the selected control law is a Function, Generator, Class, or Interactive Class.  This should be detected automatically by inspection; users should not have to adjust this.
* **Control Parameters** Any additional parameters needed by the control law are entered in this text box. It is up to the control law to prescribe what is needed to be defined in this box. The data entered into this box will be passed to the control law as a string to the "extra_parameters" argument. Control laws should parse this string to extract needed information.

### Control Channels
The `Control Channels` list allows users to select the channels in the test that will be used by the environment to perform control calculations.  These are the channels that will match the rows and columns of the specification file.

* **Control Channels** Channels that are checked will be used as the control channels for this environment.  The control channels should be ordered in the specification the same way they are ordered in this list.  For example, the first row and column of the specification CPSD matrix will correspond to the first checked channel in this list.
* **Check Selected** When clicked, any selected channels in the Control Channels list will be checked, and therefore used as control channels in the environment.
* **Uncheck Selected** When clicked, any selected channels in the Control Channels list will be unchecked, and therefore not used as control channels in the environment.

### Control and Drive Transforms
The `Control and Drive Transforms` section of the MIMO Random Vibration definition sub-tab consists of the following parameters:

* **Transformation Matrices...** Open the transformation matrix dialog to allow specification of transformations to virtual control or virtual excitation channels.  See @sec:rattlesnake_environments_transformation_matrices for more information.
* **Transform Controls** A display showing the number of virtual control channels in the environment due to transformation matrices applied to the physical control channels.
* **Transform Outputs** A display showing the number of virtual excitation channels in the environment due to transformation matrices applied to the physical excitation channels.

Note that if Transformation matrices are defined, the number of control channels ends up being the number of rows of the `Response Transformation Matrix`, rather than the number of physical control channels.  The number of physical control channels will be equal to the number of columns of the transformation matrix.  The number of rows of the specification loaded should be equal to the number of rows in the transformation.

### Signal Specification
The control signal is loaded and displayed on the right side of the MIMO Transient definition tab:

* **Signal Filename** Filename from which the control signal was loaded
* **Load Signal** Opens a file dialog allowing the user to select a file containing the signals that will be used as the control targets for this environment.
* **Specification Plot** Displays the time signals that the environment will try to reproduce on the control channels.  Individual signals can be displayed or hidden by checking the relevant box in the Specification Signal Table.
* **Specification Signal Table** A table showing the different signals defined in the specification.  Individual signals can be shown or hidden by checking or unchecking the check box in the Show? column.
* **Show All** Shows all signals in the specification plot
* **Show None** Hides all channels in the specification plot


## System Identification for the MIMO Transient Environment

When all environments are defined and the `Initialize Environments` button is pressed, Rattlesnake will proceed to the next phase of the test, which is defined on the `System Identification` tab.

MIMO Transient requires a system identification phase to compute the matrix $\mathbf{H}_{xv}$ used in the control calculations of equation @eq:forward_mimo_vibration.  Unlike the MIMO Random Vibration environment, the MIMO Transient environment's system identification phase can have a number of samples per measurement frame that is not equal to the length of the signal.  Control laws written for this environment must be able to handle this, either by doing a COLA scheme or interpolating the FRF.  @fig:transientenvironmentsystemidentification shows the GUI used to perform this phase of the test.  Given the nature of the MIMO Transient environment, it may be useful to visualize impulse response functions as the system identification is proceeding.

:::{figure} figures/transient_environment_system_identification.png
:label: fig:transientenvironmentsystemidentification
:align: center
System identification GUI used by the MIMO Transient environment.
:::

Rattlesnake's system identification phase will start with a noise floor check, where the data acquisition records data on all the channels without specifying an output signal.  After the noise floor is computed, the system identification phase will play out the specified signals to the excitation devices, and transfer functions will be computed using the responses of the control channels to those excitation signals.  @sec:using_rattlesnake_system_identification describes the System Identification tab and its various parameters and capabilities.

## Test Predictions for the MIMO Transient Environment

Once the system identification is performed, a test prediction will be performed and results displayed on the `Test Predictions` tab, shown in @fig:transientenvironmenttestprediction.  This is meant to give the user an idea of the test feasibility.  The top portion of the window displays excitation information, including peak signal levels required as well as the excitation time history that will be output.  The bottom portion of the window displays the predicted responses compared to the specification as well as the TRAC between the signals.  The TRAC is a metric that compares two time signals, and has a value of 1 if the signals are identical or 0 if the signals are not related.

:::{figure} figures/transient_environment_test_prediction.png
:label: fig:transientenvironmenttestprediction
:align: center
Test prediction GUI which gives the user some idea of the test feasibility.
:::

In the Output Voltages (Peak) section of the window:

* **Output Voltages (Peak)** Displays the peak voltage that will be supplied to each of the excitation channels in the environment based on the prescribed control law.

* **Output Voltage Prediction Plot** Displays the selected voltage signal that will be output to the excitation devices to achieve the desired response signals.
* **Go to Maximum Voltage** Select the excitation signal with the largest peak voltage and display it in the Output Voltage Prediction plot.
* **Output Signal Selector** Select which signal should be plotted on the Output Voltage Prediction Plot.
* **Go to Minimum Voltage** Select the excitation signal with the smallest peak voltage and display it in the Output Voltage Prediction plot.

In the Response Error (TRAC) section of the window:

* **Response Error (TRAC)** Time Response Assurance Criterion between the predicted response and the desired response.  Note that the TRAC may be sensitive to small time delays in the signal, so a poor TRAC value does not immediately mean a terrible test.
* **Response Prediction Plot** Displays the selected response signal.  Both the desired time signal and the predicted time signal will be plotted.
* **Go to Maximum TRAC** Select the response signal with the largest TRAC and display it in the Response Prediction plot.
* **Response Signal Selector** Select which signal should be plotted on the Response Prediction Plot.
* **Go to Minimum TRAC** Select the response signal with the smallest TRAC and display it in the Response Prediction plot.

* **Recompute Predictions** Recompute predictions by running the control law again.  If a closed loop control law is implemented, the previous predictions will be used as the "measured" data and a closed-loop update will be performed.

## Running the MIMO Transient Environment
The MIMO Transient environment is then run on the `Run Test` tab of the controller.

With the data acquisition system armed, the environment can be started manually with the `Start Environment` button.  Once running, it can be stopped manually with the `Stop Environment` button.  With the data acquisition system armed and the environment run, the GUI looks like @fig:transientenvironmentruntest.

:::{figure} figures/transient_environment_run_test.png
:label: fig:transientenvironmentruntest
:align: center
GUI for running the MIMO Transient environment.
:::

* **Start Environment** Start the environment to control to the desired transient signal
* **Stop Environment** Stop the environment manually.  The signal will ramp down using the Ramp Time specified on the Environment Definition tab to prevent a hard stop that could potentially damage test equipment or articles.

There are operations that can be performed when setting up and running the MIMO Transient environment, and many visualization operations as well.

### Test Level
The MIMO Transient Environment allows scaling of the test by modifying the `Signal Level` selector on the `Run Test` tab.  The `Signal` specifies the scaling in decibels relative to the specification level, which is 0 dB.  Note that all data and visualizations on the `Run Test` window are scaled back to full level, so users should not be surprised if for example the signals shown in the `Outputs` or `Responses` plots do not change significantly with test level.  Unlike the MIMO Random Vibration environment, the MIMO Transient environment does not allow changing test level while the environment is active.

* **Signal Level** A scale factor applied to the target signals.

### Repeating a Signal
The MIMO Transient environment also has the capability to repeat a signal continuously if the `Repeat Signal` checkbox is checked.  If this is the case, users will be required to stop the environment by clicking the `Stop Environment` button.  If the controller is not set to repeat, the environment will stop automatically after the signal has been output one time.

* **Repeat Signal** If checked, the signal will continually repeat until manually stopped.  If unchecked, the environment will automatically stop when the signal is done.

### Test Metrics and Visualizations
The MIMO Transient environment displays the output signals in the `Outputs` plot and the responses to those signals at the control channels in the `Responses` plot.  The controller will identify the position of the control signal in the response signal by using a correlation, and will draw a black box around that portion of the response signal, as shown in @fig:transientenvironmentruntest.  The errors in the response channels are reported as a TRAC value between the signal measured and the specification desired.

* **Response Error (TRAC)** Time Response Assurance Criterion between the measured and desired signals.
* **Outputs Plot** Displays the recently acquired output signals.  For long transients, displaying too much data on the screen at once can cause performance issues, so the amount of data plotted can be specified with the Display Duration selector.
* **Responses Plot** Displays the recently acquired control signals.  For long transients, displaying too much data on the screen at once can cause performance issues, so the amount of data plotted can be specified with the Display Duration selector.
* **Display Duration** Sets the maximum amount of data that will be displayed on the plot to improve performance for long transients.  Setting this value to zero will display the entirety of the signals.

To interrogate specific channels, the `Data Display` section of the `Run Test` can be used.  The specific control channel to visualize can be selected using `Control Channel` selector.  Pressing the `Create Window` button then creates the specified plot.

* **Control Channel** Selects an individual response channel to visualize.
* **Create Window** Create a window to show the signal specified by the Control Channel selector

Some convenience operations are also included to visualize all channels.  The `Show All Channels` button will bring up one window per control channel.  Be aware that showing a large number of channels can easily overwhelm the computer with plotting operations causing the GUI to become unresponsive, so use this operations with caution.  @fig:transientenvironmentchannelvisualizations shows an example displaying all channels for a test with six control degrees of freedom. 

:::{figure} figures/transient_environment_channel_visualizations.png
:label: fig:transientenvironmentchannelvisualizations
:align: center
Visualizing individual channels' time response.
:::

* **Show All Control Channels** Creates a new window for all control channels so each control channel can be visualized independently against the specification.

Further convenience operations are available in the `Window Operations:` section.  Pressing `Tile All Windows` will rearrange all channel windows neatly across the screen.  Pressing `Close All Windows` will close all open channel windows.

* **Tile All Windows** Tiles all visualization windows across the main monitor
* **Close All Windows** Closes all existing visualization windows

## Output NetCDF File Structure
When Rattlesnake saves data to a netCDF file, environment-specific parameters are stored in a netCDF group with the same name as the environment name.  Similar to the root netCDF structure described in @sec:using_rattlesnake_output_files, this group will have its own attributes, dimensions, and variables, which are described here.

### NetCDF Dimensions

* **signal_samples** The number of time samples in the specification signal provided to the MIMO Transient environment.
* **specification_channels** The number of channels in the specification signal provided to the MIMO Transient environment.
* **control_channels** The number of physical channels used for control.  Note that this may be different from the `specification_channels` due to the presence of a transformation matrix.
* **response_transformation_rows** The number of rows in the response channel transformation.  This is not defined if no response transformation is used.
* **response_transformation_cols** The number of columns in the response channel transformation.  This is not defined if no response transformation is used.
* **output_transformation_rows** The number of rows in the output transformation.  This is not defined if no output transformation is used.
* **output_transformation_cols** The number of columns in the output transformation.  This is not defined if no output transformation is used.

### NetCDF Attributes

* **sysid_frame_size** The number of samples per measurement frame in the system identification
* **sysid_averaging_type** The type of averaging used in the system identification, linear or exponential
* **sysid_noise_averages** The number of measurement frames acquired for the noise floor calculation
* **sysid_averages** The number of measurement frames acquired for the system identification calculation
* **sysid_exponential_averaging_coefficient** The weighting coefficient used for new frames in the exponential averaging scheme
* **sysid_estimator** The FRF estimator used to compute the transfer functions during the system identification
* **sysid_level** The level used by the system identification in volts RMS.
* **sysid_level_ramp_time** The time to ramp up to the test level when starting and ramp back to zero when stopping the system identification
* **sysid_signal_type** The signal type used by the system identification
* **sysid_window** The window function applied to the time data during the system identification
* **sysid_overlap** The overlap fraction between measurement frames used for system identification
* **sysid_burst_on** The fraction of a measurement frame that a burst is active for burst random excitation during system identification
* **sysid_pretrigger** The fraction of a measurement used as a pre-trigger for burst random excitation during system identification
* **sysid_burst_ramp_fraction** The fraction of a measurement frame used to ramp the burst up to full level and back to zero
* **test_level_ramp_time** The time to ramp to the test level and back to zero
* **control_python_script** The path to the Python script used to control the MIMO Random Vibration environment
* **control_python_function** The function (or class or generator function) in the Python script used to control the MIMO Transient environment
* **control_python_function_type** The type of the object used for the control law (function, generator, or class)
* **control_python_function_parameters** The extra parameters passed to the control law.

### NetCDF Variables

* **control_signal** The control signal used by the MIMO Transient environment  Type: 64-bit float; Dimensions: `specification_channels` $\times$ `signal_samples`
* **response_transformation_matrix** The response transformation matrix.  This is not defined if no response transformation is used.  Type: 64-bit float; Dimensions: `response_transformation_rows` $\times$ `response_transformation_cols`
* **output_transformation_matrix** The output transformation matrix.  This is not defined if no output transformation is used.  Type: 64-bit float; Dimensions: `output_transformation_rows` $\times$ `output_transformation_cols`
* **control_channel_indices** The indices of the active control channels in the environment.  Type: 32-bit int; Dimensions: `control_channels`

### Saving Control Data

In addition to time streaming, Rattlesnake's MIMO Transient environment can also save the previous control data directly to the disk by clicking the `Save Current Control Data`.  The control data is stored in a NetCDF file similar to the time streaming data.  The primary difference is that the control data is time-aligned to the specification, so users don't need to worry about extracting the signal from an a time stream and trying to align it with specification.  The control data also saves out the information from the system identification, including FRF and CPSD matrices.

* **Save Current Control Data** Opens a file dialog to save the current control data to a file.

The two additional dimensions are:

* **drive_channels** The number of drive channels active in the environment.
* **fft_lines** The number of frequency lines in the spectral quantities.

There are also several additional variables to store the spectral data:

* **frf_data_real** The real part of the most recently computed value for the transfer functions between the excitation signals and the control response signals.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `drive_channels`
* **frf_data_imag** The imaginary part of the most recently computed value for the transfer functions between the excitation signals and the control response signals.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `drive_channels`
* **frf_coherence** The multiple coherence of the control channels computed during the test.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels`
* **response_cpsd_real** The real part of the CPSD matrix at the control channels from the system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `specification_channels`
* **response_cpsd_imag** The imaginary part of the CPSD matrix at the control channels from the system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `specification_channels`
* **drive_cpsd_real** The real part of the CPSD matrix at the excitation channels from the system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `drive_channels` $\times$ `drive_channels`
* **drive_cpsd_imag** The imaginary part of the CPSD matrix at the excitation channels from the system identifications.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `drive_channels` $\times$ `drive_channels`
* **response_noise_cpsd_real** The real part of the CPSD matrix at the control channels during the noise floor measurement that occurred during system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `specification_channels`
* **response_noise_cpsd_imag** The imaginary part of the CPSD matrix at the control channels during the noise floor measurement that occurred during system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `specification_channels` $\times$ `specification_channels`
* **drive_noise_cpsd_real** The real part of the CPSD matrix at the excitation channels during the noise floor measurement that occurred during system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `drive_channels` $\times$ `drive_channels`
* **drive_noise_cpsd_imag** The imaginary part of the CPSD matrix at the excitation channels during the noise floor measurement that occurred during system identification.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `drive_channels` $\times$ `drive_channels`
* **control_response** The time response of the system aligned in time to the specification. Type: 64-bit float; Dimensions: `specification_channels` $\times$ `signal_samples`
* **control_drives** The drive signals aligned in time to the specification.  Type: 64-bit float; Dimensions: `drive_channels` $\times$ `signal_samples`

(sec:rattlesnake_environments_custom_control_law_transient)=
## Writing a Custom Control Law
The flexibility of the Rattlesnake framework is highlighted by the ease in which users can implement and iterate on their own ideas.  For the MIMO Transient control type, users can implement custom control laws using a custom Python function, or alternatively a generator function or class which allow state to be maintained between function calls.  This section will provide instructions and examples for implementing a custom control law.

The controller will provide various data types to the control law functions which are:

  * `sample_rate` -- The sample rate of the acquisition portion of the controller; real scalar integer
  * `specification_signals` -- The target signal for the control channels; real 2D array ($n_c \times n_s$)
  * `frequency_spacing` -- The frequency spacing of the transfer function array; real scalar float
  * `transfer_function` -- The current estimate of the transfer function between the control responses and the output voltages; complex 3D array ($n_f \times n_c \times n_o$)
  * `noise_response_cpsd` -- The levels and correlation of the noise floor measurement on the control channels obtained during system identification; complex 3D array ($n_{f} \times n_{c}\times n_{c}$)
  * `noise_reference_cpsd` -- The levels and correlation of the noise floor measurement on the excitation channels obtained during system identification; complex 3D array ($n_{f} \times n_{o}\times n_{o}$)
  * `sysid_response_cpsd` -- The levels and correlation of the control channels obtained during system identification; complex 3D array ($n_{f} \times n_{c}\times n_{c}$)
  * `sysid_reference_cpsd` -- The levels and correlation of the noise floor measurement on the excitation channels obtained during system identification; complex 3D array ($n_{f} \times n_{o}\times n_{o}$)
  * `multiple_coherence` -- The multiple coherence for each control channel; real 2D array ($n_f \times n_c$)
  * `frames` -- The number of measurement frames acquired so far, used to compute various parameters in the control law.  This can be compared to `total_frames` to determine if a full set of measurement frames has been acquired, or if the estimation of the various parameters could improve with continued averaging; real scalar integer
  * `total_frames` -- The total number of frames used to compute the CPSD and FRF matrices; real scalar integer
  * `output_oversample_factor` -- Some hardware devices and settings within Rattlesnake require output signals from Rattlesnake to be oversampled compared to the input samples.  For example, on the LAN-XI hardware, the slowest output rate is four times larger than the slowest acquisition rate.  This argument tells the control law the factor by which the output returned from the controller is oversampled compared to the acquisition; real scalar integer    
  * `extra_parameters` -- Extra parameters provided to the controller; string
  * `last_excitation_signals` -- The most recent output signal, which can be used for error-based control; real 2D array ($n_{o} \times n_{s}$)
  * `last_response_signals` -- The most recent responses to the last output signals, which can be used for error-based control; real 2D array ($n_{c} \times n_{s}$)

where size $n_s$ is the number of time samples in the control signal, $n_f$ is the number of frequency lines in the transfer function matrix, $n_c$ is the number of control channels, and $n_o$ is the number of output signals.  Note that the values passed into the function may be defined using arbitrary variable names (e.g. `transfer_function` may be instead called `H`, or any other valid variable name); however, the order of the variables passed into each function will always be constant.

### Defining a control law using a Python function
Python functions are the simplest approach to define a custom control law that can be used with the Rattlesnake software; however, they are limited in that a function's state is completely lost when a function returns.  Still, they can be used to implement relatively complex control laws as long as no state persistence is required.

A Python function used to define a MIMO Transient control law in Rattlesnake would have the following general structure within a Python script.

```{code} python
:caption: General Python function structure for defining a custom Transient control law called `control_law` in Rattlesnake
:label: lst:control_function_structure_transient
# Any module imports, initialization code, or helper functions would go here

# Now we define the control law.  It always receives the same arguments from the controller.
def control_law(
    sample_rate,
    specification_signals, # Signal to try to reproduce
    frequency_spacing,
    transfer_function, # Transfer Functions
    noise_response_cpsd,  # Noise levels and correlation 
    noise_reference_cpsd, # from the system identification
    sysid_response_cpsd,  # Response levels and correlation
    sysid_reference_cpsd, # from the system identification
    multiple_coherence, # Coherence from the system identification
    frames, # Number of frames in the CPSD and FRF matrices
    total_frames, # Total frames that could be in the CPSD and FRF matrices
    output_oversample_factor, # Oversample factor to output
    extra_parameters = '', # Extra parameters for the control law
    last_excitation_signals = None, # Last excitation signal for drive-based control
    last_response_signals = None, # Last response signal for error correction
    ):

    # Code to perform the control would go here
    # output_signal = ...

    # Finally, we need to return an output signal matrix
    return output_signal
```

The function must return an `output_signal`, which is a 2D array with size ($n_{o} \times n_{os}$) where $n_{os}$ is the number of samples in the specification signal times the `output_oversample_factor`.

One example is shown to demonstrate how a Transient control law may be written.

#### Pseudoinverse Control
Perhaps the simplest strategy to perform MIMO Transient control is to simply invert the transfer function matrix to recover the least-squares solution of the optimal output signal from the desired responses.  This example will demonstrate that approach with some additional arguments that may improve the control.

The mathematics for this control strategy are relatively simple; pre-multiply the spectra of the desired responses $\mathbf{X}$ by the pseudoinverse ($^+$) of the transfer function matrix $\mathbf{H}_{xv}$.  This calculation is performed for each frequency line.

\begin{equation}
    \mathbf{V} = {\mathbf{H}_{xv}}^+\mathbf{X}
\end{equation}

The implementation details are a bit more complex than this formula would reveal, as the controller gives and wants returned the time signals and not the spectra.  This means the FFT must be computed to transform to the frequency domain for the calculations to be performed, and the IFFT must be computed to return to the time domain.  Additionally, to handle the potential for oversampling the output, the drive signal is zero-padded to up-sample the signal appropriately.  We must also handle the case where the number of samples in the system identification is not the same as the number of samples in the control signal.  We will finally add optional arguments that may be specified using key-value pairs separated by a colon (e.g. the `rcond` parameter could be specified as `rcond:1e-5` to specify a singular value threshold of 10,000 in the inverse).  In Python code, the above mathematics and implementation details would look like

```{code} python
:caption: Computing the pseudoinverse calculation to solve for a least-squares output spectrum
:label: lst:pseudoinverse_computation_transient
import numpy as np # Import numpy to get access to its function
# Parse the input arguments in extra_parameters
rcond = 1e-15
zero_impulse_after = None
# Split it up into lines
for entry in extra_parameters.split('\n'):
    try:
        # For each entry, split the key from the value using the colon
        field,value = entry.split(':')
        # Strip any whitespace
        field = field.strip()
        # Check the field to figure out which value to assign
        if field == 'rcond':
            rcond = float(value)
        elif field == 'zero_impulse_after':
            zero_impulse_after = float(value)
        else:
            # Report if we cannot understand the parameter
            print('Unrecognized Parameter: {:}, skipping...'.format(field))
    except ValueError:
        # Report if we cannot parse the line
        print('Unable to Parse Line {:}, skipping...'.format(entry))

# Compute impulse responses using the IFFT of the transfer function
# We will zero pad the IFFT to do interpolation in the frequency domain
# to match the length of the required signal
impulse_response = np.fft.irfft(transfer_function,axis=0)

# The impulse response should be going to zero at the end of the frame,
# but practically there may be some gibbs phenomenon effects that make the
# impulse response noncausal.  If we zero pad, this might be wrong.  We
# therefore give the use the ability to zero out this non-causal poriton of
# the impulse response.
if zero_impulse_after is not None:
    # Remove noncausal portion
    impulse_response_abscissa = np.arange(impulse_response.shape[0])/sample_rate
    zero_indices = impulse_response_abscissa > zero_impulse_after
    impulse_response[zero_indices] = 0
    
# Zero pad the impulse response to create a signal that is long enough for
# the specification signal
added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) 
                       + impulse_response.shape[1:])
full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

# Compute FRFs using the FFT from the impulse response.  This is now
# interpolated such that it matches the frequency spacing of the specification
# signal
interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

# Perform convolution by frequency domain multiplication
signal_fft = np.fft.rfft(specification_signals,axis=-1)
# Invert the FRF matrix using the specified rcond parameter
inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=rcond)
# Multiply the inverted FRFs by the response spectra to get the drive spectra
drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

# Zero pad the drive FFT to oversample to the output_oversample_factor
drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
    np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)
             +drive_signals_fft.shape[1:])),axis=0)

# Finally, take the IFFT to get the time domain signal.  We need to scale
# by the output_oversample_factor due to how the IFFT is normalized.
drive_signals_oversampled = np.fft.irfft(
    drive_signals_fft_zero_padded.T,axis=-1)*output_oversample_factor
return drive_signals_oversampled
```

For users not familiar with Python and its numeric library `numpy`, the following points are clarified

  * `numpy` is imported and assigned to the alias `np`, which lets us just type in `np` rather than the longer name `numpy` when we want to access `numpy` functions.
  * To interpolate a function in the frequency domain, we can zero-pad its time response, and to interpolate a function in the time domain, we can zero-pad its frequency response.
  * The `numpy` module has a function `zeros` to create an array of zeros and a function `concatenate` to combine two arrays together.  These two functions allow us to zero-pad an array.
  * FFT calculations can be performed using the Fourier Transform package `fft` within `numpy`.  We will be using real signals, so we use `rfft` and `irfft` to do these transforms.
  * The `numpy` pseudoinverse function `pinv` is stored in the linear algebra package `linalg` within `numpy`, therefore to access `pinv`, we need to call `np.linalg.pinv`
  * The `pinv` can perform a pseudoinverse on "stacks" of matrices, so even though we are only calling the `pinv` function once, it is actually performing the pseudoinverse over all frequency lines
  * The `numpy` `einsum` function utilizes a syntax similar to Einstein Summation Notation to perform multiplication and summation over certain dimensions of the arrays.


Wrapping the above code into the function definition from @lst:control_function_structure_transient, the control law can be defined as
    
```{code} python
:caption: A pseudoinverse transient control law that can be loaded into Rattlesnake
:label: lst:pseudoinverse_transient_control
import numpy as np

def pseudoinverse_control(
        sample_rate,
        specification_signals, # Signal to try to reproduce
        frequency_spacing,
        transfer_function, # Transfer Functions
        noise_response_cpsd,  # Noise levels and correlation 
        noise_reference_cpsd, # from the system identification
        sysid_response_cpsd,  # Response levels and correlation
        sysid_reference_cpsd, # from the system identification
        multiple_coherence, # Coherence from the system identification
        frames, # Number of frames in the CPSD and FRF matrices
        total_frames, # Total frames that could be in the CPSD and FRF matrices
        output_oversample_factor, # Oversample factor to output
        extra_parameters = '', # Extra parameters for the control law
        last_excitation_signals = None, # Last excitation signal for drive-based control
        last_response_signals = None, # Last response signal for error correction
        ):
    # Parse the input arguments in extra_parameters
    rcond = 1e-15
    zero_impulse_after = None
    # Split it up into lines
    for entry in extra_parameters.split('\n'):
        try:
            # For each entry, split the key from the value using the colon
            field,value = entry.split(':')
            # Strip any whitespace
            field = field.strip()
            # Check the field to figure out which value to assign
            if field == 'rcond':
                rcond = float(value)
            elif field == 'zero_impulse_after':
                zero_impulse_after = float(value)
            else:
                # Report if we cannot understand the parameter
                print('Unrecognized Parameter: {:}, skipping...'.format(field))
        except ValueError:
            # Report if we cannot parse the line
            print('Unable to Parse Line {:}, skipping...'.format(entry))

    # Compute impulse responses using the IFFT of the transfer function
    # We will zero pad the IFFT to do interpolation in the frequency domain
    # to match the length of the required signal
    impulse_response = np.fft.irfft(transfer_function,axis=0)

    # The impulse response should be going to zero at the end of the frame,
    # but practically there may be some gibbs phenomenon effects that make the
    # impulse response noncausal.  If we zero pad, this might be wrong.  We
    # therefore give the use the ability to zero out this non-causal poriton of
    # the impulse response.
    if zero_impulse_after is not None:
        # Remove noncausal portion
        impulse_response_abscissa = np.arange(impulse_response.shape[0])/sample_rate
        zero_indices = impulse_response_abscissa > zero_impulse_after
        impulse_response[zero_indices] = 0
        
    # Zero pad the impulse response to create a signal that is long enough for
    # the specification signal
    added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) 
                           + impulse_response.shape[1:])
    full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

    # Compute FRFs using the FFT from the impulse response.  This is now
    # interpolated such that it matches the frequency spacing of the specification
    # signal
    interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

    # Perform convolution by frequency domain multiplication
    signal_fft = np.fft.rfft(specification_signals,axis=-1)
    # Invert the FRF matrix using the specified rcond parameter
    inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=rcond)
    # Multiply the inverted FRFs by the response spectra to get the drive spectra
    drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

    # Zero pad the drive FFT to oversample to the output_oversample_factor
    drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
        np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)
                 +drive_signals_fft.shape[1:])),axis=0)

    # Finally, take the IFFT to get the time domain signal.  We need to scale
    # by the output_oversample_factor due to how the IFFT is normalized.
    drive_signals_oversampled = np.fft.irfft(
        drive_signals_fft_zero_padded.T,axis=-1)*output_oversample_factor
    return drive_signals_oversampled
```
    
The requirement that transient control laws are required to be able to oversample their output and may require switching between frequency and time domains mean that the Transient control laws will generally be more complex than the Random Vibration control laws.  Note that these computations to oversample the output are still performed by the controller for the Random Vibration environment described in @sec:mimo_random; however, the user does not need to handle them explicitly.  In the Transient environment, the user-defined control laws are actually creating the time signals that will be sent to the exciters, so user-defined control laws must handle these computations.

### Defining a control law using state-persistent approaches

Similarly to the MIMO Random Vibration environment, the Transient environment may also use state-persistent approaches such as generator functions or classes.  This can be handy, for example, to compute the FFT of the desired response signals one time, as opposed to computing them every single control iteration, because the desired response signals generally do not change throughout a test.

#### Defining Control Laws with Generator Functions

The second strategy to define a control law in Rattlesnake is to use a Generator Function.  A generator function is simply a function that maintains its internal state between function calls.  @lst:pseudoinverse_transient_generator shows the Pseudoinverse Transient control law implemented as a generator function.
    
```{code} python
:caption: A Pseudoinverse Transient control law defined using a Python generator function to allow for state persistence
:label: lst:pseudoinverse_transient_generator
import numpy as np

def pseudoinverse_control_generator():
    signal_fft = None
    inverted_frf = None
    drive_signals_oversampled = None
    while True:
        (sample_rate,
         specification_signals, # Signal to try to reproduce
         frequency_spacing,
         transfer_function, # Transfer Functions
         noise_response_cpsd,  # Noise levels and correlation 
         noise_reference_cpsd, # from the system identification
         sysid_response_cpsd,  # Response levels and correlation
         sysid_reference_cpsd, # from the system identification
         multiple_coherence, # Coherence from the system identification
         frames, # Number of frames in the CPSD and FRF matrices
         total_frames, # Total frames that could be in the CPSD and FRF matrices
         output_oversample_factor, # Oversample factor to output
         extra_parameters, # Extra parameters for the control law
         last_excitation_signals, # Last excitation signal for drive-based control
         last_response_signals, # Last response signal for error correction
         ) = yield drive_signals_oversampled
        if signal_fft is None:
            # Compute the FFT of the spec if it hasn't been done yet
            signal_fft = np.fft.rfft(specification_signals).T
        # Get a tolerance if specified
        rcond = 1e-15
        zero_impulse_after = None
        for entry in extra_parameters.split('\n'):
            field,value = entry.split(':')
            field = field.strip()
            if field == 'rcond':
                rcond = float(value)
            elif field == 'zero_impulse_after':
                zero_impulse_after = float(value)
            else:
                print('Unrecognized Parameter: {:}'.format(field))
        if inverted_frf is None:
            # Compute impulse responses
            impulse_response = np.fft.irfft(transfer_function,axis=0)

            if zero_impulse_after is not None:
                # Remove noncausal portion
                impulse_response_abscissa = np.arange(impulse_response.shape[0])/sample_rate
                zero_indices = impulse_response_abscissa > zero_impulse_after
                impulse_response[zero_indices] = 0
                
            # Zero pad the impulse response to create a signal that is long enough
            added_zeros = np.zeros((specification_signals.shape[-1]-impulse_response.shape[0],) + impulse_response.shape[1:])
            full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

            # Compute FRFs
            interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

            # Perform convolution in frequency domain
            inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=rcond)
            
        drive_signals_fft = np.einsum('ijk,ki->ij',inverted_frf,signal_fft)

        # Zero pad the FFT to oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(output_oversample_factor-1)+1,)+drive_signals_fft.shape[1:])),axis=0)

        drive_signals_oversampled = np.fft.irfft(drive_signals_fft_zero_padded.T,axis=-1)*output_oversample_factor
```


Note that the generator function itself is not called with any arguments, as the initial function call simply starts up the generator.  Also note that there is no `return` statement in a generator function, only a `yield` statement.   When a program requests the next value from a generator, the generator code proceeds until it hits a `yield` statement, at which time it pauses and waits for the next value to be requested.  During this pause, all internal data is maintained inside the generator function.  The `yield` statement also accepts new data into the generator function, so this is where the same arguments used to define a control law using a Python function are passed in to the generator control law.  Therefore, by creating a while loop inside a generator function, the generator can be called infinitely many times to deliver data to the controller.
    
To implement the Pseudoinverse control as shown in @lst:pseudoinverse_transient_generator several parameters are initialized as `None`, which enables the generator to check whether or not they have been computed yet.  If they have, then there is no need to compute them again.

#### Defining Control Laws using Classes

A final way to implement more complex control laws is using a Python class.  This approach allows for the near infinite flexibility of Python's object-oriented programming at the expense of more complex syntax.  Users not familiar with Python's object-oriented programming paradigms are encouraged to \href{https://docs.python.org/3/tutorial/classes.html}{learn more} about the topic prior to reading this section.

A class in Python is effectively a container that can have functions and properties stored inside of it, so it provides a good way to encapsulate all the parameters and helper functions associated with a given control law into one place.  A class allows for arbitrary properties to be stored within it, so arbitrary data can be made persistent between function calls.

For the Rattlesnake implementation of a control law, the class must have at a minimum three functions defined.  These are the class constructor `__init__` that is called when the class is instantiated, a `system_id_update` function that is called upon completion of the System Identification portion of the controller, and a `control` function that actually computes the output CPSD matrix.  A general class structure is shown below

    
```{code} python
:caption: Structure for a class defining a transient control law in Rattlesnake
# Any module imports or constants would go here    

class ControlLawClass:
    def __init__(self,
                 sample_rate,
                 specification_signals, # Signal to try to reproduce
                 output_oversample_factor, # Oversample factor to output
                 extra_parameters, # Extra parameters for the control law
                 frequency_spacing,
                 transfer_function, # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 last_excitation_signals = None, # Last excitation signal for drive-based control
                 last_response_signals = None, # Last response signal for error correction
                 ):
        # Code to initialize the control law would go here
    
    def system_id_update(self,
                         frequency_spacing,
                         transfer_function, # Transfer Functions
                         noise_response_cpsd,  # Noise levels and correlation 
                         noise_reference_cpsd, # from the system identification
                         sysid_response_cpsd,  # Response levels and correlation
                         sysid_reference_cpsd, # from the system identification
                         multiple_coherence, # Coherence from the system identification
                         frames, # Number of frames in the CPSD and FRF matrices
                         total_frames, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Code to update the control law with system identification information would go here
    
    def control(self,
                last_excitation_signals = None, # Last excitation signal for drive-based control
                last_response_signals = None, # Last response signal for error correction
                ) -> np.ndarray:
        # Code to perform the actual control operations would go here
        
    # Any helper functions or properties that belong with the class would go here
    
```

The class's `__init__` constructor function is called whenever a class is instantiated (e.g. `control_law_object = ControlLawClass(sample_rate,specification_signals,output_oversample_factor,...)`). Note that the the constructor function accepts as arguments not only the data available at the time (e.g. the specification and any extra control parameters) but also any parameters that will eventually exist.  This is because it needs to be able to seamlessly transition in case the control law is changed during control when there is already a transfer function, last signal, etc.

The `system_id_function` is called after the system identification is complete, so inside this function is where all setup calculations that require a transfer function would go.  In the case of a typical pseudoinverse control law, the system identification will not change as the test proceeds, so the inversion can take place in this function.

The `control` function is then the function that actually performs the control operations to compute the output time history using the last response or output time histories in addition to any data that had been stored inside the class.

The class implementation of the pseudoinverse control is shown in @lst:transient_pseudoinverse_class.

```{code} python
:caption: Class implementation of the buzz test approach
:label: lst:transient_pseudoinverse_class
import numpy as np

class pseudoinverse_control_class:
    def __init__(self,
                 sample_rate,
                 specification_signals, # Signal to try to reproduce
                 output_oversample_factor, # Oversample factor to output
                 extra_parameters, # Extra parameters for the control law
                 frequency_spacing,
                 transfer_function, # Transfer Functions
                 noise_response_cpsd,  # Noise levels and correlation 
                 noise_reference_cpsd, # from the system identification
                 sysid_response_cpsd,  # Response levels and correlation
                 sysid_reference_cpsd, # from the system identification
                 multiple_coherence, # Coherence from the system identification
                 frames, # Number of frames in the CPSD and FRF matrices
                 total_frames, # Total frames that could be in the CPSD and FRF matrices
                 last_excitation_signals = None, # Last excitation signal for drive-based control
                 last_response_signals = None, # Last response signal for error correction
                 ):
        self.rcond = 1e-15
        self.zero_impulse_after = None
        for entry in extra_parameters.split('\n'):
            field,value = entry.split(':')
            field = field.strip()
            if field == 'rcond':
                self.rcond = float(value)
            elif field == 'zero_impulse_after':
                self.zero_impulse_after = float(value)
            else:
                print('Unrecognized Parameter: {:}'.format(field))
        self.sample_rate = sample_rate
        self.specification_signals = specification_signals
        self.signal_fft = np.fft.rfft(specification_signals).T
        if self.transfer_function is not None:
            self.system_id_update(
                frequency_spacing,
                transfer_function, # Transfer Functions
                noise_response_cpsd,  # Noise levels and correlation 
                noise_reference_cpsd, # from the system identification
                sysid_response_cpsd,  # Response levels and correlation
                sysid_reference_cpsd, # from the system identification
                multiple_coherence, # Coherence from the system identification
                frames, # Number of frames in the CPSD and FRF matrices
                total_frames, # Total frames that could be in the CPSD and FRF matrices
                )
        
    def system_id_update(self,
                         frequency_spacing,
                         transfer_function, # Transfer Functions
                         noise_response_cpsd,  # Noise levels and correlation 
                         noise_reference_cpsd, # from the system identification
                         sysid_response_cpsd,  # Response levels and correlation
                         sysid_reference_cpsd, # from the system identification
                         multiple_coherence, # Coherence from the system identification
                         frames, # Number of frames in the CPSD and FRF matrices
                         total_frames, # Total frames that could be in the CPSD and FRF matrices
                         ):
        # Compute impulse responses
        impulse_response = np.fft.irfft(transfer_function,axis=0)

        if self.zero_impulse_after is not None:
            # Remove noncausal portion
            impulse_response_abscissa = np.arange(impulse_response.shape[0])/self.sample_rate
            zero_indices = impulse_response_abscissa > self.zero_impulse_after
            impulse_response[zero_indices] = 0
            
        # Zero pad the impulse response to create a signal that is long enough
        added_zeros = np.zeros((self.specification_signals.shape[-1]-impulse_response.shape[0],) 
                               + impulse_response.shape[1:])
        full_impulse_response = np.concatenate((impulse_response,added_zeros),axis=0)

        # Compute FRFs
        interpolated_transfer_function = np.fft.rfft(full_impulse_response,axis=0)

        # Perform convolution in frequency domain
        self.inverted_frf = np.linalg.pinv(interpolated_transfer_function,rcond=self.rcond)
        
        drive_signals_fft = np.einsum('ijk,ki->ij',self.inverted_frf,self.signal_fft)

        # Zero pad the FFT to oversample
        drive_signals_fft_zero_padded = np.concatenate((drive_signals_fft[:-1],
            np.zeros((drive_signals_fft[:-1].shape[0]*(self.output_oversample_factor-1)+1,)
                     +drive_signals_fft.shape[1:])),axis=0)

        self.drive_signals_oversampled = np.fft.irfft(
            drive_signals_fft_zero_padded.T,axis=-1)*self.output_oversample_factor

    def control(self,
                last_excitation_signals = None, # Last excitation signal for drive-based control
                last_response_signals = None, # Last response signal for error correction
                ) -> np.ndarray:       
        # We could modify the output signal based on new data that we obtained
        # Otherwise just output the same
        
        return self.drive_signals_oversampled
```
        
Note that the number of lines of code for a class implementation of the pseudoinverse control approach is not significantly more than the simpler function implementation, therefore users should not immediately discard the class-based approach as too difficult to implement in favor of the simpler function implementation.  Each approach has its merits and limitations, so it is up to the user to decide the best approach for their control law.

## Using Transformation Matrices
Transformation matrices in the Transient environment behave identically to the the Random Vibration environment.  See @sec:rattlesnake_environments_transformation_matrices for more information.

## Identifying the Signal in the Acquired Data
As the environment may be started or stopped arbitrarily during a given test, the Transient environment will need to identify where the signal that was desired actually occurs in the acquired signals.

The Transient environment keeps a buffer twice the length of the desired control signal, so at some point during the acquisition, the entire signal should be within the buffer.

To identify the position of the signal in the buffer, a correlation is performed between the signal that was output to the exciters and the acquired output data.  Correlation is performed on the output signal as it is generally read directly back into the controller and does not depend on an accurate prediction of the part's response.  The index of the maximum value of the correlation determines the sample closest to the starting point of the signal.

A second, sub-sample alignment is then performed using phases from the FFT of the signal truncated to the starting point and signal length compared to the phases of the specified output.  The slope of the phase change vs frequency line is proportional to the sub-sample signal shift.

With the sample and subsample shift of the signal computed for the output signal, the same sample and subsample shift can be can be applied to the signal from the control gauges.  This portion of the control channel signals can then be compared directly to the desired signal to judge how well the environment is controlling.
