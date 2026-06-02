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
# Modal Testing

(sec:modal)=
# Modal Testing

Rattlesnake's Modal Testing environment is designed for dynamic characterization of a test article.  The environment computes transfer functions between a subset of reference channels and the remaining response channels.  Reference channels are traditionally measurements of the forces applied to the structure using either a modal impact hammer or modal shaker.  The modal testing environment can generate many standard modal signals such as chirp or burst random.  However, shaker signals can also be generated from other environments using Rattlesnake's combined environments capabilities.
    
## Defining the Modal Testing Environment in Rattlesnake

The Rattlesnake Modal Testing environment has many signal processing parameters to specify when setting up the modal test.  These are defined on the `Environment Definition` tab in the Rattlesnake controller on a sub-tab corresponding to the name of the modal testing environment.  @fig:modalenvironmentdefinition shows a Modal Testing sub-tab.  The following subsections describe the parameters that can be specified, as well as their effects on the analysis.

:::{figure} figures/modal_environment_definition.png
:label: fig:modalenvironmentdefinition
:align: center
GUI used to define a Modal Testing Environment
:::

### Sampling Parameters
The `Sampling Parameters` section contains information and settings that pertain to the samping in the modal test.  It consists of the following parameters:

* **Sample Rate** Sample rate in samples per second of the data acquisition hardware, for display only.  This is a global parameter and must be set in the `Data Acquisition Parameters` tab.
* **Samples per Frame** Samples per measurement frame in the controller. The measurement frame is the &quot;block&quot; of data upon which the signal processing will be performed. This value will determine the window size. A larger value will result in more frequency lines in the FFT analysis. This need not correspond to the read or write size in the data acquisition system.
* **Samples per Acquire** Number of samples that the control process processes at a time. This will be equal to the Samples per Frame * (1 - Overlap Percentage / 100). This need not correspond to the read or write size of the data acquisition system as the control process acquisition is buffered.
* **Frame Time** Time to acquire each measurement frame in seconds. This is the Samples per Frame divided by the Sample Rate.
* **Nyquist Frequency** The Nyquist Frequency is the highest frequency that can be analyzed using frequency domain techniques.  It is the Sample Rate / 2.
* **FFT Lines** The number of frequency lines in the Fast Fourier Transform output, which is the number of frequency lines that will be in the Transfer Function and CPSD matrices.
* **Frequency Spacing** The frequency resolution of the measurement, computed by 1/Frame Time.


### System ID Parameters
The `System ID Parameters` section contains information and settings that pertain to the computation of spectral quantities in the Modal Testing Environment.  This section contains the following parameters:

* **Averaging Type** Specifies how averaging is performed.  Linear averaging weights all measurement frames equally.  Exponential averaging weights more recent frames more heavily.
* **Number of Averages** The number of averages used to compute the frequency response functions.
* **Averaging Coefficient** Weighting applied to the most recent average when Exponential Averaging is used
* **FRF Technique** Technique to compute the Transfer Function. H1 averages noise on the response channels.
* **FRF Window** The window function applied to the measurement frames when computing the transfer functions.  *Rectangle:* No window is used, all samples are weighted equally. Used for periodic signals or signals that start and end at zero in the measurement frame.  *Hann:* Window that weights the start and end of the frame to zero and the middle to one. Useful for Random excitation to reduce leakage.  *Exponential:* Window that weights the beginning of the frame more heavily than the end of the frame. Useful for impact testing where the test article's response doesn't decay by itself by the end of the measurement frame.
* **Overlap Percentage** The percentage overlap used between measurement frames.  A 25% overlap means the last quarter of the previous measurement frame is used as the first quarter of the new measurement frame.
* **Window value at frame end** Value of the exponential window at the end of the measurement frame, which is used to specify how quickly the exponential window decays.

### Triggering
The `Triggering` section contains information and settings that pertain to triggering each measurement frame using a measured signal.  The parameters are:


* **Triggering Type** Type of trigger used for the analysis.  *Free Run:* No triggering is used.  *First Frame:* Triggering only starts the first measurement frame and then the measurement proceeds as if it were a Free Run.  *Every Frame:* The measurement waits for a trigger for every measurement frame.
* **Acceptance** Determines how measurement frames are accepted or rejected.  *Accept All:* Accepts all measurement frames.  *Manual:* Allows the user to specify whether the recently obtained measurement frame should be accepted or rejected. *Autoreject...:* Allows the user to load in a Python function to automatically accept or reject the measurement frame based off of arbitrary criteria.
* **Wait for Steady State** Amount of time to wait for the measurement to achieve a steady state response prior to performing signal processing operations.
* **Trigger Channel** Select the channel to use as a trigger.
* **Pretrigger** Amount of time in the measurement frame before the trigger occurs.  The system retains a running buffer of recently acquired data looking for the trigger.  When the trigger is found, the specified percentage of the frame is data that occured before the trigger.  This is used to capture, for example, the entire transient of a shock, some of which may occur before the trigger depending on the trigger level.
* **Trigger Slope** Slope of the trigger.  If the slope is Positive, it will look for a rising signal to trigger the measurement.  If the trigger slope is Negative, it will look for a falling signal to trigger the measurement.
* **Trigger Level** Level to trigger at as a percentage of the range of the channel.
* **Trigger Level (V)** Level to trigger at in volts.  This property is read only and displayed for the user's convenience.
* **Trigger Level (EU)** Level to trigger at in Engineering Units.  This property is read only and displayed for the user's convenience.
* **Hysteresis Level** Level that the signal must return to prior to a new trigger being able to be accepted.  This ensures that the system must return to some nominal state prior to a new trigger being obtained.  Specified in percentage of the range of the channel.
* **Hysteresis Level (V)** Level that the trigger channel must fall below to achieve a subsequent trigger in volts. This property is read only and displayed for the user's convenience.
* **Hysteresis Level (EU)** Level that the trigger channel must fall below to achieve a subsequent trigger in Engineering Units.  This property is read only and displayed for the user's convenience.
* **Hysteresis Frame %** Percentage of the measurement frame that the signal must be below the Hysteresis value in order to accept a new trigger.  Useful for Burst Random excitation where you don't want to accidentally trigger midway through the burst.
* **Hysteresis Samples** Number of samples that the trigger channel must be below the Hysteresis level.  This property is read only and shown for the user's convenience.
* **Hysteresis Time (s)** Amount of time that the trigger channel must be below the Hysteresis level.  This property is read only and is displayed for the user's convenience.** 

### Channel Information
The `Channel Information` section contains information regarding how many channels are active for the given test.  These quantities are read-only to inform the user about the number of channels the current setup will contain.

* **Response Channels** Number of response channels used when computing frequency response functions.
* **Reference Channels** Number of reference channels used to compute frequency response function
* **Output Channels** Number of excitation channels in the current test.

### Channel Selections
The `Channel Selections` section contains a list of all of the channels in the test and the ability to select which channels are references or responses, as well as to enable or disable channels.

To disable a channel, simply uncheck the checkbox in the `Enabled` column associated with a given channel.  Disabling a voltage channel associated with a `Feedback Device` and `Feedback Channel` on the channel table will also disable that output.

To make a channel a reference channel, simply check the checkbox in the `Reference` column associated with a given channel.  This will turn the channel from a Response channel to a Reference channel.

Multiple rows of the `Channel Selctions` table can be selected at once.  The `Enable Selected` and `Disable Selected` buttons will enable or disable all selected channels, respectively.  Similarly, the `Check Selected References` and `Uncheck Selected References` buttons will turn the selected channels into Reference Channels or Response Channels, respectively.

* **Channel Selection Table** By checking the box in the Reference column, a channel can be made into a reference channel rather than a response channel.  Unchecking the box in the Enabled column with disable the channel.  Disabling an excitation channel will disable that excitation device.  Disabling a reference will remove that reference from the FRF computation.
* **Enable Selected** Enables the channels currently selected in the Channel Selections table
* **Check Selected References** Turns the selected channels into reference channels for computing FRFs
* **Disable Selected** Disables the currently selected channels in the Channel Selections table
* **Uncheck Selected References** Turns the selected channels into response channels for computing FRFs

### Signal Generator
The `Signal Generator` section contains the parameters to determine what signal will be generated for the modal test.
The tabs at the top of the `Signal Generator` section determine the type of signal that will be applied.  Each signal type may have different parameters to define it.

* **Signal Generation Tabs** Switching tabs will change the type of excitation signal that will be output.

  * **No Signal Generation** No signal will be generated if this is selected.
  * **Random** A True Random signal will be generated.  This signal is continually generated, and being random, will not repeat for each measurement frame.  Generally a window function such as a Hann window should be used with a Random signal.
    * **Random RMS** RMS voltage level of the Random signal
    * **Random Minimum Frequency** Minimum frequency cutoff for the random vibration signal.
    * **Random Maximum Frequency** Maximum frequency cutoff for the random vibration signal.
  * **Burst Random** A Burst Random signal will be generated.  A Burst Random signal is essentially a Random signal that shuts off part of the way through a measurement frame.  This signal is continually generated, and being random, will not repeat for each measurement frame.  When using a Burst Random excitation, users will generally want to set up a `First Frame` trigger on a voltage channel with a `Hysteresis Frame %` of approximately half the time the burst will be "off".
    * **Burst Random RMS** RMS voltage of the burst portion of the excitation
    * **Burst Random Minimum Frequency** Minimum frequency of the burst excitation
    * **Burst Random Maximum Frequency** Maximum frequency of the burst excitation.
    * **Burst Random On Percentage** Percentage of the measurement frame that the burst is active for.
  * **Pseudorandom** A Pseudorandom signal will be generated.  This is a signal that appears random, but is actually deterministically constructed from the frequency lines of the FFT with randomized phases.  This signal will repeat for each measurement frame.  When using Pseudorandom excitation, users should set the `Wait For Steady State` parameter to ensure the system has reached steady state prior to acquiring data.  Pseudorandom excitation will generally not work well with multiple shaker excitation.
    * **Pseudorandom RMS** RMS voltage of the pseudorandom excitation
    * **Pseudorandom Minimum Frequency** Minimum frequency of the pseudorandom excitation
    * **Pseudorandom Maximum Frequency** Maximum frequency of the pseudorandom excitation.
  * **Chirp** A Chirp signal will be generated.  A Chirp is a fast sine sweep between two frequency values.  This signal will repeat for each measurement frame.  When using Chirp excitation, users should set the `Wait For Steady State` parameter to ensure the system has reached steady state prior to acquiring data.  Chirp excitation will generally not work well with multiple shaker excitation.
    * **Chirp Peak Level** Peak level of the chirp signal
    * **Chirp Minimum Frequency** Starting frequency of the chirp
    * **Chirp Maximum Frequency** Ending frequency of the chirp
  * **Square Pulse** A square pulse wave will be generated.  This wave is useful for triggering other test hardware, such as an automatic hammer.
    * **Square Wave Peak Level** Amplitude of the square pulse
    * **Square Wave Frequency** Frequency of the square pulse
    * **Square Wave Percent On** Percentage of the square pulse's period that it is "up".
  * **Sine** A sine pulse wave will be generated.  This is useful for characterizing a single frequency line or for performing rigid body checkouts.  Be careful when selecting a frequency to ensure there is no leakage at that frequency line.
    * **Sine Wave Peak Level** Amplitude of the sinusoidal excitation
    * **Sine Wave Frequency** Frequency of the sinusoidal excitation

The selected signal will be plotted in the `Output Signal` plot, which will give the user an idea of what the signal looks like.  If the `Automatically Regenerate Signal on Parameter Change` checkbox is checked, the signal should update automatically when different parameters are selected.  If not, the user can press the `Regenerate Signal` button to regenerate the signal.  For random signals, the `Regenerate Signal` button can also be used to visualize different realizations of the random signal.

* **Output Signal** Realization of the signals that will be generated.  Note that for random signals, the actual signal may vary slightly from this representation, as the generation of those signals is random.
* **Regenerate Signal** Generate a new realization of the output signals.
* **Automatically Regenerate Signal on Parameter Change** Check to automatically regenerate signals when signal generation parameters are updated.  If not checked, you will need to click the `Regenerate Signals` button to see changes.

## Running a Modal Test
The Modal Testing Environment is then run on the `Run Test` tab of the controller.  With the data acquisition system armed, the GUI initially looks like @fig:modalenvironmentruntestempty.  This screen looks rather empty, but users can populate it with functions of their choice, as shown in @fig:modalenvironmentruntestfull.

:::{figure} figures/modal_environment_run_test_empty.png
:label: fig:modalenvironmentruntestempty
:align: center
Empty GUI to run a modal test in Rattlesnake.
:::

:::{figure} figures/modal_environment_run_test_full.png
:label: fig:modalenvironmentruntestfull
:align: center
GUI to run a modal test in Rattlesnake populated with several data plots.
:::

### Acquisition
The `Acquisition` portion of the window contains controls for starting and stopping the measurement, as well as saving modal data to a file.  See @sec:rattlesnake_environments_modal_output_files for a description of the file format.

* **Preview Acquisition** Starts the acquisition in Preview mode which will not save data to the disk nor will it stop automatically when the requested number of averages is reached.
* **Start Acquisition** Starts the acquisition and stores data to the disk.  Acquisition will stop automatically when the requested number of averages is reached.
* **Stop Acquisition** Stops the currently active acquisition.
* **Data File** File to which modal data is written from Rattlesnake.  Note that writing modal data to a network drive can cause the user interface to hang if the files are large and the network is relatively slow.
* **Select File...** Opens a file dialog to allow the user to select a file to which data will be saved.  For large tests, it is not recommended to write data directly to a network drive, as relatively slow network speeds may cause the user interface to hang while the file is being written.
* **Auto-Increment** Select this to automatically add a 4-digit number to the file name that will increment automatically.  This can be useful for roving hammer excitation where the user does not want to accidentally overwrite previous data.


### Averaging
The `Averaging` portion of the window displays the current number of measurement frames that have been acquired, as well as the total number of measurement frames that are to be acquired.  If `Acceptance` is set to `Manual`, then the `Accept` and `Reject` buttons will become available after each measurement frame is acquired, which will allow the users to manually accept or reject a measurement frame.

If the measurement was started using the `Start Acquisition` button, the measurement will stop automatically when the current number of measurement frames is equal to the total number of measurement frames.  If `Preview Acquisition` was used, then the measurement will continue until stopped with the `Stop Acquisition` button.

* **Average Count** Current number of averages acquired
* **Total Averages** Total number of averages that will be acquired during the test, after which data acquisition will shut down automatically if not in preview mode.
* **Accept** Manually accept the current measurement frame.  This button will be disabled unless Manual acceptance is specified.
* **Reject** Manually reject the current measurement frame.  This button will be disabled unless Manual acceptance is specified.


### Windows
The `Windows` portion of the screen allows the user to customize the data that they visualize during the test.

* **Display Area** New windows will be placed in this area on the right side of the `Run Test` tab, allowing flexible visualization of modal data during the test.
* **New Window** Create a new data visualization window.
* **New Window from Template...** Create a visualization window from a set of preset templates of commonly requested visualizations, such as drive point FRFs or reference autospectra.
  * **Drive Point (Magnitude)** Rattlesnake will parse the channel names in the channel table to identify drive point FRFs, of which it will display the magnitude.  This will create one window for each drive point FRF.
  * **Drive Point (Imaginary)** Rattlesnake will parse the channel names in the channel table to identify drive point FRFs, of which it will display the imaginary part.  This will create one window for each drive point FRF.  For an acceleration to force FRF, the imaginary part of the drive point FRF should always be positive if both channels reference and response channels have polarity of + or -, or negative if one of the reference or response channels has a polarity of + and the other has a polarity of -.  If the imaginary part of the drive point `FRF` crosses zero there could be an issue with the test setup.
  * **Drive Point Coherence** Rattlesnake will parse the channel names in the channel table to identify drive point FRFs which it will display with the coherence overlaid.  This will create one window for each drive point FRF.
  * **Reciprocity** Rattlesnake will parse the channel names in the channel table to identify reciprocal FRFs to overlay.  It will find, for example the FRF with input at degree of freedom `A` and response at degree of freedom `B` and overlay it with the FRF with input at `B` and response at `A`.  It will generally create $\frac{n^2-n}{2}$ windows where $n$ is the number of drive points in the measurement.
  * **Reference Autospectrum** Rattlesnake will plot the autospectrum for each reference channel in the test.  Users can use this plot to ensure adequate excitation over the bandwidth of interest.  It will generate one window per each reference channel in the test.
  * **3x3 Channel Grid** Rattlesnake will create 9 windows which by default display time histories for the first 9 channels in the test.  This selection, combined with setting the `Increment Channels` selector to 9, can be used to quickly look through all of the channels in the test.
* **Tile** Tile all visualization windows
* **Close All** Close all visualization windows.
* **<< (Decrement Channels)** Decrement channels in visualization windows by the specified number of channels, except for windows that are locked.
* **>> (Increment Channels)** Increment channels in visualization windows by the specified number of channels, except for windows that are locked.
* **Channel Increment** Select the number of channels to increment all windows by.  This is useful to quickly move through all channels in a test.

Windows created in the `Run Test` tab of a Modal Testing environment are flexible in that they can show multiple different types of data in various formats for any channel in the test.  All windows have a `Lock` checkbox that when checked does not allow the channel to change via the `Increment Channels` arrow buttons.  A locked channel can still be changed manually, however.  Depending on the data type being visualized, the user may manually select a single channel or a reference and response channel.  Certain types may also have a selector for data display, allowing the user to investigate real, imaginary, magnitude, or phase components of the signal.

* **Signal Type Selector** Select the type of data to look at.
* **Channel Selector** Select the channel to look at.
* **Reference Channel Selector** Select the reference to look at.
* **Data Display Selector** Select how the data is displayed.
* **Lock** Locks the channel so it does not update when the increment channels buttons are clicked.  Channel can still be manually changed.

Subsequent subsections will walk through the various window types in a modal test.

#### Time Window
The `Time` window displays the time trace for a single measurement frame for the selected channel.  @fig:modaltimewindow shows an example of this window.  The channel to visualize can be chosen from the drop-down menu.  The `Lock` checkbox can be checked to ensure the channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_time_window.png
:label: fig:modaltimewindow
:align: center
Modal Testing data window showing a channel's time signal.
:::

#### Windowed Time Window
The `Windowed Time` window displays the time trace for a single measurement with the window function applied.  This allows users to visualize the effect that the window has on the data.  @fig:modalwindowedtimewindow shows an example of this window.  The channel to visualize can be chosen from the drop-down menu.  The `Lock` checkbox can be checked to ensure the channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_windowed_time_window.png
:label: fig:modalwindowedtimewindow
:align: center
Modal Testing data window showing a channel's windowed time signal.
:::

#### Spectrum Window
The `Spectrum` window displays the magnitude of the FFT of the (windowed) time trace for a single measurement.  @fig:modalspectrumwindow shows an example of this window.  The channel to visualize can be chosen from the drop-down menu.  The `Lock` checkbox can be checked to ensure the channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_spectrum_window.png
:label: fig:modalspectrumwindow
:align: center
Modal Testing data window showing a channel's spectrum.
:::

#### Autospectrum Window
The `Autospectrum` window displays the magnitude of the autospectrum for a single measurement.  This is an averaged quantity, so it will generally improve as more averages are acquired.  @fig:modalspectrumwindow shows an example of this window.  The channel to visualize can be chosen from the drop-down menu.  The `Lock` checkbox can be checked to ensure the channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_autospectrum_window.png
:label: fig:modalautospectrumwindow
:align: center
Modal Testing data window showing a channel's autospectrum.
:::

#### FRF Window
The `FRF` window displays a FRF for a reference/response channel combination.  This is an averaged quantity, so it will generally improve as more averages are acquired.  @fig:modalfrfwindow shows an example of this window.  Two channel selection menus exist to select which FRF to visualize.  The first corresponds to the `Response` channel, and the second corresponds to the `Reference` channel.  Channels selected as a Reference on the `Environment Definition` tab will show up in the reference channel selection menu, and channels that were not selected will show up in the response channel selection menu.  The FRF can be visualized by looking at Real, Imaginary, Magnitude, or Phase parts, or it can be split into two plots to visualize Magnitude and Phase or Real and Imaginary parts simultaneously.  The `Lock` checkbox can be checked to ensure the response channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_frf_window.png
:label: fig:modalfrfwindow
:align: center
Modal Testing data window showing the FRF between a reference and response channel.
:::

#### Coherence Window
The `Coherence` window displays the coherence for a given channel.  @fig:modalcoherencewindow shows an example of this window.  If only a single reference is used, this will be the regular coherence.  If multiple references are used, this will then be the Multiple Coherence function.  The channel to visualize can be chosen from the drop-down menu.  The `Lock` checkbox can be checked to ensure the channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_coherence_window.png
:label: fig:modalcoherencewindow
:align: center
Modal Testing data window showing a channel's coherence.
:::

#### FRF Coherence Window
The `FRF Coherence` window displays a FRF overlaid with the Coherence plot.  This allows users to align drops in coherence with features of the `FRF` to determine if they occur at modes of the structure, which could suggest an issue with the data.  @fig:modalfrfcoherencewindow shows an example of this window.  Two channel selection menus exist to select which FRF to visualize.  The first corresponds to the `Response` channel, and the second corresponds to the `Reference` channel.  The coherence will generally correspond to the Response channel: if multiple references exist, then the multiple coherence that is plotted will be a function of the response degree of freedom; if only a single reference exists, then the regular coherence will be with respect to the only reference in the test, and will also then only change when the response is updated.  The FRF can be visualized by looking at Real, Imaginary, Magnitude, or Phase parts, or it can be split into two plots to visualize Magnitude and Phase or Real and Imaginary parts simultaneously.  If these latter options are selected, the coherence will be overlaid on the second plot, which will be the Magnitude or Imaginary part.  The `Lock` checkbox can be checked to ensure the response channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_frfcoherence_window.png
:label: fig:modalfrfcoherencewindow
:align: center
Modal Testing data window showing a channel's FRF overlaid with the coherence.
:::

#### Reciprocity Window
The `Reciprocity` window displays two reciprocal FRFs overlaid.  For a linear system, these FRFs should be identical to one another.  @fig:modalreciprocitywindow shows an example of this window.  Two channel selection menus exist to select which FRF to visualize.  The first corresponds to the `Response` channel, and the second corresponds to the `Reference` channel.  Rattlesnake will automatically go through and find the reciprocal FRF for that measurement and overlay it.  The FRF can be visualized by looking at Real, Imaginary, Magnitude, or Phase parts, or it can be split into two plots to visualize Magnitude and Phase or Real and Imaginary parts simultaneously.  The `Lock` checkbox can be checked to ensure the response channel does not change when the `Increment Channels` functionality is utilized.


:::{figure} figures/modal_reciprocity_window.png
:label: fig:modalreciprocitywindow
:align: center
Modal Testing window showing two reciprocal FRFs.
:::

### Degree of Freedom Override
This section of the Modal Testing `Run Test` tab allows users to override the channel metadata for a given measurement.  This is particularly useful for roving hammer or roving accelerometer testing strategies.  The node identification number and direction are defined on the channel table on the first tab of the software, and it would be very tedious to need to move back to the first tab to redefine the node number, reinitialize the data acquisition, redefine the environment definition, and finally, rearm the test to take data.  Instead, the user can override the degree of freedom information without leaving the `Run Test` tab.

To add an override channel, users can press the `Add Override` button.  This will create a new row in the override table.  These can be removed by clicking the `Remove Override` button with the particular row in the table selected.  @fig:modaldofoverride shows an example of this functionality where the node information on the force channel has been updated.  Note that the windows will also be updated to display this new degree of freedom information.

:::{figure} figures/modal_dof_override.png
:label: fig:modaldofoverride
:align: center
Overriding the 16934 X+ degree of freedom information with 16950 X+ degree of freedom information.
:::

* **Degree of Freedom Override Table** This table can be used during roving hammer testing, or other testing that involves changing the channel degrees of freedom, to change channel names during a test without needing to update the channel table and re-initialize the data acquisition system.  The user selects a channel to update and provides a new node number and new direction string.  Subsequently acquired data will have degree of freedom information updated in the channel table that gets stored to the disk with the data.
* **Add Override** Adds an override to the channel table to change the degree of freedom associated with a given channel.  Useful for roving hammer or roving accelerometer testing.
* **Remove Override** Removes the selected channel override.


(sec:rattlesnake_environments_modal_output_files)=
## Output NetCDF File Structure
When Rattlesnake saves data to a netCDF file, environment-specific parameters are stored in a netCDF group with the same name as the environment name.  Similar to the root netCDF structure described in @sec:using_rattlesnake_output_files, this group will have its own attributes, dimensions, and variables, which are described here.

### NetCDF Dimensions

* **reference_channels** The number of reference channels in the measurement
* **response_channels** The number of response channels in the measurement


### NetCDF Attributes

* **samples_per_frame** The number of samples per measurement frame.
* **averaging_type** A string specifying if linear or exponential averaging was performed in the measurement.
* **num_averages** The number of averages used to compute spectral quantities like FRFs and autospectra.
* **averaging_coefficient** If Exponential Averaging is used, this is the weighting of the most recent frame compared to the weighting of the previous frames.  If the averaging coefficient is $\alpha$, then the most recent frame will be weighted $\alpha$, the frame before that will be weighted $\alpha(1-\alpha)$, the frame before that will be $\alpha(1-\alpha)^2$, etc.
* **frf_technique** A string representing the estimator used to compute FRFs, e.g. H1, H2, Hv.
* **frf_window** A string representing the window function used when computing FRFs.
* **overlap** The overlap fraction between measurement frames.
* **trigger_type** A string representing the type of triggering used in the modal test (e.g. Free Run, First Frame, Every Frame).
* **accept_type** A string representing the acceptance type used in the modal test (e.g. Accept All, Manual).
* **wait_for_steady_state** The amount of time the measurement needed to wait for steady state to occur before acquiring data.
* **trigger_channel** An integer representing the channel used to trigger the measurement.
* **pretrigger** The amount of the measurement frame that occurs prior to the trigger.
* **trigger_slope_positive** An integer equal to 1 if the slope was positive and 0 if the slope was negative.
* **trigger_level** The level that was used to trigger the system as a fraction of the total range of the channel.
* **hysteresis_level** The level that the data must return back to prior to being able to accept another trigger as a fraction of the total range of the channel.
* **hysteresis_length** The length of time that a channel must return to the hysteresis level prior to being able to accept another trigger as a fraction of the measurement frame length.
* **signal_generator_type** The type of signal being generated by the modal test
* **signal_generator_level** The signal level, which could be a peak value for signals such as a sine wave, or an RMS value for random signals
* **signal_generator_min_frequency** The minimum frequency of the excitation signal
* **signal_generator_max_frequency** The maximum frequency of the excitation signal
* **signal_generator_on_fraction** The percent of time that a square wave is "up" or a burst random is "active".
* **exponential_window_value_at_frame_end** The value of the exponential window function at the end of the measurement frame, used to set the strength of the exponential window.
* **acceptance_function** If the system is set to automatically reject a measurement, this is the script and function name used to evaluate that signal.


### NetCDF Variables

* **reference_channel_indices** The indices into the channel table that correspond to the reference channels.  Type: 32-bit int; Dimensions: `reference_channels`
* **response_channel_indices** The indices into the channel table that correspond to the response channels.  Type: 32-bit int; Dimensions: `response_channels`       


### Saving Modal Data
In addition to time streaming, Rattlesnake's Modal Testing environment will also save bespoke modal data files directly to the disk when a test is started with the `Start Acquisition` button.  The computed spectral quantities such as FRF and coherence data are in a NetCDF file, and are updated at each average.  Time data is also stored to this file, but only the time data that is used for spectral computations.  For example, if a hammer test is being performed and the user takes a minute to evaluate the data before accepting or rejecting the measurement frame, that time data measured during that pause will not be stored to the modal data file; however, it would be stored to the streaming data file described in @sec:using_rattlesnake_output_files, which is essentially an open-tape measurement.

The additional dimension is:


* **fft_lines** The number of frequency lines in the spectral quantities.


There are also several additional variables to store the spectral data:


* **frf_data_real** The real part of the most recently computed value for the transfer functions between the reference signals and the response signals.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `response_channels` $\times$ `reference_channels`
    
* **frf_data_imag** The imaginary part of the most recently computed value for the transfer functions between the excitation signals and the control response signals.  NetCDF files cannot handle complex data types so real and imaginary parts are split into two variables.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `response_channels` $\times$ `reference_channels`
    
* **coherence** The multiple or regular coherence of the channels computed during the test, depending on if there are multiple references or a single reference, respectively.  Type: 64-bit float; Dimensions: `fft_lines` $\times$ `response_channels`
    
