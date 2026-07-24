---
numbering:
  heading_2:
    start: 3
  figure:
    enumerator: 3.%s
  table:
    enumerator: 3.%s
  equation:
    enumerator: 3.%s
  code:
    enumerator: 3.%s
---
# Using Rattlesnake

(sec:using_rattlesnake)=
# Using Rattlesnake

This chapter will describe how to use Rattlesnake through its graphical user interface (GUI).  Rattlesnake is capable of running several different types of control, therefore the GUI may look different for different tests.  In general, the GUI consists of a tabbed interface across the top of the main window, and users must complete each tab before proceeding to the next.  The tabs that exist in a given test will depend on which control type is being run.  For example, in a combined environments test (see @sec:combined_environments) such as the one shown in @fig:rattlesnake_main_gui_combined_environments), there is a `Test Profile` tab that allows the user to define a testing timeline.  Additionally, environments such as the MIMO Random Vibration environment (see @sec:mimo_random) require a system identification phase where the controller identifies relationships between the output signals and the control degrees of freedom.  Therefore, tests using the MIMO Random Vibration environment will also have a `System Identification` and `Test Predictions` tab.  @fig:rattlesnake_main_gui_time_generation, on the other hand, shows the GUI for a test that only utilizes the Time History environment (see @sec:mimo_time) so these optional tabs are not displayed.

:::{figure} figures/Rattlesnake_Main_GUI_Combined_Environments.png
:label: fig:rattlesnake_main_gui_combined_environments
:align: center
Rattlesnake GUI tabs when running a combined environments test with an environment that requires a system identification.
:::

:::{figure} figures/Rattlesnake_Main_GUI_Time_Generation.png
:label: fig:rattlesnake_main_gui_time_generation
:align: center
Rattlesnake GUI tabs when running a single environment with no system identification phase.
:::

Users of Rattlesnake must be aware that depending on their test configuration, their GUI may not appear identical to images shown in this User's Manual.  Additionally, users should be aware that the GUI library used by this software will inherit stylistic features from the operating system.  There may therefore be cosmetic differences between the images of the GUI shown in this document and the GUI seen by the user.  All images in this document were created using Microsoft Windows 10 or Windows 11 operating systems, so users with Mac or Linux operating systems will note a difference in GUI appearance.

Note that the Rattlesnake enforces an order to operations when defining a particular test by enabling and disabling tabs in the GUI.  Initially, only the first tab will be enabled.  As the users complete each tab, the next tab will become available.  In @fig:rattlesnake_main_gui_combined_environments and @fig:rattlesnake_main_gui_time_generation, it can be seen that only the initial tab is enabled, and subsequent tabs are disabled.

## Environment Selection

When Rattlesnake is opened, the first GUI window that the user will see allows the user to select the environment that they will run (@fig:environment_selection).  Users can select a single environment, or alternatively select a combined environments test (see @sec:combined_environments).  The selection made in this dialog box will determine which tabs are set up in the main GUI.

:::{figure} figures/environment_selection.png
:label: fig:environment_selection
:align: center
Initial Rattlesnake dialog to select the type of control that will be run.
:::

## Global Data Acquisition Settings

The `Data Acquisition Setup` tab of the Rattlesnake GUI specifies the global test parameters that the controller will use.  Parameters are determined to be global when they affect all environments or the controller itself.  The three main sections of this portion of the interface are the Channel Table, Environment Table, and Global Data Acquisition Parameters.  @fig:data_acquisition_setup. shows this.



:::{figure} figures/data_acquisition_setup.png
:label: fig:data_acquisition_setup
:align: center
Data Acqisition Setup tab in the Rattlesnake Controller where the Channel Table, Environment Table, and Data Acquisition Parameters are specified.
:::


(sec:channel_table)=
### Channel Table

The channel table specifies how the instrument channels in a given test are connected to the data acquisition hardware, as well as how the data read from those channels are used by the software.

In general, for a given test there will be a set of excitation devices that use the output signals from Rattlesnake as well as instrumentation to record the test article's responses to those exciters.  Rattlesnake requires each instrument (or each channel on each instrument for multi-axial instruments) as well as each excitation device to have a row in the channel table.  This is perhaps contrary to other control software where only the response channels need to be set up in the channel table.  However, to maintain the flexibility to run multiple types of hardware devices, some of which having limitations to their triggering capabilities, Rattlesnake must read in the signal from its output directly in order to be able to synchronize its outputs and the responses to those outputs.  Therefore, for all Rattlesnake test setups, the output signal should be split using a tee to the exciter and the corresponding input channel.  Because of this requirement, one should keep in mind that the number of acquisition channels required on the hardware device for a given test is actually the number of responses plus the number of outputs.  @fig:lanxi_source_tee_labelled shows a schematic of a four acquisition channel, two output channel LAN-XI module set up for use with Rattlesnake.

:::{figure} figures/lanxi_source_tee_labelled.png
:label: fig:lanxi_source_tee_labelled
:align: center
Output channels teed to acquisition channels so they can be read by the controller.
:::

The required data input into the channel table varies with the physical or virtual hardware used for the test.  For device-specific channel table requirements, see the appropriate section of @sec:rattlesnake_hardware.  In general, the entries to the channel table are as follows:

* **Node Number** Determines the instrumentation position on the test article.  While not used directly by the controller except to label plots, it is important for book-keeping and test documentation.  The node number will generally correspond to a node in a test geometry or FEM.
* **Node Direction** Determines the instrumentation direction on the test article at the position specified by the Node Number.  Again, this is not used directly by the controller except to label plots, but it is important for book-keeping.  The Node Direction will generally correspond to the node's local coordinate system if one exists in the test geometry.
* **Comment** Provides space for additional information about a channel that may not be captured by the Node Number and Node Direction.
* **Serial Number** The serial number of the instrument used for the given channel.  This field is not used by the controller but will be stored with the test data and is important for data traceability to know which instruments were used to measure which channels.
* **Triax DoF** The degree of freedom on a given instrument corresponding to the given channel.  This is primarily used to distinguish between the three axes of a triaxial accelerometer, but has the potential to be used for other multi-axis instrumentation types such as strain gauge rosettes.
* **Sensitivity** The sensitivity of the instrument in millivolts per Engineering Unit.  This is used to transform the acquired data from a raw voltage to a engineering quantity such as acceleration or force.
* **Engineering Unit** The unit in which the measured signal for the given instrument will be reported.  Certain hardware will limit the units that can be specified: see @sec:rattlesnake_hardware for more information.
* **Make** The name of the instrument's manufacturer, used for data traceability.
* **Model** The product name or model number of the instrument, used for data traceability.
* **Expiration** The expiration date of the instrument's calibration certificate.  Note that this is only for data traceability; no checking of this date with the current data to ensure a valid calibration is performed by the software.
* **Physical Device** The reference to a physical device attached to the computer.  The entries in this field will be specific to the acquisition hardware being used for a given test.  For virtual control, this column must be filled to specify that a given channel is active.  See @sec:rattlesnake_hardware for more information.
* **Physical Channel** The reference to a channel on a physical device attached to the computer.  The entries in this column will be specific to the acquisition hardware being used for a given test.  See @sec:rattlesnake_hardware for more information.
* **Channel Type** The type of the channel being used for a given test, such as Acceleration, Force, or Voltage.  The allowable entries in this column will be specific to the acquisition hardware being used for a given test.  See @sec:rattlesnake_hardware for more information.
* **Minimum Value (V)** The minimum voltage that the data acquisition system can handle during a test.  This is used to set the range on the data acquisition system.  For hardware devices with symmetric ranges (e.g. $\pm$10V), this column can be left blank.
* **Maximum Value (V)]**  The maximum voltage that the data acquisition system can handle during a test.  This is used to set the range on the data acquisition system.  For hardware devices with symmetric ranges (e.g. $\pm$10V), this column is used to set the maximum and minimum voltage values.
* **Coupling** The coupling used by the data acquisition system.  This may include filtering in addition to AC/DC coupling, which is dependent on the hardware being used for a given test.  See @sec:rattlesnake_hardware for more information.
* **Excitation Source** Used to specify the signal conditioning that is required by the instrument.  This column is generally where the constant current line drive (CCLD)/integrated electronics piezoelectric (IEPE)/integrated circuit piezoelectric (ICP) is specified for a given hardware device.  See @sec:rattlesnake_hardware for more information.
* **Current Excitation (A)** Used to specify the excitation current sent to the device for signal conditioning.  Depending on whether the device has a fixed or variable excitation current, this field may be left empty.  This can also be left empty if no signal conditioning is provided by the data acquisition system.  See @sec:rattlesnake_hardware for more information.
* **Feedback Device** For output channels, this is the reference to the output or excitation device that is being fed back into the current channel's Physical device.  If the current channel is not an output channel, it should be left empty.  A populated Feedback Device column tells the controller that the given channel is an output channel.
* **Feedback Channel** For output channels, this is the reference to the output channel on the output or excitation device that is being fed back into the current channel's Physical Device.  As an example using generic device and channel names, if `Channel 2` on `Generator 1` is teed off to `Channel 3` on `Acquisition Card 2`, the corresponding row in the channel table would have `Acquisition Card 2` specified as the Physical Device, `Channel 3` specified as the Physical Channel, `Generator 1` specified as the `Feedback Device` and `Channel 2` specified as the feedback channel.
* **Warning Level** A warning level can be implemented for each channel.  The warning level is specified in the same units as the Engineering Unit column.  When a channel hits the warning limit, it will be flagged as Yellow in the Channel Monitor (see @sec:channel_monitor).  The warning level can be left blank if no warning is desired.
* **Abort Level** An abort level can be implemented for each channel.  The abort level is specified in the same units as the Engineering Unit column.  When a channel hits the abort limit, it will be flagged as Red in the Channel Monitor (see @sec:channel_monitor).  The controller will also shut down if an abort level is reached.  The abort level can be left blank if no abort is desired.

To limit the tediousness of inputting channel table information into the GUI by hand, the channel table can be loaded from an Excel spreadsheet or Comma-separated-value file.  A channel table can be loaded by clicking the `Load Channel Table` button under the channel table, which will bring up a file selection dialog, enabling the user to select a file to load.  For convenience, a template Excel spreadsheet is attached to this page:

:::{note} Template Channel Table Download
{download}`attachments/channel_table_template.xlsx`
:::

A template Excel file can also be generated by creating a test in Rattlesnake and saving the empty channel table by clicking the `Save Channel Table` button under the channel table.  If a channel table is filled out in Rattlesnake's GUI, its contents will be saved to the file as well.

A complete test can be loaded by clicking the `Load Test from File` button.  See @sec:loading_rattlesnake_tests for more details.

### Environment Table

For combined environments tests, an environment table is also provided to the right of the standard Channel Table.  This table specifies which channels are used by which environments.  A channel can be used for multiple environments, a single environment, or no environments.  Channels used by no environments will still be measured and streamed to disk, but will not be sent to any environment for use in the respective control approaches.  The environment table is also used to specify which excitation devices are used by which environment.

For single environment tests, the environment table is not visible, and the software assumes that all channels in the channel table are used by the single environment.

If importing a channel table from an Excel spreadsheet for a combined environments test, the Environment Table can be specified as the columns after the main Channel Table information (starting in Column X with one column for each environment) with the environment name specified in row 2 and an entry (e.g. an `X` or some other mark) in the row corresponding to a given channel if that channel is used for the given environment.

### Data Acquisition Parameters

The final portion of `Data Acquisition Setup` tab specifies data acquisition parameters.  These parameters may change depending on the hardware selected.

* **Hardware Selector** The physical or virtual hardware used for the test.  See @sec:rattlesnake_hardware for hardware specific details of the controller.  For some devices, a file selector window will appear will appear when the device is selected, as that device needs more information to operate.  This is primarily the case for virtual hardware where some model of the test article must be loaded.  This is also used when a specific hardware device needs to access external functionality in a library such as a `dll` file.
* **Sample Rate** The sample rate of all hardware devices used for the test.  Some devices will have arbitrary sample rates, and some devices have fixed sample rates, so the options available will depend on the acquisition hardware being used.
* **Time per Read** The amount of data that the acquisition system will acquire with each read from the hardware.  By reading data in chunks, hardware input/output operations with relatively large overhead can be limited, and the buffer gives the controller time to catch up if e.g. the operating system decides to start a computationally intensive task in the background of the computer.  Note that specifying large numbers for this quantity (e.g. 10) will reduce the responsiveness of the controller, because the controller will potentially not receive the acquired data until 10 seconds after it was acquired.  Note also that this does not need to correspond to the Samples per Analysis Frame or any other signal processing parameter used by an environment.  Each environment should be buffered such that it creates appropriately sized analysis windows from the differently sized acquisition chunks.
* **Time per Write** The amount of data that the output system will write with each write to the hardware.  By writing data in chunks, hardware input/output operations with relatively large overhead can be limited, and the buffer gives the control time to catch up if e.g. the operating system decides to start a computationally intensive task in the background of the computer.   Note that this output is also buffered, so it does not need to be equal to the size of the data that will be created during each control loop of the controller.
* **Maximum Acquisition Processes** For specific hardware devices with large channel count tests, it can be difficult to pull down large quantities of data fast enough for the controller to keep up using a single process.  This option allows the user to specify how many processes can be given to the acquisition system to be used to stream data off the hardware.  Note that too many processes will bog down the computer, and too few will result in the controller falling behind.  Generally about 20-40 channels per processor is sufficient, but this will depend on the sample rate.  For higher sample rates, more processors may be needed.
* **Integration Oversample** For synthetic hardware devices that integrate equations of motion, an integration oversample factor can be specified.  This factor will be applied to the sample rate to determine the time step for the integration.  Generally a factor of 10 is sufficient for reasonably accurate data without significant computational expense.

### Initialize Data Acquisition

With the Data Acquisition Settings specified in the GUI, the Data Acquisition can be initialized by pressing the `Initialize Data Acquisition` button.  At this point, the controller will go through and create the programming interfaces to the hardware device, specify the sampling parameters, and create the channels on the devices.  The software will then proceed to the next tab.

@fig:data_acquisition_setup_tab shows a completed `Data Acquisition Setup` tab with three response channels and one output channel for a test with two environments `A` and `B`.  The first response and output channels are used by both environments, while the second response channel is used only by environment `A` and the third response channel is used only by environment `B`.

:::{figure} figures/data_acquisition_setup_tab.png
:label: fig:data_acquisition_setup_tab
:align: center
Example of a completed `Data Acquisition Setup` tab with three response channels and one output channel.
:::


## Environment Definition

The `Environment Definition` tab is the second tab in the Rattlesnake software.  It is in this tab that the various environments are defined.  The main tab will have one sub-tab for each environment, as shown in @fig:environment_definition_subtabs.

:::{figure} figures/environment_definition_subtabs.png
:label: fig:environment_definition_subtabs
:align: center
Sub-tabs for environments `A` and `B` in the `Environment Definition` tab.
:::

Different environment types will have different parameters that can be set.  See @sec:rattlesnake_environments for a description of each environment type in Rattlesnake and the parameters that define it.

When all environments are defined, the `Initialize Environments` button can be pressed to proceed to the next portion of the controller.

(sec:using_rattlesnake_system_identification)=
## System Identification

With the environments defined, the controller proceeds to the `System Identification` tab if required by any environment, shown in @fig:system_identification.  During this phase of the controller, the controller will develop relationships between the excitation signals and the responses of the test article to those excitation signals.  It will also make a measurement of the noise floor of the test.

:::{figure} figures/system_identification.png
:label: fig:system_identification
:align: center
System identification tab showing various signals and spectral quantities that can be used to evaluate the test.
:::

Not all environment types will require a system identification.  For environments that simply stream output data, a system identification will generally not be required.  However for any environment that aims to produce an output that creates some response on the test article, a system identification will be required to understand the relationships between the excitation signals and the response signals.

There will be one sub-tab for each environment that requires a System Identification.  System identification must be run for each sub-tab before the test can be run.  When system identification is performed, the software will first perform a noise floor measurement, where all channels are recorded, but no excitation signal is provided.  After the noise floor calculation completes, the system identification will begin.

The `System Identification` tab has been significantly overhauled since the initial version of controller.  The system identification now has a number of dedicated parameters on its tab that the user can select.  These are:

* **Samples per Frame** Samples per measurement frame used in the system identification.  Some environments force system identification to use the same frame size as the signal processing performed in the environment.  If this field cannot be edited here, then check the Environment Definition tab for a similar property.
* **Averaging Type** Type of averaging used in the system identification.  Linear averaging weights each measurement frame equally.  Exponential averaging treats more recent averages with a higher weight.
* **Noise Averages** Number of measurement frames to acquire to analyze the noise level of the test.
* **System ID Averages** Number of measurement frames to acquire to identify the system and compute transfer functions.
* **Averaging Coefficient** Averaging coefficient used to weight the most recent measurement frame if exponential averaging is used.
* **Estimator** The estimator to use to compute transfer functions when performing system identification.
* **Level** Root-mean-square voltage level to play to the shakers when computing system identification
* **Ramp Time** Time to ramp the system identification voltage to the test level.

The new system identification tab also gives the option to select the signal to use for system identification.

* **Signal Type** Excitation used for system identification.  Random and Burst random excitation are suitable for MIMO control problems.  Chirp and Pseudorandom should only be used for single-shaker excitation.
* **Window** Window function to apply to the data when computing transfer functions during system identification.
* **Overlap** Overlap to use between measurement frames when computing transfer functions.
* **On Fraction** Fraction of the measurement frame that the burst random excitation is active for.
* **Pretrigger** Percentage of the measurement frame that is before the burst random excitation.
* **Ramp Fraction** Percentage of the burst that is used to ramp from zero to the test level.
* **Bandwidth Minimum Frequency** The lowest frequency content in the excitation signal used in system identification.
* **Bandwidth Maximum Frequency** The highest frequency content in the excitation signal used in system identification.

The system identification phase can stream time data to disk by selecting a streaming file and clicking the `Stream Time Data` checkbox.  If streaming time data, the noise measurement will be saved to the variable name `time_data` and the system identification measurement will be saved to the variable name `time_data_1` (see @sec:using_rattlesnake_output_files for more information on the structure of this file).

* **Stream Time Data** If checked, time data from system identification will be streamed to disk.
* **Select File...** Opens a file dialog to select the file to which the system identification data will be streamed.
* **Streaming File:** File to which system identification data will be streamed.

The spectral data from the system identification can be saved to disk by clicking the `Save System Identification Data` button and selecting the file.  When loading system identification data, one must be careful that the loaded data has the same control and excitation degrees of freedom as the current environment, and that they are in the same order.  Otherwise the channels in the loaded data will not map correctly to the current environment.

* **Save System Identification Data** Save system identification spectral data to a file on the disk.
* **Load System Identification Data** Loads system identification data from a file on the disk.  System identification being loaded must contain the same control and excitation degrees of freedom in the same order as the current environment.

To run the system identification, there are buttons to Preview the Noise or System ID characterizations.  When ready, the `Start` button can be clicked.  It will run a Noise Characterization for the specified number of `Noise Averages`, and then subsequently run the System Identification characterization for the specified number of `System ID Averages`.  If the user wishes to run either the noise or system identification phases continuously, they can click the `Preview Noise` or `Preview System ID` buttons.


* **Preview Noise** Starts the noise identification of the system identification process in preview mode.  No excitation will be applied to the shakers.  The preview will continue until stopped manually.
* **Preview System Identification** Starts the system identification process in preview mode with excitation applied to the shakers.  The preview will continue until stopped manually.
* **Start** Starts the full system identification process, including noise characterization and transfer function computation.  Each step will stop automatically when the requested number of averages has been acquired.
* **Stop** Manually stops the system identification process.

As the system identification proceeds, the displays in the `Progress` portion of the tab will be updated.

* **Current Averages** Current number of system identification averages that have been acquired.
* **Total Averages** Total number of system identification averages that will be acquired.
* **System Identification Progress Bar** Graphically displays the system identification's progress for the current noise or system identification step.

Data will be plotted as the system identification proceeds.  The signals to visualize can be selected by clicking one or more of the `References` or `Responses` channels on the right side of the screen.  In the bottom right corner, there are options to show or hide various quantities of interest.  The `System Identification` tab can show the following:

* **Time Data Plot** Displays the most recently acquired measurement frames for the selected reference and response channels.
* **Transfer Function Plot** Displays the current estimation of the system transfer functions for the currently selected reference and response channels.
* **Impulse Response Plot** Displays the current estimation of the impulse response function for the currently selected reference and response channels.
* **Coherence and Condition Number Plots** Displays the multiple coherence for the currently selected response channel, and the condition number for each frequency line in the transfer function matrix.
* **Levels Plots** Displays the levels of the selected response and reference channels compared to the noise level on that measurement.
* **Kurtosis Plot** Displays the kurtosis of all channels in the system identification.  A kurtosis of 3 suggests gaussian measurements.  A value far from 3 suggests some nonlinear effect is distorting the gaussian excitation signals.

In the `References` section of the window, the user can select the reference channels to visualize:

* **References** Select the reference channels to visualize on the plots on the System Identification tab.
* **All** Visualize all references.
* **None** Visualize no references.

In the `Responses` section of the window, the user can select the response channels to visualize:

* **Responses** Select the response channels to visualize on the plots on the System Identification tab.
* **All** Visualize all responses.
* **None** Visualize no responses.

In the `Show` section of the window, the user can select which plots are visible and which are hidden.

* **Time Data** Check to show time data during the system identification.
* **Transfer Function** Check to show transfer functions in the system identification.
* **Impulse Response** Check to show the impulse response function, which is the IFFT of the transfer function.
* **Coherence and Conditioning** Check to show multiple coherence and condition number of the transfer function matrix.
* **Levels** Check to show levels for system identification and noise characterization
* **Kurtosis** Check to show kurtosis of response and reference channels.

## Test Predictions

Once the system identification phase completes, the controller will compute a test prediction for each environment where system identification was completed.  This prediction will be based on the measured transfer functions between output signals and measured responses, as well as the environment parameters specified on the `Environment Definition` tab.  Predictions will be made both for outputs required as well as response accuracy.  These predictions will be displayed on the `Test Predictions` tab.

(sec:using_rattlesnake_test_profiles)=
## Test Profiles

The `Test Profile` tab gives the user the ability to set up a test timeline for complex combined environments tests.  The user can add a list of events that will be executed at certain times during the test.  The tab will also display a graphical representation of the test timeline.

Events can be added or removed from the test timeline by clicking the `Add Event` or `Remove Event` buttons.  Users can also load a series of events from an Excel spreadsheet.

For each event, the following parameters are defined:

* **Timestamp (s)** The time in seconds after the timeline has started that the event will be executed.
* **Environment** The environment in which the event will occur.
* **Operation** The operation that will occur to the event.  Each environment defines its own set of operations that can be executed through the test profile interface.
* **Data** Any additional data that the operation requires.  For example, if a "Set Test Level" event is chosen, the Data field should specify the value that the test level is set to.

@fig:test_profile shows an example of a test profile that ramps up the test level of environment `A` from -6 to 0 dB, and then subsequently starts environment `B`.

:::{figure} figures/test_profile.png
:label: fig:test_profile
:align: center
Example test profile showing a ramp up of test level for environment `A` and subsequently starting environment `B`.
:::

(sec:using_rattlesnake_run_test)=
## Run Test

The `Run Test` tab is where Rattlesnake finally runs the test.  This tab again has sub-tabs for the different environments in the test, however these sub-tabs will not be enabled until the data acquisition system is armed.

Rattlesnake gives the user many options to save data to the disk through a set of Radio buttons at the top of this tab.  These options are:

* **No Streaming** Do not save data to disk, just run the test.
* **Start Streaming from Test Profile Instruction** Selecting this option allows data to be saved to disk after a "Global Start Streaming" event from the test profile is executed.  This allows the user to fine tune at which point in the test data is acquired.
* **Start Streaming at <environment> Target Test Level** Selecting this option starts streaming data when the selected environment hits its target test level.  This can be useful if, for example in a random environment, the user wishes to start at a low level and slowly creep up to the target test level.  If all data is saved, it might require a large amount of file space, so instead only the data at the test level of interest can be saved.
* **Start Streaming Immediately** Saves all data from the time the first environment starts until the data acquisition system is disarmed.
* **Manually Start/Stop Streaming** Allows the user to start and stop the measurement periodically throughout the test.  A `Start Streaming` button will appear when this option is selected.  When clicked, the button will change to `Stop Streaming`.  Multiple data streams can be saved in a given test.  These will be stored to separate variables in the output NetCDF4 file (see @sec:using_rattlesnake_output_files).

When streaming data, it is important to note that the software does not stop streaming until the data acquisition system is disarmed by pressing the `Disarm Data Acquisition` button.  This is because for a combined-environments test, the environments may have down-time between them where no environment is running, and that data should still be saved.

The `Run Test` tab contains global `Arm Data Acquisition` and `Disarm Data Acquisition` buttons that start and stop the data acquisition system.  When the data acquisition system is armed, the user can no longer change streaming options, and the sub-tabs for each environment are enabled.  The user can then start or stop each environment manually using the `Start Environment` or `Stop Environment` buttons on each environment's sub-tab.  The sub-tab for each environment is described more thoroughly in @sec:rattlesnake_environments.

Alternatively, the user can start or stop the test profile by clicking on the `Start Profile` or `Stop Profile` buttons respectively.  The profile capability also includes the option to switch the active environment sub-tab when an event is executed so the user can see the results.  Note that the profile options only appear when a profile has been defined on the `Test Profile` tab.

@fig:run_test_tab shows an example `Run Test` tab with test profile events.

:::{figure} figures/run_test_tab.png
:label: fig:run_test_tab
:align: center
Run Test Tab.
:::

(sec:using_rattlesnake_output_files)=
## Rattlesnake Output Files <!--Section 3.8-->

After data is acquired, the user may wish to analyze or plot the data acquired for a given test report.  Rattlesnake stores data in a self-documenting netCDF file [@unidata2019_netcdf], which can be read by multiple platforms.  The output file is described as self-documenting because it contains all parameters necessary to reconstruct a given test using the Rattlesnake controller.  Any parameter that is set by the user in the GUI is stored to the netCDF file.

A full description of the netCDF file format is out of this document's scope, but the important points are briefly described here.  NetCDF files have a number of data structures.  Variables are multi-dimensional arrays of data.  Dimensions describe the axes of the variable arrays.  Attributes are used to store small data such as scalars or 1D arrays.  NetCDF files can be separated into different groups, and each group can have its own variables, dimensions, and attributes.

The Rattlesnake output files contain the following data members:

### NetCDF Dimensions <!--Subsection 3.8.1-->

* **`response_channels`** The number of response channels in a given test
* **`output_channels`** The number of output channels in a given test
* **`time_samples`** The number of time samples measured in the file, this dimension can expand as more data is acquired.
* **`time_samples_X`** If manual streaming is used and streaming is started multiple times, each subsequent stream will have the `time_samples` name with an underscore and appended number (e.g. `time_samples_1`, `time_samples_2`)
* **`num_environments`** The total number of environments in the test

### NetCDF Attributes <!--Subsection 3.8.2-->

* **`sample_rate`** The global sample rate of the data acquisition system
* **`time_per_write`** The amount of data put to the output hardware per write operation, in seconds
* **`time_to_read`** The amount of data read from the acquisition hardware per read operation, in seconds
* **`hardware`** The hardware index used for the test.
  * 0 -- National Instruments NI-DAQmx
  * 1 -- HBK LAN-XI Open API
  * 2 -- Data Physics Quattro
  * 3 -- Data Physics 900 Series
  * 4 -- Virtual Control defined by Exodus Modal Solution
  * 5 -- Virtual Control defined by State Space Matrices
  * 6 -- Virtual Control defined with a SDynPy System
* **`hardware_file`** The path to the file used to define the Virtual test article, or the path to the external code library used by the data acquisition hardware.  Otherwise, it will be `None`
* **`maximum_acquisition_processes`** The maximum number of processes that the LAN-XI hardware can use for acquisition
* **`output_oversample`** The oversample used either due to sample rate restrictions on the data acquisition system, or due to oversampling the integration

### NetCDF Variables <!--Subsection 3.8.3-->

* **`time_data`** The measured data from the test. Type: 64-bit float; Dimensions: `response_channels` by `time_samples`
* **`time_data_X`** If manual streaming is used and streaming is started multiple times, each subsequent stream will have the `time_data` name with an underscore and appended number (e.g. `time_data_1`, `time_data_2`)
* **`environment_names`** The name of each environment. Type: string; Dimensions: `num_environments`
* **`environment_active_channels`** The channels active in each environment.  1 if active, 0 if not. Type: 8-bit int; Dimensions: `response_channels` $\times$ `num_environments`

### Channels Group <!--Subsection 3.8.4-->

The netCDF files from Rattlesnake store all channel information into a separate group called `channels`.  Inside the `channels` group, there is a variable for each column of the channel table.  See Section #sec:channel_table for more complete descriptions of each channel variable.

* **`/channels/node_number`** The node number of each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/node_direction`** The instrument direction of each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/comment`** The commend for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/serial_number`** The serial number of the instrument for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/triax_dof`** The sensor degree of freedom for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/sensitivity`** The sensitivity of the instrument for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/unit`** The engineering unit of the instrument for each channel. Type: str; Dimensions: `response_channels`
* **`/channels/make`** The manufacturer of the instrument for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/model`** The model number or product name of the instrument for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/expiration`** The expiration date of the instrument's calibration for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/physical_device`** The physical device that the instrument is connected to for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/physical_channel`** The channel in the physical device that the instrument is attached to for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/channel_type`** The type of quantity that is measured by the channel.  Type: str; Dimensions: `response_channels`
* **`/channels/minimum_value`** The minimum voltage that the channel can accept.  Type: str; Dimensions: `response_channels`
* **`/channels/maximum_value`** The maximum voltage that the channel can accept.  Type: str; Dimensions: `response_channels`
* **`/channels/coupling`** The coupling type used by each channel (AC/DC/filter/etc.).  Type: str; Dimensions: `response_channels`
* **`/channels/excitation_source`** The excitation source for each channel, used to specify CCLD/ICP/IEPE.  Type: str; Dimensions: `response_channels`
* **`/channels/excitation`** The excitation current value used in the signal conditioning for each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/feedback_device`** The device that the channel's generator originates from if the channel is an output channel.  Type: str; Dimensions: `response_channels`
* **`/channels/feedback_channel`** The channel that the channel's generator originates from if the channel is an output channel.  Type: str; Dimensions: `response_channels`
* **`/channels/warning_level`** The warning level of each channel.  Type: str; Dimensions: `response_channels`
* **`/channels/abort_level`** The abort level of each channel.  Type: str; Dimensions: `response_channels`

### Environment Groups <!--Subsection 3.8.5-->

Environment-specific attributes, dimensions, and variables are also stored within a group corresponding to each environment.  For example, in the case where there were two environments "A" and "B", parameters specific to environment "A" would be stored within the group "A" in the netCDF file, and similarly for "B".  See @sec:rattlesnake_environments for more information on environment-specific parameters.

### Reading Rattlesnake Output Files using Python <!-- Subsection 3.8.6-->

To read data from a netCDF using Python, it is recommended to use the `netCDF4` Python package.  This library is a dependency of Rattlesnake, so if the user is not running Rattlesnake via an executable, this package should already be installed in the user's Python ecosystem.

netCDF4 provides a sleek Python interface into the data of a netCDF4 file.  This section will assume the command `import netCDF4 as nc4` was used to import the package, so `nc4` is used as a shorter alias.

A netCDF4 dataset can be opened using the following command:

```python
dataset = nc4.Dataset('path/to/netcdf4/file.nc4')
```

after which all data can be accessed through the `dataset` object.
    
Attribute names can be queried using the `dataset.ncattrs()` function and the attribute values can be accessed directly from the `dataset` object using that name.

```python
>>> dataset.ncattrs()
['sample_rate',
'samples_per_write',
'samples_per_read',
'hardware',
'hardware_file']

>>> dataset.sample_rate
2048
```

Dimensions can be accessed using the `dataset.dimensions` property, which gives a Python `dict` where the keys are the dimension names and the values are references to the dimension.  The size of the dimension can be accessed using the `size` parameter in each dimension object.

```python
>>> dataset.dimensions
{'response_channels': <class 'netCDF4._netCDF4.Dimension'>: name = 'response_channels', size = 30,
'output_channels': <class 'netCDF4._netCDF4.Dimension'>: name = 'output_channels', size = 3,
'time_samples': <class 'netCDF4._netCDF4.Dimension'> (unlimited): name = 'time_samples', size = 31745,
'num_environments': <class 'netCDF4._netCDF4.Dimension'>: name = 'num_environments', size = 2}

>>> dataset.dimensions['response_channels'].size
30
```

Variables can be accessed similarly to dimensions using the `dataset.variables` property.  Variables have many properties that may be interesting to the users, including the netCDF dimensions that were used to size the variable (accessible with the `dimensions` parameter) or the actual shape of the array (accessible with the `shape` parameter).  The data inside the dimension can be accessed by slicing or indexing the array, or simply passing it to a `numpy` array.  Note that slicing or indexing the variable returns the data in a `numpy` masked array which allows data to potentially to be missing from the array.  Rattlesnake does not use the missing data capabilities of the netCDF file, so data can safely be transformed directly to a regular `numpy` array.

```python
>>> dataset.variables
{'time_data': <class 'netCDF4._netCDF4.Variable'>
float64 time_data(response_channels, time_samples)
unlimited dimensions: time_samples
current shape = (30, 31745)
filling on, default _FillValue of 9.969209968386869e+36 used,
'environment_names': <class 'netCDF4._netCDF4.Variable'>
vlen environment_names(num_environments)
vlen data type: <class 'str'>
unlimited dimensions: 
current shape = (2,),
'environment_active_channels': <class 'netCDF4._netCDF4.Variable'>
int8 environment_active_channels(response_channels, num_environments)
unlimited dimensions: 
current shape = (30, 2)
filling on, default _FillValue of -127 ignored}

# Get the dimensions used by the variable
>>> dataset.variables['time_data'].dimensions
('response_channels', 'time_samples')

# Get the shape of the variable
>>> dataset.variables['time_data'].shape
(30, 31745)

# Access via slice returns a masked array
>>> dataset.variables['time_data'][0,0]
masked_array(data=-0.00312098,
mask=False,
fill_value=1e+20)

# Can pass directly to a numpy array to get the full variable data
>>> np.array(dataset.variables['time_data'])
array([[-3.12098493e-03,  4.26820006e-03,  3.77395182e-03, ...,
         2.00690958e-01,  3.38505511e-01,  0.00000000e+00],
        [-6.10438702e-03,  1.50628999e-02, -1.50619535e-02, ...,
         2.67639515e-01,  5.50047023e-01,  0.00000000e+00],
        [-3.42732089e-03,  7.76593927e-03, -2.66239267e-03, ...,
         2.05434816e-01,  3.21815820e-01,  0.00000000e+00],
         ...,
        [ 3.71743658e-06, -8.77497995e-08, -8.80558595e-06, ...,
         -5.26559214e-06,  0.00000000e+00,  0.00000000e+00],
        [-1.32020650e-05,  2.74453772e-05, -2.01551409e-05, ...,
         -1.02501347e-05,  0.00000000e+00,  0.00000000e+00],
        [-5.96816619e-07, -1.47868461e-05, -5.24157875e-05, ...,
         -1.61722666e-05,  0.00000000e+00,  0.00000000e+00]])
```

Group names in the netCDF dataset can be queried using `dataset.groups`, which returns a dictionary similar to the dimensions and variables.  Groups can also be accessed by indexing the dataset directly with the group name.  A group object can be treated exactly the same as the root-level dataset, and will have its own set of attributes, dimensions, and variables.

```python
>>> dataset['channels'].variables['node_number']
<class 'netCDF4._netCDF4.Variable'>
vlen control(response_channels)
vlen data type: <class 'str'>
path = /channels
unlimited dimensions: 
current shape = (30,)
```

### Reading Rattlesnake Output Files using Matlab <!-- Subsection 3.8.7-->

Matlab can also be used to read netCDF files from Rattlesnake.  The Matlab `ncdisp` function can be used to quickly determine which parameters are in a file.

```matlab
>>> ncdisp('path/to/netcdf/file.nc4')
Source:
           path/to/netcdf/file.nc4
Format:
           netcdf4
Global Attributes:
           sample_rate       = 2048
           samples_per_write = 512
           samples_per_read  = 512
           hardware          = 2
           hardware_file     = 'path/to/hardware/file.exo'
Dimensions:
           response_channels = 30
           output_channels   = 3
           time_samples      = 31745 (UNLIMITED)
           num_environments  = 2
Variables:
    time_data                  
           Size:       31745x30
           Dimensions: time_samples,response_channels
           Datatype:   double
    environment_names          
           Size:       2x1
           Dimensions: num_environments
           Datatype:   UNSUPPORTED DATATYPE
    environment_active_channels
           Size:       2x30
           Dimensions: num_environments,response_channels
           Datatype:   int8
Groups:
    /channels/
        Variables:
            node_number      
                   Size:       30x1
                   Dimensions: /response_channels
                   Datatype:   UNSUPPORTED DATATYPE
       .            
       .
       .
```

Attributes, dimensions, and other metadata can be read into Matlab using the `ncinfo` function.  Variables information must be read using the `ncread` function.

```matlap
>>> finfo = ncinfo('path/to/netcdf/file.nc4')

finfo = 

  struct with fields:

      Filename: 'C:\Users\dprohe\Documents\Local_Respositories\Combined_Environments_Controller\test_data\BARC_Exodus_Test\barc_combined.nc4'
          Name: '/'
    Dimensions: [1x4 struct]
     Variables: [1x3 struct]
    Attributes: [1x5 struct]
        Groups: [1x3 struct]
        Format: 'netcdf4'
        
>>> finfo.Dimensions(1)

ans = 

  struct with fields:

         Name: 'response_channels'
       Length: 30
    Unlimited: 0
    
>>> time_data = ncread('path/to/netcdf/file.nc4','time_data')
```

One issue with the Matlab interface is that string variables are unsupported.  This means that the majority of the channel information cannot be read through the Matlab netCDF interface.  However, they can be read using the lower level `h5read` function.

```matlab
>>> ncread(file,'channels/node_number')
Error using netcdf.getVar (line 137)
12 is not a recognized netCDF datatype.

Error in internal.matlab.imagesci.nc/read (line 605)
                data = netcdf.getVar(gid, varid);

Error in ncread (line 66)
vardata = ncObj.read(varName, varargin{:});

>>> h5read(file,'/channels/node_number')

ans =

  30x1 cell array
```

(sec:loading_rattlesnake_tests)=
## Loading Rattlesnake Tests

It can be tedious to set up a test from scratch each time a test is to be run, so Rattlesnake offers two ways to load test settings from files.
    
On the `Data Acquisition Setup` page, selecting the `Load Test From File` button allows the user to load in a netCDF data file that was output from Rattlesnake.  As all the test metadata is stored to this file, Rattlesnake can read the file and set itself up accordingly to reproduce a given test.  Note that difficulties may arise using this approach if parameters specified by file paths are no longer valid.  For example, if the control law is read from a given file on one computer, but the file is in a different place on a separate computer, Rattlesnake will not be able to find the file.
    
The second way to load in an entire test is by using the Test Profile functionality in the Combined Environments mode.  While this capability was designed to make it easier to load in complex multi-environment test setups, it can be used just as effectively for single environment tests.  See @sec:combined_environments for more information.

(sec:channel_monitor)=
## Channel Monitor <!-- Section 3.10-->

To aid with understanding the test levels and headroom available for the sensors in the test, a Channel Monitor is available where the levels are shown for each channel.  The channel monitor is displayed by clicking on the `Channel Monitor` button on the lower left side of the GUI.  The display shows both an instantaneous level (green) as well as a running historical maximum (blue).  If a channel reaches the Warning or Abort level, it will be flagged with a yellow or red tint, respectively.  These warnings "latch"; once the level is reached, it will stay highlighted in the channel monitor until the `Clear Alerts` button is clicked.  @fig:channel_monitor shows an example channel monitor.

:::{figure} figures/channel_monitor.png
:label: fig:channel_monitor
:align: center
View of the Channel Monitor dialog box showing several channels that have reached the "warning" level (highlighted yellow) and one channel that has reached the "abort" level (highlighted red).
:::

The aspect ratio of the Channel Monitor can be customized to different sizes modifying the `Channels per Row`.
