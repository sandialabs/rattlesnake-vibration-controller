---
numbering:
  heading_2:
    start: 1
    enumerator: E%s
    template: Example %s
  figure:
    enumerator: E1.%s
  table:
    enumerator: E1.%s
  equation:
    enumerator: E1.%s
  code:
    enumerator: E1.%s
---
# Example Problem using NI cDAQ Hardware and the NI DAQmx Interface

(sec:example_nidaqmx)=
# Example Problem using NI cDAQ Hardware and the NI DAQmx Interface

This appendix will demonstrate a example problem using real hardware on a simple beam structure.  This example problem will use a simple cDAQ hardware.  A NI cDAQ-9185 chassis is used with two NI-9232 acquisition modules and two NI-9260 output modules.  @fig:examplecdaqhardware shows this hardware setup.

:::{figure} figures/example_cdaq_hardware.jpg
:label: fig:examplecdaqhardware
:align: center
NI cDAQ hardware used for this demonstration.
:::

This example problem will start with a modal test using an impact hammer, then demonstrate a modal test with a shaker, and finally perform a MIMO Random and MIMO Transient test.

The beam being tested is 24 inches long by 1 inch high by 0.75 inches deep and made from steel. It is resting on soft foam to simulate a free-free boundary condition. We have six acquisition channels, so we will use four of those channels to measure response, and the other two to measure force and voltage signals as required by the testing configuration.  @fig:examplecdaqbeamstructure shows the beam, as well as a block with the coordinate axes labeled.

:::{figure} figures/example_cdaq_beam_structure.jpg
:label: fig:examplecdaqbeamstructure
:align: center
Beam structure that will be tested in this example problem.
:::

## Modal Testing with an Impact Hammer

The first demonstration will be a simple impact test using a modal hammer.  This will require no output from Rattlesnake.

### Data Acquisition Setup

We will excite the beam in the thin direction, so we will place four uniaxial accelerometers on the beam at the 1, 12, 18, and 23 inch distances from the left end. We will name these locations 1, 12, 18, and 23, respectively.

As we install instrumentation, we will fill out a channel table. Rather than filling out the channel table in the GUI, we will load it from an Excel spreadsheet, which allows us to more easily save our work. We can get a template channel table to fill out by opening Rattlesnake and selecting the Modal Testing environment (@fig:examplecdaqmodaltestingselection).

:::{figure} figures/example_cdaq_modal_testing_selection.png
:label: fig:examplecdaqmodaltestingselection
:align: center
Selecting the Modal Environment in Rattlesnake
:::

We can then click the `Save Channel Table` button to save out the empty channel table to an Excel spreadsheet (@fig:examplecdaqsavechanneltable).

:::{figure} figures/example_cdaq_save_channel_table.png
:label: fig:examplecdaqsavechanneltable
:align: center
Saving out an empty channel table to fill out for the test.
:::

As we add instrumentation and plug it into our data acquisition system, we will record that information in the channel table. In order to tell Rattlesnake the correct device, we open NI Measurement and Automation Explorer (NI MAX). In the Devices and Interfaces section, we can see our network devices, which includes our cDAQ system and its associated modules (@fig:examplecdaqnimax).

:::{figure} figures/example_cdaq_nimax.png
:label: fig:examplecdaqnimax
:align: center
NI MAX dialog showing the cDAQ devices.
:::

Our acquisition channels will be plugged into devices `cDAQ9185-217ED78Mod1` and `cDAQ9185-217ED78Mod2` which have acquisition channels `ai0`, `ai1`, and `ai2`, and our output devices are `cDAQ9185-217ED78Mod3` and `cDAQ9185-217ED78Mod4` which have output channels `ao0` and `ao1`.

After we add our accelerometers, we will also add a channel for the modal hammer. Note that we will vary the hammer excitation location, so we will simply put in an arbitrary node number that we will override later. We will hit the beam on the surface opposite the side that the accelerometers are mounted on, so the direction of the impact is the identical to the directions of the accelerometers.

When we finish our channel table it looks like @fig:examplecdaqchanneltableexcel.  Note that there is no output from Rattlesnake, so there are no entries in the Output Feedback in the channel table.

:::{figure} figures/example_cdaq_channel_table_excel.png
:label: fig:examplecdaqchanneltableexcel
:align: center
Channel Table in Excel ready to load into Rattlesnake.
:::

The test setup after attaching the accelerometers and plugging in the instrumentation is shown in @fig:examplecdaqtestsetupmodalimpact.

:::{figure} figures/example_cdaq_test_setup_modal_impact.jpg
:label: fig:examplecdaqtestsetupmodalimpact
:align: center
Modal test setup using the impact hammer.
:::

We can now load our channel table into Rattlesnake by clicking the Load Channel Table button shown in @fig:examplecdaqloadchanneltable.

:::{figure} figures/example_cdaq_load_channel_table.png
:label: fig:examplecdaqloadchanneltable
:align: center
Loading in the channel table for the test.
:::

We should see that the information from the Excel spreadsheet is populated in the Rattlesnake GUI.

In the `Data Acquisition Parameter` section we will leave `NI DAQmx` selected. We will set the `Sample Rate` to `5120` Hz, as this is a valid output rate for both the cDAQ device types we are using. We can leave the `Time per Read` and `Time per Write` to their defaults.  @fig:examplecdaqmodaltestimpactdaqsettings shows the loaded channel table and data acquisition settings.

:::{figure} figures/example_cdaq_modal_test_impact_daq_settings.png
:label: fig:examplecdaqmodaltestimpactdaqsettings
:align: center
Data acquisition settings for the Modal Test using an impact hammer.
:::

At this point, we can click the `Initialize Data Acquisition` button shown in @fig:examplecdaqinitializedataacquisition to proceed to the next stage of the test.

:::{figure} figures/example_cdaq_initialize_data_acquisition.png
:label: fig:examplecdaqinitializedataacquisition
:align: center
Click the `Initialize Data Acquisition` button to proceed.
:::

### Environment Setup

At this point, we now need to set up our modal testing environment on the Environment Definition tab. As this is a simple beam, it will likely take a while to decay, so we will set up a 2 second measurement frame by setting the `Samples per Frame` to `10240`.

We will lower the `Number of Averages` to `5`.

For `Triggering`, we will set the `Triggering Type` to `Every Frame`, and set the `Acceptance` to `Manual`. We set the `Trigger Channel` to the Force channel with a `2%` `Pretrigger`. The `Trigger Level` we will set to `1%` and the `Hysteresis Level` to half of that value. Finally, we will set the `Hysteresis Frame %` to `50%`.

In the `Channel Selections` table, we will check the `Reference` column for the Force channel.

When all of these parameters are adjusted, we can click the `Initialize Environments` button to proceed.  @fig:examplecdaqmodalimpactsettings shows the window with all settings set.

:::{figure} figures/example_cdaq_modal_impact_settings.png
:label: fig:examplecdaqmodalimpactsettings
:align: center
Modal Test parameters for the impact test.
:::

### Test Profile
We will leave the `Test Profile` tab empty. Simply click the `Initialize Profile` button to proceed on this tab.

### Run Test

We will save the standard modal data file from this test, so we won't worry about setting up streaming.

To enable the Modal Environment, first click the `Arm Data Acquisition` button. At this point, the modal environment buttons will become enabled, as shown in @fig:examplecdaqmodaltestingready

:::{figure} figures/example_cdaq_modal_testing_ready.png
:label: fig:examplecdaqmodaltestingready
:align: center
Run Test tab with data acquisition armed
:::

To start, we will use the `Degree of Freedom Override` section to choose our actual excitation location.  This allows us to more quickly adjust the impact location than returning to the `Data Acquisition Setup` page and reinitializing everything in the channel table. Click the `Add Override` button, and in the row that appears, change the `Channel` column to the force channel. In the `New Node` column, put `23`, which will be the position we excite at. For the `New Direction`, put `Y-`.  This is shown in @fig:examplecdaqchanneloverride.

:::{figure} figures/example_cdaq_channel_override.png
:label: fig:examplecdaqchanneloverride
:align: center
Using the Degree of Freedom Override to adjust the impact location metadata.
:::

Now with our channels correct, we can set up data display windows. We will use the `New from Template...` dropdown menu to create `Drive Point (Imaginary)`, `Drive Point Coherence`, and `Reference Autospectrum` channels. We will also manually create 5 windows by clicking the `New` button, and set them to visualize the time histories for each channel; these will help us determine if we need to change the trigger settings, frame length, or add a window. Clicking the `Tile` button will expand the windows to fill up the available space.  @fig:examplecdaqmodalimpactemptywindows shows the result of these operations.

:::{figure} figures/example_cdaq_modal_impact_empty_windows.png
:label: fig:examplecdaqmodalimpactemptywindows
:align: center
Run Test tab with modal windows created.
:::

We will first preview the measurement to ensure that all of the parameters we have selected are adequate. Click the `Preview Acquisition` button and impact the structure behind the 23Y- accelerometer. If the data acquisition triggers, data should appear in the windows as shown in @fig:examplecdaqtrigger.

:::{figure} figures/example_cdaq_trigger.png
:label: fig:examplecdaqtrigger
:align: center
Data acquisition triggered, waiting for acceptance or rejection of the measurement.
:::

After the data appears, the `Accept` and `Reject` buttons should become active. If the signal decays within the window and the force looks reasonable, we can click `Accept` which will then proceed to the next measurement. After a measurement is accepted, averaged quantities such as coherence, frequency response functions, and autopower spectra will also be computed as shown in @fig:examplecdaqfirstaccept. Only Spectra (FFTs) and time histories can be visualized prior to accepting a measurement.

:::{figure} figures/example_cdaq_first_accept.png
:label: fig:examplecdaqfirstaccept
:align: center
Spectral quantities appear after acceptance.
:::

After a handful of measurements are taken, we should be able to see the coherence and frequency response functions stabilize. We can evaluate the measurement by investigating the drive point FRF, coherence, reference autospectra, or any other quantities of interest.  @fig:examplecdaqallaccept shows these results.

:::{figure} figures/example_cdaq_all_accept.png
:label: fig:examplecdaqallaccept
:align: center
Quantities of interest after 5 averages.
:::

If this looks satisfactory, we can proceed with the measurement. Clicking the `Stop Acquisition` button will stop the preview, and allow us to start a real measurement. We must assign a `Data File` to save to by clicking the `Select File...` button. We will call our file `beam_hammer_impact.nc4`. To ensure we do not overwrite our data as we rove the hammer, we will click the `Auto-Increment` checkbox to automatically increment the filename.  This is shown in @fig:examplecdaqfilename.

:::{figure} figures/example_cdaq_filename.png
:label: fig:examplecdaqfilename
:align: center
Setting a file name for the modal data.
:::

With the data file selected, we can click the `Start Acquisition` button. We perform the measurement identically to the preview case. When the specified number of averages have been acquired, the measurement will stop automatically as shown in @fig:examplecdaqmodalimpactsacquired.

:::{figure} figures/example_cdaq_modal_impacts_acquired.png
:label: fig:examplecdaqmodalimpactsacquired
:align: center
Acquisition stops automatically after the desired number of impacts.
:::

Now we will acquire the next measurement at node 18. We will update the `New Node` column of the `Degree of Freedom Override` table to be `18`. Note that this will also change the labels on the data windows. We will need to update the response channels on those windows to make sure we are still visualizing the drive point data, as shown in @fig:examplecdaqupdatewindows.

:::{figure} figures/example_cdaq_update_windows.png
:label: fig:examplecdaqupdatewindows
:align: center
Updating the signal being visualized in the data visualization window.
:::

We can simply press the `Start Acquisition` button, and a new file will be created. After that measurment has been obtained, we will then repeat again, updating the `Degree of Freedom Override` table to node `12` and start a measurement exciting at that location.

When these measurements are complete, we can click `Disarm Data Acquisition` to stop the measurement, and close the Rattlesnake software.

### Analyzing Modal Data

To analyze these data, we will use SDynPy[^sdynpy], which is an open-source Python-based structural dynamics toolset. SDynPy is well-integrated with Rattlesnake, making it trivial to load in data from a Rattlesnake test.

[^sdynpy]: SDynPy can be installed via pip using `pip install sdynpy`, or otherwise downloaded from the [Github repository](https://github.com/sandialabs/sdynpy)

For example, reading in a modal dataset with SDynPy looks like:

```{code} python
# Import SDynPy and Numpy
import sdynpy as sdpy
import numpy as np

# Read a rattlesnake file using `read_modal_data`
time_data, frfs, coherence, channel_table = sdpy.rattlesnake.read_modal_data(
'beam_hammer_impact_0000.nc4')
```
    
We can plot the data easily using SDynPy's plotting functionality:
    
```{code} python
frfs.plot(one_axis=False,subplots_kwargs={'layout':'constrained'})
```

    
The results of this command are shown in @fig:examplecdaqplottedfrfs.

:::{figure} figures/example_cdaq_plotted_frfs.png
:label: fig:examplecdaqplottedfrfs
:align: center
Frequency response functions plotted in SDynPy from the modal impact test.
:::

We can combine and plot the frequency response functions from our multiple datasets:
    
```{code} python
all_frfs = []
for i in range(3):
    _, frfs, _, _ = sdpy.rattlesnake.read_modal_data('beam_hammer_impact_{:04}.nc4'.format(i))
all_frfs.append(frfs)
all_frfs = np.concatenate(all_frfs)
all_frfs.plot(one_axis=False,subplots_kwargs={'layout':'constrained','figsize':(10,6)})
```

The results of this command are shown in @fig:examplecdaqallfrfs.

:::{figure} figures/example_cdaq_all_frfs.png
:label: fig:examplecdaqallfrfs
:align: center
All FRFs from the three impacts
:::

Ideally we would like to plot mode shapes of this test, so as a first step towards that goal we will construct a test geometry. This contains nodes, coordinate systems, and optionally tracelines or elements.  We can construct nodes with the ID numbers given in the test and coordinates specified by each node's position along the beam.  Here we construct and print a `node_array` in SDynPy
    
```{code} python
nodes = sdpy.node_array(id = [1,12,18,23],
                        coordinate = [[1,0,0],
                                      [12,0,0],
                                      [18,0,0],
                                      [23,0,0]])
print(nodes)
```

which has output
    
```

    Index,     ID,        X,        Y,        Z, DefCS, DisCS
     (0,),      1,    1.000,    0.000,    0.000,     1,     1
     (1,),     12,   12.000,    0.000,    0.000,     1,     1
     (2,),     18,   18.000,    0.000,    0.000,     1,     1
     (3,),     23,   23.000,    0.000,    0.000,     1,     1
```

    
We will do similarly with a global coordinate system
    
```{code} python
css = sdpy.coordinate_system_array(id=1,name='global')
print(css)
```

which has output
    
```

   Index,     ID,                 Name, Color,       Type
      (),      1,               global,     1,  Cartesian
```
    
We can then construct a Geometry object. We will then add a traceline to connect the nodes together.  We will then plot the geometry.
    
```{code} python
geometry = sdpy.Geometry(nodes,css)
geometry.add_traceline([1,12,18,23])
geometry.plot()
```

The plotted geometry is shown in @fig:examplecdaqgeometry.

:::{figure} figures/example_cdaq_geometry.png
:label: fig:examplecdaqgeometry
:align: center
Geometry used to plot mode shapes in the modal test.
:::

Now that we have a geometry, we can fit modes and plot mode shapes. We will only fit up to approximately 2000 Hz, which is where the antialiasing filter of the cDAQ device is clearly active.  We will use the `PolyPy_GUI` in SDynPy to do the mode fits.  We can add our geometry directly to the fitter by assigning to the `geometry` attribute.
    
```{code} python
ppgui = sdpy.PolyPy_GUI(all_frfs)
ppgui.geometry = geometry
```

    
We can select frequency ranges and polynomial orders to solve for, then click `Compute Stabilization` to proceed, as shown in @fig:examplecdaqpolypyranges.

:::{figure} figures/example_cdaq_polypy_ranges.png
:label: fig:examplecdaqpolypyranges
:align: center
Setting up frequency ranges and polynomial orders in PolyPy.
:::

We can then select stable poles from the diagram and plot resynthesized FRFs or mode indicator functions.  This operation and the results are shown in @fig:examplecdaqpolypyselectpoles.

:::{figure}
:label: fig:examplecdaqpolypyselectpoles
:align: center

![Selecting stable poles](figures/example_cdaq_polypy_select_poles.png)

![Visualizing resynthesized FRFs](figures/example_cdaq_polypy_resynthesis.png)

Selecting stable poles and visualizing resynthesized FRFs.
::: 

We can then plot mode shapes from the test.  Note that with only four sensors, our mode shapes will not be very highly resolved; however, they clearly correspond to the first three bending modes of the beam.  The first mode shape is shown in @fig:examplecdaqmodalfits.

:::{figure} figures/example_cdaq_modal_fits.png
:label: fig:examplecdaqmodalfits
:align: center
First mode shape fit to the modal data from the impact test.
:::

(sec:example_nidaqmx_modal_shaker)=
## Modal Testing with a Modal Shaker

Next we will demonstrate how to run a modal test with a shaker. A cap is placed over the accelerometer at node 1, and a force sensor is threaded into the cap. A receptacle is mounted to the force gauge to accept the shaker stinger.

### Data Acquisition Settings

Recall that because Rattlesnake is required to measure the voltage associated with any drive signals it outputs, adding a shaker to a modal test requires two channels: one for the force sensor to measure the reference force, and one to measure the shaker voltage. Therefore, if we aim to measure the force for this test, we can only accommodate a single shaker input, as four of the six available channels channels are already taken by the accelerometers.

When we set up the shaker, we put a tee into the output channel from the data acquisition system, and we tee off the shaker drive signal into the last acquisition channel and to the shaker amplifier, as shown in @fig:examplecdaqmodalshakerhardwaresetup.  @fig:examplecdaqmodalshakersetup shows a wider view with the modal shaker visible.

:::{figure} figures/example_cdaq_modal_shaker_hardware_setup.jpg
:label: fig:examplecdaqmodalshakerhardwaresetup
:align: center
Data acquisition setup for the shaker test showing the shaker signal teed into the last acquisition channel.
:::

:::{figure} figures/example_cdaq_modal_shaker_setup.jpg
:label: fig:examplecdaqmodalshakersetup
:align: center
Modal shaker setup for running a modal test.
:::

As we fill out the channel table, we enter the physical device and channel that the shaker voltage is teed into in the `Channel Definition` section of the channel table, and enter the physical device and channel that the shaker voltage originates from in the `Output Feedback` section. In our case, the shaker signal comes from the `ao0` channel on the third module `cDAQ9185-217ED78Mod3`. The shaker signal is teed into acquisition channel `ai2` on the second module `cDAQ9185-217ED78Mod2`. We also enter the force sensor into the channel table.  note that the direction associated with the force is now in the `Y+` direction, as the push direction of the shaker is opposite the measurement direction of the accelerometer that is covered by the cap.  @fig:examplecdaqmodalshakerchanneltable shows the channel table set up in Excel.

:::{figure} figures/example_cdaq_modal_shaker_channel_table.png
:label: fig:examplecdaqmodalshakerchanneltable
:align: center
Channel table set up for the modal shaker test.
:::

Note that when assigning a node and direction to the voltage signal, you do not want to use the same values as for the drive point, otherwise Rattlesnake will not be able to determine which frequency response function is the drive point FRF. The author's practice is to use the same node number as the force sensor, but offset by some value (e.g. 100 in this case). In the case of multiple shakers, it will be clear which voltage corresponds to which shaker if such a convention is followed.

With the channel table set up, we can now start Rattlesnake and load the channel table. We will again select the Modal Testing environment, click the `Load Channel Table` button and load our file. We will again set the `Sample Rate` to `5120` Hz. With these parameters set, we can click the `Initialize Data Acquisition` button to proceed.  @fig:examplecdaqmodalshakerdaqsettings shows these settings in the Rattlesnake GUI.

:::{figure} figures/example_cdaq_modal_shaker_daq_settings.png
:label: fig:examplecdaqmodalshakerdaqsettings
:align: center
Data acquisition settings for the Modal Shaker Test
:::

### Environment Definition for Pseudorandom Excitation

Now we set up the Modal Testing environment parameters. Because we only have one shaker, we can use the `Pseudorandom` excitation, so we will plan for that.

We will leave the `Samples per Frame` at its default value. We will change the `Number of Averages` to 10. We will leave `Triggering Type` set to `Free Run`, but we will add a delay of `5` in the `Wait for Steady State` box, to allow the system to come to steady state, as assumed by the Pseudorandom excitation.

In the `Channel Selections` box, we select the Force channel as the reference.

In the `Signal Generator` portion of the window, we select the `Pseudorandom` tab to set up a Pseudorandom signal. We can truncate the `Frequency Range` from `10 -- 2000` to remove the large rigid body motion that can occur at low frequency, as well as the portion of the excitation that is in the antialiasing filters of the system. We will set the `RMS` value to `0.1` V.

With these parameters set, we can click the `Initialize Environments` button to proceed.  @fig:examplecdaqmodalpseudorandomsettings shows these settings in the Rattlesnake GUI.

:::{figure} figures/example_cdaq_modal_pseudorandom_settings.png
:label: fig:examplecdaqmodalpseudorandomsettings
:align: center
Parameters set for the Pseudorandom shaker excitation.
:::

### Test Profile
We will leave the `Test Profile` tab empty. Simply click the `Initialize Profile` button to proceed on this tab.

### Run Test

:::{warning} cDAQ Pulses
cDAQ devices can output a pulse when plugged in, turned on, or activated for the first time, so make sure the shaker amplifier is off or the gain is set to zero before pressing the `Arm Data Acquisition` button for the first time. Once we `Arm Data Acquisition`, then we can unmute the shaker amplifier.
:::

To run the test we can again `Arm Data Acquisition` to enable the modal controls. Because we will not be roving the shaker, we have entered the proper node number into the channel table, and therefore do not need to utilize the `Degree of Freedom Override` table. We will again display the standard diagnostics of drive point FRF and coherence, reference autospectra, and time response from all sensors and the voltage signal.  This is shown in @fig:examplecdaqmodalshakerempty.

:::{figure} figures/example_cdaq_modal_shaker_empty.png
:label: fig:examplecdaqmodalshakerempty
:align: center
Shaker testing with data acquisition armed and data visualization windows created.
:::

We can then `Preview Acquisition` to see how the test parameters look. Data will take a few seconds to appear after the shaker starts due to the `Wait for Steady State` parameter that was set.  Ideally if the pseudorandom excitation has reached steady state, we should see no appreciable difference between the time response of measurement frames, as the exact same signal is being applied and exact same responses are being measured.  This is shown in @fig:examplecdaqspeudorandompreview.

:::{figure} figures/example_cdaq_speudorandom_preview.png
:label: fig:examplecdaqspeudorandompreview
:align: center
Previewing the data acquisition with the pseudorandom excitation.
:::

If we are happy with the preview, we can move to `Start Acquisition`. We define a file to save the data to called `beam_shaker_pseudorandom.nc4`.  This is shown in @fig:examplecdaqpseudorandomacquire.  The measurement and shakers should stop automatically when the requested number of averages have been acquired.

:::{figure} figures/example_cdaq_pseudorandom_acquire.png
:label: fig:examplecdaqpseudorandomacquire
:align: center
Acquiring the pseudorandom modal data.
:::

Once the test completes, we can mute our amplifier and Disarm Data Acquisition.

### Other Shaker Excitation Types
We will return to the `Environment Definition` tab and set up a `Burst Random` test. We saw that the hammer ring-down took approximately 2 seconds, so we will set the `Samples Per Frame` value to `15360` to achieve a 3 second frame. We will set the `Number of Averages` to `30`. We will set the `Triggering Type` to `First Frame`, remove the `Wait for Steady State` value, and set the `Trigger Channel` to the voltage channel with a `Pretrigger` of `1%`. We will set the `Trigger Level` to `1%`, the `Hysteresis Level` to half that value, and the `Hysteresis Frame %` to `25%`.

In the `Signal Generator` section, we will select a `Burst Random` signal and set the `On Percentage` to `33%` to provide a 1 second burst and 2 second ring down. We will keep the other parameters the same as our previous Pseudorandom test.

@fig:examplecdaqburstrandomparameters shows the parameters in the Rattlesnake GUI.  Click the `Initialize Environments` button to proceed, and return to the `Run Test` tab.  

:::{figure} figures/example_cdaq_burst_random_parameters.png
:label: fig:examplecdaqburstrandomparameters
:align: center
Test parameters for the Burst Random excitation
:::

We can preview the signal by clicking the `Preview Acquisition` button to ensure that the triggering has been set up correctly and the signals ring down appropriately.  @fig:examplecdaqburstrandompreview shows the previewed data.

:::{figure} figures/example_cdaq_burst_random_preview.png
:label: fig:examplecdaqburstrandompreview
:align: center
Previewing the measurement with the initial parameter set.
:::

Apparently the shaker adds enough damping that the structure stops ringing by approximately 1/2 second after the burst finishes. Let's therefore `Stop Acquisition`, `Disarm Data Acquisition`, and return to the `Environment Definition` tab.  We will update the `Samples Per Frame` value to `5120` to achieve a 1 second measurement frame.  We will also change the burst `On Percentage` to `50%`.

@fig:examplecdaqburstrandomparametersv2 shows the updated parameters in the Rattlesnake GUI.  Click the `Initialize Environments` button to confirm these settings, and return to the `Run Test` tab.  

:::{figure} figures/example_cdaq_burst_random_parameters_v2.png
:label: fig:examplecdaqburstrandomparametersv2
:align: center
Updated burst random parameters after previewing the initial guess at the parameters.
:::

We can then preview this new configuration by `Arm Data Acquisition` and `Start Preview`.  The results are shown in @fig:examplecdaqburstrandompreviewv2.  This configuration looks good and runs 3x faster, so we will keep it.

:::{figure} figures/example_cdaq_burst_random_preview_v2.png
:label: fig:examplecdaqburstrandompreviewv2
:align: center
Previewing the measurement with the updated parameter set.
:::

Press the `Stop Acquisition` button to stop the preview, set a file to store the results to (`beam_shaker_burst.nc4`) and `Start Acquisition`. The acquisition should stop automatically when 30 averages are obtained.  @fig:examplecdaqburstrandomacquire shows the measurement midway through acquisition.

:::{figure} figures/example_cdaq_burst_random_acquire.png
:label: fig:examplecdaqburstrandomacquire
:align: center
Acquiring the burst random data
:::

### Analyzing Modal Shaker Data

We can now analyze the shaker data similarly to the hammer data using SDynPy[^sdynpy].  We will also load in the impact data from the tip.  While we did not excite the structure at node 1 with the hammer, we can use symmetry of the beam to compare between the two measurement types.
    
```{code} python
# Read a rattlesnake file using `read_modal_data`
time_data_pr, frfs_pr, coherence_pr, channel_table_pr = sdpy.rattlesnake.read_modal_data(
    'beam_shaker_pseudorandom.nc4')
time_data_br, frfs_br, coherence_br, channel_table_br = sdpy.rattlesnake.read_modal_data(
    'beam_shaker_burst.nc4')
# Compare to the impact
time_data_im, frfs_im, coherence_im, channel_table_im = sdpy.rattlesnake.read_modal_data(
    'beam_hammer_impact_0000.nc4')
```
    
We can use SDynPy to extract common degrees of freedom from each test and plot the FRFs for each.  
    
```{code} python
import matplotlib.pyplot as plt

ref_resp_dofs = sdpy.coordinate_array([1,23],'Y+')
compare_frfs = [frfs_pr[ref_resp_dofs[::-1]][np.newaxis],
frfs_br[ref_resp_dofs[::-1]][np.newaxis],
frfs_im[ref_resp_dofs][np.newaxis]]
fig,ax = plt.subplots(2)
for frf in compare_frfs:
frf.plot(ax)
ax[-1].legend(['Psuedorandom','Burst','Hammer'])
ax[-1].set_yscale('log')
```

This gives the following plot:

:::{figure} figures/example_cdaq_modal_comparison.png
:label: fig:examplecdaqmodalcomparison
:align: center
Comparison of hammer impact testing compared to shaker excitation with two types of signals.
:::

We can note that the peaks are shifted slightly lower in frequency for the shaker testing, likely due to the small amount of mass loading of the shaker.  Additionally, there appears to be some shaker-structure interactions going on near the third mode, but otherwise there is reasonably good agreement between the three tests.

## Running a Vibration Test with Rattlesnake

Now we will demonstrate how we can run a controlled vibration test with Rattlesnake.  We will demonstrate both Random and Transient vibration control in this section.

(sec:example_nidaqmx_spec)=
### Constructing Vibration Specifications

Before we can run a test, we will first generate some environment data.  This is the response that we will control to during the vibration testing. We will do this using the modal hammer to randomly impact the structure, and then we will try to get the shaker to control to that response.

To acquire environment data, we will simply run the modal environment in preview mode, but now use Rattlesnake's `Streaming` functionality to record time data, from which we will construct specifications for vibration testing.

We will disconnect the shaker from our test article. Then we return to the `Environment Definition` page and set the `Triggering Type` to `Free Run`.

On the `Run Test` tab, before selecting `Arm Data Acquisition` we select `Start Streaming Immediately` and `Select Streaming File...` to save a time history file called `environment_streaming.nc4`.  This is shown in @fig:examplecdaqsetupstreaming.

:::{figure} figures/example_cdaq_set_up_streaming.png
:label: fig:examplecdaqsetupstreaming
:align: center
Run Test tab with Streaming enabled
:::

Now we can `Arm Data Acquisition` and then `Preview Acquisition`. Once the data acquisition is running, we will tap the structure with the modal hammer repeatedly along the length of the beam.  Don't worry that the hammer is not plugged in, as we only care about the accelerometer response. Once we have recorded a few seconds of impacts, we can select `Stop Acquisition` and `Disarm Data Acquisition`. Note that the streaming file will not be available until the data acquisition is disarmed.  We can now close Rattlesnake because we are done with the Modal environment.

We will now load the streaming file with SDynPy to create an environment.
    
```{code} python
time_history, channel_table = sdpy.rattlesnake.read_rattlesnake_output(
    'environment_streaming.nc4')
```
    
We will extract just the first four acceleration channels for our environment, which are shown in @fig:examplecdaqspecificationsignal
    
```{code} python
accelerations = time_history[:4]
accelerations.plot()
```

:::{figure} figures/example_cdaq_specification_signal.png
:label: fig:examplecdaqspecificationsignal
:align: center
Measured acceleration data that we will turn into vibration specifications.
:::

Looking at the data, we can truncate it to just the time range of interest, shown in @fig:examplecdaqspecificationsignaltruncated.
    
```{code} python
truncated_accelerations = accelerations.extract_elements_by_abscissa(2.3,16.3)
truncated_accelerations.plot()
```
    
:::{figure} figures/example_cdaq_specification_signal_truncated.png
:label: fig:examplecdaqspecificationsignaltruncated
:align: center
Truncated acceleration data that we will turn into vibration specifications.
:::

To construct a random vibration environment, we will compute CPSDs from this signal. While this isn't really an appropriate signal for a random vibration specification, it is good enough to demonstrate Rattlesnake's capabilities.  @fig:examplecdaqcpsdspecification shows the APSDs, which are the diagonal entries of the CPSD specification matrix.
    
```{code} python
random_spec = truncated_accelerations.cpsd(samples_per_frame = 5120, overlap=0.5, window = 'hann')
# Remove low frequency and antialiasing filters
random_spec = random_spec.extract_elements_by_abscissa(20,2000)
# Scale it down so we don't break shakers trying to hit it
random_spec = random_spec / 100

ax = random_spec.plot_asds()
fig = ax[0,0].figure
fig.tight_layout()
```

:::{figure} figures/example_cdaq_cpsd_specification.png
:label: fig:examplecdaqcpsdspecification
:align: center
Diagonal entries of the random vibration specification CPSD matrix.
:::

We will also construct a transient specification that we will use to test the transient vibration control, shown in @fig:examplecdaqtransientspecification.
    
```{code} python
# We will make it shorter and again scale it down to make it so we don't break anything
transient_spec = truncated_accelerations.extract_elements_by_abscissa(11.48,16.3) / 5
transient_spec.plot()
```

:::{figure} figures/example_cdaq_transient_specification.png
:label: fig:examplecdaqtransientspecification
:align: center
Transient specification used for vibration control.
:::

We can now save these specifications to disk.
    
```{code} python
random_spec.to_rattlesnake_specification('random_specification.npz')
transient_spec.to_rattlesnake_specification('transient_specification.npz')
```

    
### Setting up the Shakers and Channel Table

We will now set up our vibration test.  We add a second shaker to the other end of the beam using a second drive cap over the accelerometer at node 23, shown in @fig:examplecdaqtwoshakersetup.

:::{figure} figures/example_cdaq_two_shaker_setup.jpg
:label: fig:examplecdaqtwoshakersetup
:align: center
Vibration test setup showing two shakers attached to the test article.
:::

Because we only have 6 acquisition channels, we must remove the force gauge to make room for the second shaker drive signal. We have now teed the first shaker's drive from `cDAQ9185-217ED78Mod3`'s `ao0` channel to the `ai1` channel of `cDAQ9185-217ED78Mod2`. The second shaker will be driven by `cDAQ9185-217ED78Mod3`'s `ao1` channel and is teed to the `ai2` channel of `cDAQ9185-217ED78Mod2`.  This is shown in @fig:examplecdaqtwoshakerhardwaresetup.

:::{figure} figures/example_cdaq_two_shaker_hardware_setup.jpg
:label: fig:examplecdaqtwoshakerhardwaresetup
:align: center
cDAQ hardware setup using two shakers.
:::

Our channel table now looks like @fig:examplecdaqtwoshakerchanneltable.

:::{figure} figures/example_cdaq_two_shaker_channel_table.png
:label: fig:examplecdaqtwoshakerchanneltable
:align: center
Channel table for the random vibration test setup
:::

### Setting up the Random Vibration Test
Now that the test hardware and channel table are set up, we will set up our random vibration test. We will open Rattlesnake and select the `MIMO Random Vibration` environment as shown in @fig:examplecdaqmimorandomselection.

:::{figure} figures/example_cdaq_mimo_random_selection.png
:label: fig:examplecdaqmimorandomselection
:align: center
Selecting the MIMO Random Vibration environment
:::

On the `Data Acquisition Setup` tab, we will `Load Channel Table` to import our Excel channel spreadsheet. We will also set the `Sample Rate` to `5120`. The other parameters can be left at their default values.  @fig:examplecdaqvibrationdataacquisitionsetup shows these parameters.

:::{figure} figures/example_cdaq_vibration_data_acquisition_setup.png
:label: fig:examplecdaqvibrationdataacquisitionsetup
:align: center
Data acquisition setup for the vibration tests.
:::

Click the `Initialize Data Acquisition` button to proceed.

On the `Environment Definition` tab, we will keep the `Samples Per Frame` set to `5120`, which is consistent with the parameters used to construct the specification. In the `Control Python Script` section, we will click `Load` to select a control law. We will use the `control_laws.py` file, which is located in the `control_laws` folder of the main Rattlesnake directory. In the `Control Python Function` drop-down, we will select `pseudoinverse_control`.

In the `Control Channels` section, we will select all four acceleration channels, which we will use to control to the environment.

Finally we will click the `Load Spec` button and navigate to our random specification file `random_specification.npz` from SDynPy. The specification should then appear in the plots.

With these parameters selected, we can click the Initialize Environments button to proceed.  @fig:examplecdaqrandomvibrationparameters shows the parameters in the Rattlesnake GUI.

:::{figure} figures/example_cdaq_random_vibration_parameters.png
:label: fig:examplecdaqrandomvibrationparameters
:align: center
Random vibration parameters specified in the Rattlesnake GUI.
:::

At this point, we must run the `System Identification` phase of the controller. We will set the Level to `0.1 V RMS`, then click `Start` to start the system identification. A noise floor check will first be run, then the system identification follows. Rattlesnake will determine the transfer functions between the two shaker drive signals and the four accelerometer channels. When the specified number of averages have been obtained, the System Identification will stop automatically.  @fig:examplecdaqrandomvibrationsystemid shows the system identification phase of the controller.

:::{figure} figures/example_cdaq_random_vibration_system_id.png
:label: fig:examplecdaqrandomvibrationsystemid
:align: center
System identification for the Random Vibration environment.
:::

On the `Test Prediction` tab, Rattlesnake will make predictions of the voltage that it will need to output to achieve the control specified, as well as the responses predicted to be obtained. Here it is important to note that the output signals are not too large for the shakers.  @fig:examplecdaqrandomvibrationpredictions shows the predictions for this test.

:::{figure} figures/example_cdaq_random_vibration_predictions.png
:label: fig:examplecdaqrandomvibrationpredictions
:align: center
Test predictions for the random vibration environment
:::

If the predictions look good, we can proceed to the Test Profile tab, which we will again leave blank and click the Initialize Test Profile button to proceed to the Run Test tab.

### Running a Random Vibration Test
On the `Run Test` tab, we see a different interface than the one we saw for the modal test. We can click the `Arm Data Acquisition` button to enable the Random environment.

To start, we will set the `Current Test Level` to `-12` dB to ensure when we start the test we don't break anything. We will then step up slowly to the full `0` dB level test.

We can then click the `Start Environment` button to start the test.  @fig:examplecdaqrandomvibrationstart shows the GUI prior to starting the test with the test level turned down to -12 dB.

:::{figure} figures/example_cdaq_random_vibration_start.png
:label: fig:examplecdaqrandomvibrationstart
:align: center
Ready to start a random vibration test in Rattlesnake
:::

As the test is running, the main Rattlesnake interface will show the sum of APSDs, which is the trace of the CPSD matrix and can be thought of as an average response of the test (@fig:examplecdaqrandomvibrationsumasds). To visualize individual channels, windows can be created in the `Data Display` section of the window (@fig:examplecdaqrandomvibrationindividualchannels).

:::{figure} figures/example_cdaq_random_vibration_sum_asds.png
:label: fig:examplecdaqrandomvibrationsumasds
:align: center
Rattlesnake GUI showing the average test response
:::

:::{figure} figures/example_cdaq_random_vibration_individual_channels.png
:label: fig:examplecdaqrandomvibrationindividualchannels
:align: center
Individual channel APSDs showing how well each channel is matching the test level.
:::

If the controller is controlling sufficiently well, we can step up the level to `0` dB as shown in @fig:examplecdaqrandomvibrationsumasdsfulllevel.

:::{figure} figures/example_cdaq_random_vibration_sum_asds_full_level.png
:label: fig:examplecdaqrandomvibrationsumasdsfulllevel
:align: center
Vibration control at full specification level.
:::

Feel free to investigate other control laws in the `control_laws.py` file, as they can result in a better control than the standard pseudoinverse control. @fig:examplecdaqrandomvibrationsumasdsmatchtrace shows the results using the `match_trace_pseudoinverse` approach.  In particular, we see the regions between the peaks improve in this case.

:::{figure} figures/example_cdaq_random_vibration_sum_asds_match_trace.png
:label: fig:examplecdaqrandomvibrationsumasdsmatchtrace
:align: center
Improved control using a closed loop control law found in `match_trace_pseudoinverse`.
:::

Clearly, Rattlesnake has obtained relatively good control for this test.

To save data from the test, we can either stream time data using the same approach used previously, or we can simply use the `Save Current Spectral Data` to save out the current realizations of the control CPSDs, which we will call `random_vibration_spectral_results.nc4`. This is the quickest way to do comparisons between the specification and the control achieved. We can load this file with SDynPy and use SDynPy to make comparisons between the test and specification.  The results are shown in @fig:examplecdaqrandomvibrationerrorsummary.


```{code} python
control, spec, drives = sdpy.rattlesnake.read_random_spectral_data(
    'random_vibration_spectral_results.nc4')
spec = spec.extract_elements_by_abscissa(20,2000)
control = control.extract_elements_by_abscissa(20,2000)
spec.error_summary(figure_kwargs={'figsize':(10,6)},Control=control)
```

    
:::{figure} figures/example_cdaq_random_vibration_error_summary.png
:label: fig:examplecdaqrandomvibrationerrorsummary
:align: center
Error summary between the test and specification for the random vibration control.
:::

We can then `Stop Environment`, `Disarm Data Acquisition` and close the software.

### Setting up the Transient Vibration Test

The last demonstration will show how we can perform a MIMO Transient environment test. Open Rattlesnake and select the `MIMO Transient` environment as shown in @fig:examplecdaqmimotransientselection.

:::{figure} figures/example_cdaq_mimo_transient_selection.png
:label: fig:examplecdaqmimotransientselection
:align: center
Selecting the MIMO Transient environment
:::

We will set the `Data Acquisition Setup` tab identical to what was used before, as shown in @fig:examplecdaqvibrationdataacquisitionsetup. Click the `Initialize Data Acquisition` button to proceed.

On the `Environment Definition tab`, we will again `Load` a control law file. This time we will load the `transient_control_laws.py` file, which contains transient control laws for Rattlesnake. We will select the `pseudoinverse_control` function.

In the `Control Channels` section, we will again select all four accelerometer channels as the control channels.

Finally, we will select the `Load Signal` button to load in our specification file `transient_specification.npz`.  @fig:examplecdaqtransientparameters shows these parameters set in the Rattlesnake GUI.  With these parameters set, we will click the `Initialize Environments` button to proceed.

:::{figure} figures/example_cdaq_transient_parameters.png
:label: fig:examplecdaqtransientparameters
:align: center
Parameters set for the Transient environment in the Rattlesnake GUI.
:::

The Transient environment will also need to perform a `System Identification`. Again set the `Level` to `0.1 V RMS` and click `Start` to start the noise floor and system identification. Here it may be useful to look at the `Impulse Response`, as that is useful to understand how transient control will be performed.  This is shown in @fig:examplecdaqtransientsystemid.

:::{figure} figures/example_cdaq_transient_system_id.png
:label: fig:examplecdaqtransientsystemid
:align: center
System Identification for the Transient environment with Impulse Response function shown.
:::

Once the system identification has been performed, Rattlesnake will again make a prediction as to the results of the test and show it on the `Test Predictions` tab. It will show the peak voltage it aims to output, as well as the response it predicts that it will get compared to the desired response.  @fig:examplecdaqtransientpredictions shows these predictions.

:::{figure} figures/example_cdaq_transient_predictions.png
:label: fig:examplecdaqtransientpredictions
:align: center
Test predictions and voltage that will be output for the transient control.
:::

We can then proceed to the `Test Profile` tab, click the `Initialize Profile` button, and we are ready to run the test.

### Running a Transient Test

We can now click the `Arm Data Acquisition` button to enable the Transient control functionality. We can set the `Signal Level` to `-12` dB to start so we don't break anything when we run. If it runs successfully, then we can step back up to full level.

Rattlesnake will identify the control signal as it comes through the data acquisition and surround it with a box, as shown in @fig:examplecdaqtransientcontrolresults.

:::{figure} figures/example_cdaq_transient_control_results.png
:label: fig:examplecdaqtransientcontrolresults
:align: center
Running the transient control with the control signal identified.
:::

To visualize individual channels, new windows can be created in the `Data Display` portion of the window.  The four control channels can be seen in @fig:examplecdaqtransientcontrolchannels

:::{figure} figures/example_cdaq_transient_control_channels.png
:label: fig:examplecdaqtransientcontrolchannels
:align: center
Control achieved on each control channel in the Transient vibration test.
:::

To save out data, we can click the `Save Current Control Data` button. We will save the data to the file `transient_vibration_control_results.nc4`. This can be loaded into SDynPy to do comparisons to the desired signal.  It can be useful to zoom in to the starts of each impulse to see that while the ring down does match quite well, the initial impulse cannot be achieved as well with the modal shakers as it was with the impact hammer.  These results are shown in @fig:examplecdaqtransientzoom.

    
```{code} python
control, specification, drives = sdpy.rattlesnake.read_transient_control_data(
'transient_vibration_control_results.nc4')

ax = specification.plot(
    one_axis=False, plot_kwargs = {'linewidth':0.5},
    subplots_kwargs = {'figsize':(10,6),'sharex':True,'sharey':True})
for a,c in zip(ax.flatten(),control):
    c.plot(a,plot_kwargs = {'linewidth':0.5})
# Zoom in on a specific portion
a.set_xlim(0.45,0.55)
```
    
:::{figure} figures/example_cdaq_transient_zoom.png
:label: fig:examplecdaqtransientzoom
:align: center
Zoom into an impact during the transient control showing good results achieved after the impact.
:::

## Summary
This appendix has walked through several example tests using real hardware with Rattlesnake.  Users are encouraged to use this appendix as a quick-start guide to using Rattlesnake.  A relatively inexpensive NI cDAQ device was used to perform modal and vibration testing on a simple beam structure with minimal instrumentation.

Modal testing was performed using both hammer impacts and shaker excitation.  Modes were fit to the data using the SDynPy Python package.

Vibration testing was performed using modal shakers.  An impact hammer was used to create an environment that the vibration test would aim to replicate.  Both MIMO Random and MIMO transient vibration tests were run, where the MIMO Random attempted replicate a prescribed CPSD matrix, and the MIMO Transient attempted to replicate the time history directly.  Both control schemes achieved good results.
