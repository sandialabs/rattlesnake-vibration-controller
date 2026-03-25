---
numbering:
  figure: false
---
# random_vibration_run



(sec:random_vibration_run:test_output_voltages_groupbox)=
* [**Output Voltages (RMS)**](#fig:random_vibration_run:test_output_voltage_list) Current excitation voltages being output from the controller.


(sec:random_vibration_run:test_response_error_groupbox)=
* [**Response Error (dB)**](#fig:random_vibration_run:test_response_error_list) Current control RMS dB error.  Double clicking on an item will open up a window showing that channel's magnitude.  Channels will be highlighted yellow if they hit a warning limit and red if they hit an abort limit.

* [**Current Test Level**](#fig:random_vibration_run:current_test_level_selector) Current test level in dB.  0 dB is the actual test level from the specification.
* [**Target Test Level**](#fig:random_vibration_run:target_test_level_selector) Target test level in dB.  This can be used to automatically trigger streaming or used to stop the controller after a specified amount of time.
* [**Total Test Time**](#fig:random_vibration_run:total_test_time_display) Total time that the environment has been running for.
* [**Time at Level**](#fig:random_vibration_run:time_at_level_display) Time that the environment has been running at the current test level.
* [**Environment Progress**](#fig:random_vibration_run:test_progress_bar) When the bar reaches 100%, the environment will stop automatically.  Will not be active during a continuous run.
* [**Continuous Run**](#fig:random_vibration_run:continuous_test_radiobutton) Run the environment until it is manually stopped.
* [**Run Time**](#fig:random_vibration_run:test_time_selector) Amount of time that the environment will run for.
* [**Run for (timed run)**](#fig:random_vibration_run:timed_test_radiobutton) Run the environment for a specified amount of time
* [**at Target Test Level**](#fig:random_vibration_run:time_test_at_target_level_checkbox) If checked, the timer will only run when the test is at the target test level.
* [**Start Environment**](#fig:random_vibration_run:start_test_button) Starts the environment controlling to the specification.
* [**Stop Environment**](#fig:random_vibration_run:stop_test_button) Stops the environment manually
* [**Save Current Spectral Data**](#fig:random_vibration_run:save_current_spectral_data_button) Saves current spectral data to a NetCDF4 file.
* [**Sum of ASDs Display**](#fig:random_vibration_run:global_test_performance_plot) This plot shows the trace of the response CPSD matrix compared to the specification.  This is essentially an "average" quantity, and individual channels may be higher or lower at each frequency line.

(sec:random_vibration_run:data_display_groupbox)=
* [**Control Channel 1**](#fig:random_vibration_run:control_channel_1_selector) Row of the CPSD matrix to create a window for.
* [**Control Channel 2**](#fig:random_vibration_run:control_channel_2_selector) Column of the CPSD matrix to create a window for
* [**Data Type**](#fig:random_vibration_run:data_type_selector) Data type to display on the new visualization window.
* [**Create Window**](#fig:random_vibration_run:create_window_button) Creates a new window to visualize the response of a given entry in the CPSD matrix with Control Channel 1 and Control Channel 2 specifying the row and column of the CPSD matrix, and Data Type specifying how the channel is displayed.
* [**Show All Autospectral Densities**](#fig:random_vibration_run:show_all_asds_button) Show Autospectral Densities (ASDs) for all control channels. This will create a window for every channel in the test.
* [**Show All Spectral Densities (phase/coh)**](#fig:random_vibration_run:show_all_csds_phscoh_button) Show the entire CPSD matrix.  This will show Autospectral Densities on the diagonal, and phase and coherence on the off-diagonals.  WARNING: this will create a number of windows equal to the number of control channels squared, which for large tests could be a huge number of windows that can bog down the user interface of the software and make it unresponsive.
* [**Show All Spectral Densities (real/imag)**](#fig:random_vibration_run:show_all_csds_realimag_button) Show the entire CPSD matrix. This will show Autospectral Densities on the diagonal, and real and imaginary parts on the off-diagonals.  WARNING: this will create a number of windows equal to the number of control channels squared, which for large tests could be a huge number of windows that can bog down the user interface of the software and make it unresponsive.
* [**Tile All Windows**](#fig:random_vibration_run:tile_windows_button) Tiles all windows over the main monitor
* [**Close All Windows**](#fig:random_vibration_run:close_windows_button) Closes all visualization windows




:::{figure} figures/random_vibration_run__test_output_voltages_groupbox.png
:label: fig:random_vibration_run:test_output_voltages_groupbox
 Output Voltages (RMS) Settings
:::

:::{figure} figures/random_vibration_run__test_output_voltage_list.png
:label: fig:random_vibration_run:test_output_voltage_list
 **Output Voltages (RMS)** Current excitation voltages being output from the controller.
:::

:::{figure} figures/random_vibration_run__test_response_error_groupbox.png
:label: fig:random_vibration_run:test_response_error_groupbox
 Response Error (dB) Settings
:::

:::{figure} figures/random_vibration_run__test_response_error_list.png
:label: fig:random_vibration_run:test_response_error_list
 **Response Error (dB)** Current control RMS dB error.  Double clicking on an item will open up a window showing that channel's magnitude.  Channels will be highlighted yellow if they hit a warning limit and red if they hit an abort limit.
:::

:::{figure} figures/random_vibration_run__current_test_level_selector.png
:label: fig:random_vibration_run:current_test_level_selector
 **Current Test Level** Current test level in dB.  0 dB is the actual test level from the specification.
:::

:::{figure} figures/random_vibration_run__target_test_level_selector.png
:label: fig:random_vibration_run:target_test_level_selector
 **Target Test Level** Target test level in dB.  This can be used to automatically trigger streaming or used to stop the controller after a specified amount of time.
:::

:::{figure} figures/random_vibration_run__total_test_time_display.png
:label: fig:random_vibration_run:total_test_time_display
 **Total Test Time** Total time that the environment has been running for.
:::

:::{figure} figures/random_vibration_run__time_at_level_display.png
:label: fig:random_vibration_run:time_at_level_display
 **Time at Level** Time that the environment has been running at the current test level.
:::

:::{figure} figures/random_vibration_run__test_progress_bar.png
:label: fig:random_vibration_run:test_progress_bar
 **Environment Progress** When the bar reaches 100%, the environment will stop automatically.  Will not be active during a continuous run.
:::

:::{figure} figures/random_vibration_run__continuous_test_radiobutton.png
:label: fig:random_vibration_run:continuous_test_radiobutton
 **Continuous Run** Run the environment until it is manually stopped.
:::

:::{figure} figures/random_vibration_run__test_time_selector.png
:label: fig:random_vibration_run:test_time_selector
 **Run Time** Amount of time that the environment will run for.
:::

:::{figure} figures/random_vibration_run__timed_test_radiobutton.png
:label: fig:random_vibration_run:timed_test_radiobutton
 **Run for (timed run)** Run the environment for a specified amount of time
:::

:::{figure} figures/random_vibration_run__time_test_at_target_level_checkbox.png
:label: fig:random_vibration_run:time_test_at_target_level_checkbox
 **at Target Test Level** If checked, the timer will only run when the test is at the target test level.
:::

:::{figure} figures/random_vibration_run__start_test_button.png
:label: fig:random_vibration_run:start_test_button
 **Start Environment** Starts the environment controlling to the specification.
:::

:::{figure} figures/random_vibration_run__stop_test_button.png
:label: fig:random_vibration_run:stop_test_button
 **Stop Environment** Stops the environment manually
:::

:::{figure} figures/random_vibration_run__save_current_spectral_data_button.png
:label: fig:random_vibration_run:save_current_spectral_data_button
 **Save Current Spectral Data** Saves current spectral data to a NetCDF4 file.
:::

:::{figure} figures/random_vibration_run__global_test_performance_plot.png
:label: fig:random_vibration_run:global_test_performance_plot
 **Sum of ASDs Display** This plot shows the trace of the response CPSD matrix compared to the specification.  This is essentially an "average" quantity, and individual channels may be higher or lower at each frequency line.
:::

:::{figure} figures/random_vibration_run__data_display_groupbox.png
:label: fig:random_vibration_run:data_display_groupbox
 Data Display Settings
:::

:::{figure} figures/random_vibration_run__control_channel_1_selector.png
:label: fig:random_vibration_run:control_channel_1_selector
 **Control Channel 1** Row of the CPSD matrix to create a window for.
:::

:::{figure} figures/random_vibration_run__control_channel_2_selector.png
:label: fig:random_vibration_run:control_channel_2_selector
 **Control Channel 2** Column of the CPSD matrix to create a window for
:::

:::{figure} figures/random_vibration_run__data_type_selector.png
:label: fig:random_vibration_run:data_type_selector
 **Data Type** Data type to display on the new visualization window.
:::

:::{figure} figures/random_vibration_run__create_window_button.png
:label: fig:random_vibration_run:create_window_button
 **Create Window** Creates a new window to visualize the response of a given entry in the CPSD matrix with Control Channel 1 and Control Channel 2 specifying the row and column of the CPSD matrix, and Data Type specifying how the channel is displayed.
:::

:::{figure} figures/random_vibration_run__show_all_asds_button.png
:label: fig:random_vibration_run:show_all_asds_button
 **Show All Autospectral Densities** Show Autospectral Densities (ASDs) for all control channels. This will create a window for every channel in the test.
:::

:::{figure} figures/random_vibration_run__show_all_csds_phscoh_button.png
:label: fig:random_vibration_run:show_all_csds_phscoh_button
 **Show All Spectral Densities (phase/coh)** Show the entire CPSD matrix.  This will show Autospectral Densities on the diagonal, and phase and coherence on the off-diagonals.  WARNING: this will create a number of windows equal to the number of control channels squared, which for large tests could be a huge number of windows that can bog down the user interface of the software and make it unresponsive.
:::

:::{figure} figures/random_vibration_run__show_all_csds_realimag_button.png
:label: fig:random_vibration_run:show_all_csds_realimag_button
 **Show All Spectral Densities (real/imag)** Show the entire CPSD matrix. This will show Autospectral Densities on the diagonal, and real and imaginary parts on the off-diagonals.  WARNING: this will create a number of windows equal to the number of control channels squared, which for large tests could be a huge number of windows that can bog down the user interface of the software and make it unresponsive.
:::

:::{figure} figures/random_vibration_run__tile_windows_button.png
:label: fig:random_vibration_run:tile_windows_button
 **Tile All Windows** Tiles all windows over the main monitor
:::

:::{figure} figures/random_vibration_run__close_windows_button.png
:label: fig:random_vibration_run:close_windows_button
 **Close All Windows** Closes all visualization windows
:::