---
numbering:
  figure: false
---
# random_vibration_prediction



(sec:random_vibration_prediction:excitation_voltage_groupbox)=
* [**Output Voltage (RMS)**](#fig:random_vibration_prediction:excitation_voltage_list) RMS Voltage predicted for each excitation channel

* [**Excitation Display Plot**](#fig:random_vibration_prediction:excitation_display_plot) Shows the specified portion of the CPSD matrix.  If an off-diagonal term is selected, both real and imaginary parts will be plotted.
* [**Go to Maximum Excitation**](#fig:random_vibration_prediction:maximum_voltage_button) Shows the excitation channel with the largest voltage
* [**Excitation CPSD Row Channel**](#fig:random_vibration_prediction:excitation_row_selector) Select the row of the excitation CPSD matrix to visualize
* [**Go to Minimum Excitation**](#fig:random_vibration_prediction:minimum_voltage_button) Shows the excitation channel with the smallest voltage
* [**Excitation CPSD Column Channel**](#fig:random_vibration_prediction:excitation_column_selector) Select the column of the excitation CPSD matrix to visualize

(sec:random_vibration_prediction:response_error_groupbox)=
* [**Response Error (dB)**](#fig:random_vibration_prediction:response_error_list) RMS dB error predicted at each control channel.  Channels will be highlighted yellow if they hit a warning limit and red if they hit an abort limit.  Double clicking on an item will show its response prediction.

* [**Response Prediction Display Plot**](#fig:random_vibration_prediction:response_display_plot) Shows the specified portion of the response CPSD matrix predicted using the computed excitation CPSD and system identification information compared to the specification.  If an off-diagonal term is selected, both real and imaginary parts will be plotted.
* [**Go to Maximum Response Error**](#fig:random_vibration_prediction:maximum_error_button) Show the control channel prediction with the largest predicted error
* [**Response CPSD Row Channel**](#fig:random_vibration_prediction:response_row_selector) Select the row of the response CPSD matrix to visualize
* [**Go to Minimum Response Error**](#fig:random_vibration_prediction:minimum_error_button) Show the control channel prediction with the smallest predicted error
* [**Response CPSD Column Channel**](#fig:random_vibration_prediction:response_column_selector) Select the column of the response CPSD matrix to visualize
* [**Recompute Prediction**](#fig:random_vibration_prediction:recompute_prediction_button) Click to recompute the prediction by running the control law again.



:::{figure} figures/random_vibration_prediction__excitation_voltage_groupbox.png
:label: fig:random_vibration_prediction:excitation_voltage_groupbox
 Output Voltages (RMS) Settings
:::

:::{figure} figures/random_vibration_prediction__excitation_voltage_list.png
:label: fig:random_vibration_prediction:excitation_voltage_list
 **Output Voltage (RMS)** RMS Voltage predicted for each excitation channel
:::

:::{figure} figures/random_vibration_prediction__excitation_display_plot.png
:label: fig:random_vibration_prediction:excitation_display_plot
 **Excitation Display Plot** Shows the specified portion of the CPSD matrix.  If an off-diagonal term is selected, both real and imaginary parts will be plotted.
:::

:::{figure} figures/random_vibration_prediction__maximum_voltage_button.png
:label: fig:random_vibration_prediction:maximum_voltage_button
 **Go to Maximum Excitation** Shows the excitation channel with the largest voltage
:::

:::{figure} figures/random_vibration_prediction__excitation_row_selector.png
:label: fig:random_vibration_prediction:excitation_row_selector
 **Excitation CPSD Row Channel** Select the row of the excitation CPSD matrix to visualize
:::

:::{figure} figures/random_vibration_prediction__minimum_voltage_button.png
:label: fig:random_vibration_prediction:minimum_voltage_button
 **Go to Minimum Excitation** Shows the excitation channel with the smallest voltage
:::

:::{figure} figures/random_vibration_prediction__excitation_column_selector.png
:label: fig:random_vibration_prediction:excitation_column_selector
 **Excitation CPSD Column Channel** Select the column of the excitation CPSD matrix to visualize
:::

:::{figure} figures/random_vibration_prediction__response_error_groupbox.png
:label: fig:random_vibration_prediction:response_error_groupbox
 Response Error (dB) Settings
:::

:::{figure} figures/random_vibration_prediction__response_error_list.png
:label: fig:random_vibration_prediction:response_error_list
 **Response Error (dB)** RMS dB error predicted at each control channel.  Channels will be highlighted yellow if they hit a warning limit and red if they hit an abort limit.  Double clicking on an item will show its response prediction.
:::

:::{figure} figures/random_vibration_prediction__response_display_plot.png
:label: fig:random_vibration_prediction:response_display_plot
 **Response Prediction Display Plot** Shows the specified portion of the response CPSD matrix predicted using the computed excitation CPSD and system identification information compared to the specification.  If an off-diagonal term is selected, both real and imaginary parts will be plotted.
:::

:::{figure} figures/random_vibration_prediction__maximum_error_button.png
:label: fig:random_vibration_prediction:maximum_error_button
 **Go to Maximum Response Error** Show the control channel prediction with the largest predicted error
:::

:::{figure} figures/random_vibration_prediction__response_row_selector.png
:label: fig:random_vibration_prediction:response_row_selector
 **Response CPSD Row Channel** Select the row of the response CPSD matrix to visualize
:::

:::{figure} figures/random_vibration_prediction__minimum_error_button.png
:label: fig:random_vibration_prediction:minimum_error_button
 **Go to Minimum Response Error** Show the control channel prediction with the smallest predicted error
:::

:::{figure} figures/random_vibration_prediction__response_column_selector.png
:label: fig:random_vibration_prediction:response_column_selector
 **Response CPSD Column Channel** Select the column of the response CPSD matrix to visualize
:::

:::{figure} figures/random_vibration_prediction__recompute_prediction_button.png
:label: fig:random_vibration_prediction:recompute_prediction_button
 **Recompute Prediction** Click to recompute the prediction by running the control law again.
:::