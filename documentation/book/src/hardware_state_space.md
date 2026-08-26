---
numbering:
  heading_2:
    start: 8
  figure:
    enumerator: 8.%s
  table:
    enumerator: 8.%s
  equation:
    enumerator: 8.%s
  code:
    enumerator: 8.%s
---
# Virtual Control using State Space Matrices

(sec:state_space_hardware)=
# Virtual Control using State Space Matrices

If no data acquisition hardware is available, it can still be advantageous to run Rattlesnake in a "virtual" mode by simulating responses to virtual forces using some kind of model.  This would allow users of the software to develop new control laws and environments without risking damage to potentially expensive test hardware if the control law or environment is not implemented correctly.  Additionally, synthetic control can be used to determine proper test parameters prior to the actual test occurring, which can help determine if a given test is feasible or not.

Rattlesnake is able to integrate linear time-invariant equations of motion using general state space matrices.  The general form for state space equations of motion is 

\begin{equation}\label{eq:state_eoms}
    \dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}
\end{equation}
\begin{equation}\label{eq:output_eoms}
    \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}
\end{equation}

where $\mathbf{A}$ is the state matrix, $\mathbf{B}$ is the input matrix, $\mathbf{C}$ is the output matrix, $\mathbf{D}$ is the feedthrough matrix, $\mathbf{x}$ is the state vector, $\mathbf{u}$ is the input vector, and $\mathbf{y}$ is the output vector.  Equation @eq:state_eoms defines how the state changes with time $\dot{\mathbf{x}}$ given the current state $\mathbf{x}$ and inputs $\mathbf{u}$ to the system.  Equation @eq:output_eoms describes how the output degrees of freedom $\mathbf{y}$ are constructed from the current state and inputs to the system.  When used within Rattlesnake, the output signals from Rattlesnake will be supplied to the system as $\mathbf{u}$, and the output degrees of freedom $\mathbf{y}$ will be acquired by Rattlesnake.  A complete example for running this type of test can be found in @sec:example_state_space.
    
For many structural dynamic analyses, the user will have mass, stiffness, and damping matrices $\mathbf{M}$, $\mathbf{K}$, and $\mathbf{C}$, respectively, in the typical 2nd-order differential equations of motion

\begin{equation}\label{eq:second_order_eoms}
\mathbf{M}\ddot{\mathbf{z}} + \mathbf{C}\dot{\mathbf{z}} + \mathbf{K}\mathbf{z} = \mathbf{u}
\end{equation}

where $\mathbf{z}$ is treated as the displacement degrees of freedom of the system and $\mathbf{u}$ are the input forces.  The state of the system is represented by 

$$
\mathbf{x} = \begin{bmatrix}
        \mathbf{z} \\
        \dot{\mathbf{z}}
        \end{bmatrix}
$$

and therefore the derivative of the state is

$$
\dot{\mathbf{x}} = \begin{bmatrix}
        \dot{\mathbf{z}} \\
        \ddot{\mathbf{z}}
        \end{bmatrix}
$$

Given the state $\mathbf{x}$ and the inputs $\mathbf{u}$, equation @eq:second_order_eoms can be transformed into @eq:state_eoms with

\begin{equation}\label{eq:state_space_a}
\mathbf{A} = \begin{bmatrix}
\mathbf{0} & \mathbf{I} \\
-\mathbf{M}^{-1}\mathbf{K} & -\mathbf{M}^{-1}\mathbf{C}
\end{bmatrix}
\end{equation}
\begin{equation}\label{eq:state_space_b}
\mathbf{B} = \begin{bmatrix}
\mathbf{0} \\ \mathbf{M}^{-1}
\end{bmatrix}
\end{equation}

where the first row partition is the trivial equation $\dot{\mathbf{z}} = \dot{\mathbf{z}}$.

The output degrees of freedom will then depend on which types of quantities are desired from the analysis.  Displacements, velocities, accelerations, and forces can be recovered in the output degrees of freedom $\mathbf{y}$ using

\begin{equation}\label{eq:state_space_c}
\mathbf{C} = \begin{bmatrix}
\mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{I} \\
-\mathbf{M}^{-1}\mathbf{K} & -\mathbf{M}^{-1}\mathbf{C} \\
\mathbf{0} & \mathbf{0}
\end{bmatrix}
\end{equation}
\begin{equation}\label{eq:state_space_d}
\mathbf{D} = \begin{bmatrix}
\mathbf{0} \\\mathbf{0} \\ \mathbf{M}^{-1} \\ \mathbf{I}
\end{bmatrix}
\end{equation}

where the first row partition recovers displacements $\mathbf{z}$, the second row partition recovers velocities $\dot{\mathbf{z}}$, the third row partition recovers accelerations $\ddot{\mathbf{z}}$, and the fourth row partition recovers forces $\mathbf{u}$.  Note that Rattlesnake requires its output signals (corresponding to the input signals to the system $\mathbf{u}$) to be measured in order to function (see @sec:channel_table) so the input signals to the system $\mathbf{u}$ must be passed through directly to output degrees of freedom $\mathbf{y}$ which are the acquired data in Rattlesnake.  Practically, this means that there should always be at least one row partition of $\mathbf{C}$ containing zeros, and the same row partition of $\mathbf{D}$ should contain the identity matrix.
    
This chapter describes the setup and implementation details for running a synthetic control problem using state space matrices.  For a complete example of using this type of control, see @sec:example_state_space.

## Setting up the Channel Table for Virtual Control using State Space Matrics <!--Section 8.1-->

Required channel table inputs for virtual control using state space matrices are minimal, as the $\mathbf{C}$ and $\mathbf{D}$ matrices effectively define the measured degrees of freedom.  Still users are encouraged to fill out the channel table as much as possible for test documentation's sake.  The only required parameters for virtual control are have an entry in the Physical Device column of the channel table.  The authors recommend simply using the word "Virtual" to reinforce the fact that these are not real hardware channels, but instead virtual channels that exist only in software.  The excitation channels must also have an entry in the Feedback Device column.  The authors recommend simply using the word "Input" to make it clear that these channels are input channels to the system.  One important consideration is that the number of channels in the channel table must be the same as the number of output degrees of freedom in $\mathbf{y}$ (and therefore the number of rows of matrices $\mathbf{C}$ and $\mathbf{D}$), and they must also be ordered identically.  Similarly, the number of input channels to the system specified in the channel table must be the same as the number of input degrees of freedom in $\mathbf{u}$ (and therefore the number of columns of matrices $\mathbf{B}$ and $\mathbf{D}$), also ordered identically.

## Hardware Parameters <!--Section 8.2-->

Hardware parameters are similar to those found in @sec:exodus_hardware.  When the `State Space Integration...` hardware device is selected, Rattlesnake will bring up a file dialog box that will allow the user to specify a Numpy or Matlab file from which the state space matrices will be loaded.  This file should contain fields `A`, `B`, `C`, and `D` containing the appropriately sized matrices.  An additional parameter will appear in the `Data Acquisition Parameters` portion of the `Data Acquisition Setup` tab that allows the user to specify an `Integration Oversample` amount, as the integration time steps should generally be finer than the sample rate of the controller.  A value of 10 is typically sufficient for integration of a linear system using `lsim` from `scipy.signal`.

(sec:state_space_implementation_details)=
## Implementation Details <!-- Section 8.3-->

A `scipy.signal.StateSpace` model is created from the loaded matrices which are used for integration using the `scipy.signal.lsim` function.

### Output Signal Force Buffering <!-- Subsection 8.3.1-->

Excitation signals coming into the state space virtual hardware are buffered to enable different read and write sizes.  The excitation signals will be delivered to the hardware based on the specified `Time per Write` on the `Data Acquisition Setup` tab of the controller.  The virtual hardware will then remove an amount of excitation signal from the buffer based on the specified `Time per Read`, as the acquisition task will perform the integration over time blocks equivalent to `Time per Read`.

### Integration using LSIM <!--Subsection 8.3.2-->

The `scipy.signal.lsim` function is used to integrate the state space model.  When the acquisition is started, the state of the integration is initialized to zero.  For subsequent integrations, the initial state is set to the final state of the previous integration.  The excitation signals are taken from the force buffer discussed previously.  Note that integration is performed at the rate specified by the `Sample Rate` of the data acquisition system multiplied by the `Integration Oversample` amount.  After the integration is performed, the output is downsampled by the `Integration Oversample` amount to return to the sample rate of the controller.  For subsequent integration blocks, the initial conditions of the integration are equivalent to the final value of the last integration block.
