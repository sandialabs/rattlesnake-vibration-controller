---
numbering:
  heading_2:
    start: 9
  figure:
    enumerator: 9.%s
  table:
    enumerator: 9.%s
  equation:
    enumerator: 9.%s
  code:
    enumerator: 9.%s
---
# Virtual Control using Finite Element Results in Exodus Files

(sec:exodus_hardware)=
# Virtual Control using Finite Element Results in Exodus Files

In addition to virtual control using state space matrices, Rattlesnake can also perform virtual control using results from FEM analyses.  This is perhaps less complex to set up than the virtual control using state space matrices described in @sec:state_space_hardware, but it offers less flexibility to the user and relies on the user having results in the Exodus file format.
    
Modern FEMs can contain thousands or even millions of degrees of freedom, so it is not realistic to be able to integrate the full finite element model's equations of motion in real time.  Some model reduction is necessary to reduce the number of degrees of freedom that must be integrated.  Rattlesnake instead integrates modal equations of motion rather than the full set of equations of motion.  The modal transformation has the advantage of generally only requiring a small number of modal degrees of freedom to characterize the part over some frequency bandwidth.  Additionally, modal degrees of freedom are uncoupled from one another, which makes for simpler integration strategies.
    
This chapter describes the setup and implementation details for running a synthetic control problem using Exodus FEM results.

## Setting up the Channel Table for Virtual Control using a finite element model (FEM) <!--Section 9.1-->

Setting up the channel table is very straightforward for a virtual control problem.  The node number that is specified in the channel table corresponds to the node in the finite element model.  Note that this corresponds to the node *number* not the node *index*, so users should be aware when picking out nodes for virtual control using visualization software whether or not the software is reporting node number or node index.  The node direction can be specified as either a principal direction (X+, Y+, Z+, X-, Y-, or Z-) or a comma separated 3-vector with unit magnitude that specifies the measurement direction.
    
The modal damping value that will be applied to all modes in the model should be placed in the Comment column of the first channel.  This should be specified as a fraction of critical damping (e.g. 0.01) rather than a percentage (e.g. 1\%).  For more complex damping capabilities, a State Space formulation should be used instead, see @sec:state_space_hardware.
    
All active measurement degrees of freedom and output degrees of freedom must have an entry in the `Physical Device` column of the channel table.  The authors recommend simply using the word `Virtual` to reinforce the fact that these are virtual channels.  Any excitation channel must have an entry in the `Feedback Device` column.  The authors here recommend simply using the word `Input` to make it clear that these channels are excitation channels.
    
All other columns of the channel table can be left empty, though it may be useful to put entries into them to document the virtual test better.

## Hardware Parameters <!--Section 9.2-->

When the `Exodus Modal Solution...` hardware device is selected, Rattlesnake will bring up a file dialog box that will allow the user to specify the Exodus file from which the modal solution will be loaded.  An additional parameter will appear in the `Data Acquisition Parameters` portion of the `Data Acquisition Setup` tab that allows the user to specify an `Integration Oversample` amount, as the integration time steps should generally be finer than the sample rate of the controller.  A value of 10 is typically sufficient for integration of a linear system using `lsim` from `scipy.signal`.

## Implementation Details <!--Section 9.3-->

This section contains implementation details of the virtual control using modal analysis solutions found in exodus files. 

### Creating State Space Matrices from Modal EOMs <!--Subsection 9.3.1-->

To assemble the state space equations of motion, the mode shape matrix is used to transform between modal and physical degrees of freedom.  A 3D displacement array is constructed from the `DispX`, `DispY`, and `DispZ` node variables in the Exodus file.  A mode shape matrix is computed by taking the dot product of the direction vector specified by the direction in the channel table with the displacement vector for the node specified by the channel table for each mode shape.

#### State Degrees of Freedom <!--Subsubsection 9.3.1.1-->

States for the virtual control using an Exodus modal solution are the modal displacements and velocities for the modes in the exodus file below 1.5$\times$ the Nyquist frequency of the controller.  Mass normalized mode shapes are assumed, so the modal mass matrix is assumed to be the identify matrix, modal damping is assumed to be a diagonal matrix of damping coefficients $2\zeta_n\omega_n$, and modal stiffness is assumed to be a diagonal matrix of stiffness coefficients ${\omega_n}^2$.  The state equation matrices $\mathbf{A}$ and $\mathbf{B}$ are then defined by equations @eq:state_space_a and @eq:state_space_b.

#### Response Degrees of Freedom <!--Subsubsection 9.3.1.2-->

Response degrees of freedom are defined using the $\mathbf{C}$ and $\mathbf{D}$ matrices defined by equations @eq:state_space_c and @eq:state_space_a.  This hardware device assumes responses are accelerations, so the third row partition of equations @eq:state_space_c and @eq:state_space_d are used.  These are modal accelerations, so they are multiplied by the mode shape matrix to produce physical accelerations.

#### Excitation Degrees of Freedom <!--Subsubsecion 9.3.1.3-->

Excitation to the system is specified as modal forces, so the physical excitation forces provided by Rattlesnake are multiplied by the mode shape matrix transposed to create modal forces.  To accommodate potentially different read and write sizes in the controller, physical forces to be output are appended to a force buffer and one write worth of forces are pulled from the buffer to be transformed to modal forces and applied to the system.
    
Rattlesnake's measurement strategy requires that excitation signals also be measured by the data acquisition, and this is also true for virtual control.  Therefore, excitation signals must be returned as a measured response.

### Integration of Equations of Motion <!--Subsection 9.3.2-->

Once state space matrices are formed, the implementation of this virtual hardware is largely the same as the virtual control described in @sec:state_space_hardware. See @sec:state_space_implementation_details for details on the integration scheme.
