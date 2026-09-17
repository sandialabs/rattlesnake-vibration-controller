---
numbering:
  heading_2:
    start: 10
  figure:
    enumerator: 10.%s
  table:
    enumerator: 10.%s
  equation:
    enumerator: 10.%s
  code:
    enumerator: 10.%s
---
# Virtual Control using SDYNPY System Objects

(sec:sdynpy_hardware)=
# Virtual Control using SDYNPY System Objects

The final option for synthetic control in Rattlesnake is to load in a SDynPy [`System`](https://sandialabs.github.io/sdynpy/sdynpy-system/) object, which gets stored from SDynPy using the [`sdyn.System.save`](https://sandialabs.github.io/sdynpy/sdynpy-system/#api-sdynpy-system-save) method.  SDynPy `System` objects store mass, stiffness, and damping matrices.  They also store transformation matrices which transform its internal system space into the physical space.  This allows SDynPy `System` objects to represent both "full" physical systems as well as "reduced" systems (e.g., Craig-Bampton or Modal systems).  The final part of the SDynPy `System` is the degree of freedom information that is stored with the matrices.  This maps rows of the transformation matrix (or rows of the mass, stiffness, and damping matrices if the transformation is the identity matrix) to physical degrees of freedom.

A complete example problem using a SDynPy `System` object can be found in @sec:example_sdynpy.

## Setting up the Channel Table <!--Section 10.1-->

Setting up the channel table is very straightforward for a SDynPy System.  Degrees of freedom in the SDynPy system are selected in the Rattlesnake test by specifying the node and direction of the degree of freedom in the `Node` and `Direction` of the `coordinate` field of the SDynPy system.  The node is specified as an integer, and the direction is specified as a string (X+, Y+, Z+, X-, Y-, or Z-).
    
All active measurement degrees of freedom and output degrees of freedom must have an entry in the `Physical Device` column of the channel table.  The authors recommend simply using the word `Virtual` to reinforce the fact that these are virtual channels.  Any excitation channel must have an entry in the `Feedback Device` column.  The authors here recommend simply using the word `Input` to make it clear that these channels are excitation channels.
    
The `Channel Type` must also be specified.  It is used to specify the derivative of the degree of freedom that will be acquired in the response channels.  The `Channel Type` can be specified as `Acceleration`, `Velocity`, or `Displacement`, and the output matrices of the state space system that will be integrated will be assembled such that it will return the specified channel type.  If an unknown channel type is specified or if none are specified, then displacement will be assumed.  Note excitation degrees of freedom are not modified by a `Channel Type` modifier, they are applied directly as-is to the SDynPy system.
    
All other columns of the channel table can be left empty, though it may be useful to put entries into them to document the virtual test better.

## Hardware Parameters <!--Section 10.2-->

 When the `SDynPy System Integration...` hardware device is selected, Rattlesnake will bring up a file dialog box that will allow the user to specify the file from which the SDynPy system will be loaded.  An additional parameter will appear in the `Data Acquisition Parameters` portion of the `Data Acquisition Setup` tab that allows the user to specify an `Integration Oversample` amount, as the integration time steps should generally be finer than the sample rate of the controller.  A value of 10 is typically sufficient for integration of a linear system using `lsim` from `scipy.signal`.
    

## Implementation Details <!--Section 10.3-->

This section describes the implementation details for the SDynPy System integration.

### Setting up the State Space System <!--Subsection 10.3.1-->
    
Rattlesnake is able to integrate linear time-invariant equations of motion using general state space matrices.  The general form for state space equations of motion is given in Equations @eq:state_eoms and @eq:output_eoms.
    
@sec:state_space_hardware describes how the state space equations can be constructed from mass, stiffness, and damping matrices.  The state space formulation can be extended to allow for a transformation matrix between the mass, stiffness, and damping matrices, and the physical degrees of freedom.
    
$$
\mathbf{C} = \begin{bmatrix}
            \mathbf{\Phi}_r & \mathbf{0} \\
            \mathbf{0} & \mathbf{\Phi}_r \\
            -\mathbf{\Phi}_r\mathbf{M}^{-1}\mathbf{K} & -\mathbf{\Phi}_r\mathbf{M}^{-1}\mathbf{C} \\
            \mathbf{0} & \mathbf{0}
        \end{bmatrix}
$$
$$
\mathbf{D} = \begin{bmatrix}
            \mathbf{0} \\\mathbf{0} \\ \mathbf{\Phi}_r\mathbf{M}^{-1}{\mathbf{\Phi}_e}^T \\ \mathbf{I}
        \end{bmatrix}
$$

Here the system transformation $\mathbf{\Phi}$ is partitioned into the response degrees of freedom $\mathbf{\Phi}_r$ and excitation degrees of freedom $\mathbf{\Phi}_e$.
    
Matrices $\mathbf{A}$ and $\mathbf{B}$ are unchanged and defined in equations @eq:state_space_a and @eq:state_space_b.
    
The state space matrices are then integrated using `scipy.signal.lsim`, similar to the State Space formulation described in @sec:state_space_hardware.
