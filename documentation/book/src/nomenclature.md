(sec:glossary)=
# Glossary

:::{glossary}
6DoF 
: six degree-of-freedom; it is common for {term}`MIMO` tests to be performed on a shaker table that can be moved in six degrees of freedom (three translations and three rotations).  Generally these shaker tables will have 6 to 12 shakers attached to the table in different directions to facilitate motions in these directions.

API
: application programming interface; a programming interface into a specific application.

APSD
: auto-power spectral density; the diagonal terms of a cross-power spectral density matrix which correspond to a signal referenced to itself.

CCLD 
: constant current line drive; a hardware configuration where the data acquisition system supplies power and signal conditioning to a sensor that requires it.  See also {term}`IEPE`.

COLA 
: constant overlap and add; a signal processing algorithm in which the signal is split up into overlapping sections which are analyzed separately.  The results of the separate analyses are then summed back together, typically with a window function applied to smooth transitions between segments.  See <wiki:Overlap–add_method>.

CPSD 
: cross-power spectral density; a function describing the distribution of power into frequency components of that signal.  See <wiki:Spectral_density#Cross_power_spectral_density>.

FEM 
: finite element model; a model built using the <wiki:Finite_element_method>, which subdivides the domain into many small elements, each of which solves a small portion of the domain.

FFT 
: fast Fourier transform; an algorithm computing the discrete Fourier transform of a signal or its inverse.  See <wiki:Fast_Fourier_transform>

FRF 
: frequency response function; a function describing the frequency response of a system.  This is a complex-valued function describing the frequency-domain relationship between some input and some response.  See [What is a Frequency Response Function (FRF)](https://community.sw.siemens.com/s/article/what-is-a-frequency-response-function-frf) for further information.

GUI 
: graphical user interface; the user interface used to control a piece of software represented in graphical form, including various windows, buttons, and text entry fields.  Rattlesnake's primary input is through its graphical user interface.

ICP 
: integrated circuit piezoelectric; a hardware configuration where the data acquisition system supplies power and signal conditioning to a sensor that requires it.  See also {term}`IEPE`.

IDE 
: integrated development environment; a programming environment that aids the user in developing software, typically including features such as find/replace, documentation, and code completion.  [Spyder](https://www.spyder-ide.org/) and [VSCode](https://code.visualstudio.com/) are two common Python {term}`IDE`s.

IEPE 
: integrated electronics piezoelectric; a hardware configuration where the data acquisition system supplies power and signal conditioning to a sensor that requires it.  See <wiki:Integrated_Electronics_Piezo-Electric>.

IFFT 
: inverse fast Fourier transform; see {term}`FFT`.

JSON 
: javascript object notation; an open standard file format and data iterchange format that uses human readable format.  See <wiki:JSON>

MIMO 
: multiple input/multiple output: referencing the fact that a vibration test might have multiple vibration shakers attached to it (inputs) and multiple accelerometers measuring responses from it (outputs)

ReST 
: representational state transfer; an approach for computer systems to communicate with each other.  See <wiki:Representational_state_transfer>

RMS 
: root-mean-square; values in a signal are first squared, the mean value is then taken, then a square root is computed of that value.  Also known as the quadratic mean.  See <wiki:Root_mean_square>

SVD 
: singular value decomposition; a linear algebra operation that decomposes a matrix $\mathbf{M}$ into two unitary matrices $\mathbf{U}$ and $\mathbf{V}$ and one diagonal matrix $\mathbf{S}$ of the form $\mathbf{M} = \mathbf{U}\mathbf{S}\mathbf{V}^*$.  Matrices $\mathbf{U}$ and $\mathbf{V}$ are called the left and right singular vectors, and entries of $\mathbf{S}$ are called the singular values.  See <wiki:Singular_value_decomposition>.

TRAC 
: time response assurance criterion; a comparison metric applicable to time data that judges how similar two time histories are to one another.  A TRAC of 1 indicates the time histories are identical to within a scale factor.  A TRAC of 0 indicates the time histories are not similar.  

UI
: user interface; an interface into software that allows a user to issue commands.  It may be a GUI or a command line interface.
:::