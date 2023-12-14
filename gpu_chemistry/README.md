# gpu_chemistry

## Compilation
Everything in this folder such compile with the Allwmake script. Before running it, ensure that you have a valid OpenFOAM installation and compilers available by typing the following commands

```
foamRun -help
g++ --version
nvcc --version
```

## Performance

Some performance results are presented below. The results are from ```../tutorials/scalingTest``` which solves a simple shear layer combustion case in a cubic domain. The cell count in the test case is N=100^3 for the HPC node test and N=30^3 in the desktop PC case. As can be seen, the GPU implementation is faster, especially, when the number of cells per mpi process is high.

##### On a HPC node: 2 x AMD Rome 7H12 CPU (128 CPU cores) with 4 x A100 GPU
<img src="../tutorials/scalingTest/gri_results_a100.png" alt="asd" width="370"/>
<img src="../tutorials/scalingTest/h2_results_a100.png" alt="asd" width="370"/>

##### On Desktop PC: With AMD Ryzen Threadripper 3960X (24 CPU cores) with a RTX 3080Ti GPU
<img src="../tutorials/scalingTest/gri_results_rtx3080ti.png" alt="asd" width="370"/>
<img src="../tutorials/scalingTest/h2_results_rtx3080ti.png" alt="asd" width="370"/>

## Supported features
Not everything from the official OpenFOAM release is supported. If you need more support for features, please make a feature request.

#### Supported thermo models
Currently, only the following thermodynamics model combination is supported.

```
thermoType
{
    type            heRhoThermo;
    mixture         multicomponentMixture;
    transport       sutherland;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState perfectGas;
    specie          specie;
}
```
This solves the energy equation for sensible enthalpy and the transport properties (vicosity and conductivity) are computed using Sutherland's law. Heat capacity is taken from Janaf polynomials and the ideal gas law is assumed for the density.


#### Supported ODE solvers
* Rosenbrock12
* Rosenbrock23
* Rosenbrock34

Only the Rosenbrock family of ODE solvers is supported at this point.


#### Supported reaction types
The following list of reactions are supported:

* reversibleArrhenius
* irreversibleArrhenius
* reversibleThirdBodyArrhenius
* reversibleArrheniusLindemannFallOff
* reversibleArrheniusTroeFallOff



