# gpu_chemistry 

## Compilation 
Everything in this folder such compile with the Allwmake script. Before running it, ensure that you have a valid OpenFOAM installation and compilers available by typing the following commands

```
foamRun -help
g++ --version
nvcc --version
```

## Performance

Best performance is obtained for small mechanisms (small number of species and reactions). 


## Supported features  
Not everything from the official OpenFOAM release is supported. If you need more support for features, please make a feature request.

### Supported thermo models
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


### Supported ODE solvers
* Rosenbrock12
* Rosenbrock23
* Rosenbrock34

Only the Rosenbrock family of ODE solvers is supported at this point.


### Supported reaction types
The following list of reactions are supported:

* reversibleArrhenius
* irreversibleArrhenius
* reversibleThirdBodyArrhenius
* reversibleArrheniusLindemannFallOff
* reversibleArrheniusTroeFallOff



