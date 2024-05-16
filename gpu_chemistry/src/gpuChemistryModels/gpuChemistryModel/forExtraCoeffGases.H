
#pragma once

#include "forGases.H" //forCoeffGasEnergiesAndThermos
#include "logPolynomialTransport.H"


/*
//These are the same as in forGases.H
#define forExtraGasEqns(Mu, He, Cp, Macro, Args...)                           \
    forThermo(Mu, He, Cp, perfectGas, specie, Macro, Args);

#define forExtraGasEnergiesAndThermos(Mu, Macro, Args...)                     \
    forExtraGasEqns(Mu, sensibleEnthalpy, hConstThermo, Macro, Args);         \
    forExtraGasEqns(Mu, sensibleEnthalpy, janafThermo, Macro, Args);          \
    forExtraGasEqns(Mu, sensibleInternalEnergy, eConstThermo, Macro, Args);   \
    forExtraGasEqns(Mu, sensibleInternalEnergy, hConstThermo, Macro, Args);   \
    forExtraGasEqns(Mu, sensibleInternalEnergy, janafThermo, Macro, Args)

*/


//This calls a macro from forGases.H. To use this properly, it is required to call this for:
// * all wanted thermo/mixture combinations
// * all wanted reaction types
// * all wanted chemistry models
//Note that it is important to _not_ define any of the transport models which are already
//defined in forCoeffGasTransports (forGases.H file) because of duplicates
#define forExtraCoeffGasTransports(Macro, Args...)                                 \
    forCoeffGasEnergiesAndThermos(logPolynomialTransport, Macro, Args);        


#define forExtraCoeffGases(Macro, Args...)                                         \
    forExtraCoeffGasTransports(Macro, Args);








