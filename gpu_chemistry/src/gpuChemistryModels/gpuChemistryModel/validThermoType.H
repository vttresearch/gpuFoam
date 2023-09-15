#pragma once

#include "janafThermo.H"
#include "perfectGas.H"
#include "specie.H"
#include "thermo.H"
#include "sensibleEnthalpy.H"
#include "sutherlandTransport.H"
namespace FoamGpu
{

    //This is the corresponding cpu thermo type for the gpuThermo.
    using validThermoType =
        Foam::sutherlandTransport
        <
            Foam::species::thermo
            <
                Foam::janafThermo
                <
                    Foam::perfectGas
                    <
                        Foam::specie
                    >
                >,
                Foam::sensibleEnthalpy
            >
        >;


}