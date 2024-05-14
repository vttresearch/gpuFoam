#pragma once

#include "mechanisms.H"
#include "testThermoType.H"

#include "speciesTable.H"

#include "ReactionList.H"
#include "ReversibleReaction.H"
#include "sensibleEnthalpy.H"

#include "sutherlandTransport.H"
#include "thermo.H"

using FoamThermoType = FoamGpu::testThermoType1;

namespace TestData {

Foam::dictionary makeThermoDict(TestData::Mechanism m);

Foam::speciesTable makeSpeciesTable(TestData::Mechanism m);

Foam::PtrList<FoamThermoType> makeCpuThermos(TestData::Mechanism m);

Foam::dictionary makeReactionDict(TestData::Mechanism m);

Foam::ReactionList<FoamThermoType>
makeCpuReactions(TestData::Mechanism m);
} // namespace TestData
