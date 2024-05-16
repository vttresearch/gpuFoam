// This file defines any additional combinations of chemistryModel -
// thermo pairs that the user wishes to use and are not pre-compiled
// in OpenFOAM. Because of a bug in basicChemistryModel::New all pairs
// need to be pre-compiled. Currently only the ability to use
// coefficientWileMulticomponent mixture with logPolynomial transport
// is added. With standard chemistryModel this combination can be used
// through the dynamic code routines.

#include "makeGpuChemistrySolver.H"
#include "noChemistrySolver.H"

// Thermo types
#include "makeFluidMulticomponentThermo.H"
#include "psiMulticomponentThermo.H"
#include "rhoFluidMulticomponentThermo.H"
#include "rhoThermo.H"

#include "coefficientMulticomponentMixture.H"
#include "coefficientWilkeMulticomponentMixture.H"
#include "singleComponentMixture.H"

// Reaction types
#include "JanevReactionRate.H"
#include "LandauTellerReactionRate.H"
#include "makeReaction.H"
#include "powerSeriesReactionRate.H"
#include "thirdBodyArrheniusReactionRate.H"

#include "ChemicallyActivatedReactionRate.H"
#include "FallOffReactionRate.H"

#include "LindemannFallOffFunction.H"
#include "SRIFallOffFunction.H"
#include "TroeFallOffFunction.H"

#include "forExtraCoeffGases.H"

namespace Foam {

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   rhoFluidThermo,
                   rhoFluidMulticomponentThermo,
                   coefficientMulticomponentMixture);

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   rhoFluidThermo,
                   rhoFluidMulticomponentThermo,
                   coefficientWilkeMulticomponentMixture);

forExtraCoeffGases(makeFluidMulticomponentThermos,
                   psiThermo,
                   psiMulticomponentThermo,
                   coefficientMulticomponentMixture);
forExtraCoeffGases(makeFluidMulticomponentThermos,
                   psiThermo,
                   psiMulticomponentThermo,
                   coefficientWilkeMulticomponentMixture);

forExtraCoeffGases(defineReaction, nullArg);
forExtraCoeffGases(makeIRNReactions, ArrheniusReactionRate);
forExtraCoeffGases(makeIRNReactions, LandauTellerReactionRate);
forExtraCoeffGases(makeIRNReactions, thirdBodyArrheniusReactionRate);
forExtraCoeffGases(makeIRNReactions, JanevReactionRate);
forExtraCoeffGases(makeIRNReactions, powerSeriesReactionRate);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   FallOffReactionRate,
                   ArrheniusReactionRate,
                   LindemannFallOffFunction);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   FallOffReactionRate,
                   ArrheniusReactionRate,
                   TroeFallOffFunction);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   FallOffReactionRate,
                   ArrheniusReactionRate,
                   SRIFallOffFunction);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   ChemicallyActivatedReactionRate,
                   ArrheniusReactionRate,
                   LindemannFallOffFunction);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   ChemicallyActivatedReactionRate,
                   ArrheniusReactionRate,
                   TroeFallOffFunction);

forExtraCoeffGases(makeIRTemplate2Reactions,
                   ChemicallyActivatedReactionRate,
                   ArrheniusReactionRate,
                   SRIFallOffFunction);

// Finally add the chemistry models
forExtraCoeffGases(defineGpuChemistrySolvers, nullArg);
forExtraCoeffGases(makeGpuChemistrySolvers, noChemistrySolver)

} // namespace Foam
