#include "makeGpuChemistrySolver.H"
#include "noChemistrySolver.H"

#include "forGases.H" //The standard forCoeffGases macro
// #include "forLiquids.H"

// This file defines the chemistryModel - thermo combinations which
// are also precompiled in OpenFOAM. Any other compilation will not
// work unless also pre-compiled (in makeMoreGpuChemistryModels.C).
// To test which models are pre-compiled it is possible to put:
//
// InfoSwitches
//{
//    // Allow case-supplied C++ code (#codeStream, codedFixedValue)
//    allowSystemOperations   0;
//}
// In controlDict to avoid OpenFOAM triggering the dynamic code
// routines which can not be used with custom chemistry models due to
// a bug in basicChemistryModel::New

namespace Foam {

forCoeffGases(defineGpuChemistrySolvers, nullArg);
forCoeffGases(makeGpuChemistrySolvers, noChemistrySolver);

// TODO: Consider adding forCoeffLiquids

} // namespace Foam
