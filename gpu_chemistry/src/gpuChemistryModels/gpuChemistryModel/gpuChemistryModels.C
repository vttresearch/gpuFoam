#include "addToRunTimeSelectionTable.H"
#include "gpuChemistryModel.H"
#include "noChemistrySolver.H"
#include "makeChemistrySolver.H"
#include "forGases.H"
#include "forLiquids.H"

#define defineGpuChemistrySolvers(nullArg, ThermoPhysics)                         \
    defineChemistrySolver                                                      \
    (                                                                          \
        gpuChemistryModel,                                                        \
        ThermoPhysics                                                          \
    )

#define makeGpuChemistrySolvers(Solver, ThermoPhysics)                            \
    makeChemistrySolver                                                        \
    (                                                                          \
        Solver,                                                                \
        gpuChemistryModel,                                                        \
        ThermoPhysics                                                          \
    )



namespace Foam {

    forCoeffGases(defineGpuChemistrySolvers, nullArg);
    forCoeffLiquids(defineGpuChemistrySolvers, nullArg);

    forCoeffGases(makeGpuChemistrySolvers, noChemistrySolver);
    forCoeffLiquids(makeGpuChemistrySolvers, noChemistrySolver);

} // namespace Foam
