#pragma once

#include "addToRunTimeSelectionTable.H"
#include "makeChemistrySolver.H"
#include "gpuChemistryModel.H"


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

