#pragma once

#include "makeChemistrySolver.H"



#define defineLbChemistrySolvers(nullArg, ThermoPhysics)                       \
    defineChemistrySolver                                                      \
    (                                                                          \
        gpuChemistryModel,                                                      \
        ThermoPhysics                                                          \
    )

#define makeLbChemistrySolvers(Solver, ThermoPhysics)                          \
    makeChemistrySolver                                                        \
    (                                                                          \
        Solver,                                                                \
        gpuChemistryModel,                                                      \
        ThermoPhysics                                                          \
    )


