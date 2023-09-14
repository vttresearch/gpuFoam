#pragma once

#include "gpuConstants.H"
#include <string>

namespace FoamGpu {


struct gpuODESolverInputs
{
    //name of the solver
    std::string             name;
    //base class inputs
    gScalar                 absTol;
    gScalar                 relTol;
    gLabel                  maxSteps;
    //adaptive solver inputs
    gScalar safeScale;
    gScalar alphaInc;
    gScalar alphaDec;
    gScalar minScale;
    gScalar maxScale;
};

}