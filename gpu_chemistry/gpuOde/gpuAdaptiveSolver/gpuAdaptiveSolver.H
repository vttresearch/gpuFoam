#pragma once

#include "cuda_host_dev.H"
#include "gpuODESystem.H"
#include "gpuConstants.H"

namespace FoamGpu{


struct gpuAdaptiveSolverInputs{
    gScalar safeScale;
    gScalar alphaInc;
    gScalar alphaDec;
    gScalar minScale;
    gScalar maxScale;
};

struct gpuAdaptiveSolver{

    //- Step-size adjustment controls
    gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;


    gpuAdaptiveSolver() = default;


    gpuAdaptiveSolver(gpuAdaptiveSolverInputs i) :
         safeScale_(i.safeScale)
        , alphaInc_(i.alphaInc)
        , alphaDec_(i.alphaDec)
        , minScale_(i.minScale)
        , maxScale_(i.maxScale) {}

};

}