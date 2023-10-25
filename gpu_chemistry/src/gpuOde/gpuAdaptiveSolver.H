#pragma once

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuODESolverInputs.H"
#include "gpuODESystem.H"
#include "gpu_constants.H"

namespace FoamGpu {

struct gpuAdaptiveSolver {

    //- Step-size adjustment controls
    gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

    gpuAdaptiveSolver() = default;

    gpuAdaptiveSolver(const gpuODESolverInputs& i)
        : safeScale_(i.safeScale)
        , alphaInc_(i.alphaInc)
        , alphaDec_(i.alphaDec)
        , minScale_(i.minScale)
        , maxScale_(i.maxScale) {}


    CUDA_HOSTDEV gScalar safeScale() const {return safeScale_;}
    CUDA_HOSTDEV gScalar alphaInc() const {return alphaInc_;}
    CUDA_HOSTDEV gScalar alphaDec() const {return alphaDec_;}
    CUDA_HOSTDEV gScalar minScale() const {return minScale_;}
    CUDA_HOSTDEV gScalar maxScale() const {return maxScale_;}



};

} // namespace FoamGpu