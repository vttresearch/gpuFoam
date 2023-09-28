#include "error_handling.H"
#include "gpuMemoryResource.H"

namespace FoamGpu {

gpuMemoryResource::gpuMemoryResource(gLabel nCells,
                                     gLabel nSpecie,
                                     gLabel nEqns)
    : nCells_(nCells)
    , nSpecie_(nSpecie)
    , nEqns_(nEqns) {

    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        const auto bytesize = nCells_ * nEqns_ * sizeof(gScalar);
        gScalar*   ptr      = data_[i];
        CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, bytesize));
    }

    const auto bytesize = nCells_ * nEqns_ * sizeof(gLabel);
    CHECK_CUDA_ERROR(cudaMalloc((void**)&pivotIndices_, bytesize));
}

std::array<gScalar*, N_SCALAR_ARRAYS>
gpuMemoryResource::getScalarArrays(gLabel celli) {

    assert(celli < nCells_);

    std::array<gScalar*, N_SCALAR_ARRAYS> ret;
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        ret[i] = data_[nEqns_ * celli];
    }
    return ret;
}

void gpuMemoryResource::free() {

    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        CHECK_CUDA_ERROR(cudaFree(data_[i]));
    }
    CHECK_CUDA_ERROR(cudaFree(pivotIndices_));
}

} // namespace FoamGpu
