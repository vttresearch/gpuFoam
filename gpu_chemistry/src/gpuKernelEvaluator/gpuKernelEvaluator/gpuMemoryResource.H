#pragma once

#include "gpuBuffer.H"
#include "gpu_constants.H"
#include "mdspan.H"
#include "memoryResource.H"
#include "pointer_casts.hpp"
#include <array>
#include <vector>

namespace FoamGpu {

struct gpuMemoryResource : public memoryResource {

    gpuMemoryResource() = default;

    gpuMemoryResource(gLabel nCells, gLabel nSpecie);

    ~gpuMemoryResource();

    std::array<gLabel*, N_LABEL_ARRAYS>& getLabelData() override {
        return labelData_;
    }

    std::array<gScalar*, N_SCALAR_ARRAYS>& getScalarData() override {
        return scalarData_;
    }

    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS>&
    getTwodScalarData() override {
        return twodScalarData_;
    }

    void resize(gLabel nCells, gLabel nSpecie) override;


private:
    void allocate();
    void deallocate();

    std::array<gLabel*, N_LABEL_ARRAYS>        labelData_{};
    std::array<gScalar*, N_SCALAR_ARRAYS>      scalarData_{};
    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData_{};
};

} // namespace FoamGpu