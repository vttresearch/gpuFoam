#pragma once

#include "gpuBuffer.H"
#include "gpuConstants.H"
#include "mdspan.H"
#include "memoryResource.H"
#include "pointer_casts.hpp"
#include <array>
#include <vector>

namespace FoamGpu {

struct gpuMemoryResource2 : public memoryResource {

    gpuMemoryResource2() = default;

    gpuMemoryResource2(gLabel nCells, gLabel nSpecie);

    ~gpuMemoryResource2();

    std::array<gLabel*, N_LABEL_ARRAYS>& getLabelData() override {
        return labelData_;
    }

    std::array<gScalar*, N_SCALAR_ARRAYS>& getScalarData() override {
        return scalarData_;
    }

    virtual std::array<gScalar*, N_TWOD_SCALAR_ARRAYS>&
    getTwodScalarData() override {
        return twodScalarData_;
    }

private:
    void allocate();
    void deallocate();

    std::array<gLabel*, N_LABEL_ARRAYS>        labelData_{};
    std::array<gScalar*, N_SCALAR_ARRAYS>      scalarData_{};
    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData_{};
};

} // namespace FoamGpu