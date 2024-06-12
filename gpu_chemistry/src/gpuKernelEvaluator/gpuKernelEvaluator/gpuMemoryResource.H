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

    device_array<gLabel*, N_LABEL_ARRAYS>& getLabelData() override {
        return labelData_;
    }

    device_array<gScalar*, N_SCALAR_ARRAYS>& getScalarData() override {
        return scalarData_;
    }

    device_array<gScalar*, N_TWOD_SCALAR_ARRAYS>&
    getTwodScalarData() override {
        return twodScalarData_;
    }

    void resize(gLabel nCells, gLabel nSpecie) override;


private:
    void allocate();
    void deallocate();

    device_array<gLabel*, N_LABEL_ARRAYS>        labelData_{};
    device_array<gScalar*, N_SCALAR_ARRAYS>      scalarData_{};
    device_array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData_{};
};

} // namespace FoamGpu