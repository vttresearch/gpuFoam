#pragma once

#include "gpuBuffer.H"
#include "gpuConstants.H"
#include "memoryResource.H"
#include "pointer_casts.hpp"
#include <array>
#include <vector>

namespace FoamGpu {

struct cpuMemoryResource : public memoryResource {

    using labelAllocator  = std::allocator<gLabel>;
    using scalarAllocator = std::allocator<gScalar>;

    cpuMemoryResource() = default;

    cpuMemoryResource(gLabel nCells, gLabel nSpecie)
        : memoryResource(nCells, nSpecie) {
        this->allocate();
    }

    ~cpuMemoryResource() { deallocate(); }

    std::array<gLabel*, N_LABEL_ARRAYS>& getLabelData() override {
        return labelData_;
    }

    std::array<gScalar*, N_SCALAR_ARRAYS>& getScalarData() override {
        return scalarData_;
    }

    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS>& getTwodScalarData() override {
        return twodScalarData_;
    }

    void resize(gLabel nCells, gLabel nSpecie) override {
        if (shouldReallocate(nCells, nSpecie)) {
            this->deallocate();
            nCells_  = nCells;
            nSpecie_ = nSpecie;
            nEqns_   = nSpecie + 2;
            this->allocate();
        }
    }

private:
    void allocate() {

        labelAllocator  lAllocator;
        scalarAllocator sAllocator;

        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            labelData_[i] = lAllocator.allocate(labelArrayLength());
        }
        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            scalarData_[i] = sAllocator.allocate(scalarArrayLength());
        }
        for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
            twodScalarData_[i] = sAllocator.allocate(twodScalarArrayLength());
        }
    }

    void deallocate() {
        labelAllocator  lAllocator;
        scalarAllocator sAllocator;


        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            auto ptr = labelData_[i];
            lAllocator.deallocate(ptr, labelArrayLength());
        }
        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            auto ptr = scalarData_[i];
            sAllocator.deallocate(ptr, scalarArrayLength());
        }
        for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
            auto ptr = twodScalarData_[i];
            sAllocator.deallocate(ptr, twodScalarArrayLength());
        }
    }

    std::array<gLabel*, N_LABEL_ARRAYS>        labelData_{};
    std::array<gScalar*, N_SCALAR_ARRAYS>      scalarData_{};
    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData_{};
};

} // namespace FoamGpu