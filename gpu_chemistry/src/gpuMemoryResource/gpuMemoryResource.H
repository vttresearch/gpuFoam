#pragma once

#include "gpuBuffer.H"
#include "gpuConstants.H"
#include "mdspan.H"
#include "pointer_casts.hpp"
#include <array>
#include <vector>

namespace FoamGpu {

template <class labelAllocator, class scalarAllocator>
struct gpuMemoryResource {

    labelAllocator  lAllocator_;
    scalarAllocator sAllocator_;

    gpuMemoryResource() = default;

    gpuMemoryResource(gLabel nCells, gLabel nSpecie)
        : nCells_(nCells)
        , nSpecie_(nSpecie)
        , nEqns_(nSpecie + 2) {

        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            labelData_[i] =
                make_raw_pointer(lAllocator_.allocate(nEqns_ * nCells_));
        }

        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            scalarData_[i] =
                make_raw_pointer(sAllocator_.allocate(nEqns_ * nCells_));
        }

        for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
            twodScalarData_[i] =
                make_raw_pointer(sAllocator_.allocate(nEqns_ * nEqns_ * nCells_));
        }

    }

    ~gpuMemoryResource() { this->deallocate(); }

    gLabel nCells() const { return nCells_; }
    gLabel nEqns() const { return nEqns_; }
    gLabel nSpecie() const { return nSpecie_; }

    std::array<gLabel*, N_LABEL_ARRAYS> getLabelData() { return labelData_; }

    std::array<gScalar*, N_SCALAR_ARRAYS> getScalarData() {
        return scalarData_;
    }

    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> getTwodScalarData() {
        return twodScalarData_;
    }

private:
    gLabel nCells_;
    gLabel nSpecie_;
    gLabel nEqns_;

    std::array<gLabel*, N_LABEL_ARRAYS>   labelData_{};
    std::array<gScalar*, N_SCALAR_ARRAYS> scalarData_{};
    std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData_{};

    void deallocate() {

        // TODO: these casts dont make any sense because of the templated
        //       allocators
        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            auto ptr = make_device_pointer(labelData_[i]);
            lAllocator_.deallocate(ptr, nEqns_ * nCells_);
        }

        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            auto ptr = make_device_pointer(scalarData_[i]);
            sAllocator_.deallocate(ptr, nEqns_ * nCells_);
        }

        for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
            auto ptr = make_device_pointer(twodScalarData_[i]);
            sAllocator_.deallocate(ptr, nEqns_* nEqns_ * nCells_);
        }

    }
};

template <class A1, class A2>
static inline std::vector<gpuBuffer>
splitToBuffers(gpuMemoryResource<A1, A2>& mr) {

    std::vector<gpuBuffer> ret(mr.nCells());

    auto ld = mr.getLabelData();
    auto sd = mr.getScalarData();
    auto twod_sd = mr.getTwodScalarData();

    for (gLabel j = 0; j < mr.nCells(); ++j) {
        std::array<gLabel*, N_LABEL_ARRAYS> labelData{};
        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            labelData[i]  = ld[i] + mr.nEqns() * j;
        }

        std::array<gScalar*, N_SCALAR_ARRAYS> scalarData{};
        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            scalarData[i] = sd[i] + mr.nEqns() * j;
        }

        std::array<gScalar*, N_TWOD_SCALAR_ARRAYS> twodScalarData{};
        for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
            twodScalarData[i] = twod_sd[i] + mr.nEqns() * mr.nEqns() * j;
        }

        ret[j] = gpuBuffer(labelData, scalarData, twodScalarData, mr.nSpecie());
    }

    return ret;
}



} // namespace FoamGpu