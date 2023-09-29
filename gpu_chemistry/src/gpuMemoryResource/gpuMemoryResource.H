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
    }

    ~gpuMemoryResource() { this->deallocate(); }

    gLabel nCells() const { return nCells_; }
    gLabel nEqns() const { return nEqns_; }
    gLabel nSpecie() const { return nSpecie_; }

    std::array<gLabel*, N_LABEL_ARRAYS> getLabelData() { return labelData_; }

    std::array<gScalar*, N_SCALAR_ARRAYS> getScalarData() {
        return scalarData_;
    }

private:
    gLabel nCells_;
    gLabel nSpecie_;
    gLabel nEqns_;

    std::array<gLabel*, N_LABEL_ARRAYS>   labelData_{};
    std::array<gScalar*, N_SCALAR_ARRAYS> scalarData_{};

    void deallocate() {

        // TODO: these casts dont make any sense because of the templated
        //       allocators
        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            auto ptr = make_device_pointer(labelData_[i]);
            lAllocator_.deallocate(ptr, nEqns_ * nCells_);
        }

        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            auto ptr = make_device_pointer(scalarData_[i]);
            sAllocator_.deallocate(ptr, nEqns_ * nCells_);
        }
    }
};

template <class A1, class A2>
static inline std::vector<gpuBuffer>
splitToBuffers(gpuMemoryResource<A1, A2>& mr) {

    std::vector<gpuBuffer> ret(mr.nCells());

    auto ld = mr.getLabelData();
    auto sd = mr.getScalarData();

    for (gLabel j = 0; j < mr.nCells(); ++j) {
        std::array<gLabel*, N_LABEL_ARRAYS> labelData{};
        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            //gLabel* begin = ld[i];
            labelData[i]  = ld[i] + mr.nEqns() * j;
        }

        std::array<gScalar*, N_SCALAR_ARRAYS> scalarData{};
        for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
            // gScalar* begin = sd[i];
            scalarData[i] = sd[i] + mr.nEqns() * j;
        }

        ret[j] = gpuBuffer(labelData, scalarData, mr.nSpecie());
    }

    return ret;
}

/*
template <class A1, class A2>
static inline gpuMemoryResource<std::allocator<gLabel>, std::allocator<gScalar>>
toHost(gpuMemoryResource<A1, A2>& mr) {


    using return_type = gpuMemoryResource<std::allocator<gLabel>, std::allocator<gScalar>>;
    return_type ret(mr.nCells(), nSpecie());
    auto ld = mr.getLabelData();
    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {



            gLabel* begin = ld[i];
            labelData[i]  = begin + mr.nEqns() * j;

    }

}
*/

} // namespace FoamGpu