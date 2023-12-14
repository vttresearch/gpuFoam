#pragma once

#include "gpuBuffer.H"
#include "gpu_constants.H"
#include "mdspan.H"
#include "pointer_casts.hpp"
#include <array>
#include <vector>

namespace FoamGpu {

struct memoryResource {

    memoryResource() = default;

    memoryResource(gLabel nCells, gLabel nSpecie)
        : nCells_(nCells)
        , nSpecie_(nSpecie)
        , nEqns_(nSpecie + 2) {}

    virtual ~memoryResource() {}

    gLabel nCells() const { return nCells_; }
    gLabel nEqns() const { return nEqns_; }
    gLabel nSpecie() const { return nSpecie_; }

    virtual std::array<gLabel*, N_LABEL_ARRAYS>& getLabelData() = 0;

    virtual std::array<gScalar*, N_SCALAR_ARRAYS>& getScalarData() = 0;

    virtual std::array<gScalar*, N_TWOD_SCALAR_ARRAYS>& getTwodScalarData() = 0;

    virtual void resize(gLabel nCells, gLabel nSpecie) = 0;

protected:
    gLabel labelArrayLength() const { return nEqns() * nCells(); }

    gLabel scalarArrayLength() const { return nEqns() * nCells(); }

    gLabel twodScalarArrayLength() const {
        return nEqns() * nEqns() * nCells();
    }

    bool shouldReallocate(gLabel nCells, gLabel nSpecie) const {

        if ((nCells != this->nCells()) || (nSpecie != this->nSpecie()) ||
            (nCells + 2 != this->nEqns())) {
            return true;
        }
        return false;
    }


    gLabel nCells_;
    gLabel nSpecie_;
    gLabel nEqns_;
};

static inline std::vector<gpuBuffer> splitToBuffers(memoryResource& mr) {

    std::vector<gpuBuffer> ret(mr.nCells());

    auto& ld      = mr.getLabelData();
    auto& sd      = mr.getScalarData();
    auto& twod_sd = mr.getTwodScalarData();

    for (gLabel j = 0; j < mr.nCells(); ++j) {
        std::array<gLabel*, N_LABEL_ARRAYS> labelData{};
        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
            labelData[i] = ld[i] + mr.nEqns() * j;
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