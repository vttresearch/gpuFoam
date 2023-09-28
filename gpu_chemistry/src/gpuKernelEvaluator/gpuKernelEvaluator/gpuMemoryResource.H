#pragma once

#include "gpuConstants.H"
#include "mdspan.H"
#include <array>

namespace FoamGpu {

enum scalarArrays {
    dydx0,
    yTemp,
    dydx,
    dfdx,
    k1,
    k2,
    k3,
    k4,
    err,
    lubuffer,
    c,
    tempField1,
    tempField2,
    N_SCALAR_ARRAYS
};

struct gpuBuffer2 {

    gpuBuffer2(std::array<gScalar*, N_SCALAR_ARRAYS> data,
               gLabel*                               pivotIndices,
               gLabel                                nSpecie,
               gLabel                                nEqns)
        : data_(data)
        , pivotIndices_(pivotIndices)
        , nSpecie_(nSpecie)
        , nEqns_(nEqns) {}

    CUDA_HOSTDEV auto dydx0() {
        return mdspan<gScalar, 1>(data_[scalarArrays::dydx0],
                                  extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto yTemp() {
        return mdspan<gScalar, 1>(data_[scalarArrays::yTemp],
                                  extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto dydx() {
        return mdspan<gScalar, 1>(data_[scalarArrays::dydx],
                                  extents<1>{nEqns_});
    }
    CUDA_HOSTDEV auto dfdx() {
        return mdspan<gScalar, 1>(data_[scalarArrays::dfdx],
                                  extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto pivotIndices() {
        return mdspan<gLabel, 1>(pivotIndices_, extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto k1() {
        return mdspan<gScalar, 1>(data_[scalarArrays::k1], extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto k2() {
        return mdspan<gScalar, 1>(data_[scalarArrays::k2], extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto k3() {
        return mdspan<gScalar, 1>(data_[scalarArrays::k3], extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto k4() {
        return mdspan<gScalar, 1>(data_[scalarArrays::k4], extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto err() {
        return mdspan<gScalar, 1>(data_[scalarArrays::err], extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto lubuffer() {
        return mdspan<gScalar, 1>(data_[scalarArrays::lubuffer],
                                  extents<1>{nEqns_});
    }

    // Note! Length is nSpecie
    CUDA_HOSTDEV auto c() {
        return mdspan<gScalar, 1>(data_[scalarArrays::c], extents<1>{nSpecie_});
    }

    CUDA_HOSTDEV auto tempField1() {
        return mdspan<gScalar, 1>(data_[scalarArrays::tempField1],
                                  extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto tempField2() {
        return mdspan<gScalar, 1>(data_[scalarArrays::tempField2],
                                  extents<1>{nEqns_});
    }

    std::array<gScalar*, N_SCALAR_ARRAYS> data_;
    gLabel*                               pivotIndices_;

    gLabel nSpecie_;
    gLabel nEqns_;
};

struct gpuMemoryResource {

    gpuMemoryResource(gLabel nCells, gLabel nSpecie, gLabel nEqns);

    void free();


private:
    std::array<gScalar*, N_SCALAR_ARRAYS> getScalarArrays(gLabel celli);

    gLabel nCells_;
    gLabel nSpecie_;
    gLabel nEqns_;

    std::array<gScalar*, N_SCALAR_ARRAYS> data_;
    gLabel*                               pivotIndices_;

};

} // namespace FoamGpu
