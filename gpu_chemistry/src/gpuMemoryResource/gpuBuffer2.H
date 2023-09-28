#pragma once

#include "mdspan.H"
#include "arrays.H"
#include "gpuConstants.H"
#include "check_ptr.H"
namespace FoamGpu {

struct gpuBuffer2 {

    CUDA_HOSTDEV
    gpuBuffer2() = default;

    CUDA_HOSTDEV
    gpuBuffer2(std::array<gLabel*, N_LABEL_ARRAYS>   lData,
               std::array<gScalar*, N_SCALAR_ARRAYS> sData,
               gLabel                                nSpecie)
        : lData_(lData)
        , sData_(sData)
        , nSpecie_(nSpecie)
        , nEqns_(nSpecie + 2) {

            for (size_t i = 0; i < lData_.size(); ++i)
            {
                check_ptr(lData[i], "label pointer");
            }
            for (size_t i = 0; i < sData_.size(); ++i)
            {
                check_ptr(sData[i], "scalar pointer");
            }
        }


    CUDA_HOSTDEV
    auto getLabelData(labelArrays a, gLabel size) {
        return mdspan<gLabel, 1>(lData_[a], extents<1>{size});
    }

    CUDA_HOSTDEV
    auto getScalarData(scalarArrays a, gLabel size) {
        return mdspan<gScalar, 1>(sData_[a], extents<1>{size});
    }

    CUDA_HOSTDEV auto pivotIndices() {
        return getLabelData(labelArrays::pivotIndices, nEqns_);
    }


    CUDA_HOSTDEV auto dydx0() {
        return getScalarData(scalarArrays::dydx0, nEqns_);
    }

    CUDA_HOSTDEV auto yTemp() {
        return getScalarData(scalarArrays::yTemp, nEqns_);
    }

    CUDA_HOSTDEV auto dydx() {
        return getScalarData(scalarArrays::dydx, nEqns_);
    }
    CUDA_HOSTDEV auto dfdx() {
        return getScalarData(scalarArrays::dfdx, nEqns_);
    }

    CUDA_HOSTDEV auto k1() {
        return getScalarData(scalarArrays::k1, nEqns_);
    }

    CUDA_HOSTDEV auto k2() {
        return getScalarData(scalarArrays::k2, nEqns_);
    }

    CUDA_HOSTDEV auto k3() {
        return getScalarData(scalarArrays::k3, nEqns_);
    }

    CUDA_HOSTDEV auto k4() {
        return getScalarData(scalarArrays::k4, nEqns_);
    }

    CUDA_HOSTDEV auto err() {
        return getScalarData(scalarArrays::err, nEqns_);
    }

    CUDA_HOSTDEV auto lubuffer() {
        return getScalarData(scalarArrays::lubuffer, nEqns_);
    }

    // Note! Length is nSpecie
    CUDA_HOSTDEV auto c() {
        return getScalarData(scalarArrays::c, nSpecie_);
    }

    CUDA_HOSTDEV auto tempField1() {
        return getScalarData(scalarArrays::tempField1, nEqns_);
    }

    CUDA_HOSTDEV auto tempField2() {
        return getScalarData(scalarArrays::tempField2, nEqns_);
    }


private:


    std::array<gLabel*, N_LABEL_ARRAYS>   lData_;
    std::array<gScalar*, N_SCALAR_ARRAYS> sData_;

    gLabel nSpecie_;
    gLabel nEqns_;
};

} // namespace FoamGpu
