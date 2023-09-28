#pragma once

#include <array>
#include "gpuConstants.H"
#include "mdspan.H"

namespace FoamGpu{

enum labelArrays{
    pivotIndices,
    N_LABEL_ARRAYS
};

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


template<class labelAllocator, class scalarAllocator>
struct gpuMemoryResource{


    labelAllocator lAllocator_;
    scalarAllocator sAllocator_;

    gpuMemoryResource() = default;

    gpuMemoryResource(gLabel nCells, gLabel nSpecie) :
    nCells_(nCells),
    nSpecie_(nSpecie),
    nEqns_(nSpecie + 2)
    {

        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i)
        {
            labelData_[i] = lAllocator_.allocate(nEqns_ * nCells_);
        }

        for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i)
        {
            scalarData_[i] = sAllocator_.allocate(nEqns_ * nCells_);
        }


    }

private:

    gLabel nCells_;
    gLabel nSpecie_;
    gLabel nEqns_;

    std::array<gLabel*, N_LABEL_ARRAYS> labelData_;
    std::array<gScalar*, N_SCALAR_ARRAYS> scalarData_;


};

}