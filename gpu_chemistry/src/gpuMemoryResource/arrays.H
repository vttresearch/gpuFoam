

#pragma once

namespace FoamGpu {

enum labelArrays { pivotIndices, N_LABEL_ARRAYS };

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

enum twoDScalarArrays {
    J,
    N_TWOD_SCALAR_ARRAYS
};

} // namespace FoamGpu
