#pragma once

#include "cuda_host_dev.H"
#include "gpu_constants.H"
#include "swap.H"

namespace FoamGpu {

static inline CUDA_HOSTDEV void LUDecompose(mdspan<gScalar, 2> matrix,
                                            mdspan<gLabel, 1>  pivotIndices,
                                            mdspan<gScalar, 1> vv) {

    const gLabel m = static_cast<gLabel>(matrix.extents().extent(1));

    // gLabel sign = 1;
    for (gLabel i = 0; i < m; i++) {
        gScalar largestCoeff = 0.0;

        for (gLabel j = 0; j < m; j++) {
            gScalar temp = std::abs(matrix(i, j));
            if (temp > largestCoeff) { largestCoeff = temp; }
        }

        vv[i] = 1.0 / largestCoeff;
    }

    //std::array<gScalr, SPECIE_MAX> matrix_i{};

    for (gLabel j = 0; j < m; j++) {
        for (gLabel i = 0; i < j; i++) {

            for (gLabel k = 0; k < i; k++) {
                matrix(i, j) -= matrix(i, k) * matrix(k, j);
            }
        }

        gLabel iMax = 0;

        gScalar largestCoeff = 0.0;
        for (gLabel i = j; i < m; i++) {

            for (gLabel k = 0; k < j; k++) {
                matrix(i,j) -= matrix(i, k) * matrix(k, j);
            }

            gScalar temp = vv[i] * std::abs(matrix(i, j));
            if (temp >= largestCoeff) {
                largestCoeff = temp;
                iMax         = i;
            }
        }

        pivotIndices[j] = iMax;

        if (j != iMax) {

            for (gLabel k = 0; k < m; k++) {
                swap(matrix(j, k), matrix(iMax, k));
            }

            // sign *= -1;
            vv[iMax] = vv[j];
        }

        // matrix(j,j) = std::max(matrix(j,j), small);

        if (matrix(j, j) == 0.0) { matrix(j, j) = gpuSmall; }

        if (j != m - 1) {
            gScalar rDiag = 1.0 / matrix(j, j);

            for (gLabel i = j + 1; i < m; i++) { matrix(i, j) *= rDiag; }
        }
    }
}

static inline CUDA_HOSTDEV void
LUBacksubstitute(mdspan<gScalar, 2>            luMatrix,
                 const mdspan<const gLabel, 1> pivotIndices,
                 mdspan<gScalar, 1>            sourceSol) {

    const gLabel m  = static_cast<gLabel>(luMatrix.extents().extent(1));
    gLabel       ii = 0;

    for (gLabel i = 0; i < m; i++) {
        gLabel  ip    = pivotIndices[i];
        gScalar sum   = sourceSol[ip];
        sourceSol[ip] = sourceSol[i];

        if (ii != 0) {
            for (gLabel j = ii - 1; j < i; j++) {
                sum -= luMatrix(i, j) * sourceSol[j];
            }
        } else if (sum != gScalar(0)) {
            ii = i + 1;
        }

        sourceSol[i] = sum;
    }

    for (gLabel i = m - 1; i >= 0; i--) {
        gScalar sum = sourceSol[i];

        for (gLabel j = i + 1; j < m; j++) {
            sum -= luMatrix(i, j) * sourceSol[j];
        }

        sourceSol[i] = sum / luMatrix(i, i);
    }
}

} // namespace FoamGpu