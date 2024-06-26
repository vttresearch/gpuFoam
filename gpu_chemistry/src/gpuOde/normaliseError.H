#pragma once
#include "gpu_constants.H"
#include "mdspan.H"
namespace FoamGpu{

static inline CUDA_HOSTDEV gScalar normaliseError(
    gScalar y0, gScalar y, gScalar err, gScalar absTol, gScalar relTol) {
    gScalar tol = absTol + relTol * fmax(fabs(y0), fabs(y));
    gScalar ret = fabs(err) / tol;
    return ret;
}

static inline CUDA_HOSTDEV gScalar
normaliseError(const mdspan<const gScalar, 1> y0,
               const mdspan<const gScalar, 1> y,
               const mdspan<const gScalar, 1> err,
               gScalar                        absTol,
               gScalar                        relTol) {
    // Calculate the maximum error
    gScalar maxErr = 0.0;

    for (gLabel i = 0; i < gLabel(y.size()); ++i) {

        gScalar t = normaliseError(y0[i], y[i], err[i], absTol, relTol);
        maxErr    = fmax(maxErr, t);
    }

    return maxErr;
}


}