#pragma once

#include "gpuBuffer.H"
#include "gpuConstants.H"
#include "mdspan.H"
#include "stepState.H"

namespace FoamGpu {

static inline CUDA_HOSTDEV gScalar normaliseError(
    gScalar y0, gScalar y, gScalar err, gScalar absTol, gScalar relTol) {
    double tol = absTol + relTol * std::max(std::abs(y0), std::abs(y));
    double ret = std::abs(err) / tol;
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
        maxErr    = std::max(maxErr, t);
    }

    return maxErr;
}

template <class System, class Ode>
static inline CUDA_HOSTDEV void adaptiveSolve(const System&      system,
                                              const Ode&         ode,
                                              gScalar&           x,
                                              mdspan<gScalar, 1> y,
                                              const gLabel       li,
                                              gScalar&           dxTry,
                                              gpuBuffer&         buffer) {

    auto dydx0 = buffer.dydx0();
    auto yTemp = buffer.yTemp();

    gScalar dx  = dxTry;
    gScalar err = 0.0;

    system.derivatives(x, y, li, dydx0, buffer);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do {
        // Solve step and provide error estimate
        err = ode.solve(x, y, li, dydx0, dx, yTemp, buffer);

        // If error is large reduce dx
        if (err > 1) {
            gScalar scale = max(ode.safeScale() * pow(err, -ode.alphaDec()),
                                ode.minScale());
            dx *= scale;

            if (dx < gpuVSmall) {
                printf("Small time step in ode. \n");
                assert(0);
            }
        }
    } while (err > 1.0);

    // Update the state
    x += dx;

    // y = yTemp;
    for (gLabel i = 0; i < gLabel(y.size()); ++i) { y[i] = yTemp[i]; }

    // If the error is small increase the step-size
    if (err > pow(ode.maxScale() / ode.safeScale(), -1.0 / ode.alphaInc())) {
        dxTry = min(max(ode.safeScale() * pow(err, -ode.alphaInc()),
                        ode.minScale()),
                    ode.maxScale()) *
                dx;
    } else {
        dxTry = ode.safeScale() * ode.maxScale() * dx;
    }
}

/*
// This is the main solve call from gpuODESolverBase
template <class Ode>
static inline CUDA_HOSTDEV void odeSolve(const Ode&         ode,
                                         const gScalar      xStart,
                                         const gScalar      xEnd,
                                         mdspan<gScalar, 1> y,
                                         const gLabel       li,
                                         gScalar&           dxTry,
                                         gpuBuffer&         buffer,
                                         gLabel             maxSteps) {

    stepState step(dxTry);
    gScalar   x = xStart;

    // printf("dxTry value = %lf\n", dxTry);

    for (gLabel nStep = 0; nStep < maxSteps; nStep++) {

        // Store previous iteration dxTry
        gScalar dxTry0 = step.dxTry;

        step.reject = false;

        // Check if this is a truncated step and set dxTry to integrate to
        // xEnd
        if ((x + step.dxTry - xEnd) * (x + step.dxTry - xStart) > 0) {
            step.last  = true;
            step.dxTry = xEnd - x;
        }

        // Integrate as far as possible up to step.dxTry
        ode.interface(x, y, li, step, buffer);

        // Check if reached xEnd
        if ((x - xEnd) * (xEnd - xStart) >= 0) {
            if (nStep > 0 && step.last) { step.dxTry = dxTry0; }

            dxTry = step.dxTry;

            return;
        }

        step.first = false;

        // If the step.dxTry was reject set step.prevReject
        if (step.reject) { step.prevReject = true; }
    }
}
*/

} // namespace FoamGpu
