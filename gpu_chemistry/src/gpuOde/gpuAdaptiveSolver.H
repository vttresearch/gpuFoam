#pragma once

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuODESolverBase.H"
#include "gpuODESolverInputs.H"
#include "gpu_constants.H"

namespace FoamGpu {

template <class System>
struct gpuAdaptiveSolver : public gpuODESolverBase<System> {

    //- Step-size adjustment controls
    gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

    gpuAdaptiveSolver() = default;

    gpuAdaptiveSolver(const System& system, const gpuODESolverInputs& i)
        : gpuODESolverBase<System>(system, i)
        , safeScale_(i.safeScale)
        , alphaInc_(i.alphaInc)
        , alphaDec_(i.alphaDec)
        , minScale_(i.minScale)
        , maxScale_(i.maxScale) {}

    CUDA_HOSTDEV gScalar safeScale() const { return safeScale_; }
    CUDA_HOSTDEV gScalar alphaInc() const { return alphaInc_; }
    CUDA_HOSTDEV gScalar alphaDec() const { return alphaDec_; }
    CUDA_HOSTDEV gScalar minScale() const { return minScale_; }
    CUDA_HOSTDEV gScalar maxScale() const { return maxScale_; }
};

template <class Ode>
static inline CUDA_HOSTDEV void adaptiveSolve(const Ode&         ode,
                                              gScalar&           x,
                                              mdspan<gScalar, 1> y,
                                              gScalar&           dxTry,
                                              gpuBuffer&         buffer) {

    auto dydx0 = buffer.dydx0();
    auto yTemp = buffer.yTemp();

    gScalar dx  = dxTry;
    gScalar err = 0.0;

    ode.system().derivatives(x, y, dydx0, buffer);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do {
        // Solve step and provide error estimate
        err = ode.solve(x, y, dydx0, dx, yTemp, buffer);

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

} // namespace FoamGpu