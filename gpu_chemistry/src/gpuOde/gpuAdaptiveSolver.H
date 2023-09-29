#pragma once

#include "cuda_host_dev.H"
#include "gpuODESolverInputs.H"
#include "gpuODESystem.H"
#include "gpu_constants.H"

namespace FoamGpu {

template <class Derived> struct gpuAdaptiveSolver {

    //- Step-size adjustment controls
    gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

    gpuAdaptiveSolver() = default;

    gpuAdaptiveSolver(const gpuODESolverInputs& i)
        : safeScale_(i.safeScale)
        , alphaInc_(i.alphaInc)
        , alphaDec_(i.alphaDec)
        , minScale_(i.minScale)
        , maxScale_(i.maxScale) {}

private:
    //- Solve a single step dx and return the error

    CUDA_HOSTDEV gScalar interface(const gScalar             x0,
                                   const mdspan<gScalar, 1>& y0,
                                   const gLabel              li,
                                   const mdspan<gScalar, 1>& dydx0,
                                   const gScalar             dx,
                                   mdspan<gScalar, 1>&       y,
                                   mdspan<gScalar, 2>&       a,
                                   gpuBuffer&                buffer) const {

        return static_cast<const Derived*>(this)->solve(
            x0, y0, li, dydx0, dx, y, a, buffer);
    }


public:
    //- Solve the ODE system and the update the state
    // template <class System, class Span, class TwoDSpan>
    template <class System>
    CUDA_HOSTDEV void adaptiveSolve(const System&       odes,
                                    gScalar&            x,
                                    mdspan<gScalar, 1>& y,
                                    const gLabel        li,
                                    gScalar&            dxTry,
                                    mdspan<gScalar, 2>& a,
                                    gpuBuffer&          buffer) const {

        auto dydx0 = buffer.dydx0();
        auto yTemp = buffer.yTemp();

        gScalar dx  = dxTry;
        gScalar err = 0.0;

        odes.derivatives(x, y, li, dydx0, buffer);

        // Loop over solver and adjust step-size as necessary
        // to achieve desired error
        do {
            // Solve step and provide error estimate
            err = interface(x, y, li, dydx0, dx, yTemp, a, buffer);

            // If error is large reduce dx
            if (err > 1) {
                gScalar scale =
                    std::max(safeScale_ * std::pow(err, -alphaDec_), minScale_);
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
        if (err > std::pow(maxScale_ / safeScale_, -1.0 / alphaInc_)) {
            dxTry = std::min(std::max(safeScale_ * std::pow(err, -alphaInc_),
                                      minScale_),
                             maxScale_) *
                    dx;
        } else {
            dxTry = safeScale_ * maxScale_ * dx;
        }
    }
};

} // namespace FoamGpu