#pragma once

#include "cuda_host_dev.H"
#include "gpuODESolverBase.H"
#include "gpuODESolverInputs.H"
#include "ludecompose.H"
#include "gpu_constants.H"

namespace FoamGpu {

template <class System>
struct gpuAdaptiveSolver : public gpuODESolverBase<System> {

    //- Step-size adjustment controls
    gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

    gpuAdaptiveSolver() = default;

    /// @brief Construct from template ode system and solver inputs
    /// @param system the ODE system used
    /// @param i the tolerance inputs
    gpuAdaptiveSolver(const System&             system,
                      const gpuODESolverInputs& i)
        : gpuODESolverBase<System>(system, i)
        , safeScale_(i.safeScale)
        , alphaInc_(i.alphaInc)
        , alphaDec_(i.alphaDec)
        , minScale_(i.minScale)
        , maxScale_(i.maxScale) {}

    /// @brief Get the safeScale value
    /// @return Value of the safeScale
    CUDA_HOSTDEV gScalar safeScale() const { return safeScale_; }

    /// @brief Get the alpha increase value
    /// @return Value of alpha increase
    CUDA_HOSTDEV gScalar alphaInc() const { return alphaInc_; }

    /// @brief Get the alpha decrease value
    /// @return Value of alpha decrease
    CUDA_HOSTDEV gScalar alphaDec() const { return alphaDec_; }

    /// @brief Get the minimum scale value
    /// @return Minimum scale
    CUDA_HOSTDEV gScalar minScale() const { return minScale_; }

    /// @brief Get the maximum scale value
    /// @return Maximum scale
    CUDA_HOSTDEV gScalar maxScale() const { return maxScale_; }

    CUDA_HOSTDEV void
    computeDecomposedJacobian(const gScalar                  dx,
                              const gScalar                 gamma,
                              const mdspan<const gScalar, 1> y0,
                              mdspan<gScalar, 1>             dfdx,
                              mdspan<gScalar, 2>             a,
                              mdspan<gLabel, 1> pivotIndices,
                              gpuBuffer&        buffer) const {

        this->system().jacobian(y0, dfdx, a, buffer);

        const gScalar temp = gScalar(1.0) / (gamma * dx);
        // Negate the jacobian and add 1/(gamma*dx) to diagonal
        for (gLabel i = 0; i < this->nEqns(); i++) {
            for (gLabel j = 0; j < this->nEqns(); j++) {
                a(i, j) = -a(i, j);
            }

            a(i, i) += temp; // gScalar(1.0)/(gamma*dx);
        }

        FoamGpu::LUDecompose(a, pivotIndices, buffer.lubuffer());
    }
};

/// @brief Integrate the ode as far as possible between [x, x+dxTry]
/// while adjusting the step size.
/// @param ode The ode to integrate
/// @param x Start point.
/// @param y The modified solution vector of size ode.nEqns().
/// @param dxTry Initial guess for the step size.
/// @param buffer A preallocated memory buffer.
template <class Ode>
static inline CUDA_HOSTDEV void adaptiveSolve(const Ode&         ode,
                                              gScalar&           x,
                                              mdspan<gScalar, 1> y,
                                              gScalar&   dxTry,
                                              gpuBuffer& buffer) {

    auto dydx0 = buffer.dydx0();
    auto yTemp = buffer.yTemp();

    gScalar dx  = dxTry;
    gScalar err = 0.0;

    ode.system().derivatives(y, dydx0, buffer);

    // Loop over solver and adjust step-size as necessary
    // to achieve desired error
    do {
        // Solve step and provide error estimate
        err = ode.solve(y, dydx0, dx, yTemp, buffer);

        // If error is large reduce dx
        if (err > 1) {
            gScalar scale =
                fmax(ode.safeScale() * pow(err, -ode.alphaDec()),
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
    if (err > pow(ode.maxScale() / ode.safeScale(),
                  -1.0 / ode.alphaInc())) {
        dxTry = fmin(fmax(ode.safeScale() * pow(err, -ode.alphaInc()),
                        ode.minScale()),
                    ode.maxScale()) *
                dx;
    } else {
        dxTry = ode.safeScale() * ode.maxScale() * dx;
    }
}

} // namespace FoamGpu