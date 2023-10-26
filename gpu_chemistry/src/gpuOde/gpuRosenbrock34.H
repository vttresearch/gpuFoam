#pragma once

#include "check_ptr.H"
#include "gpuAdaptiveSolver.H"
#include "gpuBuffer.H"
#include "gpuODESolverBase.H"
#include "gpu_constants.H"
#include "ludecompose.H"
#include "normaliseError.H"

namespace FoamGpu {

namespace Rosenbrock34Coeffs {

// Constants by Shampine
// More accurate than the L-Stable coefficients for small step-size
// but less stable for large step-size
constexpr gScalar a21 = 2.0;
constexpr gScalar a31 = 48.0 / 25.0;
constexpr gScalar a32 = 6.0 / 25.0;

constexpr gScalar c21 = -8.0;
constexpr gScalar c31 = 372.0 / 25.0;
constexpr gScalar c32 = 12.0 / 5.0;

constexpr gScalar c41 = -112.0 / 125.0;
constexpr gScalar c42 = -54.0 / 125.0;
constexpr gScalar c43 = -2.0 / 5.0;

constexpr gScalar b1 = 19.0 / 9.0;
constexpr gScalar b2 = 1.0 / 2.0;
constexpr gScalar b3 = 25.0 / 108.0;
constexpr gScalar b4 = 125.0 / 108.0;

constexpr gScalar e1 = 34.0 / 108.0;
constexpr gScalar e2 = 7.0 / 36.0;
constexpr gScalar e3 = 0.0;
constexpr gScalar e4 = 125.0 / 108.0;

constexpr gScalar gamma = 1.0 / 2.0;
constexpr gScalar c2    = 1.0;
constexpr gScalar c3    = 3.0 / 5.0;

constexpr gScalar d1 = 1.0 / 2.0;
constexpr gScalar d2 = -3.0 / 2.0;
constexpr gScalar d3 = 605.0 / 250.0;
constexpr gScalar d4 = 29.0 / 250.0;
} // namespace Rosenbrock34Coeffs

template <class System>
class gpuRosenbrock34 : public gpuAdaptiveSolver<System> {

public:
    gpuRosenbrock34() = default;

    gpuRosenbrock34(const System& system, const gpuODESolverInputs& i)
        : gpuAdaptiveSolver<System>(system, i) {}

    CUDA_HOSTDEV void solve(gScalar&           x,
                            mdspan<gScalar, 1> y,
                            stepState&         step,
                            gpuBuffer&         buffer) const {
        gScalar x0 = x;
        adaptiveSolve(*this, x, y, step.dxTry, buffer);
        step.dxDid = x - x0;
    }

    //- Solve a single step dx and return the error
    CUDA_HOSTDEV gScalar solve(const gScalar                  x0,
                               const mdspan<const gScalar, 1> y0,
                               const mdspan<const gScalar, 1> dydx0,
                               const gScalar                  dx,
                               mdspan<gScalar, 1>             y,
                               gpuBuffer&                     buffer) const {

        using namespace Rosenbrock34Coeffs;

        auto dydx         = buffer.dydx();
        auto dfdx         = buffer.dfdx();
        auto pivotIndices = buffer.pivotIndices();
        auto k1           = buffer.k1();
        auto k2           = buffer.k2();
        auto k3           = buffer.k3();
        auto k4           = buffer.k4();
        auto err          = buffer.err();
        auto lubuffer     = buffer.lubuffer();
        auto a            = buffer.J();

        this->system().jacobian(x0, y0, dfdx, a, buffer);

        const gScalar temp = gScalar(1.0) / (gamma * dx);
        // Negate the jacobian and add 1/(gamma*dx) to diagonal
        for (gLabel i = 0; i < this->nEqns(); i++) {
            for (gLabel j = 0; j < this->nEqns(); j++) { a(i, j) = -a(i, j); }

            a(i, i) += temp; // gScalar(1.0)/(gamma*dx);
        }

        FoamGpu::LUDecompose(a, pivotIndices, lubuffer);

        // Calculate k1:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k1[i] = dydx0[i] + dx * d1 * dfdx[i];
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k1);

        // Calculate k2:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i] = y0[i] + a21 * k1[i];
        }

        this->system().derivatives(x0 + c2 * dx, y, dydx, buffer);

        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k2[i] = dydx[i] + dx * d2 * dfdx[i] + c21 * k1[i] / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k2);

        // Calculate k3:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i] = y0[i] + a31 * k1[i] + a32 * k2[i];
        }

        this->system().derivatives(x0 + c3 * dx, y, dydx, buffer);

        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k3[i] =
                dydx[i] + dx * d3 * dfdx[i] + (c31 * k1[i] + c32 * k2[i]) / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k3);

        // Calculate k4:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k4[i] = dydx[i] + dx * d4 * dfdx[i] +
                    (c41 * k1[i] + c42 * k2[i] + c43 * k3[i]) / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k4);

        // Calculate error and update state:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i]   = y0[i] + b1 * k1[i] + b2 * k2[i] + b3 * k3[i] + b4 * k4[i];
            err[i] = e1 * k1[i] + e2 * k2[i] + e4 * k4[i];
        }

        return normaliseError(y0, y, err, this->absTol(), this->relTol());
    }
};

} // namespace FoamGpu
