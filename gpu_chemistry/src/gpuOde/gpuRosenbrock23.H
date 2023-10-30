#pragma once

#include "check_ptr.H"
#include "gpuAdaptiveSolver.H"
#include "gpuBuffer.H"
#include "gpuODESolverBase.H"
#include "gpu_constants.H"
#include "ludecompose.H"
#include "normaliseError.H"

namespace FoamGpu {

namespace Rosenbrock23Coeffs {

constexpr gScalar a21 = 1;
constexpr gScalar a31 = 1;
constexpr gScalar a32 = 0;

constexpr gScalar c21 = -1.0156171083877702091975600115545;
constexpr gScalar c31 = 4.0759956452537699824805835358067;
constexpr gScalar c32 = 9.2076794298330791242156818474003;

constexpr gScalar b1 = 1;
constexpr gScalar b2 = 6.1697947043828245592553615689730;
constexpr gScalar b3 = -0.4277225654321857332623837380651;

constexpr gScalar e1 = 0.5;
constexpr gScalar e2 = -2.9079558716805469821718236208017;
constexpr gScalar e3 = 0.2235406989781156962736090927619;

constexpr gScalar gamma = 0.43586652150845899941601945119356;
constexpr gScalar c2    = 0.43586652150845899941601945119356;

constexpr gScalar d1 = 0.43586652150845899941601945119356;
constexpr gScalar d2 = 0.24291996454816804366592249683314;
constexpr gScalar d3 = 2.1851380027664058511513169485832;

} // namespace Rosenbrock23Coeffs

template <class System>
class gpuRosenbrock23 : public gpuAdaptiveSolver<System> {

public:
    gpuRosenbrock23() = default;

    gpuRosenbrock23(const System& system, const gpuODESolverInputs& i)
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

        using namespace Rosenbrock23Coeffs;

        auto dydx         = buffer.dydx();
        auto dfdx         = buffer.dfdx();
        auto pivotIndices = buffer.pivotIndices();
        auto k1           = buffer.k1();
        auto k2           = buffer.k2();
        auto k3           = buffer.k3();
        //auto err          = buffer.err();
        auto lubuffer     = buffer.lubuffer();
        auto a            = buffer.J();

        this->system().jacobian(y0, dfdx, a, buffer);

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

        this->system().derivatives(y, dydx, buffer);

        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k2[i] = dydx[i] + dx * d2 * dfdx[i] + c21 * k1[i] / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k2);

        // Calculate k3:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k3[i] =
                dydx[i] + dx * d3 * dfdx[i] + (c31 * k1[i] + c32 * k2[i]) / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k3);

        // Calculate error and update state:
        auto err = buffer.err();
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i]   = y0[i] + b1 * k1[i] + b2 * k2[i] + b3 * k3[i];
            err[i] = e1 * k1[i] + e2 * k2[i] + e3 * k3[i];
        }

        return normaliseError(y0, y, err, this->absTol(), this->relTol());
    }
};

} // namespace FoamGpu
