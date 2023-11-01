#pragma once

#include "check_ptr.H"
#include "gpuAdaptiveSolver.H"
#include "gpuBuffer.H"
#include "gpuODESolverBase.H"
#include "gpu_constants.H"
#include "ludecompose.H"
#include "normaliseError.H"

namespace FoamGpu {


namespace Rosenbrock12Coeffs {

    //constexpr gScalar gamma = 1 + 1.0/std::sqrt(2.0);
    //constexpr gScalar gamma = 1.0 + 1.0/1.41421356237;
    constexpr gScalar gamma = 1.0 + 1.0/1.414213562373095;
    constexpr gScalar a21 = 1.0/gamma;
    constexpr gScalar c2 = 1.0;
    constexpr gScalar c21 = -2.0/gamma;
    constexpr gScalar b1 = (3.0/2.0)/gamma;
    constexpr gScalar b2 = (1.0/2.0)/gamma;
    constexpr gScalar e1 = b1 - 1.0/gamma;
    constexpr gScalar e2 = b2;
    constexpr gScalar d1 = gamma;
    constexpr gScalar d2 = -gamma;



} // namespace Rosenbrock12Coeffs


template <class System>
class gpuRosenbrock12 : public gpuAdaptiveSolver<System> {

public:
    gpuRosenbrock12() = default;

    gpuRosenbrock12(const System& system, const gpuODESolverInputs& i)
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
    CUDA_HOSTDEV gScalar solve(const mdspan<const gScalar, 1> y0,
                               const mdspan<const gScalar, 1> dydx0,
                               const gScalar                  dx,
                               mdspan<gScalar, 1>             y,
                               gpuBuffer& buffer) const {
        
        using namespace Rosenbrock12Coeffs;
        
        auto dydx         = buffer.dydx();
        auto dfdx         = buffer.dfdx();
        auto pivotIndices = buffer.pivotIndices();
        auto a            = buffer.J();

        this->computeDecomposedJacobian(
            dx, gamma, y0, dfdx, a, pivotIndices, buffer);
        
        // Calculate k1:
        auto k1 = buffer.k1();
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k1[i] = dydx0[i] + dx * d1 * dfdx[i];
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k1);


        // Calculate k2:
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i] = y0[i] + a21 * k1[i];
        }

        this->system().derivatives(y, dydx, buffer);

        auto k2 = buffer.k2();
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            k2[i] = dydx[i] + dx * d2 * dfdx[i] + c21 * k1[i] / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k2);

        // Calculate error and update state:
        auto err = buffer.err();
        for (gLabel i = 0; i < this->nEqns(); ++i) {
            y[i]   = y0[i] + b1 * k1[i] + b2 * k2[i];
            err[i] = e1 * k1[i] + e2 * k2[i];
        }

        return normaliseError(
            y0, y, err, this->absTol(), this->relTol());

        
    }
};

} // namespace FoamGpu
