#pragma once

#include "check_ptr.H"
#include "gpuAdaptiveSolver.H"
#include "gpuBuffer.H"
#include "gpuConstants.H"
#include "gpuODESolver.H"
#include "ludecompose.H"

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
    constexpr gScalar c2 = 0.43586652150845899941601945119356;

    constexpr gScalar d1 = 0.43586652150845899941601945119356;
    constexpr gScalar d2 = 0.24291996454816804366592249683314;
    constexpr gScalar d3 = 2.1851380027664058511513169485832;


} // namespace RosenbrockCoeffs

struct gpuRosenbrock23Inputs {

    gpuAdaptiveSolverInputs adaptiveInputs;
    gScalar                 absTol;
    gScalar                 relTol;
    gLabel                  maxSteps;
};

template <class System>
class gpuRosenbrock23 : public gpuODESolver<gpuRosenbrock23<System>>, 
public gpuAdaptiveSolver<gpuRosenbrock23<System>> {
    // Private Data

    CUDA_HOSTDEV gLabel nEqns() const 
    {
        return odes_.nEqns();
    }


public:

    using myType = gpuRosenbrock23<System>;
    using typename gpuODESolver<myType>::stepState;

    System odes_;



    gpuRosenbrock23(const System& ode, const gpuRosenbrock23Inputs& i)
        : gpuODESolver<gpuRosenbrock23<System>>(i.absTol, i.relTol, i.maxSteps)
        , gpuAdaptiveSolver<gpuRosenbrock23<System>>(i.adaptiveInputs)
        , odes_(ode)
        {}

    

    

    using gpuODESolver<myType>::solve;
    

    template <class Span, class TwoDSpan>
    CUDA_HOSTDEV void solve(gScalar&     x,
                            Span&        y,
                            const gLabel li,
                            stepState&   step,
                            TwoDSpan&    a,
                            gpuBuffer&   buffer) const {
        gScalar x0 = x;
        this->adaptiveSolve(odes_, x, y, li, step.dxTry, a, buffer);
        step.dxDid = x - x0;
    }
    
    //- Solve a single step dx and return the error
    template <class Span1, class Span2, class Span3, class TwoDSpan>
    CUDA_HOSTDEV gScalar solve(const gScalar x0,
                               const Span1&  y0,
                               const gLabel  li,
                               const Span2&  dydx0,
                               const gScalar dx,
                               Span3&        y,
                               TwoDSpan&     a,
                               gpuBuffer&    buffer) const {

        using namespace Rosenbrock23Coeffs;

       
        auto dydx         = buffer.dydx();
        auto dfdx         = buffer.dfdx();
        auto pivotIndices = buffer.pivotIndices();
        auto k1           = buffer.k1();
        auto k2           = buffer.k2();
        auto k3           = buffer.k3();
        auto err          = buffer.err();
        auto lubuffer     = buffer.lubuffer();

        odes_.jacobian(x0, y0, li, dfdx, a, buffer);

        const gScalar temp = gScalar(1.0) / (gamma * dx);
        // Negate the jacobian and add 1/(gamma*dx) to diagonal
        for (gLabel i = 0; i < nEqns(); i++) {
            for (gLabel j = 0; j < nEqns(); j++) 
            {
                a(i, j) = -a(i, j);
            }

            a(i, i) += temp; // gScalar(1.0)/(gamma*dx);
        }

        FoamGpu::LUDecompose(a, pivotIndices, lubuffer);

        // Calculate k1:
        for (gLabel i = 0; i < nEqns(); ++i) {
            k1[i] = dydx0[i] + dx * d1 * dfdx[i];
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k1);

        // Calculate k2:
        for (gLabel i = 0; i < nEqns(); ++i) { y[i] = y0[i] + a21 * k1[i]; }

        odes_.derivatives(x0 + c2 * dx, y, li, dydx, buffer);

        for (gLabel i = 0; i < nEqns(); ++i) {
            k2[i] = dydx[i] + dx * d2 * dfdx[i] + c21 * k1[i] / dx;
        }

        FoamGpu::LUBacksubstitute(a, pivotIndices, k2);

        //Calculate k3:
        for (gLabel i = 0; i < nEqns(); ++i) {
            k3[i] =
                dydx[i] + dx * d3 * dfdx[i] + (c31 * k1[i] + c32 * k2[i]) / dx;
        }
        
        FoamGpu::LUBacksubstitute(a, pivotIndices, k3);

       

        // Calculate error and update state:
        for (gLabel i = 0; i < nEqns(); ++i) {
            y[i] = y0[i] + b1*k1[i] + b2*k2[i] + b3*k3[i];
            err[i] = e1*k1[i] + e2*k2[i] + e3*k3[i];
            
        }

        return gpuODESolver<myType>::normaliseError(y0, y, err);
        
    }
};

} // namespace FoamGpu
