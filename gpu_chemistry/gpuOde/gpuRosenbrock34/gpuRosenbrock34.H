#pragma once

#include "gpuODESolver.H"
#include "gpuConstants.H"
#include "ludecompose.H"
#include "gpuBuffer.H"
#include "check_ptr.H"

namespace FoamGpu
{

namespace RosenbrockCoeffs
{

    // Constants by Shampine
    // More accurate than the L-Stable coefficients for small step-size
    // but less stable for large step-size
     constexpr gScalar a21 = 2.0;
     constexpr gScalar a31 = 48.0/25.0;
     constexpr gScalar a32 = 6.0/25.0;

     constexpr gScalar c21 = -8.0;
     constexpr gScalar c31 = 372.0/25.0;
     constexpr gScalar c32 = 12.0/5.0;

     constexpr gScalar c41 = -112.0/125.0;
     constexpr gScalar c42 = -54.0/125.0;
     constexpr gScalar c43 = -2.0/5.0;

     constexpr gScalar b1 = 19.0/9.0;
     constexpr gScalar b2 = 1.0/2.0;
     constexpr gScalar b3 = 25.0/108.0;
     constexpr gScalar b4 = 125.0/108.0;

     constexpr gScalar e1 = 34.0/108.0;
     constexpr gScalar e2 = 7.0/36.0;
     constexpr gScalar e3 = 0.0;
     constexpr gScalar e4 = 125.0/108.0;

     constexpr gScalar gamma = 1.0/2.0;
     constexpr gScalar c2 = 1.0;
     constexpr gScalar c3  = 3.0/5.0;

     constexpr gScalar d1 = 1.0/2.0;
     constexpr gScalar d2 = -3.0/2.0;
     constexpr gScalar d3 = 605.0/250.0;
     constexpr gScalar d4 = 29.0/250.0;
}

struct gpuRosenbrock34Inputs
{
    gScalar absTol;
    gScalar relTol;
    gLabel maxSteps;
    gScalar safeScale;
    gScalar alphaInc;
    gScalar alphaDec;
    gScalar minScale;
    gScalar maxScale;

};

template<class System>
class gpuRosenbrock34
:
    public gpuODESolver
{
    // Private Data

public:

        System odes_;

        gLabel n_;

        //- Step-size adjustment controls
        gScalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;



    // Constructors



        gpuRosenbrock34(const System& ode, gpuRosenbrock34Inputs i)
            : gpuODESolver(i.absTol, i.relTol, i.maxSteps)
            , odes_(ode)
            , n_(ode.nEqns())
            , safeScale_(i.safeScale)
            , alphaInc_(i.alphaInc)
            , alphaDec_(i.alphaDec)
            , minScale_(i.minScale)
            , maxScale_(i.maxScale) {}

        // Member Functions

        //- Solve the ODE system from the current state xStart, y
        //  and the optional index into the list of systems to solve li
        //  to xEnd and return an estimate for the next step in dxTry
        template<class Span, class TwoDSpan>
        CUDA_HOSTDEV void solve
        (
            const gScalar xStart,
            const gScalar xEnd,
            Span& y,
            const gLabel li,
            gScalar& dxTry,
            TwoDSpan& a,
            gpuBuffer buffer
        ) const
        {


            stepState step(dxTry);
            gScalar x = xStart;

            //printf("dxTry value = %lf\n", dxTry);

            for (gLabel nStep=0; nStep<maxSteps_; nStep++)
            {


                // Store previous iteration dxTry
                gScalar dxTry0 = step.dxTry;

                step.reject = false;

                // Check if this is a truncated step and set dxTry to integrate to xEnd
                if ((x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0)
                {
                    step.last = true;
                    step.dxTry = xEnd - x;
                }

                // Integrate as far as possible up to step.dxTry
                solve(x, y, li, step, a, buffer);

                // Check if reached xEnd
                if ((x - xEnd)*(xEnd - xStart) >= 0)
                {
                    if (nStep > 0 && step.last)
                    {
                        step.dxTry = dxTry0;
                    }

                    dxTry = step.dxTry;

                    return;
                }

                step.first = false;

                // If the step.dxTry was reject set step.prevReject
                if (step.reject)
                {
                    step.prevReject = true;
                }
            }


        }


        template<class Span, class TwoDSpan>
        CUDA_HOSTDEV void solve
        (
            gScalar& x,
            Span& y,
            const gLabel li,
            stepState& step,
            TwoDSpan& a,
            gpuBuffer& buffer
        ) const
        {
            gScalar x0 = x;
            solve(x, y, li, step.dxTry, a, buffer);
            step.dxDid = x - x0;
        }



        //- Solve the ODE system and the update the state
        template<class Span, class TwoDSpan>
        CUDA_HOSTDEV void solve
        (
            gScalar& x,
            Span& y,
            const gLabel li,
            gScalar& dxTry,
            TwoDSpan& a,
            gpuBuffer& buffer
        ) const
        {


            using namespace RosenbrockCoeffs;


            auto dydx0 = buffer.dydx0();
            auto yTemp = buffer.yTemp();


            gScalar dx = dxTry;
            gScalar err = 0.0;


            odes_.derivatives(x, y, li, dydx0, buffer);


            // Loop over solver and adjust step-size as necessary
            // to achieve desired error
            do
            {
                // Solve step and provide error estimate
                err = solve(x, y, li, dydx0, dx, yTemp, a, buffer);



                // If error is large reduce dx
                if (err > 1)
                {
                    gScalar scale = std::max(safeScale_*std::pow(err, -alphaDec_), minScale_);
                    dx *= scale;

                    if (dx < gpuVSmall)
                    {
                        printf("Small time step in ode. \n");
                        assert(0);
                    }

                }
            } while (err > 1.0);

            // Update the state
            x += dx;

            //y = yTemp;
            for (gLabel i = 0; i < n_; ++i){y[i] = yTemp[i];}

            // If the error is small increase the step-size
            if (err > std::pow(maxScale_/safeScale_, -1.0/alphaInc_))
            {
                dxTry =
                    std::min(std::max(safeScale_*std::pow(err, -alphaInc_), minScale_), maxScale_)*dx;
            }
            else
            {
                dxTry = safeScale_*maxScale_*dx;
            }

        }

        //- Solve a single step dx and return the error
        template<class Span1, class Span2, class Span3, class TwoDSpan>
        CUDA_HOSTDEV gScalar solve
        (
            const gScalar x0,
            const Span1& y0,
            const gLabel li,
            const Span2& dydx0,
            const gScalar dx,
            Span3& y,
            TwoDSpan& a,
            gpuBuffer& buffer
        ) const
        {

            using namespace RosenbrockCoeffs;


            auto dydx = buffer.dydx();
            auto dfdx = buffer.dfdx();
            auto pivotIndices = buffer.pivotIndices();
            auto k1 = buffer.k1();
            auto k2 = buffer.k2();
            auto k3 = buffer.k3();
            auto k4 = buffer.k4();
            auto err = buffer.err();
            auto lubuffer = buffer.lubuffer();


            odes_.jacobian(x0, y0, li, dfdx, a, buffer);

            const gScalar temp = gScalar(1.0)/(gamma*dx);
            //Negate the jacobian and add 1/(gamma*dx) to diagonal
            for (gLabel i=0; i<n_; i++)
            {
                for (gLabel j=0; j<n_; j++)
                {
                    a(i, j) = -a(i, j);
                }

                a(i, i) += temp; //gScalar(1.0)/(gamma*dx);
            }


            FoamGpu::LUDecompose(a, pivotIndices, lubuffer);



            // Calculate k1:
            for(gLabel i = 0; i < n_; ++i)
            {
                k1[i] = dydx0[i] + dx*d1*dfdx[i];
            }

            FoamGpu::LUBacksubstitute(a, pivotIndices, k1);

            // Calculate k2:
            for(gLabel i = 0; i < n_; ++i)
            {
                y[i] = y0[i] + a21*k1[i];
            }

            odes_.derivatives(x0 + c2*dx, y, li, dydx, buffer);

            for(gLabel i = 0; i < n_; ++i)
            {
                k2[i] = dydx[i] + dx*d2*dfdx[i] + c21*k1[i]/dx;
            }

            FoamGpu::LUBacksubstitute(a, pivotIndices, k2);

            // Calculate k3:
            for(gLabel i = 0; i < n_; ++i)
            {
                y[i] = y0[i] + a31*k1[i] + a32*k2[i];
            }

            odes_.derivatives(x0 + c3*dx, y, li, dydx, buffer);

            for(gLabel i = 0; i < n_; ++i)
            {
                k3[i] = dydx[i] + dx*d3*dfdx[i] + (c31*k1[i] + c32*k2[i])/dx;
            }

            FoamGpu::LUBacksubstitute(a, pivotIndices, k3);

            // Calculate k4:
            for(gLabel i = 0; i < n_; ++i)
            {
                k4[i] = dydx[i] + dx*d4*dfdx[i]
                + (c41*k1[i] + c42*k2[i] + c43*k3[i])/dx;
            }

            FoamGpu::LUBacksubstitute(a, pivotIndices, k4);

            // Calculate error and update state:
            for(gLabel i = 0; i < n_; ++i)
            {
                y[i] = y0[i] + b1*k1[i] + b2*k2[i] + b3*k3[i] + b4*k4[i];
                err[i] = e1*k1[i] + e2*k2[i] + e4*k4[i];
            }

            return normaliseError(y0, y, err, n_);


        }
};



} // End namespace Foam

