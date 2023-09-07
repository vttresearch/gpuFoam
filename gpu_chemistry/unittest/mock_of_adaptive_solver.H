#pragma once

#include "ODESolver.H"
#include "ODESystem.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class MockAdaptiveSolver Declaration
\*---------------------------------------------------------------------------*/

class MockAdaptiveSolver
{
    // Private Data

        //- Step-size adjustment controls
        scalar safeScale_, alphaInc_, alphaDec_, minScale_, maxScale_;

        //- Cache for dydx at the initial time
        mutable scalarField dydx0_;

        //- Temporary for the test-step solution
        mutable scalarField yTemp_;


public:

    // Constructors

        //- Construct from ODESystem
        MockAdaptiveSolver(const ODESystem& ode, const dictionary& dict)
        :
        safeScale_(dict.lookupOrDefault<scalar>("safeScale", 0.9)),
        alphaInc_(dict.lookupOrDefault<scalar>("alphaIncrease", 0.2)),
        alphaDec_(dict.lookupOrDefault<scalar>("alphaDecrease", 0.25)),
        minScale_(dict.lookupOrDefault<scalar>("minScale", 0.2)),
        maxScale_(dict.lookupOrDefault<scalar>("maxScale", 10)),
        dydx0_(ode.nEqns()),
        yTemp_(ode.nEqns())
        {}

    //- Destructor
    virtual ~MockAdaptiveSolver()
    {}


    // Member Functions

        //- Resize the ODE solver
        bool resize(const label n)
        {
            return true;
        }

        //- Solve a single step dx and return the error
        virtual scalar solve
        (
            const scalar x0,
            const scalarField& y0,
            const label li,
            const scalarField& dydx0,
            const scalar dx,
            scalarField& y
        ) const = 0;

        //- Solve the ODE system and the update the state
        void solve
        (
            const ODESystem& ode,
            scalar& x,
            scalarField& y,
            const label li,
            scalar& dxTry
        ) const
        {
            scalar dx = dxTry;
            scalar err = 0.0;

            ode.derivatives(x, y, li, dydx0_);

            // Loop over solver and adjust step-size as necessary
            // to achieve desired error
            do
            {
                // Solve step and provide error estimate
                err = solve(x, y, li, dydx0_, dx, yTemp_);

                printf("err : %.17g \n", err);

                // If error is large reduce dx
                if (err > 1)
                {
                    scalar scale = max(safeScale_*pow(err, -alphaDec_), minScale_);
                    dx *= scale;

                    printf("dx : %.17g \n", dx);

                    if (dx < vSmall)
                    {
                        FatalErrorInFunction
                            << "stepsize underflow"
                            << exit(FatalError);
                    }
                }
            } while (err > 1);

            // Update the state
            x += dx;
            y = yTemp_;

            // If the error is small increase the step-size
            if (err > pow(maxScale_/safeScale_, -1.0/alphaInc_))
            {
                dxTry =
                    min(max(safeScale_*pow(err, -alphaInc_), minScale_), maxScale_)*dx;
            }
            else
            {
                dxTry = safeScale_*maxScale_*dx;
            }

        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// ************************************************************************* //