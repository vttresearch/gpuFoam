#pragma once

#include "ODESolver.H"
#include "mock_of_adaptive_solver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class MockRosenbrock34 Declaration
\*---------------------------------------------------------------------------*/

class MockRosenbrock34
:
    public ODESolver,
    public MockAdaptiveSolver
{
    // Private Data

        mutable scalarField k1_;
        mutable scalarField k2_;
        mutable scalarField k3_;
        mutable scalarField k4_;
        mutable scalarField err_;
        mutable scalarField dydx_;
        mutable scalarField dfdx_;
        mutable scalarSquareMatrix dfdy_;
        mutable scalarSquareMatrix a_;
        mutable labelList pivotIndices_;

    static constexpr scalar a21 = 2;
    static constexpr scalar a31 = 48.0/25.0;
    static constexpr scalar a32 = 6.0/25.0;

    static constexpr scalar c21 = -8;
    static constexpr scalar c31 = 372.0/25.0;
    static constexpr scalar c32 = 12.0/5.0;

    static constexpr scalar c41 = -112.0/125.0;
    static constexpr scalar c42 = -54.0/125.0;
    static constexpr scalar c43 = -2.0/5.0;

    static constexpr scalar b1 = 19.0/9.0;
    static constexpr scalar b2 = 1.0/2.0;
    static constexpr scalar b3 = 25.0/108.0;
    static constexpr scalar b4 = 125.0/108.0;

    static constexpr scalar e1 = 34.0/108.0;
    static constexpr scalar e2 = 7.0/36.0;
    static constexpr scalar e3 = 0;
    static constexpr scalar e4 = 125.0/108.0;

    static constexpr scalar gamma = 1.0/2.0;
    static constexpr scalar c2 = 1;
    static constexpr scalar c3  = 3.0/5.0;

    static constexpr scalar d1 = 1.0/2.0;
    static constexpr scalar d2 = -3.0/2.0;
    static constexpr scalar d3 = 605.0/250.0;
    static constexpr scalar d4 = 29.0/250.0;


public:

    //- Runtime type information
    //TypeName("MockRosenbrock34");

        //- Return the normalised scalar error
        scalar normaliseErrorNew
        (
            const scalarField& y0,
            const scalarField& y,
            const scalarField& err
        ) const
        {

            // Calculate the maximum error
            scalar maxErr = 0.0;
            forAll(err, i)
            {
                scalar tol = absTol_[i] + relTol_[i]*max(mag(y0[i]), mag(y[i]));
                maxErr = max(maxErr, mag(err[i])/tol);
                //scalar maxMag = relTol_[i] * max(mag(y0[i]), mag(y[i]));
                //printf("tol: %.17g \n", tol);
                //printf("maxMag: %.17g \n", maxMag);
                //printf("y0: %.17g \n", std::abs(y0[i]));
                //printf("y: %.17g \n", std::abs(y[i]));
                //printf("maxErr: %.17g \n", maxErr);
                //printf("tol, maxErr, y0, y, absTol, relTol: %.17g %.17g %.17g %.17g %.17g %.17g \n", tol, maxErr, y0[i], y[i], absTol_, relTol_);
                //printf("y0, y1, err, maxErr: %.17g %.17g %.17g %.17g \n", y0[i], y[i], err[i], maxErr );
                //printf("err: %.17g \n", err[i]);
            }

            return maxErr;
        }

    // Constructors

        //- Construct from ODESystem
        MockRosenbrock34(const ODESystem& ode, const dictionary& dict)
        :
        ODESolver(ode, dict),
        MockAdaptiveSolver(ode, dict),
        k1_(n_),
        k2_(n_),
        k3_(n_),
        k4_(n_),
        err_(n_),
        dydx_(n_),
        dfdx_(n_),
        dfdy_(n_, n_),
        a_(n_, n_),
        pivotIndices_(n_)
        {}


    //- Destructor
    virtual ~MockRosenbrock34()
    {}


    // Member Functions

        //- Inherit solve from ODESolver
        using ODESolver::solve;

        //- Resize the ODE solver
        virtual bool resize()
        {
            return false;
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
        ) const
        {
            odes_.jacobian(x0, y0, li, dfdx_, dfdy_);

            for (label i=0; i<n_; i++)
            {
                for (label j=0; j<n_; j++)
                {
                    a_(i, j) = -dfdy_(i, j);
                }

                a_(i, i) += 1.0/(gamma*dx);
            }

            LUDecompose(a_, pivotIndices_);

            // Calculate k1:
            forAll(k1_, i)
            {
                k1_[i] = dydx0[i] + dx*d1*dfdx_[i];
            }

            LUBacksubstitute(a_, pivotIndices_, k1_);

            // Calculate k2:
            forAll(y, i)
            {
                y[i] = y0[i] + a21*k1_[i];
            }

            odes_.derivatives(x0 + c2*dx, y, li, dydx_);

            forAll(k2_, i)
            {
                k2_[i] = dydx_[i] + dx*d2*dfdx_[i] + c21*k1_[i]/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, k2_);

            // Calculate k3:
            forAll(y, i)
            {
                y[i] = y0[i] + a31*k1_[i] + a32*k2_[i];
            }

            odes_.derivatives(x0 + c3*dx, y, li, dydx_);

            forAll(k3_, i)
            {
                k3_[i] = dydx_[i] + dx*d3*dfdx_[i] + (c31*k1_[i] + c32*k2_[i])/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, k3_);

            // Calculate k4:
            forAll(k4_, i)
            {
                k4_[i] = dydx_[i] + dx*d4*dfdx_[i]
                    + (c41*k1_[i] + c42*k2_[i] + c43*k3_[i])/dx;
            }

            LUBacksubstitute(a_, pivotIndices_, k4_);

            // Calculate error and update state:
            //scalar err_sum = 0.0;
            //scalar y_sum = 0.0;
            forAll(y, i)
            {
                y[i] = y0[i] + b1*k1_[i] + b2*k2_[i] + b3*k3_[i] + b4*k4_[i];
                err_[i] = e1*k1_[i] + e2*k2_[i] + e4*k4_[i];
                //y_sum += y[i];
                //err_sum += err_[i];
            }

            return normaliseError(y0, y, err_);
            //scalar normal = normaliseErrorNew(y0, y, err_);
            //printf("y_sum: %.17g \n", y_sum);
            //printf("err_sum: %.17g \n", err_sum);
            //printf("err_norm: %.17g \n", normal);


//            return normal;
            //return normaliseError(y0, y, err_);

        }

        //- Solve the ODE system and the update the state
        virtual void solve
        (
            scalar& x,
            scalarField& y,
            const label li,
            scalar& dxTry
        ) const
        {
            MockAdaptiveSolver::solve(odes_, x, y, li, dxTry);
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// ************************************************************************* //