/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::gpuSeulex

Description
    An extrapolation-algorithm, based on the linearly implicit Euler method
    with step size control and order selection.

    Reference:
    \verbatim
        Hairer, E., Nørsett, S. P., & Wanner, G. (1996).
        Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems, second edition",
        Springer-Verlag, Berlin.
    \endverbatim

SourceFiles
    gpuSeulex.C

\*---------------------------------------------------------------------------*/

#ifndef gpuSeulex_H
#define gpuSeulex_H

#include "gpuODESolver.H"
#include "ludecompose.H"
#include "scalarMatrices.H"
#include "labelField.H"
#include "gpuConstants.H"
#include "mdspan.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class gpuSeulex Declaration
\*---------------------------------------------------------------------------*/

template<class System>
class gpuSeulex
:
    public gpuODESolver
{


    // Private Data



        // Static constants

            static constexpr label kMaxx_ = 12;
            static constexpr label iMaxx_ = kMaxx_ + 1;


            static constexpr scalar stepFactor1_ = 0.6;
            static constexpr scalar stepFactor2_ = 0.93;
            static constexpr scalar stepFactor3_ = 0.1;
            static constexpr scalar stepFactor4_ = 4;
            static constexpr scalar stepFactor5_ = 0.5;
            static constexpr scalar kFactor1_ = 0.7;
            static constexpr scalar kFactor2_ = 0.9;

        static constexpr std::array<label, iMaxx_> makeNseq()
        {
            std::array<label, iMaxx_> nSeq{};

            nSeq[0] = 2;
            nSeq[1] = 3;

            for (int i=2; i<iMaxx_; i++)
            {
                nSeq[i] = 2*nSeq[i-2];
            }

            return nSeq;
        }


        static constexpr std::array<scalar, iMaxx_*iMaxx_>
        makeCoffTable()
        {
            std::array<scalar, iMaxx_*iMaxx_> table{};

            auto coeff = mdspan<scalar, 2>(table.data(), extents<2>{iMaxx_, iMaxx_});

            constexpr auto nSeq = makeNseq();

            // Set the extrapolation coefficients array
            for (int k=0; k<iMaxx_; k++)
            {
                for (int l=0; l<k; l++)
                {
                    scalar ratio = scalar(nSeq[k])/nSeq[l];
                    coeff(k, l) = 1/(ratio - 1);
                }
            }


            return table;
        }

        static constexpr std::array<scalar, iMaxx_> makeCpu()
        {
            std::array<scalar, iMaxx_> cpu{};

            constexpr auto nseq = makeNseq();

            // The CPU time factors for the major parts of the algorithm
            constexpr scalar cpuFunc = 1, cpuJac = 5, cpuLU = 1, cpuSolve = 1;

            cpu[0] = cpuJac + cpuLU + nseq[0]*(cpuFunc + cpuSolve);

            for (int k=0; k<kMaxx_; k++)
            {
                cpu[k+1] = cpu[k] + (nseq[k+1]-1)*(cpuFunc + cpuSolve) + cpuLU;
            }
            return cpu;
        }

        static constexpr std::array<scalar, iMaxx_*iMaxx_> COEFF =
            makeCoffTable();

        static constexpr std::array<label, iMaxx_> NSEQ =
            makeNseq();

        static constexpr std::array<scalar, iMaxx_> CPU =
            makeCpu();

        //The ode system
        System odes_;

        //Size of the system
        label n_;

        // Evaluated constants



            scalar jacRedo_;

        // Temporary storage
        // held to avoid dynamic memory allocation between calls
        // and to transfer internal values between functions

            mutable scalar theta_;
            mutable label kTarg_;
            mutable scalarRectangularMatrix table_;

            mutable scalarSquareMatrix dfdy_;

            // Fields space for "solve" function
            mutable scalarField dxOpt_;




    // Private Member Functions


        //- Computes the j-th line of the extrapolation table
        template<class Span1, class Span2, class Span3>
        bool seul
        (
            const scalar x0,
            const Span1& y0,
            const label li,
            const scalar dxTot,
            const label k,
            Span2& y,
            const Span3& scale
        ) const
        {
            label nSteps = NSEQ[k];
            scalar dx = dxTot/nSteps;

            /*
            scalarSquareMatrix a(n_);
            labelList pivotIndices(n_);

            scalarField dy(n_);
            scalarField dydx(n_);
            */
            using namespace gpuConstants;
            static constexpr int N = SPECIE_MAX + 2;

            std::array<scalar, N*N> a_arr;
            auto a = make_mdspan(a_arr, extents<2>{n_, n_});

            std::array<label, N> pivotIndices_arr;
            auto pivotIndices = make_mdspan(pivotIndices_arr, extents<1>{n_});

            std::array<scalar, N> dy_arr;
            auto dy = make_mdspan(dy_arr, extents<1>{n_});

            std::array<scalar, N> dydx_arr;
            auto dydx = make_mdspan(dydx_arr, extents<1>{n_});

            std::array<scalar, N> lubuffer_arr;
            auto lubuffer = make_mdspan(lubuffer_arr, extents<1>{n_});

            std::array<scalar, N> yTemp_arr;
            auto yTemp = make_mdspan(yTemp_arr, extents<1>{n_});


            for (label i=0; i<n_; i++)
            {
                for (label j=0; j<n_; j++)
                {
                    a(i, j) = -dfdy_(i, j);
                }

                a(i, i) += 1/dx;
            }


            ::gpu::LUDecompose(a, pivotIndices, lubuffer);

            scalar xnew = x0 + dx;
            odes_.derivatives(xnew, y0, li, dy);
            ::gpu::LUBacksubstitute(a, pivotIndices, dy);


            for (label i=0; i<n_; i++)
            {
                yTemp[i] = y0[i];
            }

            //auto yTemp = y0;

            for (label nn=1; nn<nSteps; nn++)
            {
                //yTemp += dy;
                for (label i=0; i<n_; i++)
                {
                    yTemp[i] += dy[i];
                }


                xnew += dx;

                if (nn == 1 && k<=1)
                {
                    scalar dy1 = 0;
                    for (label i=0; i<n_; i++)
                    {
                        dy1 += sqr(dy[i]/scale[i]);
                    }
                    dy1 = sqrt(dy1);

                    odes_.derivatives(x0 + dx, yTemp, li, dydx);
                    for (label i=0; i<n_; i++)
                    {
                        dy[i] = dydx[i] - dy[i]/dx;
                    }

                    ::gpu::LUBacksubstitute(a, pivotIndices, dy);

                    // This form from the original paper is unreliable
                    // step size underflow for some cases
                    // const scalar denom = max(1, dy1);

                    // This form is reliable but limits how large the step size can be
                    const scalar denom = std::min(1.0, dy1 + gpuSmall);

                    scalar dy2 = 0;
                    for (label i=0; i<n_; i++)
                    {
                        // Test of dy_[i] to avoid overflow
                        if (std::abs(dy[i]) > scale[i]*denom)
                        {
                            theta_ = 1;
                            return false;
                        }

                        dy2 += sqr(dy[i]/scale[i]);
                    }
                    dy2 = sqrt(dy2);
                    theta_ = dy2/denom;

                    if (theta_ > 1)
                    {
                        return false;
                    }
                }

                odes_.derivatives(xnew, yTemp, li, dy);
                ::gpu::LUBacksubstitute(a, pivotIndices, dy);
            }

            for (label i=0; i<n_; i++)
            {
                y[i] = yTemp[i] + dy[i];
            }

            return true;

        }





        //- Polynomial extrapolation
        void extrapolate
        (
            const label k,
            scalarRectangularMatrix& table,
            scalarField& y
        ) const
        {

            for (int j=k-1; j>0; j--)
            {
                for (label i=0; i<n_; i++)
                {
                    table(j-1, i) =
                        table(j, i) + COEFF[k*iMaxx_ +j]*(table(j, i) - table(j-1, i));
                }
            }

            for (int i=0; i<n_; i++)
            {
                y[i] = table(0, i) + COEFF[k*iMaxx_]*(table(0, i) - y[i]);
            }
        }


public:



    // Constructors




        gpuSeulex(const System& ode, scalar absTol, scalar relTol, label maxSteps)
        :
            gpuODESolver(absTol, relTol, maxSteps),
            odes_(ode),
            n_(ode.nEqns()),
            jacRedo_(std::min(1e-4, relTol_)),
            theta_(2*jacRedo_),
            table_(kMaxx_, kMaxx_),
            dfdy_(n_),
            dxOpt_(iMaxx_)
        {}

        //- Construct from gpuODESystem
        gpuSeulex(const System& ode, const dictionary& dict)
        :
            gpuSeulex
            (
                ode,
                dict.lookupOrDefault<scalar>("absTol", gpuConstants::gpuSmall),
                dict.lookupOrDefault<scalar>("relTol", 1e-4),
                dict.lookupOrDefault<scalar>("maxSteps", 10000)
            )
        {}



    // Member Functions


        //- Solve the ODE system from the current state xStart, y
        //  and the optional index into the list of systems to solve li
        //  to xEnd and return an estimate for the next step in dxTry
        template<class Span>
        void solve
        (
            const scalar xStart,
            const scalar xEnd,
            Span& y,
            const label li,
            scalar& dxTry
        ) const
        {

            stepState step(dxTry);
            scalar x = xStart;

            for (label nStep=0; nStep<maxSteps_; nStep++)
            {
                // Store previous iteration dxTry
                scalar dxTry0 = step.dxTry;

                step.reject = false;

                // Check if this is a truncated step and set dxTry to integrate to xEnd
                if ((x + step.dxTry - xEnd)*(x + step.dxTry - xStart) > 0)
                {
                    step.last = true;
                    step.dxTry = xEnd - x;
                }

                // Integrate as far as possible up to step.dxTry
                solve(x, y, li, step);

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




        //- Solve the ODE system from the current state xStart, y
        //  and the optional index into the list of systems to solve li
        //  as far as possible up to dxTry adjusting the step as necessary
        //  to provide a solution within the specified tolerance.
        //  Update the state and return an estimate for the next step in dxTry
        template<class Span>
        void solve
        (
            scalar& x,
            Span& y,
            const label li,
            scalar& dxTry
        ) const
        {
            stepState step(dxTry);
            solve(x, y, li, step);
            dxTry = step.dxTry;
        }






        //- Solve the ODE system and the update the state
        template<class Span>
        void solve
        (
            scalar& x,
            Span& y,
            const label li,
            stepState& step
        ) const
        {
            scalarField temp(iMaxx_);
            scalarField dfdx(n_);

            temp[0] = gpuConstants::gpuGreat;

            scalar dx = step.dxTry;
            auto y0 = y;
            dxOpt_[0] = std::abs(0.1*dx);

            if (step.first || step.prevReject)
            {
                theta_ = 2*jacRedo_;
            }

            if (step.first)
            {
                // NOTE: the first element of relTol_ and absTol_ are used here.
                scalar logTol = -std::log10(relTol_ + absTol_)*0.6 + 0.5;
                kTarg_ = std::max(1, std::min(kMaxx_ - 1, int(logTol)));
            }

            scalarField scale(n_);

            forAll(scale, i)
            {
                scale[i] = absTol_ + relTol_*std::abs(y[i]);
            }

            bool jacUpdated = false;

            //Careful here, Jacobian is not updated every time this function is called in
            // official OF.
            if (theta_ > jacRedo_)
            {
                odes_.jacobian(x, y, li, dfdx, dfdy_);
                jacUpdated = true;
            }

            int k;
            scalar dxNew = std::abs(dx);
            bool firstk = true;

            scalarField ySequence(n_);

            while (firstk || step.reject)
            {
                dx = step.forward ? dxNew : -dxNew;
                firstk = false;
                step.reject = false;


                scalar errOld = 0;

                for (k=0; k<=kTarg_+1; k++)
                {
                    bool success = seul(x, y0, li, dx, k, ySequence, scale);

                    if (!success)
                    {
                        step.reject = true;
                        dxNew = std::abs(dx)*stepFactor5_;
                        break;
                    }

                    if (k == 0)
                    {
                        y = ySequence;
                    }
                    else
                    {
                        forAll(ySequence, i)
                        {
                            table_[k-1][i] = ySequence[i];
                        }
                    }

                    if (k != 0)
                    {
                        extrapolate(k, table_, y);
                        scalar err = 0;
                        forAll(scale, i)
                        {
                            scale[i] = absTol_ + relTol_*std::abs(y0[i]);
                            err += sqr((y[i] - table_(0, i))/scale[i]);
                        }
                        err = sqrt(err/n_);
                        if (err > 1/small || (k > 1 && err >= errOld))
                        {
                            step.reject = true;
                            dxNew = std::abs(dx)*stepFactor5_;
                            break;
                        }
                        errOld = min(4*err, 1);
                        scalar expo = 1.0/(k + 1);
                        scalar facmin = pow(stepFactor3_, expo);
                        scalar fac;
                        if (err == 0)
                        {
                            fac = 1/facmin;
                        }
                        else
                        {
                            fac = stepFactor2_/pow(err/stepFactor1_, expo);
                            fac = max(facmin/stepFactor4_, min(1/facmin, fac));
                        }
                        dxOpt_[k] = std::abs(dx*fac);
                        temp[k] = CPU[k]/dxOpt_[k];

                        if ((step.first || step.last) && err <= 1)
                        {
                            break;
                        }

                        if
                        (
                            k == kTarg_ - 1
                        && !step.prevReject
                        && !step.first && !step.last
                        )
                        {
                            if (err <= 1)
                            {
                                break;
                            }
                            else if (err > NSEQ[kTarg_]*NSEQ[kTarg_ + 1]*4)
                            {
                                step.reject = true;
                                kTarg_ = k;
                                if (kTarg_>1 && temp[k-1] < kFactor1_*temp[k])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                                break;
                            }
                        }

                        if (k == kTarg_)
                        {
                            if (err <= 1)
                            {
                                break;
                            }
                            else if (err > NSEQ[k + 1]*2)
                            {
                                step.reject = true;
                                if (kTarg_>1 && temp[k-1] < kFactor1_*temp[k])
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                                break;
                            }
                        }

                        if (k == kTarg_+1)
                        {
                            if (err > 1)
                            {
                                step.reject = true;
                                if
                                (
                                    kTarg_ > 1
                                && temp[kTarg_-1] < kFactor1_*temp[kTarg_]
                                )
                                {
                                    kTarg_--;
                                }
                                dxNew = dxOpt_[kTarg_];
                            }
                            break;
                        }
                    }
                }
                if (step.reject)
                {
                    step.prevReject = true;
                    if (!jacUpdated)
                    {
                        theta_ = 2*jacRedo_;

                        if (theta_ > jacRedo_ && !jacUpdated)
                        {
                            odes_.jacobian(x, y, li, dfdx, dfdy_);
                            jacUpdated = true;
                        }
                    }
                }
            }

            jacUpdated = false;

            step.dxDid = dx;
            x += dx;

            label kopt;
            if (k == 1)
            {
                kopt = 2;
            }
            else if (k <= kTarg_)
            {
                kopt=k;
                if (temp[k-1] < kFactor1_*temp[k])
                {
                    kopt = k - 1;
                }
                else if (temp[k] < kFactor2_*temp[k - 1])
                {
                    kopt = min(k + 1, kMaxx_ - 1);
                }
            }
            else
            {
                kopt = k - 1;
                if (k > 2 && temp[k-2] < kFactor1_*temp[k - 1])
                {
                    kopt = k - 2;
                }
                if (temp[k] < kFactor2_*temp[kopt])
                {
                    kopt = min(k, kMaxx_ - 1);
                }
            }

            if (step.prevReject)
            {
                kTarg_ = std::min(kopt, k);
                dxNew = std::min(std::abs(dx), dxOpt_[kTarg_]);
                step.prevReject = false;
            }
            else
            {
                if (kopt <= k)
                {
                    dxNew = dxOpt_[kopt];
                }
                else
                {
                    if (k < kTarg_ && temp[k] < kFactor2_*temp[k - 1])
                    {
                        dxNew = dxOpt_[k]*CPU[kopt + 1]/CPU[k];
                    }
                    else
                    {
                        dxNew = dxOpt_[k]*CPU[kopt]/CPU[k];
                    }
                }
                kTarg_ = kopt;
            }

            step.dxTry = step.forward ? dxNew : -dxNew;
        }
};



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
