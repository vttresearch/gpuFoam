/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2021 OpenFOAM Foundation
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
    Foam::gpuODESolver

Description
    Abstract base-class for ODE system solvers

SourceFiles
    gpuODESolver.C

\*---------------------------------------------------------------------------*/

#ifndef gpuODESolver_H
#define gpuODESolver_H

#include "cuda_host_dev.H"
#include "gpuODESystem.H"
#include "double_intrinsics.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{

/*---------------------------------------------------------------------------*\
                          Class gpuODESolver Declaration
\*---------------------------------------------------------------------------*/


class gpuODESolver
{

protected:

    // Protected data


        //- Absolute convergence tolerance per step
        gScalar absTol_;

        //- Relative convergence tolerance per step
        gScalar relTol_;

        //- The maximum number of sub-steps allowed for the integration step
        gLabel maxSteps_;


    // Protected Member Functions


        CUDA_HOSTDEV
        double newError
        (
            double y0, double y, double err
        ) const
        {
            //constexpr double absTol = std::numeric_limits<double>::epsilon();
            //constexpr double relTol = 1E-4;

            double tol = absTol_ + relTol_*std::max(std::abs(y0), std::abs(y));
            double ret = std::abs(err)/tol;
            return ret;
        }

        //- Return the normalised gScalar error
        template<class Span1, class Span2, class Span3>
        CUDA_HOSTDEV gScalar normaliseError
        (
            const Span1& y0,
            const Span2& y,
            const Span3& err,
            gLabel n
        ) const
        {
            // Calculate the maximum error
            gScalar maxErr = 0.0;


            for (gLabel i = 0; i < n; ++i)
            {

                gScalar t = newError(y0[i], y[i], err[i]);
                maxErr = std::max(maxErr, t);
            }

            return maxErr;

        }


public:


    class stepState
    {
        public:

        const bool forward;
        gScalar dxTry;
        gScalar dxDid;
        bool first;
        bool last;
        bool reject;
        bool prevReject;

        CUDA_HOSTDEV stepState(const gScalar dx)
        :
            forward(dx > 0 ? true : false),
            dxTry(dx),
            dxDid(0),
            first(true),
            last(false),
            reject(false),
            prevReject(false)
        {}
    };




    // Constructors
        //- Construct for given ODESystem specifying tolerances
        gpuODESolver
        (
            const gScalar absTol,
            const gScalar relTol,
            const gLabel maxSteps
        )
        :
            absTol_(absTol),
            relTol_(relTol),
            maxSteps_(maxSteps)
        {}



    // Member Functions


        //- Return access to the absolute tolerance field
        inline CUDA_HOSTDEV gScalar& absTol()
        {
            return absTol_;
        }

        //- Return access to the relative tolerance field
        inline CUDA_HOSTDEV gScalar& relTol()
        {
            return relTol_;
        }




};



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
