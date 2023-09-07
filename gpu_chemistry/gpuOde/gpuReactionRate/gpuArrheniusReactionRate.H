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
    Foam::gpuArrheniusReactionRate

Description
    Arrhenius reaction rate given by:

        k = A * T^beta * exp(-Ta/T)

SourceFiles
    gpuArrheniusReactionRateI.H

\*---------------------------------------------------------------------------*/

#ifndef gpuArrheniusReactionRate_H
#define gpuArrheniusReactionRate_H

#include "gpuConstants.H"
#include "mdspan.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{


/*---------------------------------------------------------------------------*\
                    Class gpuArrheniusReactionRate Declaration
\*---------------------------------------------------------------------------*/

class gpuArrheniusReactionRate
{
    // Private Data

        gScalar A_;
        gScalar beta_;
        gScalar Ta_;



    inline CUDA_HOSTDEV gScalar 
    computeAk(const gScalar p, const gScalar T) const
    {
        gScalar ak = A_;

        if (fabs(beta_) > gpuVSmall)
        {
            ak *= pow(T, beta_);
        }

        if (fabs(Ta_) > gpuVSmall)
        {
            ak *= exp(-Ta_/T);
        }

        return ak;
    }

public:

    // Constructors

        gpuArrheniusReactionRate() = default;

        //- Construct from components
        inline gpuArrheniusReactionRate
        (
            const gScalar A,
            const gScalar beta,
            const gScalar Ta
        )
        :
            A_(A),
            beta_(beta),
            Ta_(Ta)
        {}


    // Member Functions


        //- Pre-evaluation hook
        inline CUDA_HOSTDEV void preEvaluate() const
        {}

        //- Post-evaluation hook
        inline CUDA_HOSTDEV void postEvaluate() const
        {}

        //- Return the rate
        template<class Span>
        inline CUDA_HOSTDEV gScalar operator()
        (
            const gScalar p,
            const gScalar T,
            const Span& c,
            const gLabel li
        ) const
        {
            (void) c;
            (void) li;
            return computeAk(p, T);

        }

        //- The derivative of the rate w.r.t. temperature
        template<class Span>
        inline CUDA_HOSTDEV gScalar ddT
        (
            const gScalar p,
            const gScalar T,
            const Span& c,
            const gLabel li
        ) const
        {
            (void) c;
            (void) li;
            return computeAk(p, T)*(beta_+Ta_/T)/T;
        }

        //- Is the rate a function of concentration?
        inline CUDA_HOSTDEV bool hasDdc() const
        {
            return false;
        }

        //- The derivative of the rate w.r.t. concentration
        template<class Span1, class Span2>
        inline CUDA_HOSTDEV void ddc
        (
            const gScalar p,
            const gScalar T,
            const Span1& c,
            const gLabel li,
            Span2& ddc
        ) const
        {
            for (gLabel i = 0; i < gLabel(std::size(ddc)); ++i)
            {
                ddc[i] = 0;
            }
        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
