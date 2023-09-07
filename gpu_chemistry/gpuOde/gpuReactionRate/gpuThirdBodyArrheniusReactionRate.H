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
    Foam::gpuThirdBodyArrheniusReactionRate

Description
    Arrhenius reaction rate enhanced by third-body interaction.

SourceFiles
    gpuThirdBodyArrheniusReactionRateI.H

\*---------------------------------------------------------------------------*/

#ifndef gpuThirdBodyArrheniusReactionRate_H
#define gpuThirdBodyArrheniusReactionRate_H

#include "gpuArrheniusReactionRate.H"
#include "gpuThirdBodyEfficiencies.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{




/*---------------------------------------------------------------------------*\
               Class gpuThirdBodyArrheniusReactionRate Declaration
\*---------------------------------------------------------------------------*/

class gpuThirdBodyArrheniusReactionRate
:
    gpuArrheniusReactionRate
{
    // Private Data

        gpuThirdBodyEfficiencies thirdBodyEfficiencies_;


public:

    // Constructors
        gpuThirdBodyArrheniusReactionRate() = default;

        //- Construct from components
        inline gpuThirdBodyArrheniusReactionRate
        (
            const gScalar A,
            const gScalar beta,
            const gScalar Ta,
            const gpuThirdBodyEfficiencies& tbes
        )
        :
            gpuArrheniusReactionRate(A, beta, Ta),
            thirdBodyEfficiencies_(tbes)
        {}

        //- Construct from components
        inline gpuThirdBodyArrheniusReactionRate
        (
            const gpuArrheniusReactionRate& arhenius,
            const gpuThirdBodyEfficiencies& tbes
        )
        :
            gpuArrheniusReactionRate(arhenius),
            thirdBodyEfficiencies_(tbes)
        {}


    // Member Functions


        //- Pre-evaluation hook
        inline CUDA_HOSTDEV void preEvaluate() const
        {
            gpuArrheniusReactionRate::preEvaluate();
        }

        //- Post-evaluation hook
        inline CUDA_HOSTDEV void postEvaluate() const
        {
            gpuArrheniusReactionRate::postEvaluate();
        }

        //- Return the rate
        template<class Span>
        inline CUDA_HOSTDEV gScalar operator()
        (
            const gScalar p,
            const gScalar T,
            const Span c,
            const gLabel li
        ) const
        {
            return
                thirdBodyEfficiencies_.M(c)
            *gpuArrheniusReactionRate::operator()(p, T, c, li);
        }

        //- The derivative of the rate w.r.t. temperature
        template<class Span>
        inline CUDA_HOSTDEV gScalar ddT
        (
            const gScalar p,
            const gScalar T,
            const Span c,
            const gLabel li
        ) const
        {
            return
                thirdBodyEfficiencies_.M(c)
            *gpuArrheniusReactionRate::ddT(p, T, c, li);
        }

        //- Is the rate a function of concentration?
        inline CUDA_HOSTDEV bool hasDdc() const
        {
            return true;
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
            const gScalar k = gpuArrheniusReactionRate::operator()(p, T, c, li);

            for (gLabel i = 0; i < gLabel(std::size(ddc)); ++i)
            {
                ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
            }
        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //