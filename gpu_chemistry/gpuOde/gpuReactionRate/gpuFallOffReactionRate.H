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
    Foam::gpuFallOffReactionRate

Description
    General class for handling unimolecular/recombination fall-off reactions.

SourceFiles
    gpuFallOffReactionRateI.H

\*---------------------------------------------------------------------------*/

#ifndef gpuFallOffReactionRate_H
#define gpuFallOffReactionRate_H

#include "gpuThirdBodyEfficiencies.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{



/*---------------------------------------------------------------------------*\
                     Class gpuFallOffReactionRate Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionRate, class FallOffFunction>
class gpuFallOffReactionRate
{
    // Private Data

        ReactionRate k0_;

        ReactionRate kInf_;

        FallOffFunction F_;

        gpuThirdBodyEfficiencies thirdBodyEfficiencies_;


public:

    // Constructors

        gpuFallOffReactionRate() = default;

        //- Construct from components
        inline gpuFallOffReactionRate
        (
            const ReactionRate& k0,
            const ReactionRate& kInf,
            const FallOffFunction& F,
            const gpuThirdBodyEfficiencies& tbes
        )
        :
            k0_(k0),
            kInf_(kInf),
            F_(F),
            thirdBodyEfficiencies_(tbes)
        {}



    // Member Functions


        //- Pre-evaluation hook
        inline CUDA_HOSTDEV void preEvaluate() const
        {
            k0_.preEvaluate();
            kInf_.preEvaluate();
        }

        //- Post-evaluation hook
        inline CUDA_HOSTDEV void postEvaluate() const
        {
            k0_.postEvaluate();
            kInf_.postEvaluate();
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
            const gScalar k0 = k0_(p, T, c, li);
            const gScalar kInf = kInf_(p, T, c, li);
            const gScalar M = thirdBodyEfficiencies_.M(c);
            const gScalar Pr = k0/kInf*M;
            const gScalar F = F_(T, Pr);

            return kInf*(Pr/(1.0 + Pr))*F;
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
            const gScalar k0 = k0_(p, T, c, li);
            const gScalar kInf = kInf_(p, T, c, li);
            const gScalar M = thirdBodyEfficiencies_.M(c);
            const gScalar Pr = k0/kInf*M;
            const gScalar F = F_(T, Pr);

            const gScalar dkInfdT = kInf_.ddT(p, T, c, li);

            #if FALL_OFF_FUNCTION_JACOBIAN

            const gScalar dk0dT = k0_.ddT(p, T, c, li);
            const gScalar dPrdT = (M*dk0dT - Pr*dkInfdT)/kInf;
            const gScalar dFdT = F_.ddT(T, Pr, F) + F_.ddPr(T, Pr, F)*dPrdT;

            return
                dkInfdT*(Pr/(1 + Pr))*F
            + kInf*dPrdT/sqr(1 + Pr)*F
            + kInf*(Pr/(1 + Pr))*dFdT;

            #else

            return dkInfdT*(Pr/(1.0 + Pr))*F;

            #endif

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

            const gScalar k0 = k0_(p, T, c, li);
            const gScalar kInf = kInf_(p, T, c, li);
            const gScalar M = thirdBodyEfficiencies_.M(c);
            const gScalar Pr = k0/kInf*M;
            const gScalar F = F_(T, Pr);

            #if FALL_OFF_FUNCTION_JACOBIAN

            gScalarField dk0dc(c.size(), 0);
            k0_.ddc(p, T, c, li, dk0dc);
            gScalarField dkInfdc(c.size(), 0);
            kInf_.ddc(p, T, c, li, dkInfdc);
            tmp<gScalarField> tdMdc = thirdBodyEfficiencies_.dMdc(c);
            const gScalarField& dMdc = tdMdc();
            gScalarField dPrdc((M*dk0dc - Pr*dkInfdc + k0*dMdc)/kInf);
            const gScalar dFdPr = F_.ddPr(T, Pr, F);

            ddc =
                dkInfdc*(Pr/(1 + Pr))*F
            + kInf*dPrdc/sqr(1 + Pr)*F
            + kInf*(Pr/(1 + Pr))*dFdPr*dPrdc;

            #else

            kInf_.ddc(p, T, c, li, ddc);

            for (gLabel i = 0; i < gLabel(std::size(ddc)); ++i)
            {
                ddc[i] *= (Pr/(1.0 + Pr))*F;
            }

            //ddc *= (Pr/(1 + Pr))*F;

            #endif
        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam



#endif

// ************************************************************************* //