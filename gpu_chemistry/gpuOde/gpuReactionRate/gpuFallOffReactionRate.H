#pragma once


#include "gpuThirdBodyEfficiencies.H"


namespace FoamGpu
{



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


            kInf_.ddc(p, T, c, li, ddc);

            for (gLabel i = 0; i < gLabel(std::size(ddc)); ++i)
            {
                ddc[i] *= (Pr/(1.0 + Pr))*F;
            }
        }

};



} // End namespace Foam

