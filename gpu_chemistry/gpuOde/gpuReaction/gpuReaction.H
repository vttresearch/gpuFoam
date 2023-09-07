/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2022 OpenFOAM Foundation
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
    Foam::gpuReaction

Description
    Simple extension of MulticomponentThermo to handle gpuReaction kinetics in
    addition to the equilibrium thermodynamics already handled.

SourceFiles
    gpuReactionI.H
    gpuReaction.C

\*---------------------------------------------------------------------------*/

#ifndef gpuReaction_H
#define gpuReaction_H

#include <array>

#include "gpuConstants.H"
#include "cuda_math.H"
//#include "utilities.H"
#include "cuda_host_dev.H"
#include "gpuReactionRate.H"
#include "gpuThermo.H"
#include "mdspan.H"
//#include "double_intrinsics.H"

//#include "specieExponent.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{



struct gpuSpecieCoeffs{

    gpuSpecieCoeffs() = default;

    //- Index of the specie
    gLabel index;

    //- Stoichiometric coefficient
    gScalar stoichCoeff;

    //- Exponent of the specie concentration
    gScalar exponent;
};






/*---------------------------------------------------------------------------*\
                          Class gpuReaction Declaration
\*---------------------------------------------------------------------------*/

class gpuReaction : public gpuThermo
{


public:

    // Static data


        using specieCoeffArray
            = std::array<gpuSpecieCoeffs, MAX_REACTION_WIDTH>;

        //- Default temperature limits of applicability of gpuReaction rates
        static constexpr gScalar TlowDefault = 0;
        static constexpr gScalar ThighDefault = std::numeric_limits<double>::max() / 10;


        specieCoeffArray lhs_;
        specieCoeffArray rhs_;


        gLabel lhs_size_;
        gLabel rhs_size_;

        gpuReactionRate k_;

        //Flags if the specie coeffs contain small (< 1)  exponents
        bool small_lhs_exp_;
        bool small_rhs_exp_;


    // Private Data

        //- Temperature limits of applicability of gpuReaction rates
        gScalar Tlow_, Thigh_;



    //Note! These return spans and not the underlying data.
    inline CUDA_HOSTDEV const auto lhs() const
    {
        return make_mdspan(lhs_, extents<1>{lhs_size_});
    }
    inline CUDA_HOSTDEV const auto rhs() const {

        return make_mdspan(rhs_, extents<1>{rhs_size_});
    }


    inline CUDA_HOSTDEV bool isIrreversible() const
    {
        return k_.isIrreversible();
    }



    inline CUDA_HOSTDEV bool lhs_small_exp() const
    {
        return small_lhs_exp_;
    }

    template<class Span>
    inline CUDA_HOSTDEV bool lhs_small_concentration(const Span& c) const
    {
        const auto llhs = lhs();
        for (size_t i = 0; i < llhs.size(); ++i)
        {
            const auto& si = llhs[i].index;
            if (c[si] < gpuSmall)
            {
                return true;
            }

        }
        return false;
    }

    inline CUDA_HOSTDEV bool rhs_small_exp() const
    {
        return small_rhs_exp_;
    }

    template<class Span>
    inline CUDA_HOSTDEV bool rhs_small_concentration(const Span& c) const
    {
        const auto rrhs = rhs();
        for (size_t i = 0; i < rrhs.size(); ++i)
        {
            const auto& si = rrhs[i].index;
            if (c[si] < gpuSmall)
            {
                return true;
            }
        }
        return false;
    }

    template<class Span>
    inline CUDA_HOSTDEV
    std::array<gScalar, MAX_REACTION_WIDTH> get_lhs_exps
    (
        const Span& c
    ) const
    {

        const auto llhs = lhs();
        std::array<gScalar, MAX_REACTION_WIDTH> ret{};
        if (!lhs_small_exp())
        {
            for (size_t i = 0; i < llhs.size(); ++i)
            {
                const auto& el = llhs[i].exponent;
                const auto& si = llhs[i].index;
                ret[i] = pow(c[si], el);
            }
        }
        return ret;
    }

    template<class Span>
    inline CUDA_HOSTDEV
    std::array<gScalar, MAX_REACTION_WIDTH> get_rhs_exps
    (
        const Span& c
    ) const
    {

        const auto rrhs = rhs();
        std::array<gScalar, MAX_REACTION_WIDTH> ret{};
        if (!rhs_small_exp())
        {
            for (size_t i = 0; i < rrhs.size(); ++i)
            {
                const auto& er = rrhs[i].exponent;
                const auto& si = rrhs[i].index;
                ret[i] = pow(c[si], er);
            }
        }
        return ret;
    }


public:

    gpuReaction() = default;

    CUDA_HOSTDEV
    gpuReaction
    (
        const gpuThermo& thermo,
        const specieCoeffArray& lhs,
        const specieCoeffArray& rhs,
        const gLabel lhs_size,
        const gLabel rhs_size,
        const gpuReactionRate& k,
        const gScalar Tlow,
        const gScalar Thigh
    ) :
    gpuThermo(thermo),
    lhs_(lhs),
    rhs_(rhs),
    lhs_size_(lhs_size),
    rhs_size_(rhs_size),
    k_(k),
    Tlow_(Tlow),
    Thigh_(Thigh)
    {

        const auto llhs = this->lhs();
        small_lhs_exp_ = false;
        for(size_t i = 0; i < llhs.size(); ++i)
        {
            if (llhs[i].exponent < gScalar(1))
            {
                small_lhs_exp_ = true;
                break;
            }
        }
        const auto rrhs = this->rhs();
        small_rhs_exp_ = false;
        for(size_t i = 0; i < rrhs.size(); ++i)
        {
            if (rrhs[i].exponent < gScalar(1))
            {
                small_rhs_exp_ = true;
                break;
            }
        }



    }

    gpuThermo& get_thermo()
    {
        return *this;
    }



    // Member Functions

        // Access


            //- Return the lower temperature limit for the gpuReaction
            inline CUDA_HOSTDEV gScalar Tlow() const
            {
                return Tlow_;
            }

            //- Return the upper temperature limit for the gpuReaction
            inline CUDA_HOSTDEV gScalar Thigh() const
            {
                return Thigh_;
            }


        // Hooks

            //- Pre-evaluation hook
            inline CUDA_HOSTDEV void preEvaluate() const
            {
                //TODO: make sure this is feasible
            }

            //- Post-evaluation hook
            inline CUDA_HOSTDEV void postEvaluate() const
            {
                //TODO: make sure this is feasible
            }


        // gpuReaction rate coefficients

            //- Concentration powers
            template<class Span>
            inline CUDA_HOSTDEV void C
            (
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li,
                gScalar& Cf,
                gScalar& Cr
            ) const
            {
                (void) li;


                Cf = Cr = 1.0;

                const auto llhs = lhs();

                for(size_t i = 0; i < llhs.size(); ++i)
                {
                    const gLabel si = llhs[i].index;
                    const auto el = llhs[i].exponent;
                    Cf *= c[si] >= gpuSmall || el >= 1 ? pow(max(c[si], 0.0), el) : 0.0;
                    //Cf *= pow(max(c[si], 0.0), el);
                    //Cf *= pow(c[si], el);

                }

                const auto rrhs = rhs();

                for(size_t i = 0; i < rrhs.size(); ++i)
                {
                    const gLabel si = rrhs[i].index;
                    const auto er = rrhs[i].exponent;
                    Cr *= c[si] >= gpuSmall || er >= 1 ? pow(max(c[si], 0.0), er) : 0.0;
                    //Cr *= pow(max(c[si], 0.0), er);
                    //Cr *= pow(c[si], er);
                }

            }

            inline CUDA_HOSTDEV gScalar fast_Cf
            (
                const std::array<gScalar, MAX_REACTION_WIDTH>& lhs_exp
            ) const
            {
                gScalar Cf = 1.0;
                const auto llhs = lhs();

                for(size_t i = 0; i < llhs.size(); ++i)
                {
                    Cf *= lhs_exp[i];
                }
                return Cf;

            }


            template<class Span>
            inline CUDA_HOSTDEV auto fast_dCfdcj
            (
                const std::array<gScalar, MAX_REACTION_WIDTH>& lhs_exp,
                const Span& c
            ) const
            {

                std::array<gScalar, MAX_REACTION_WIDTH> ret{};

                const auto llhs = lhs();
                for (size_t j = 0; j < llhs.size(); ++j)
                {

                    gScalar dCfdcj = 1;
                    for(size_t i = 0; i < llhs.size(); ++i)
                    {
                        const auto& el = llhs[i].exponent;
                        const auto& si = llhs[i].index;
                        if (i == j)
                        {
                            dCfdcj *= el * pow(c[si], el - gScalar(1));
                        }
                        else
                        {
                            dCfdcj *= lhs_exp[i];
                        }
                    }
                    ret[j] = dCfdcj;
                }

                return ret;

            }

            template<class Span>
            inline CUDA_HOSTDEV auto veryfast_dCfdcj
            (
                const gScalar Cf,
                const Span& c
            ) const
            {

                std::array<gScalar, MAX_REACTION_WIDTH> ret{};


                const auto llhs = lhs();

                for (size_t j = 0; j < llhs.size(); ++j)
                {

                    const auto& el = llhs[j].exponent;
                    const auto& si = llhs[j].index;

                    ret[j] = el * Cf / c[si]; //overflow for small c


                }

                return ret;

            }

            template<class Span>
            inline CUDA_HOSTDEV auto fast_dCrdcj
            (
                const std::array<gScalar, MAX_REACTION_WIDTH>& rhs_exp,
                const Span& c
            ) const
            {
                std::array<gScalar, MAX_REACTION_WIDTH> ret{};

                const auto rrhs = rhs();
                for (size_t j = 0; j < rrhs.size(); ++j)
                {

                    gScalar dCrdcj = 1;
                    for(size_t i = 0; i < rrhs.size(); ++i)
                    {
                        const auto& er = rrhs[i].exponent;
                        const auto& si = rrhs[i].index;
                        if (i == j)
                        {
                            dCrdcj *= er * pow(c[si], er - gScalar(1));
                        }
                        else
                        {
                            dCrdcj *= rhs_exp[i];
                        }
                    }
                    ret[j] = dCrdcj;
                }

                return ret;

            }

            template<class Span>
            inline CUDA_HOSTDEV auto veryfast_dCrdcj
            (
                const gScalar Cr,
                const Span& c
            ) const
            {

                std::array<gScalar, MAX_REACTION_WIDTH> ret{};


                const auto rrhs = rhs();
                for (size_t j = 0; j < rrhs.size(); ++j)
                {

                    const auto& er = rrhs[j].exponent;
                    const auto& si = rrhs[j].index;
                    ret[j] = er * Cr / c[si]; //overflow for small c
                }

                return ret;

            }


            inline CUDA_HOSTDEV gScalar fast_Cr
            (
                const std::array<gScalar, MAX_REACTION_WIDTH>& rhs_exp
            ) const
            {
                gScalar Cr = 1.0;
                const auto rrhs = rhs();

                for(size_t i = 0; i < rrhs.size(); ++i)
                {
                    Cr *= rhs_exp[i];
                }
                return Cr;

            }





            //- Net gpuReaction rate
            template<class Span>
            inline CUDA_HOSTDEV gScalar omega
            (
                const gScalar p,
                const gScalar T,
                const Span& c
            ) const
            {

                const gScalar clippedT = min(max(T, this->Tlow()), this->Thigh());

                // Rate constants
                const gLabel li = 0;
                const gScalar kf = this->kf(p, clippedT, c, li);
                const gScalar kr = this->kr(kf, p, clippedT, c, li);

                const auto lhs_exps = get_lhs_exps(c);
                const auto rhs_exps = get_rhs_exps(c);

                const gScalar Cf = fast_Cf(lhs_exps);
                const gScalar Cr = fast_Cr(rhs_exps);


                return kf*Cf - kr*Cr;

            }

            template<class Span>
            inline CUDA_HOSTDEV void dNdtByV
            (
                const gScalar omega,
                Span& dNdtByV
            ) const
            {
                const auto llhs = lhs();

                for(size_t i = 0; i < llhs.size(); ++i)
                {
                    const gLabel si = llhs[i].index;
                    const gScalar sl = llhs[i].stoichCoeff;
                    dNdtByV[si] -= sl*omega;
                }

                const auto rrhs = rhs();

                for(size_t i = 0; i < rrhs.size(); ++i)
                {
                    const gLabel si = rrhs[i].index;
                    const gScalar sr = rrhs[i].stoichCoeff;
                    dNdtByV[si] += sr*omega;
                }
            }

            //- The net gpuReaction rate for each species involved
            template<class Span1, class Span2>
            inline CUDA_HOSTDEV void dNdtByV
            (
                const gScalar p,
                const gScalar T,
                const Span1& c,
                Span2& dNdtByV
            ) const
            {
                const gScalar omega = this->omega(p, T, c);

                this->dNdtByV(omega, dNdtByV);

            }


        // gpuReaction rate coefficients

            //- Forward rate constant
            template<class Span>
            inline CUDA_HOSTDEV gScalar kf
            (
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li
            ) const
            {
                //return 43.0;
                (void) li;
                return k_(p, T, c, li);
            }

            //- Reverse rate constant from the given forward rate constant
            template<class Span>
            inline CUDA_HOSTDEV gScalar kr
            (
                const gScalar kfwd,
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li
            ) const
            {

                (void) li;
                if (isIrreversible()){
                    return 0.0;
                }
                //return kfwd/max(this->Kc(p, T), gpuRootSmall);
                //return kfwd/max(this->Kc(p, T), gpuRootSmall);
                const double rSmall = sqrt(gpuSmall);
                return kfwd/max(this->Kc(p, T), rSmall);

            }

            //- Reverse rate constant
            template<class Span>
            inline CUDA_HOSTDEV gScalar kr
            (
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li
            ) const
            {
                (void) li;
                if (isIrreversible()){
                    return 0.0;
                }
                //reversible

                return kr(kf(p, T, c, li), p, T, c, li);
            }


        // Jacobian coefficients

            //- Temperature derivative of forward rate
            template<class Span>
            inline CUDA_HOSTDEV gScalar dkfdT
            (
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li
            ) const
            {
                (void) li;
                return k_.ddT(p, T, c, li);
            }

            //- Temperature derivative of reverse rate
            template<class Span>
            inline CUDA_HOSTDEV gScalar dkrdT
            (
                const gScalar p,
                const gScalar T,
                const Span& c,
                const gLabel li,
                const gScalar dkfdT,
                const gScalar kr
            ) const
            {

                (void) li;
                if (isIrreversible()){
                    return 0.0;
                }
                const double rSmall = sqrt(gpuSmall);
                const gScalar Kc = max(this->Kc(p, T), rSmall);

                return dkfdT/Kc - (Kc > rSmall ? kr*this->dKcdTbyKc(p, T) : 0.0);

            }
            //- Does this gpuReaction have concentration-dependent rate constants?
            inline CUDA_HOSTDEV bool hasDkdc() const
            {
                return k_.hasDdc();
            }

            //- Concentration derivative of forward rate
            template<class Span1, class Span2>
            inline CUDA_HOSTDEV void dkfdc
            (
                const gScalar p,
                const gScalar T,
                const Span1& c,
                const gLabel li,
                Span2& dkfdc
            ) const
            {
                (void) li;
                k_.ddc(p, T, c, li, dkfdc);
            }

            //- Concentration derivative of reverse rate
            template<class Span1, class Span2, class Span3>
            inline CUDA_HOSTDEV void dkrdc
            (
                const gScalar p,
                const gScalar T,
                const Span1& c,
                const gLabel li,
                const Span2& dkfdc,
                const gScalar kr,
                Span3& dkrdc
            ) const
            {

                (void) li;
                if (isIrreversible())
                {

                    for (gLabel i = 0; i < gLabel(std::size(dkrdc)); ++i)
                    {
                        dkrdc[i] = 0.0;
                    }
                }
                else
                {
                    const gScalar rSmall = sqrt(gpuSmall);
                    const gScalar Kc = max(this->Kc(p, T), rSmall);

                    for (gLabel i = 0; i < gLabel(std::size(dkrdc)); ++i)
                    {
                        dkrdc[i] = dkfdc[i]/Kc;
                    }

                }
                //dkrdc = dkfdc/Kc;

            }


            // Jacobian contributions from the derivative of the lhs concentration products
            // w.r.t. concentration
            template<class Span, class TwoDSpan>
            inline CUDA_HOSTDEV void jac_dCfdcj_contribution
            (
                const gScalar kf,
                const gScalar Cf,
                const std::array<gScalar, MAX_REACTION_WIDTH>& lhs_exp,
                const Span& c,
                TwoDSpan& ddNdtByVdcTp
            ) const
            {
                /*
                auto t1 = fast_dCfdcj(lhs_exp, c);
                auto t2 =
                */

                std::array<gScalar, MAX_REACTION_WIDTH> dCfdjs;
                if (!lhs_small_concentration(c))
                {
                    dCfdjs = veryfast_dCfdcj(Cf, c);
                }
                else
                {
                    dCfdjs = fast_dCfdcj(lhs_exp, c);
                }


                //const auto dCfdjs = fast_dCfdcj(lhs_exp, c);
                //const auto dCfdjs = veryfast_dCfdcj(Cf, lhs_exp, c);

                const auto llhs = lhs();
                const auto rrhs = rhs();


                for (size_t j = 0; j < llhs.size(); ++j)
                {
                    const gScalar dCfdcj = dCfdjs[j];


                    const gLabel sj = llhs[j].index;
                    for(size_t i = 0; i < llhs.size(); ++i)
                    {
                        const gLabel si = llhs[i].index;
                        const gScalar sl = llhs[i].stoichCoeff;
                        ddNdtByVdcTp(si, sj) -= sl*kf*dCfdcj;
                    }
                    for(size_t i = 0; i < rrhs.size(); ++i)
                    {
                        const gLabel si = rrhs[i].index;
                        const gScalar sr = rrhs[i].stoichCoeff;
                        ddNdtByVdcTp(si, sj) += sr*kf*dCfdcj;
                    }
                }

            }


            // Jacobian contributions from the derivative of the lhs concentration products
            // w.r.t. concentration
            template<class Span, class TwoDSpan>
            inline CUDA_HOSTDEV void jac_dCrdcj_contribution
            (
                const gScalar kr,
                const gScalar Cr,
                const std::array<gScalar, MAX_REACTION_WIDTH>& rhs_exp,
                const Span& c,
                TwoDSpan& ddNdtByVdcTp
            ) const
            {
                //const auto dCrdjs = fast_dCrdcj(rhs_exp, c);
                //const auto dCrdjs = veryfast_dCrdcj(Cr, c);

                std::array<gScalar, MAX_REACTION_WIDTH> dCrdjs;
                if (!rhs_small_concentration(c))
                {
                    dCrdjs = veryfast_dCrdcj(Cr, c);
                }
                else
                {
                    dCrdjs = fast_dCrdcj(rhs_exp, c);
                }

                const auto llhs = lhs();
                const auto rrhs = rhs();

                for (size_t j = 0; j < rrhs.size(); ++j)
                {
                    const gScalar dCrcj = dCrdjs[j];

                    const gLabel sj = rrhs[j].index;

                    for(size_t i = 0; i < llhs.size(); ++i)
                    {
                        const gLabel si =  llhs[i].index;
                        const gScalar sl = llhs[i].stoichCoeff;
                        ddNdtByVdcTp(si, sj) += sl*kr*dCrcj;
                    }
                    for(size_t i = 0; i < rrhs.size(); ++i)
                    {
                        const gLabel si = rrhs[i].index;
                        const gScalar sr = rrhs[i].stoichCoeff;
                        ddNdtByVdcTp(si, sj) -= sr*kr*dCrcj;
                    }
                }
            }

            // Jacobian contributions from the derivative of the rate constants
            // w.r.t. temperature
            template<class Span, class TwoDSpan>
            inline CUDA_HOSTDEV void jac_dCdT_contribution
            (
                const gScalar p,
                const gScalar T,
                const gScalar Cf,
                const gScalar Cr,
                const gScalar kr,
                const gLabel Tsi,
                const Span& c,
                const TwoDSpan& ddNdtByVdcTp
            ) const
            {

                const gScalar dkfdT = this->dkfdT(p, T, c, 0);
                const gScalar dkrdT = this->dkrdT(p, T, c, 0, dkfdT, kr);

                const gScalar dwdT = dkfdT*Cf - dkrdT*Cr;

                const auto llhs = lhs();
                for(size_t i = 0; i < llhs.size(); ++i)
                {
                    const gLabel si = llhs[i].index;
                    const gScalar sl = llhs[i].stoichCoeff;
                    ddNdtByVdcTp(si, Tsi) -= sl*dwdT;
                }
                const auto rrhs = rhs();
                for(size_t i = 0; i < rrhs.size(); ++i)
                {
                    const gLabel si = rrhs[i].index;
                    const gScalar sr = rrhs[i].stoichCoeff;
                    ddNdtByVdcTp(si, Tsi) += sr*dwdT;
                }

            }


            // Jacobian contributions from the derivative of the rate constants
            // w.r.t. concentration
            template<class Span1, class Span2, class Span3, class TwoDSpan>
            inline CUDA_HOSTDEV void jac_dCdC_contribution
            (
                const gScalar p,
                const gScalar T,
                const gScalar Cf,
                const gScalar Cr,
                const gScalar kr,
                const Span1& c,
                Span2& cTpWork0,
                Span3& cTpWork1,
                TwoDSpan& ddNdtByVdcTp
            ) const
            {
                if (hasDkdc())
                {
                    auto& dkfdc = cTpWork0;
                    auto& dkrdc = cTpWork1;

                    this->dkfdc(p, T, c, 0, dkfdc);
                    this->dkrdc(p, T, c, 0, dkfdc, kr, dkrdc);

                    //Do the compute here once and reuse the first work array
                    for(gLabel j = 0; j < gLabel(std::size(c)); ++j)
                    {
                        dkfdc[j] = dkfdc[j]*Cf - dkrdc[j]*Cr;
                    }

                    const auto llhs = lhs();


                    for(size_t i = 0; i < llhs.size(); ++i)
                    {
                        const gLabel si = llhs[i].index;
                        const gScalar sl = llhs[i].stoichCoeff;

                        for(gLabel j = 0; j < gLabel(std::size(c)); ++j)
                        {
                            ddNdtByVdcTp(si, j) -= sl*dkfdc[j];
                        }
                    }

                    const auto rrhs = rhs();
                    for(size_t i = 0; i < rrhs.size(); ++i)
                    {
                        const gLabel si = rrhs[i].index;
                        const gScalar sr = rrhs[i].stoichCoeff;
                        for(gLabel j = 0; j < gLabel(std::size(c)); ++j)
                        {
                            ddNdtByVdcTp(si, j) += sr*dkfdc[j];
                        }

                    }

                }

            }


            //- Derivative of the net gpuReaction rate for each species involved
            //  w.r.t. the concentration and temperature
            template<class Span1, class Span2, class TwoDSpan, class Span3, class Span4>
            inline CUDA_HOSTDEV void ddNdtByVdcTp
            (
                const gScalar p,
                const gScalar T,
                const Span1& c,
                const gLabel li,
                Span2& dNdtByV,
                TwoDSpan& ddNdtByVdcTp,
                const gLabel Nsi0,
                const gLabel Tsi,
                Span3& cTpWork0,
                Span4& cTpWork1
            ) const
            {




                (void) li;


                const auto lhs_exps = get_lhs_exps(c);
                const auto rhs_exps = get_rhs_exps(c);

                const gScalar Cf = fast_Cf(lhs_exps);
                const gScalar Cr = fast_Cr(rhs_exps);

                // Rate constants
                const gScalar kf = this->kf(p, T, c, li);
                const gScalar kr = this->kr(kf, p, T, c, li);

                // Overall reaction rate
                const gScalar omega = kf*Cf - kr*Cr;

                // Specie reaction rates

                //Derivative contribuiton
                this->dNdtByV(omega, dNdtByV);


                // Jacobian contributions from the derivative of the concentration products
                // w.r.t. concentration

                jac_dCfdcj_contribution(kf, Cf, lhs_exps, c, ddNdtByVdcTp);
                jac_dCrdcj_contribution(kr, Cr, rhs_exps, c, ddNdtByVdcTp);


                jac_dCdT_contribution
                (
                    p, T, Cf, Cr, kr, Tsi, c, ddNdtByVdcTp
                );

                jac_dCdC_contribution(p, T, Cf, Cr, kr, c, cTpWork0, cTpWork1, ddNdtByVdcTp);


            }

};






// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
