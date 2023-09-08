#pragma once

#include <array>

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuConstants.H"
#include "gpuReactionRate.H"
#include "gpuThermo.H"
#include "mdspan.H"

namespace FoamGpu {

struct gpuSpecieCoeffs {

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

class gpuReaction : public gpuThermo {

public:
    using specieCoeffArray = std::array<gpuSpecieCoeffs, MAX_REACTION_WIDTH>;

    //- Default temperature limits of applicability of gpuReaction rates
    static constexpr gScalar TlowDefault = 0;
    static constexpr gScalar ThighDefault =
        std::numeric_limits<double>::max() / 10;

private:

    //- LHS specie coefficients
    specieCoeffArray lhs_;

    //- RHS specie coefficients
    specieCoeffArray rhs_;

    //- Number of species on the LHS. This is different than the size of the
    // lhs_ array because of static memory.
    gLabel lhs_size_;

    //- Number of species on the RHS. This is different than the size of the
    // rhs_ array because of static memory.
    gLabel rhs_size_;

    //- The underlying gpuReactionRate object.
    gpuReactionRate k_;

    // Flags if the specie coeffs contain small (< 1)  exponents
    bool small_lhs_exp_;
    bool small_rhs_exp_;

    //- Temperature limits of applicability of gpuReaction rates
    gScalar Tlow_, Thigh_;

public:

    gpuReaction() = default;

    CUDA_HOSTDEV
    gpuReaction(const gpuThermo&        thermo,
                const specieCoeffArray& lhs,
                const specieCoeffArray& rhs,
                const gLabel            lhs_size,
                const gLabel            rhs_size,
                const gpuReactionRate&  k,
                const gScalar           Tlow,
                const gScalar           Thigh)
        : gpuThermo(thermo)
        , lhs_(lhs)
        , rhs_(rhs)
        , lhs_size_(lhs_size)
        , rhs_size_(rhs_size)
        , k_(k)
        , Tlow_(Tlow)
        , Thigh_(Thigh) {

        small_lhs_exp_ = false;
        for (gLabel i = 0; i < lhs_size; ++i) {
            if (lhs[i].exponent < gScalar(1)) {
                small_lhs_exp_ = true;
                break;
            }
        }
        small_rhs_exp_ = false;
        for (gLabel i = 0; i < rhs_size; ++i) {
            if (rhs[i].exponent < gScalar(1)) {
                small_rhs_exp_ = true;
                break;
            }
        }
    }

    /// @brief Returns a span to the LHS specie coeffcients.
    /// @return A span to the LHS specie coefficients.
    inline CUDA_HOSTDEV auto lhs() const {
        return make_mdspan(lhs_, extents<1>{lhs_size_});
    }

    /// @brief Returns a span to the LHS specie coeffcients.
    /// @return A span to the LHS specie coefficients.
    inline CUDA_HOSTDEV auto rhs() const {
        return make_mdspan(rhs_, extents<1>{rhs_size_});
    }

    /// @brief Checks if the reaction rate of this reaction is irreversible.
    /// @return Returns true if the reaction rate of this reaction is
    /// irreversible. Returns false otherwise.
    inline CUDA_HOSTDEV bool isIrreversible() const {
        return k_.isIrreversible();
    }

    /// @brief Convert this reaction to the inherited gpuThermo object.
    /// @return Inherited gpuThermo object.
    gpuThermo& get_thermo() { return *this; }

    /// @brief Returns the lower temperature limit for this reaction.
    /// @return The lower temperature limit.
    inline CUDA_HOSTDEV gScalar Tlow() const { return Tlow_; }

    /// @brief Returns the upper temperature limit for this reaction.
    /// @return The upper temperature limit.
    inline CUDA_HOSTDEV gScalar Thigh() const { return Thigh_; }


    /// @brief Checks if this reaction has concentration dependent rate
    /// constants.
    /// @return True if has concetration dependent rate constants, false
    /// otherwise.
    inline CUDA_HOSTDEV bool hasDkdc() const { return k_.hasDdc(); }

    /// @brief Computes the net reaction rate of this reaction based on the
    /// input pressure, temperature and concentrations.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell.
    /// @return The net reaction rate of this reaction.
    template <class Span>
    inline CUDA_HOSTDEV gScalar omega(const gScalar p,
                                      const gScalar T,
                                      const Span&   c) const;

    /// @brief Assigns the precomputed net reaction rate for each specie
    /// involved to the input dNdtByV span.
    /// @param omega Precomputed net reaction rate of this reaction.
    /// @param dNdtByV The span of net reaction rates which are modified by this
    /// function.
    template <class Span>
    inline CUDA_HOSTDEV void dNdtByV(const gScalar omega, Span& dNdtByV) const;

    /// @brief Computes and assigns the net reaction rate for each specie
    /// involved to the input dNdtByV span.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell.
    /// @param dNdtByV The span of net reaction rates which are modified by this
    /// function.
    template <class Span1, class Span2>
    inline CUDA_HOSTDEV void dNdtByV(const gScalar p,
                                     const gScalar T,
                                     const Span1&  c,
                                     Span2&        dNdtByV) const;

    /// @brief Computes the forward reaction rate constant of this reaction.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @return The forward reaction rate constant of this reaction.
    template <class Span>
    inline CUDA_HOSTDEV gScalar
    kf(const gScalar p, const gScalar T, const Span& c, const gLabel li) const;

    /// @brief Computes the reverse reaction rate constant of this reaction.
    /// @param kfwd Forward reaction rate contant.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @return The reverse reaction rate constant of this reaction.
    template <class Span>
    inline CUDA_HOSTDEV gScalar kr(const gScalar kfwd,
                                   const gScalar p,
                                   const gScalar T,
                                   const Span&   c,
                                   const gLabel  li) const;

    /// @brief Computes the reverse reaction rate constant of this reaction.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @return The reverse reaction rate constant of this reaction.
    template <class Span>
    inline CUDA_HOSTDEV gScalar
    kr(const gScalar p, const gScalar T, const Span& c, const gLabel li) const;

    /// @brief Computes the temperature derivative of the forward rate.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @return The temperature derivative of the forward rate.
    template <class Span>
    inline CUDA_HOSTDEV gScalar dkfdT(const gScalar p,
                                      const gScalar T,
                                      const Span&   c,
                                      const gLabel  li) const;

    /// @brief Computes the temperature derivative of the reverse rate.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @param dkfdT Temperature derivative of the forward rate.
    /// @param kr The reverse rate constant.
    /// @return The temperature derivative of the reverse rate.
    template <class Span>
    inline CUDA_HOSTDEV gScalar dkrdT(const gScalar p,
                                      const gScalar T,
                                      const Span&   c,
                                      const gLabel  li,
                                      const gScalar dkfdT,
                                      const gScalar kr) const;

    /// @brief Computes the concentration derivative of the forward rate.
    /// @param p Pressure
    /// @param T Temperature
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @param dkfdc A span of concentation derivatives to be modified.
    template <class Span1, class Span2>
    inline CUDA_HOSTDEV void dkfdc(const gScalar p,
                                   const gScalar T,
                                   const Span1&  c,
                                   const gLabel  li,
                                   Span2&        dkfdc) const;

    /// @brief Computes the concentration derivative of the reverse rate.
    /// @param p Pressure
    /// @param T Temperature
    /// @param c A span of concentrations in a cell
    /// @param li Arbitrary integer TODO: remove.
    /// @param dkfdc Concentration derivatives of the forward rate.
    /// @param kr The reverse rate constant.
    /// @param dkrdc A span of concentation derivatives to be modified.
    template <class Span1, class Span2, class Span3>
    inline CUDA_HOSTDEV void dkrdc(const gScalar p,
                                   const gScalar T,
                                   const Span1&  c,
                                   const gLabel  li,
                                   const Span2&  dkfdc,
                                   const gScalar kr,
                                   Span3&        dkrdc) const;

    /// @brief Computes and assigns the jacobian contributions from the
    /// derivative of the rate constants w.r.t concentration and temperature.
    /// Pressure is assumed constant and thus the contributions are zero.
    /// @param p Pressure
    /// @param T Temperature
    /// @param c A span of concentrations in a cell.
    /// @param li Arbitrary integer TODO: remove.
    /// @param dNdtByV A one dimensional span where the derivative contributions
    /// are assigned.
    /// @param ddNdtByVdcTp A two dimensional span where the jacobian
    /// contributions are assigned.
    /// @param Nsi0 Starting index of the concentrations in the solution array
    /// (0).
    /// @param Tsi0 Starting index of the temperature in the solution array
    /// (nSpecie).
    /// @param cTpWork0 A a span of at least size nSpecie + 2 used as a
    /// temporary.
    /// @param cTpWork1 A a span of at least size nSpecie + 2 used as a
    /// temporary.
    template <class Span1,
              class Span2,
              class TwoDSpan,
              class Span3,
              class Span4>
    inline CUDA_HOSTDEV void ddNdtByVdcTp(const gScalar p,
                                          const gScalar T,
                                          const Span1&  c,
                                          const gLabel  li,
                                          Span2&        dNdtByV,
                                          TwoDSpan&     ddNdtByVdcTp,
                                          const gLabel  Nsi0,
                                          const gLabel  Tsi,
                                          Span3&        cTpWork0,
                                          Span4&        cTpWork1) const;

private:
    /// @brief Checks if the LHS specie coefficients contain small (< 1)
    /// exponents.
    /// @return Returns true if the LHS species of the reaction contain less
    /// than unity exponents. Returns false otherwise.
    inline CUDA_HOSTDEV bool lhsSmallExp() const { return small_lhs_exp_; }

    /// @brief Checks if the RHS specie coefficients contain small (< 1)
    /// exponents.
    /// @return Returns true if the RHS species of the reaction contain less
    /// than unity exponents. Returns false otherwise.
    inline CUDA_HOSTDEV bool rhsSmallExp() const { return small_rhs_exp_; }

    /// @brief Checks if any of the species involved in the LHS of this reaction
    /// has small (< gpuSmall) concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return True if any of the concentrations on the LHS of this reaction
    /// have small values. False otherwise.
    template <class Span>
    inline CUDA_HOSTDEV bool lhsSmallConcentration(const Span& c) const;

    /// @brief Checks if any of the species involved in the LHS of this reaction
    /// has small (< gpuSmall) concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return True if any of the concentrations on the LHS of this reaction
    /// have small values. False otherwise.
    template <class Span>
    inline CUDA_HOSTDEV bool rhsSmallConcentration(const Span& c) const;

    /// @brief Computes the powers pow(c_i, e_i) of the LHS specie
    /// concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return An array of powers of the LHS species.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                        lhsPowers(const Span& c) const;

    /// @brief Computes the powers pow(c_i, e_i) of the RHS specie
    /// concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return An array of powers of the RHS species.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                        rhsPowers(const Span& c) const;

    /// @brief Given an array of LHS concentration powers, computes the product
    /// of them all.
    /// @param lhsPow The input LHS powers to compute the product for.
    /// @return The product of the input powers.
    inline CUDA_HOSTDEV gScalar
    calcCf(const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow) const;

    /// @brief Given an array of RHS concentration powers, computes the product
    /// of them all.
    /// @param lhsPow The input RHS powers to compute the product for.
    /// @return The product of the input powers.
    inline CUDA_HOSTDEV gScalar
    calcCr(const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow) const;

    /// @brief Given precomputed LHS powers, computes the derivatives of the
    /// product of the powers w.r.t all concentrations on the LHS. This
    /// version has no limit on the values of c.
    /// @param lhsPow Precomputed concentrations powers.
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of the product of concentration powers w.r.t to
    /// all concentrations on the LHS.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    calcdCfdcj(const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow,
               const Span&                                    c) const;

    /// @brief Given precomputed RHS powers, computes the derivatives of the
    /// product of the powers w.r.t all concentrations on the RHS. This
    /// version has no limit on the values of c.
    /// @param lhsPow Precomputed concentrations powers.
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of the product of concentration powers w.r.t to
    /// all concentrations on the RHS.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    calcdCrdcj(const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow,
               const Span&                                    c) const;

    /// @brief Given precomputed product of the LHS powers, computes the
    /// derivatives of the product w.r.t all concentrations on the LHS. This
    /// version requires no pow() calls but may overflow for small
    /// concentrations.
    /// @param Cf Precomputed product of LHS powers.
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of Cf w.r.t all concentrations on the LHS.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                        calcdCfdcjLargeC(const gScalar Cf, const Span& c) const;

    /// @brief Given precomputed product of the RHS powers, computes the
    /// derivatives of the product w.r.t all concentrations on the RHS. This
    /// version requires no pow() calls but may overflow for small
    /// concentrations.
    /// @param Cr Precomputed product of RHS powers.
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of Cr w.r.t all concentrations on the RHS.
    template <class Span>
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                        calcdCrdcjLargeC(const gScalar Cr, const Span& c) const;

    /// @brief Computes and assigns the jacobian contributions from the
    /// derivative of the LHS concentration products.
    /// @param kf The forward rate constant.
    /// @param Cf The product of the LHS concentration powers.
    /// @param lhsPow The LHS concentration powers.
    /// @param c A span of concentrations in a cell.
    /// @param ddNdtByVdcTp A two dimensional span where the result is assigned.
    template <class Span, class TwoDSpan>
    inline CUDA_HOSTDEV void jac_dCfdcj_contribution(
        const gScalar                                  kf,
        const gScalar                                  Cf,
        const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow,
        const Span&                                    c,
        TwoDSpan&                                      ddNdtByVdcTp) const;

    /// @brief Computes and assigns the jacobian contributions from the
    /// derivative of the RHS concentration products.
    /// @param kr The reverse rate constant.
    /// @param Cr The product of the RHS concentration powers.
    /// @param rhsPow The RHS concentration powers.
    /// @param c A span of concentrations in a cell.
    /// @param ddNdtByVdcTp A two dimensional span where the result is assigned.
    template <class Span, class TwoDSpan>
    inline CUDA_HOSTDEV void jac_dCrdcj_contribution(
        const gScalar                                  kr,
        const gScalar                                  Cr,
        const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow,
        const Span&                                    c,
        TwoDSpan&                                      ddNdtByVdcTp) const;

    /// @brief Computes and assigns the jacobian contributions from the
    /// derivative of the rate constants w.r.t temperature.
    /// @param p Pressure
    /// @param T Temperature
    /// @param Cf The product of the LHS concentration powers.
    /// @param Cr The product of the RHS concentration powers.
    /// @param kr The reverse rate constant.
    /// @param Tsi Starting index of temperature in the solution array (nSpecie)
    /// @param c A span of concentrations in a cell.
    /// @param ddNdtByVdcTp A two dimensional span where the result is assigned.
    template <class Span, class TwoDSpan>
    inline CUDA_HOSTDEV void
    jac_dCdT_contribution(const gScalar   p,
                          const gScalar   T,
                          const gScalar   Cf,
                          const gScalar   Cr,
                          const gScalar   kr,
                          const gLabel    Tsi,
                          const Span&     c,
                          const TwoDSpan& ddNdtByVdcTp) const;

    /// @brief Computes and assigns the jacobian contributions from the
    /// derivative of the rate constants w.r.t concentration.
    /// @param p Pressure
    /// @param T Temperature
    /// @param Cf The product of the LHS concentration powers.
    /// @param Cr The product of the RHS concentration powers.
    /// @param kr The reverse rate constant.
    /// @param c A span of concentrations in a cell.
    /// @param cTpWork0 A a span of at least size nSpecie used as a temporary.
    /// @param cTpWork1 A a span of at least size nSpecie used as a temporary.
    /// @param ddNdtByVdcTp A two dimensional span where the result is assigned.
    template <class Span1, class Span2, class Span3, class TwoDSpan>
    inline CUDA_HOSTDEV void
    jac_dCdC_contribution(const gScalar p,
                          const gScalar T,
                          const gScalar Cf,
                          const gScalar Cr,
                          const gScalar kr,
                          const Span1&  c,
                          Span2&        cTpWork0,
                          Span3&        cTpWork1,
                          TwoDSpan&     ddNdtByVdcTp) const;
};

//////////////////////////////////////////////////////////////////////////////////////////////

template <class Span>
inline CUDA_HOSTDEV bool
gpuReaction::lhsSmallConcentration(const Span& c) const {
    const auto llhs = lhs();
    for (size_t i = 0; i < llhs.size(); ++i) {
        const auto& si = llhs[i].index;
        if (c[si] < gpuSmall) { return true; }
    }
    return false;
}

template <class Span>
inline CUDA_HOSTDEV bool
gpuReaction::rhsSmallConcentration(const Span& c) const {
    const auto rrhs = rhs();
    for (size_t i = 0; i < rrhs.size(); ++i) {
        const auto& si = rrhs[i].index;
        if (c[si] < gpuSmall) { return true; }
    }
    return false;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                    gpuReaction::lhsPowers(const Span& c) const {

    const auto                              llhs = lhs();
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};
    if (!lhsSmallExp()) {
        for (size_t i = 0; i < llhs.size(); ++i) {
            const auto& el = llhs[i].exponent;
            const auto& si = llhs[i].index;
            ret[i]         = pow(c[si], el);
        }
    }
    return ret;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                    gpuReaction::rhsPowers(const Span& c) const {

    const auto                              rrhs = rhs();
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};
    if (!rhsSmallExp()) {
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const auto& er = rrhs[i].exponent;
            const auto& si = rrhs[i].index;
            ret[i]         = pow(c[si], er);
        }
    }
    return ret;
}

inline CUDA_HOSTDEV gScalar gpuReaction::calcCf(
    const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow) const {
    gScalar Cf = 1.0;
    for (gLabel i = 0; i < lhs_size_; ++i) { Cf *= lhsPow[i]; }
    return Cf;
}

inline CUDA_HOSTDEV gScalar gpuReaction::calcCr(
    const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow) const {
    gScalar Cr = 1.0;
    for (gLabel i = 0; i < rhs_size_; ++i) { Cr *= rhsPow[i]; }
    return Cr;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::calcdCfdcj(const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow,
                        const Span& c) const {

    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    const auto llhs = lhs();
    for (size_t j = 0; j < llhs.size(); ++j) {

        gScalar dCfdcj = 1;
        for (size_t i = 0; i < llhs.size(); ++i) {
            const auto& el = llhs[i].exponent;
            const auto& si = llhs[i].index;
            if (i == j) {
                dCfdcj *= el * pow(c[si], el - gScalar(1));
            } else {
                dCfdcj *= lhsPow[i];
            }
        }
        ret[j] = dCfdcj;
    }

    return ret;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::calcdCrdcj(const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow,
                        const Span& c) const {
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    const auto rrhs = rhs();
    for (size_t j = 0; j < rrhs.size(); ++j) {

        gScalar dCrdcj = 1;
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const auto& er = rrhs[i].exponent;
            const auto& si = rrhs[i].index;
            if (i == j) {
                dCrdcj *= er * pow(c[si], er - gScalar(1));
            } else {
                dCrdcj *= rhsPow[i];
            }
        }
        ret[j] = dCrdcj;
    }

    return ret;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::calcdCfdcjLargeC(const gScalar Cf, const Span& c) const {

    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    const auto llhs = lhs();
    for (size_t j = 0; j < llhs.size(); ++j) {

        const auto& el = llhs[j].exponent;
        const auto& si = llhs[j].index;

        ret[j] = el * Cf / c[si]; // overflow for small c
    }

    return ret;
}

template <class Span>
inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::calcdCrdcjLargeC(const gScalar Cr, const Span& c) const {

    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    const auto rrhs = rhs();
    for (size_t j = 0; j < rrhs.size(); ++j) {

        const auto& er = rrhs[j].exponent;
        const auto& si = rrhs[j].index;
        ret[j]         = er * Cr / c[si]; // overflow for small c
    }

    return ret;
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::omega(const gScalar p,
                                               const gScalar T,
                                               const Span&   c) const {

    const gScalar clippedT = min(max(T, this->Tlow()), this->Thigh());

    // Rate constants
    const gLabel  li = 0;
    const gScalar kf = this->kf(p, clippedT, c, li);
    const gScalar kr = this->kr(kf, p, clippedT, c, li);

    const gScalar Cf = calcCf(lhsPowers(c));
    const gScalar Cr = calcCr(rhsPowers(c));

    return kf * Cf - kr * Cr;
}

template <class Span>
inline CUDA_HOSTDEV void gpuReaction::dNdtByV(const gScalar omega,
                                              Span&         dNdtByV) const {
    const auto llhs = lhs();

    for (size_t i = 0; i < llhs.size(); ++i) {
        const gLabel  si = llhs[i].index;
        const gScalar sl = llhs[i].stoichCoeff;
        dNdtByV[si] -= sl * omega;
    }

    const auto rrhs = rhs();

    for (size_t i = 0; i < rrhs.size(); ++i) {
        const gLabel  si = rrhs[i].index;
        const gScalar sr = rrhs[i].stoichCoeff;
        dNdtByV[si] += sr * omega;
    }
}

template <class Span1, class Span2>
inline CUDA_HOSTDEV void gpuReaction::dNdtByV(const gScalar p,
                                              const gScalar T,
                                              const Span1&  c,
                                              Span2&        dNdtByV) const {
    const gScalar omega = this->omega(p, T, c);

    this->dNdtByV(omega, dNdtByV);
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::kf(const gScalar p,
                                            const gScalar T,
                                            const Span&   c,
                                            const gLabel  li) const {
    // return 43.0;
    (void)li;
    return k_(p, T, c, li);
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::kr(const gScalar kfwd,
                                            const gScalar p,
                                            const gScalar T,
                                            const Span&   c,
                                            const gLabel  li) const {

    (void)li;
    if (isIrreversible()) { return 0.0; }
    const double rSmall = sqrt(gpuSmall);
    return kfwd / max(this->Kc(p, T), rSmall);
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::kr(const gScalar p,
                                            const gScalar T,
                                            const Span&   c,
                                            const gLabel  li) const {
    (void)li;
    if (isIrreversible()) { return 0.0; }
    return kr(kf(p, T, c, li), p, T, c, li);
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::dkfdT(const gScalar p,
                                               const gScalar T,
                                               const Span&   c,
                                               const gLabel  li) const {
    (void)li;
    return k_.ddT(p, T, c, li);
}

template <class Span>
inline CUDA_HOSTDEV gScalar gpuReaction::dkrdT(const gScalar p,
                                               const gScalar T,
                                               const Span&   c,
                                               const gLabel  li,
                                               const gScalar dkfdT,
                                               const gScalar kr) const {

    (void)li;
    if (isIrreversible()) { return 0.0; }
    const double  rSmall = sqrt(gpuSmall);
    const gScalar Kc     = max(this->Kc(p, T), rSmall);

    return dkfdT / Kc - (Kc > rSmall ? kr * this->dKcdTbyKc(p, T) : 0.0);
}

template <class Span1, class Span2>
inline CUDA_HOSTDEV void gpuReaction::dkfdc(const gScalar p,
                                            const gScalar T,
                                            const Span1&  c,
                                            const gLabel  li,
                                            Span2&        dkfdc) const {
    (void)li;
    k_.ddc(p, T, c, li, dkfdc);
}

template <class Span1, class Span2, class Span3>
inline CUDA_HOSTDEV void gpuReaction::dkrdc(const gScalar p,
                                            const gScalar T,
                                            const Span1&  c,
                                            const gLabel  li,
                                            const Span2&  dkfdc,
                                            const gScalar kr,
                                            Span3&        dkrdc) const {

    (void)li;
    if (isIrreversible()) {

        for (gLabel i = 0; i < gLabel(std::size(dkrdc)); ++i) {
            dkrdc[i] = 0.0;
        }
    } else {
        const gScalar rSmall = sqrt(gpuSmall);
        const gScalar Kc     = max(this->Kc(p, T), rSmall);

        for (gLabel i = 0; i < gLabel(std::size(dkrdc)); ++i) {
            dkrdc[i] = dkfdc[i] / Kc;
        }
    }
}

template <class Span, class TwoDSpan>
inline CUDA_HOSTDEV void gpuReaction::jac_dCfdcj_contribution(
    const gScalar                                  kf,
    const gScalar                                  Cf,
    const std::array<gScalar, MAX_REACTION_WIDTH>& lhsPow,
    const Span&                                    c,
    TwoDSpan&                                      ddNdtByVdcTp) const {

    std::array<gScalar, MAX_REACTION_WIDTH> dCfdjs;
    if (!lhsSmallConcentration(c)) {
        dCfdjs = calcdCfdcjLargeC(Cf, c);
    } else {
        dCfdjs = calcdCfdcj(lhsPow, c);
    }

    const auto llhs = lhs();
    const auto rrhs = rhs();

    for (size_t j = 0; j < llhs.size(); ++j) {
        const gScalar dCfdcj = dCfdjs[j];

        const gLabel sj = llhs[j].index;
        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) -= sl * kf * dCfdcj;
        }
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) += sr * kf * dCfdcj;
        }
    }
}

template <class Span, class TwoDSpan>
inline CUDA_HOSTDEV void gpuReaction::jac_dCrdcj_contribution(
    const gScalar                                  kr,
    const gScalar                                  Cr,
    const std::array<gScalar, MAX_REACTION_WIDTH>& rhsPow,
    const Span&                                    c,
    TwoDSpan&                                      ddNdtByVdcTp) const {

    std::array<gScalar, MAX_REACTION_WIDTH> dCrdjs;
    if (!rhsSmallConcentration(c)) {
        dCrdjs = calcdCrdcjLargeC(Cr, c);
    } else {
        dCrdjs = calcdCrdcj(rhsPow, c);
    }

    const auto llhs = lhs();
    const auto rrhs = rhs();

    for (size_t j = 0; j < rrhs.size(); ++j) {
        const gScalar dCrcj = dCrdjs[j];

        const gLabel sj = rrhs[j].index;
        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) += sl * kr * dCrcj;
        }
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) -= sr * kr * dCrcj;
        }
    }
}

template <class Span, class TwoDSpan>
inline CUDA_HOSTDEV void
gpuReaction::jac_dCdT_contribution(const gScalar   p,
                                   const gScalar   T,
                                   const gScalar   Cf,
                                   const gScalar   Cr,
                                   const gScalar   kr,
                                   const gLabel    Tsi,
                                   const Span&     c,
                                   const TwoDSpan& ddNdtByVdcTp) const {

    const gScalar dkfdT = this->dkfdT(p, T, c, 0);
    const gScalar dkrdT = this->dkrdT(p, T, c, 0, dkfdT, kr);

    const gScalar dwdT = dkfdT * Cf - dkrdT * Cr;

    const auto llhs = lhs();
    for (size_t i = 0; i < llhs.size(); ++i) {
        const gLabel  si = llhs[i].index;
        const gScalar sl = llhs[i].stoichCoeff;
        ddNdtByVdcTp(si, Tsi) -= sl * dwdT;
    }
    const auto rrhs = rhs();
    for (size_t i = 0; i < rrhs.size(); ++i) {
        const gLabel  si = rrhs[i].index;
        const gScalar sr = rrhs[i].stoichCoeff;
        ddNdtByVdcTp(si, Tsi) += sr * dwdT;
    }
}

template <class Span1, class Span2, class Span3, class TwoDSpan>
inline CUDA_HOSTDEV void
gpuReaction::jac_dCdC_contribution(const gScalar p,
                                   const gScalar T,
                                   const gScalar Cf,
                                   const gScalar Cr,
                                   const gScalar kr,
                                   const Span1&  c,
                                   Span2&        cTpWork0,
                                   Span3&        cTpWork1,
                                   TwoDSpan&     ddNdtByVdcTp) const {
    if (hasDkdc()) {
        auto& dkfdc = cTpWork0;
        auto& dkrdc = cTpWork1;

        this->dkfdc(p, T, c, 0, dkfdc);
        this->dkrdc(p, T, c, 0, dkfdc, kr, dkrdc);

        // Do the compute here once and reuse the first work array
        for (gLabel j = 0; j < gLabel(std::size(c)); ++j) {
            dkfdc[j] = dkfdc[j] * Cf - dkrdc[j] * Cr;
        }

        const auto llhs = lhs();

        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;

            for (gLabel j = 0; j < gLabel(std::size(c)); ++j) {
                ddNdtByVdcTp(si, j) -= sl * dkfdc[j];
            }
        }

        const auto rrhs = rhs();
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            for (gLabel j = 0; j < gLabel(std::size(c)); ++j) {
                ddNdtByVdcTp(si, j) += sr * dkfdc[j];
            }
        }
    }
}

template <class Span1, class Span2, class TwoDSpan, class Span3, class Span4>
inline CUDA_HOSTDEV void gpuReaction::ddNdtByVdcTp(const gScalar p,
                                                   const gScalar T,
                                                   const Span1&  c,
                                                   const gLabel  li,
                                                   Span2&        dNdtByV,
                                                   TwoDSpan&     ddNdtByVdcTp,
                                                   const gLabel  Nsi0,
                                                   const gLabel  Tsi,
                                                   Span3&        cTpWork0,
                                                   Span4& cTpWork1) const {

    (void)li;

    const auto lhsPow = lhsPowers(c);
    const auto rhsPow = rhsPowers(c);

    const gScalar Cf = calcCf(lhsPow);
    const gScalar Cr = calcCr(rhsPow);

    // Rate constants
    const gScalar kf = this->kf(p, T, c, li);
    const gScalar kr = this->kr(kf, p, T, c, li);

    // Overall reaction rate
    const gScalar omega = kf * Cf - kr * Cr;

    // Specie reaction rates

    // Derivative contribuiton
    this->dNdtByV(omega, dNdtByV);

    // Jacobian contributions from the derivative of the concentration
    // products w.r.t. concentration

    jac_dCfdcj_contribution(kf, Cf, lhsPow, c, ddNdtByVdcTp);
    jac_dCrdcj_contribution(kr, Cr, rhsPow, c, ddNdtByVdcTp);

    jac_dCdT_contribution(p, T, Cf, Cr, kr, Tsi, c, ddNdtByVdcTp);

    jac_dCdC_contribution(
        p, T, Cf, Cr, kr, c, cTpWork0, cTpWork1, ddNdtByVdcTp);
}

} // namespace FoamGpu
