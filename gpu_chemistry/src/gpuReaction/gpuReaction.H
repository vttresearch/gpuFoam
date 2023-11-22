#pragma once

#include <array>

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuReactionRate.H"
#include "gpuSpecieExponent.H"
#include "gpuThermo.H"
#include "gpu_constants.H"
#include "mdspan.H"



#define MAX_REACTION_WIDTH 10


namespace FoamGpu {

struct reactionParams {


    gScalar omega = 0.0;
    gScalar dwdT   = 0.0;
    std::array<gScalar, MAX_REACTION_WIDTH> dCfdjs{};
    std::array<gScalar, MAX_REACTION_WIDTH> dCrdjs{};
    mdspan<gScalar, 1>                      ddc;
};

struct gpuSpecieCoeffs {

    gpuSpecieCoeffs() = default;

    //- Index of the specie
    gLabel index;

    //- Stoichiometric coefficient
    gScalar stoichCoeff;

    //- Exponent of the specie concentration
    gpuSpecieExponent exponent;
};

/*---------------------------------------------------------------------------*\
                          Class gpuReaction Declaration
\*---------------------------------------------------------------------------*/

class gpuReaction : public gpuThermo {

public:
    using specieCoeffArray =
        std::array<gpuSpecieCoeffs, MAX_REACTION_WIDTH>;

    //- Default temperature limits of applicability of gpuReaction
    // rates
    static constexpr gScalar TlowDefault = 0;
    static constexpr gScalar ThighDefault =
        std::numeric_limits<double>::max() / 10;

    //- LHS specie coefficients
    specieCoeffArray lhs_;

    //- RHS specie coefficients
    specieCoeffArray rhs_;

    //- Number of species on the LHS. This is different than the size
    // of the
    // lhs_ array because of static memory.
    gLabel lhs_size_;

    //- Number of species on the RHS. This is different than the size
    // of the
    // rhs_ array because of static memory.
    gLabel rhs_size_;

    //- The underlying gpuReactionRate object.
    gpuReactionRate k_;

    // Flags if the specie coeffs contain small (< 1)  exponents
    bool small_lhs_exp_;
    bool small_rhs_exp_;

    //- Temperature limits of applicability of gpuReaction rates
    gScalar Tlow_, Thigh_;

    gScalar RSMALL{sqrt(gpuSmall)};

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

    /// @brief Checks if the reaction rate of this reaction is
    /// irreversible.
    /// @return Returns true if the reaction rate of this reaction is
    /// irreversible. Returns false otherwise.
    inline CUDA_HOSTDEV bool isIrreversible() const {
        return k_.isIrreversible();
    }

    /// @brief Convert this reaction to the inherited gpuThermo
    /// object.
    /// @return Inherited gpuThermo object.
    inline gpuThermo& get_thermo() { return *this; }

    /// @brief Returns the lower temperature limit for this reaction.
    /// @return The lower temperature limit.
    inline CUDA_HOSTDEV gScalar Tlow() const { return Tlow_; }

    /// @brief Returns the upper temperature limit for this reaction.
    /// @return The upper temperature limit.
    inline CUDA_HOSTDEV gScalar Thigh() const { return Thigh_; }

    /// @brief Checks if this reaction has concentration dependent
    /// rate constants.
    /// @return True if has concetration dependent rate constants,
    /// false otherwise.
    inline CUDA_HOSTDEV bool hasDkdc() const { return k_.hasDdc(); }

    /// @brief Computes the net reaction rate of this reaction based
    /// on the input pressure, temperature and concentrations.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell.
    /// @return The net reaction rate of this reaction.
    inline CUDA_HOSTDEV gScalar
    omega(const gScalar                  p,
          const gScalar                  T,
          const mdspan<const gScalar, 1> c) const;

    /// @brief Assigns the precomputed net reaction rate for each
    /// specie involved to the input dNdtByV span.
    /// @param omega Precomputed net reaction rate of this reaction.
    /// @param dNdtByV The span of net reaction rates which are
    /// modified by this function.
    inline CUDA_HOSTDEV void
    dNdtByV(const gScalar omega, mdspan<gScalar, 1> dNdtByV) const;

    inline CUDA_HOSTDEV void
    dNdtByV(const reactionParams& r,
            mdspan<gScalar, 1>    dNdtByV) const {

        this->dNdtByV(r.omega, dNdtByV);
    }

    /// @brief Computes and assigns the net reaction rate for each
    /// specie involved to the input dNdtByV span.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell.
    /// @param dNdtByV The span of net reaction rates which are
    /// modified by this function.
    inline CUDA_HOSTDEV void
    dNdtByV(const gScalar                  p,
            const gScalar                  T,
            const mdspan<const gScalar, 1> c,
            mdspan<gScalar, 1>             dNdtByV) const;

    /// @brief Computes the forward reaction rate constant of this
    /// reaction.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @return The forward reaction rate constant of this reaction.
    inline CUDA_HOSTDEV gScalar
    kf(const gScalar                  p,
       const gScalar                  T,
       const mdspan<const gScalar, 1> c) const;

    /// @brief Computes the reverse reaction rate constant of this
    /// reaction.
    /// @param kfwd Forward reaction rate contant.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param Kc Precomputed eqilibrium constant.
    /// @param c A span of concentrations in a cell
    /// @return The reverse reaction rate constant of this reaction.
    inline CUDA_HOSTDEV gScalar
    kr(const gScalar                  kfwd,
       const gScalar                  p,
       const gScalar                  T,
       const gScalar                  Kc,
       const mdspan<const gScalar, 1> c) const;

    /// @brief Computes the reverse reaction rate constant of this
    /// reaction.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @param c A span of concentrations in a cell
    /// @return The reverse reaction rate constant of this reaction.
    inline CUDA_HOSTDEV gScalar
    kr(const gScalar                  p,
       const gScalar                  T,
       const mdspan<const gScalar, 1> c) const;


    inline CUDA_HOSTDEV void
    ddNdtByVdcTp(const gScalar                  p,
                 const gScalar                  T,
                 const mdspan<const gScalar, 1> c,
                 mdspan<gScalar, 2>             ddNdtByVdcTp,
                 const reactionParams&          r) const {

        jac_dCfdcj_contribution(r, ddNdtByVdcTp);

        jac_dCrdcj_contribution(r, ddNdtByVdcTp);

        const gLabel Tsi = static_cast<gLabel>(c.size());

        jac_dCdT_contribution(r, Tsi, ddNdtByVdcTp);

        jac_dCdC_contribution(p, T, r, c, ddNdtByVdcTp);
    }

    /// @brief Checks if the LHS specie coefficients contain small (<
    /// 1) exponents.
    /// @return Returns true if the LHS species of the reaction
    /// contain less than unity exponents. Returns false otherwise.
    inline CUDA_HOSTDEV bool lhsSmallExp() const {
        return small_lhs_exp_;
    }

    /// @brief Checks if the RHS specie coefficients contain small (<
    /// 1) exponents.
    /// @return Returns true if the RHS species of the reaction
    /// contain less than unity exponents. Returns false otherwise.
    inline CUDA_HOSTDEV bool rhsSmallExp() const {
        return small_rhs_exp_;
    }

    /// @brief Checks if any of the species involved in the LHS of
    /// this reaction has small (< gpuSmall) concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return True if any of the concentrations on the LHS of this
    /// reaction have small values. False otherwise.
    inline CUDA_HOSTDEV bool
    lhsSmallConcentration(const mdspan<const gScalar, 1> c) const;

    /// @brief Checks if any of the species involved in the LHS of
    /// this reaction has small (< gpuSmall) concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return True if any of the concentrations on the LHS of this
    /// reaction have small values. False otherwise.
    inline CUDA_HOSTDEV bool
    rhsSmallConcentration(const mdspan<const gScalar, 1> c) const;

    /// @brief Computes the powers speciePow(c_i, e_i) of the LHS
    /// specie concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return An array of powers of the LHS species.
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    lhsPowers(const mdspan<const gScalar, 1> c) const;

    /// @brief Computes the powers speciePow(c_i, e_i) of the RHS
    /// specie concentrations.
    /// @param c A span of concentrations in a cell.
    /// @return An array of powers of the RHS species.
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    rhsPowers(const mdspan<const gScalar, 1> c) const;

    /// @brief Given an array of LHS concentration powers, computes
    /// the product of them all.
    /// @param lhsPow The input LHS powers to compute the product for.
    /// @return The product of the input powers.
    inline CUDA_HOSTDEV gScalar calcCf(
        const std::array<gScalar, MAX_REACTION_WIDTH> lhsPow) const;

    /// @brief Given an array of RHS concentration powers, computes
    /// the product of them all.
    /// @param lhsPow The input RHS powers to compute the product for.
    /// @return The product of the input powers.
    inline CUDA_HOSTDEV gScalar calcCr(
        const std::array<gScalar, MAX_REACTION_WIDTH> rhsPow) const;

    /// @brief Given precomputed LHS powers, computes the derivatives
    /// of the product of the powers w.r.t all concentrations on the
    /// LHS. This version has no limit on the values of c.
    /// @param lhsPow Precomputed concentrations powers.
    /// @param Cf Precomputed product of lhs powers
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of the product of concentration powers
    /// w.r.t to all concentrations on the LHS.
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    calcdCfdcj(const std::array<gScalar, MAX_REACTION_WIDTH> lhsPow,
               const gScalar                                 Cf,
               const mdspan<const gScalar, 1>                c) const;

    /// @brief Given precomputed RHS powers, computes the derivatives
    /// of the product of the powers w.r.t all concentrations on the
    /// RHS. This version has no limit on the values of c.
    /// @param rhsPow Precomputed concentrations powers.
    /// @param Cr Precomputed product of rhs powers
    /// @param c A span of concentrations in a cell.
    /// @return The derivative of the product of concentration powers
    /// w.r.t to all concentrations on the RHS.
    inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
    calcdCrdcj(const std::array<gScalar, MAX_REACTION_WIDTH> rhsPow,
               const gScalar                                 Cr,
               const mdspan<const gScalar, 1>                c) const;

    inline CUDA_HOSTDEV void
    jac_dCfdcj_contribution(const reactionParams& r,
                            mdspan<gScalar, 2>    ddNdtByVdcTp) const;

    inline CUDA_HOSTDEV void
    jac_dCrdcj_contribution(const reactionParams& r,
                            mdspan<gScalar, 2>    ddNdtByVdcTp) const;

    inline CUDA_HOSTDEV void
    jac_dCdT_contribution(const reactionParams& r,
                          const gLabel          Tsi,
                          mdspan<gScalar, 2>    ddNdtByVdcTp) const;

    inline CUDA_HOSTDEV void
    jac_dCdC_contribution(const gScalar         p,
                          const gScalar         T,
                          const reactionParams& r,
                          const mdspan<const gScalar, 1> c,
                         mdspan<gScalar, 2> ddNdtByVdcTp) const;
};

//////////////////////////////////////////////////////////////////////////////////////////////

inline CUDA_HOSTDEV bool gpuReaction::lhsSmallConcentration(
    const mdspan<const gScalar, 1> c) const {
    const auto llhs = lhs();
    for (size_t i = 0; i < llhs.size(); ++i) {
        const auto& si = llhs[i].index;
        if (c[si] < gpuSmall) { return true; }
    }
    return false;
}

inline CUDA_HOSTDEV bool gpuReaction::rhsSmallConcentration(
    const mdspan<const gScalar, 1> c) const {
    const auto rrhs = rhs();
    for (size_t i = 0; i < rrhs.size(); ++i) {
        const auto& si = rrhs[i].index;
        if (c[si] < gpuSmall) { return true; }
    }
    return false;
}

inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::lhsPowers(const mdspan<const gScalar, 1> c) const {

    const auto                              llhs = lhs();
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};
    for (size_t i = 0; i < llhs.size(); ++i) {
        const auto& el = llhs[i].exponent;
        const auto& si = llhs[i].index;
        ret[i]         = speciePow(c[si], el);
    }
    return ret;
}

inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
gpuReaction::rhsPowers(const mdspan<const gScalar, 1> c) const {

    const auto                              rrhs = rhs();
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};
    for (size_t i = 0; i < rrhs.size(); ++i) {
        const auto& er = rrhs[i].exponent;
        const auto& si = rrhs[i].index;
        ret[i]         = speciePow(c[si], er);
    }
    return ret;
}

inline CUDA_HOSTDEV gScalar gpuReaction::calcCf(
    const std::array<gScalar, MAX_REACTION_WIDTH> lhsPow) const {
    gScalar Cf = 1.0;
    for (gLabel i = 0; i < lhs_size_; ++i) { Cf *= lhsPow[i]; }
    return Cf;
}

inline CUDA_HOSTDEV gScalar gpuReaction::calcCr(
    const std::array<gScalar, MAX_REACTION_WIDTH> rhsPow) const {
    gScalar Cr = 1.0;
    for (gLabel i = 0; i < rhs_size_; ++i) { Cr *= rhsPow[i]; }
    return Cr;
}

inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                    gpuReaction::calcdCfdcj(
    const std::array<gScalar, MAX_REACTION_WIDTH> lhsPow,
    const gScalar                                 Cf,
    const mdspan<const gScalar, 1>                c) const {

    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    /*
    //Should be just this but overflows
    const auto llhs = lhs();
        for (size_t j = 0; j < llhs.size(); ++j) {

            const auto& el = llhs[j].exponent;
            const auto& si = llhs[j].index;

            ret[j] = el * Cf / c[si]; // overflow for small c
        }
    */

    (void)Cf;

    const auto llhs = lhs();
    for (size_t j = 0; j < llhs.size(); ++j) {

        gScalar dCfdcj = 1;
        for (size_t i = 0; i < llhs.size(); ++i) {
            const auto& el = llhs[i].exponent;
            const auto& si = llhs[i].index;
            if (i == j) {
                dCfdcj *=
                    el * speciePow(c[si],
                                   el - gpuSpecieExponent(gLabel(1)));
            } else {
                dCfdcj *= lhsPow[i];
            }
        }
        ret[j] = dCfdcj;
    }

    return ret;
}

inline CUDA_HOSTDEV std::array<gScalar, MAX_REACTION_WIDTH>
                    gpuReaction::calcdCrdcj(
    const std::array<gScalar, MAX_REACTION_WIDTH> rhsPow,
    const gScalar                                 Cr,
    const mdspan<const gScalar, 1>                c) const {
    std::array<gScalar, MAX_REACTION_WIDTH> ret{};

    /*
    //Should be just this but overflows
    const auto rrhs = rhs();
    for (size_t j = 0; j < rrhs.size(); ++j) {

        const auto& er = rrhs[j].exponent;
        const auto& si = rrhs[j].index;
        ret[j]         = er * Cr / c[si]; // overflow for small c
    }
    */

    (void)Cr;

    const auto rrhs = rhs();
    for (size_t j = 0; j < rrhs.size(); ++j) {

        gScalar dCrdcj = 1;
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const auto& er = rrhs[i].exponent;
            const auto& si = rrhs[i].index;
            if (i == j) {
                dCrdcj *=
                    er * speciePow(c[si],
                                   er - gpuSpecieExponent(gLabel(1)));
            } else {
                dCrdcj *= rhsPow[i];
            }
        }
        ret[j] = dCrdcj;
    }

    return ret;
}

inline CUDA_HOSTDEV gScalar
gpuReaction::omega(const gScalar                  p,
                   const gScalar                  T,
                   const mdspan<const gScalar, 1> c) const {

    const gScalar clippedT = min(max(T, this->Tlow()), this->Thigh());

    const gScalar Kc = max(RSMALL, this->Kc(p, T));
    // Rate constants
    const gScalar kf = this->kf(p, clippedT, c);
    const gScalar kr = this->kr(kf, p, clippedT, Kc, c);

    const gScalar Cf = calcCf(lhsPowers(c));
    const gScalar Cr = calcCr(rhsPowers(c));

    return kf * Cf - kr * Cr;
}

inline CUDA_HOSTDEV void
gpuReaction::dNdtByV(const gScalar      omega,
                     mdspan<gScalar, 1> dNdtByV) const {
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

inline CUDA_HOSTDEV void
gpuReaction::dNdtByV(const gScalar                  p,
                     const gScalar                  T,
                     const mdspan<const gScalar, 1> c,
                     mdspan<gScalar, 1>             dNdtByV) const {
    const gScalar omega = this->omega(p, T, c);

    this->dNdtByV(omega, dNdtByV);
}

inline CUDA_HOSTDEV gScalar
gpuReaction::kf(const gScalar                  p,
                const gScalar                  T,
                const mdspan<const gScalar, 1> c) const {
    // return 43.0;
    return k_(p, T, c);
}

inline CUDA_HOSTDEV gScalar
gpuReaction::kr(const gScalar                  kfwd,
                const gScalar                  p,
                const gScalar                  T,
                const gScalar                  Kc,
                const mdspan<const gScalar, 1> c) const {

    if (isIrreversible()) { return 0.0; }
    return kfwd / Kc;
}

inline CUDA_HOSTDEV gScalar
gpuReaction::kr(const gScalar                  p,
                const gScalar                  T,
                const mdspan<const gScalar, 1> c) const {
    if (isIrreversible()) { return 0.0; }
    const gScalar Kc = max(this->Kc(p, T), RSMALL);
    return kr(kf(p, T, c), p, T, Kc, c);
}



inline CUDA_HOSTDEV void gpuReaction::jac_dCfdcj_contribution(
    const reactionParams& r, mdspan<gScalar, 2> ddNdtByVdcTp) const {

    const auto llhs = lhs();
    const auto rrhs = rhs();

    for (size_t j = 0; j < llhs.size(); ++j) {

        //const gScalar dCfdcj_kf = r.dCfdjs[j] * r.kf;
        const gLabel  sj        = llhs[j].index;

        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) -= sl * r.dCfdjs[j];
        }
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) += sr * r.dCfdjs[j];
        }
    }
}

inline CUDA_HOSTDEV void gpuReaction::jac_dCrdcj_contribution(
    const reactionParams& r, mdspan<gScalar, 2> ddNdtByVdcTp) const {

    const auto llhs = lhs();
    const auto rrhs = rhs();

    for (size_t j = 0; j < rrhs.size(); ++j) {

        //const gScalar dCrcj_kr = r.dCrdjs[j] * r.kr;
        const gLabel  sj       = rrhs[j].index;

        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) += sl * r.dCrdjs[j];
        }
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            ddNdtByVdcTp(si, sj) -= sr * r.dCrdjs[j];
        }
    }
}

inline CUDA_HOSTDEV void gpuReaction::jac_dCdT_contribution(
    const reactionParams& r,
    const gLabel          Tsi,
    mdspan<gScalar, 2>    ddNdtByVdcTp) const {


    const auto llhs = lhs();
    for (size_t i = 0; i < llhs.size(); ++i) {
        const gLabel  si = llhs[i].index;
        const gScalar sl = llhs[i].stoichCoeff;
        ddNdtByVdcTp(si, Tsi) -= sl * r.dwdT;
    }
    const auto rrhs = rhs();
    for (size_t i = 0; i < rrhs.size(); ++i) {
        const gLabel  si = rrhs[i].index;
        const gScalar sr = rrhs[i].stoichCoeff;
        ddNdtByVdcTp(si, Tsi) += sr * r.dwdT;
    }
}

inline CUDA_HOSTDEV void gpuReaction::jac_dCdC_contribution(
    const gScalar         p,
    const gScalar         T,
    const reactionParams& r,
    const mdspan<const gScalar, 1> c,
    mdspan<gScalar, 2> ddNdtByVdcTp) const {

    if (hasDkdc()) {

        const auto llhs = lhs();

        const gScalar* __restrict__ ddc = &r.ddc[0];

        for (size_t i = 0; i < llhs.size(); ++i) {
            const gLabel  si = llhs[i].index;
            const gScalar sl = llhs[i].stoichCoeff;
            gScalar* __restrict__ ddN_si = &ddNdtByVdcTp(si, 0);

            for (gLabel j = 0; j < gLabel(c.size()); ++j) {
                ddN_si[j] -= sl * ddc[j];
            }
        }

        const auto rrhs = rhs();
        for (size_t i = 0; i < rrhs.size(); ++i) {
            const gLabel  si = rrhs[i].index;
            const gScalar sr = rrhs[i].stoichCoeff;
            gScalar* __restrict__ ddN_si = &ddNdtByVdcTp(si, 0);

            for (gLabel j = 0; j < gLabel(c.size()); ++j) {
                ddN_si[j] += sr * ddc[j];
            }
        }
    }
}

static inline CUDA_HOSTDEV reactionParams
computeReactionParameters(const gpuReaction&             r,
                          const mdspan<const gScalar, 1> c,
                          const gScalar                  p,
                          const gScalar                  T,
                          mdspan<gScalar, 1>             ddc) {
    reactionParams ret;


    const auto [kf, ddT] = r.k_.everything(p, T, c, ddc);


    const auto lhsPow = r.lhsPowers(c);
    const auto Cf     = r.calcCf(lhsPow);
    ret.dCfdjs        = r.calcdCfdcj(lhsPow, Cf, c);
    ret.dwdT          = Cf * ddT;
    ret.omega         = Cf * kf;

    for (auto& e : ret.dCfdjs){
        e *= kf;
    }


    gScalar Cr = 0.0;
    gScalar Kc = 1.0;
    if (!r.isIrreversible()) {

        const auto [Kc_t, dKcdTbyV] = r.KcAnddKcTbyKc(p, T);
        Kc  = max(Kc_t, r.RSMALL);
        const auto kr     = kf / Kc;

        const auto rhsPow = r.rhsPowers(c);
        Cr            = r.calcCr(rhsPow);

        const auto dkrdT = ddT / Kc -
           (Kc > r.RSMALL ? kr * dKcdTbyV : 0.0);

        ret.dwdT          -= Cr * dkrdT;
        ret.omega         -= Cr * kr;
        ret.dCrdjs        = r.calcdCrdcj(rhsPow, Cr, c);


        for (auto& e : ret.dCrdjs){
            e *= kr;
        }

    }


    if (r.hasDkdc()) {
        const gScalar temp = (Cf - Cr/Kc);
        for (gLabel j = 0; j < gLabel(c.size()); ++j) {
            ddc[j] *= temp;
        }

    }

    ret.ddc = ddc;

    return ret;
}

} // namespace FoamGpu
