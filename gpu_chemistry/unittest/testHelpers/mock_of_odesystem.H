#pragma once

#include "test_utilities.H"

#include "ODESystem.H"
#include "create_foam_inputs.H"

namespace Foam {

enum jacobianType { fast = 0, exact = 1 };

class MockOFSystem : public ODESystem {

    const PtrList<FoamThermoType>            specieThermos_;
    const ReactionList<FoamThermoType>       reactions_;
    label                                    nSpecie_;
    mutable scalarField                      Y_;
    mutable scalarField                      c_;
    mutable FixedList<scalarField, 5>        YTpWork_;
    mutable FixedList<scalarSquareMatrix, 2> YTpYTpWork_;

    const bool  reduction_ = false;
    List<label> cTos_{};

    //- Type of the Jacobian to be calculated
    const jacobianType jacobianType_ = jacobianType::fast;

public:
    MockOFSystem(TestData::Mechanism m)
        : specieThermos_(TestData::makeCpuThermos(m))
        , reactions_(TestData::makeSpeciesTable(m),
                     specieThermos_,
                     TestData::makeReactionDict(m))
        , nSpecie_(TestData::speciesCount(m))
        , Y_(nSpecie_)
        , c_(nSpecie_)
        , YTpWork_(scalarField(nSpecie_ + 2))
        , YTpYTpWork_(scalarSquareMatrix(nSpecie_ + 2)) {}

    virtual ~MockOFSystem() {}

    label nEqns() const override { return nSpecie_ + 2; }

    void derivatives(const scalar       time,
                     const scalarField& YTp,
                     const label        li,
                     scalarField&       dYTpdt) const override {

        forAll(Y_, i) { Y_[i] = max(YTp[i], 0); }

        const scalar T = YTp[nSpecie_];
        const scalar p = YTp[nSpecie_ + 1];

        // printf("derivative T: %.17g \n", T);
        // printf("derivative p: %.17g \n", p);

        // Evaluate the mixture density
        scalar rhoM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            rhoM += Y_[i] / specieThermos_[i].rho(p, T);
        }
        rhoM = 1 / rhoM;

        // printf("derivative rhoM: %.17g \n", rhoM);

        // Evaluate the concentrations
        for (label i = 0; i < Y_.size(); i++) {
            c_[i] = rhoM / specieThermos_[i].W() * Y_[i];
        }

        // Evaluate contributions from reactions
        dYTpdt = Zero;
        forAll(reactions_, ri) {
            reactions_[ri].dNdtByV(
                p, T, c_, li, dYTpdt, reduction_, cTos_, 0);
        }

        // Reactions return dNdtByV, so we need to convert the result
        // to dYdt
        for (label i = 0; i < nSpecie_; i++) {
            const scalar WiByrhoM =
                specieThermos_[sToc(i)].W() / rhoM;
            scalar& dYidt = dYTpdt[i];
            dYidt *= WiByrhoM;
        }

        // Evaluate the effect on the thermodynamic system ...

        // Evaluate the mixture Cp
        scalar CpM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            CpM += Y_[i] * specieThermos_[i].Cp(p, T);
        }

        // printf("derivative CpM: %.17g \n", CpM);

        // dT/dt
        scalar& dTdt = dYTpdt[nSpecie_];
        for (label i = 0; i < nSpecie_; i++) {
            dTdt -= dYTpdt[i] * specieThermos_[sToc(i)].ha(p, T);
        }
        dTdt /= CpM;

        // printf("derivative dTdt: %.17g \n", dTdt);

        // dp/dt = 0 (pressure is assumed constant)
        scalar& dpdt = dYTpdt[nSpecie_ + 1];
        dpdt         = 0;
    }

    void jacobian(const scalar        t,
                  const scalarField&  YTp,
                  const label         li,
                  scalarField&        dYTpdt,
                  scalarSquareMatrix& J) const override {

        forAll(c_, i) { Y_[i] = max(YTp[i], 0); }

        const scalar T = YTp[nSpecie_];
        const scalar p = YTp[nSpecie_ + 1];

        // printf("jacobian T: %.17g \n", T);
        // printf("jacobian p: %.17g \n", p);

        // Evaluate the specific volumes and mixture density
        scalarField& v = YTpWork_[0];
        for (label i = 0; i < Y_.size(); i++) {
            v[i] = 1 / specieThermos_[i].rho(p, T);
        }
        scalar rhoM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            rhoM += Y_[i] * v[i];
        }
        rhoM = 1 / rhoM;

        // printf("jacobian rhoM: %.17g \n", rhoM);

        // Evaluate the concentrations
        for (label i = 0; i < Y_.size(); i++) {
            c_[i] = rhoM / specieThermos_[i].W() * Y_[i];
        }

        // Evaluate the derivatives of concentration w.r.t. mass
        // fraction
        scalarSquareMatrix& dcdY = YTpYTpWork_[0];
        for (label i = 0; i < nSpecie_; i++) {
            const scalar rhoMByWi =
                rhoM / specieThermos_[sToc(i)].W();
            switch (jacobianType_) {
            case jacobianType::fast: {
                dcdY(i, i) = rhoMByWi;
            } break;
            case jacobianType::exact:
                for (label j = 0; j < nSpecie_; j++) {
                    dcdY(i, j) =
                        rhoMByWi *
                        ((i == j) - rhoM * v[sToc(j)] * Y_[sToc(i)]);
                }
                break;
            }
        }

        // Evaluate the mixture thermal expansion coefficient
        scalar alphavM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            alphavM +=
                Y_[i] * rhoM * v[i] * specieThermos_[i].alphav(p, T);
        }

        // printf("jacobian alphavM: %.17g \n", alphavM);

        // Evaluate contributions from reactions
        dYTpdt = Zero;

        scalarSquareMatrix& ddNdtByVdcTp = YTpYTpWork_[1];
        for (label i = 0; i < nSpecie_ + 2; i++) {
            for (label j = 0; j < nSpecie_ + 2; j++) {
                ddNdtByVdcTp[i][j] = 0;
            }
        }
        forAll(reactions_, ri) {
            reactions_[ri].ddNdtByVdcTp(p,
                                        T,
                                        c_,
                                        li,
                                        dYTpdt,
                                        ddNdtByVdcTp,
                                        reduction_,
                                        cTos_,
                                        0,
                                        nSpecie_,
                                        YTpWork_[1],
                                        YTpWork_[2]);
        }

        // Reactions return dNdtByV, so we need to convert the result
        // to dYdt
        for (label i = 0; i < nSpecie_; i++) {
            const scalar WiByrhoM =
                specieThermos_[sToc(i)].W() / rhoM;
            scalar& dYidt = dYTpdt[i];
            dYidt *= WiByrhoM;

            for (label j = 0; j < nSpecie_; j++) {
                scalar ddNidtByVdYj = 0;
                switch (jacobianType_) {
                case jacobianType::fast: {
                    const scalar ddNidtByVdcj = ddNdtByVdcTp(i, j);
                    ddNidtByVdYj = ddNidtByVdcj * dcdY(j, j);
                } break;
                case jacobianType::exact:
                    for (label k = 0; k < nSpecie_; k++) {
                        const scalar ddNidtByVdck =
                            ddNdtByVdcTp(i, k);
                        ddNidtByVdYj += ddNidtByVdck * dcdY(k, j);
                    }
                    break;
                }

                scalar& ddYidtdYj = J(i, j);
                ddYidtdYj         = WiByrhoM * ddNidtByVdYj +
                            rhoM * v[sToc(j)] * dYidt;
            }

            scalar ddNidtByVdT = ddNdtByVdcTp(i, nSpecie_);
            for (label j = 0; j < nSpecie_; j++) {
                const scalar ddNidtByVdcj = ddNdtByVdcTp(i, j);
                ddNidtByVdT -= ddNidtByVdcj * c_[sToc(j)] * alphavM;
            }

            scalar& ddYidtdT = J(i, nSpecie_);
            ddYidtdT = WiByrhoM * ddNidtByVdT + alphavM * dYidt;

            scalar& ddYidtdp = J(i, nSpecie_ + 1);
            ddYidtdp         = 0;
        }

        // Evaluate the effect on the thermodynamic system ...

        // Evaluate the mixture Cp and its derivative
        scalarField& Cp  = YTpWork_[3];
        scalar       CpM = 0, dCpMdT = 0;
        for (label i = 0; i < Y_.size(); i++) {
            Cp[i] = specieThermos_[i].Cp(p, T);
            CpM += Y_[i] * Cp[i];
            dCpMdT += Y_[i] * specieThermos_[i].dCpdT(p, T);
        }

        // printf("jacobian CpM: %.17g \n", CpM);
        // printf("jacobian dCpMdT: %.17g \n", dCpMdT);

        // dT/dt
        scalarField& Ha   = YTpWork_[4];
        scalar&      dTdt = dYTpdt[nSpecie_];
        for (label i = 0; i < nSpecie_; i++) {
            Ha[sToc(i)] = specieThermos_[sToc(i)].ha(p, T);
            dTdt -= dYTpdt[i] * Ha[sToc(i)];
        }
        dTdt /= CpM;

        // printf("jacobian dTdt: %.17g \n", dTdt);

        // dp/dt = 0 (pressure is assumed constant)
        scalar& dpdt = dYTpdt[nSpecie_ + 1];
        dpdt         = 0;

        // d(dTdt)/dY
        for (label i = 0; i < nSpecie_; i++) {
            scalar& ddTdtdYi = J(nSpecie_, i);
            ddTdtdYi         = 0;
            for (label j = 0; j < nSpecie_; j++) {
                const scalar ddYjdtdYi = J(j, i);
                ddTdtdYi -= ddYjdtdYi * Ha[sToc(j)];
            }
            ddTdtdYi -= Cp[sToc(i)] * dTdt;
            ddTdtdYi /= CpM;
        }

        // d(dTdt)/dT
        scalar& ddTdtdT = J(nSpecie_, nSpecie_);
        ddTdtdT         = 0;
        for (label i = 0; i < nSpecie_; i++) {
            const scalar dYidt    = dYTpdt[i];
            const scalar ddYidtdT = J(i, nSpecie_);
            ddTdtdT -= dYidt * Cp[sToc(i)] + ddYidtdT * Ha[sToc(i)];
        }
        ddTdtdT -= dTdt * dCpMdT;
        ddTdtdT /= CpM;

        // d(dTdt)/dp = 0 (pressure is assumed constant)
        scalar& ddTdtdp = J(nSpecie_, nSpecie_ + 1);
        ddTdtdp         = 0;

        // d(dpdt)/dYiTp = 0 (pressure is assumed constant)
        for (label i = 0; i < nSpecie_ + 2; i++) {
            scalar& ddpdtdYiTp = J(nSpecie_ + 1, i);
            ddpdtdYiTp         = 0;
        }
    }

    inline label sToc(const label si) const { return si; }

    inline label cTos(const label ci) const { return ci; }
};

} // namespace Foam