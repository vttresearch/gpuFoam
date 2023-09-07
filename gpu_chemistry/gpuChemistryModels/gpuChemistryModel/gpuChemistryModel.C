

#include "UniformField.H"
#include "addToRunTimeSelectionTable.H"
#include "clockTime.H"
#include "localEulerDdtScheme.H"

#include <algorithm>
#include <vector>

template <class ThermoType>
void Foam::gpuChemistryModel<ThermoType>::calculate() {
    if (!this->chemistry_) { return; }

    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    scalarField c(this->nSpecie());

    List<label> cTos; // Note! not allocated!

    forAll(rhovf, celli) {
        const scalar rho = rhovf[celli];
        const scalar T   = Tvf[celli];
        const scalar p   = pvf[celli];

        for (label i = 0; i < this->nSpecie(); i++) {
            const scalar Yi = Yvf[i][celli];
            c[i]            = rho * Yi / specieThermos_[i].W();
        }

        dNdtByV = Zero;

        forAll(reactions_, ri) {
            reactions_[ri].dNdtByV(p, T, c, celli, dNdtByV, false, cTos, 0);
        }

        for (label i = 0; i < this->nSpecie(); i++) {
            RR_[i][celli] = dNdtByV[i] * specieThermos_[i].W();
        }
    }
}

template <class ThermoType>
Foam::PtrList<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::gpuChemistryModel<ThermoType>::reactionRR(const label reactioni) const {
    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    PtrList<volScalarField::Internal> RR(this->nSpecie());
    for (label i = 0; i < this->nSpecie(); i++) {
        RR.set(i,
               volScalarField::Internal::New(
                   "RR." + Yvf[i].name(),
                   this->mesh(),
                   dimensionedScalar(dimMass / dimVolume / dimTime, 0))
                   .ptr());
    }

    if (!this->chemistry_) { return RR; }

    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);

    scalarField c(this->nSpecie());

    List<label> cTos; // Note! not allocated!

    const auto& R = reactions_[reactioni];

    forAll(rhovf, celli) {
        const scalar rho = rhovf[celli];
        const scalar T   = Tvf[celli];
        const scalar p   = pvf[celli];

        for (label i = 0; i < this->nSpecie(); i++) {
            const scalar Yi = Yvf[i][celli];
            c[i]            = rho * Yi / specieThermos_[i].W();
        }

        dNdtByV = Zero;

        R.dNdtByV(p, T, c, celli, dNdtByV, false, cTos, 0);

        for (label i = 0; i < this->nSpecie(); i++) {
            RR[i][celli] = dNdtByV[i] * specieThermos_[i].W();
        }
    }

    return RR;
}

template <class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solve(const scalar deltaT) {
    // Don't allow the time-step to change more than a factor of 2
    return min(
        // this->doSolve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        this->doSolve(deltaT),
        2 * deltaT);
}

template <class ThermoType>
Foam::scalar
Foam::gpuChemistryModel<ThermoType>::solve(const scalarField& deltaT) {
    throw std::logic_error("LTS time step not yet supported");
    // return this->doSolve<scalarField>(deltaT);
    return 0;
}

template <class ThermoType>
Foam::scalar
Foam::gpuChemistryModel<ThermoType>::doSolve(const scalar& deltaT) {

    if (!this->chemistry_) { return great; }

    auto [RR, deltaTChem, minDeltaT] = evaluator_.computeRR(deltaT,
                                                            deltaTChemMax_,
                                                            device_rho0(),
                                                            device_p0(),
                                                            device_T0(),
                                                            device_deltaTChem(),
                                                            device_Y0());

    auto RRs = make_mdspan(RR, extents<2>{nCells(), nSpecie()});

    for (label celli = 0; celli < nCells(); ++celli) {
        for (label i = 0; i < this->nSpecie(); ++i) {
            RR_[i][celli] = RRs(celli, i);
        }
    }

    for (label celli = 0; celli < nCells(); ++celli) {
        deltaTChem_[celli] = deltaTChem[celli];
    }

    return minDeltaT;
}

template <class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::gpuChemistryModel<ThermoType>::tc() const {
    tmp<volScalarField> ttc(volScalarField::New(
        "tc",
        this->mesh(),
        dimensionedScalar(dimTime, small),
        extrapolatedCalculatedFvPatchScalarField::typeName));
    scalarField&        tc = ttc.ref();

    if (!this->chemistry_) {
        ttc.ref().correctBoundaryConditions();
        return ttc;
    }

    tmp<volScalarField>   trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    scalarField c(this->nSpecie());

    List<label> cTos; // Note! not allocated!

    forAll(rhovf, celli) {
        const scalar rho = rhovf[celli];
        const scalar T   = Tvf[celli];
        const scalar p   = pvf[celli];

        for (label i = 0; i < this->nSpecie(); i++) {
            c[i] = rho * Yvf[i][celli] / specieThermos_[i].W();
        }

        // A reaction's rate scale is calculated as it's molar
        // production rate divided by the total number of moles in the
        // system.
        //
        // The system rate scale is the average of the reactions' rate
        // scales weighted by the reactions' molar production rates. This
        // weighting ensures that dominant reactions provide the largest
        // contribution to the system rate scale.
        //
        // The system time scale is then the reciprocal of the system rate
        // scale.
        //
        // Contributions from forward and reverse reaction rates are
        // handled independently and identically so that reversible
        // reactions produce the same result as the equivalent pair of
        // irreversible reactions.

        scalar sumW = 0, sumWRateByCTot = 0;
        forAll(reactions_, i) {
            const Reaction<ThermoType>& R = reactions_[i];
            scalar                      omegaf, omegar;
            R.omega(p, T, c, celli, omegaf, omegar);

            scalar wf = 0;
            forAll(R.rhs(), s) { wf += R.rhs()[s].stoichCoeff * omegaf; }
            sumW += wf;
            sumWRateByCTot += sqr(wf);

            scalar wr = 0;
            forAll(R.lhs(), s) { wr += R.lhs()[s].stoichCoeff * omegar; }
            sumW += wr;
            sumWRateByCTot += sqr(wr);
        }

        tc[celli] =
            sumWRateByCTot == 0 ? vGreat : sumW / sumWRateByCTot * sum(c);
    }

    ttc.ref().correctBoundaryConditions();
    return ttc;
}

template <class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::gpuChemistryModel<ThermoType>::Qdot() const {
    tmp<volScalarField> tQdot(volScalarField::New(
        "Qdot",
        this->mesh_,
        dimensionedScalar(dimEnergy / dimVolume / dimTime, 0)));

    if (!this->chemistry_) { return tQdot; }

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    scalarField& Qdot = tQdot.ref();

    forAll(Yvf, i) {
        forAll(Qdot, celli) {
            const scalar hi = specieThermos_[i].Hf();
            Qdot[celli] -= hi * RR_[i][celli];
        }
    }

    return tQdot;
}

// ************************************************************************* //
