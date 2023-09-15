


namespace Foam {

template <class ThermoType>
std::vector<gScalar> gpuChemistryModel<ThermoType>::getRho0() const {
    const volScalarField& rho = this->mesh()
                                    .template lookupObject<volScalarField>(
                                        this->thermo().phasePropertyName("rho"))
                                    .oldTime();
    return std::vector<gScalar>(rho.begin(), rho.end());
}

template <class ThermoType>
std::vector<gScalar> gpuChemistryModel<ThermoType>::getP0() const {
    const volScalarField& p = this->thermo().p().oldTime();
    return std::vector<gScalar>(p.begin(), p.end());
}

template <class ThermoType>
std::vector<gScalar> gpuChemistryModel<ThermoType>::getT0() const {
    const volScalarField& T = this->thermo().T().oldTime();
    return std::vector<gScalar>(T.begin(), T.end());
}

template <class ThermoType>
std::vector<gScalar> gpuChemistryModel<ThermoType>::getDeltaTChem() const {
    return std::vector<gScalar>(deltaTChem_.begin(), deltaTChem_.end());
}

template <class ThermoType>
std::vector<gScalar> gpuChemistryModel<ThermoType>::getY0() const {
    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    const volScalarField& p = this->thermo().p().oldTime();
    const volScalarField& T = this->thermo().T().oldTime();

    std::vector<gScalar> Y0(nCells() * nEqns());

    auto s = make_mdspan(Y0, extents<2>{nCells(), nEqns()});

    for (label celli = 0; celli < nCells(); ++celli) {
        for (label i = 0; i < nSpecie(); ++i) {
            s(celli, i) = Yvf[i].oldTime()[celli];
        }
        s(celli, nSpecie())     = T[celli];
        s(celli, nSpecie() + 1) = p[celli];
    }

    return Y0;
}

template <class ThermoType>
FoamGpu::GpuKernelEvaluator gpuChemistryModel<ThermoType>::makeEvaluator(
    label                                    nEqns,
    label                                    nSpecie,
    const ReactionList<ThermoType>&          reactions,
    const dictionary&                        physicalProperties,
    const dictionary&                        chemistryProperties,
    const multicomponentMixture<ThermoType>& mixture) {

    auto gpu_thermos =
        FoamGpu::makeGpuThermos(mixture.specieThermos(), physicalProperties);
    auto gpu_reactions = FoamGpu::makeGpuReactions(
        mixture.specieNames(), chemistryProperties, gpu_thermos, reactions);

    return FoamGpu::GpuKernelEvaluator(
        nEqns,
        nSpecie,
        gpu_thermos,
        gpu_reactions,
        FoamGpu::read_gpuODESolverInputs(
            chemistryProperties.subDict("odeCoeffs")));
}

template <class ThermoType>
gpuChemistryModel<ThermoType>::gpuChemistryModel(
    const fluidMulticomponentThermo& thermo)
    : basicChemistryModel(thermo)
    , mixture_(
          dynamicCast<const multicomponentMixture<ThermoType>>(this->thermo()))
    , specieThermos_(mixture_.specieThermos())
    , reactions_(mixture_.specieNames(), specieThermos_, this->mesh(), *this)
    , RR_(this->thermo().Y().size())
    , evaluator_(makeEvaluator(
          nEqns(),
          nSpecie(),
          reactions_,
          this->mesh().lookupObject<dictionary>("physicalProperties"),
          *this,
          mixture_)) {

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    // Create the fields for the chemistry sources
    forAll(RR_, fieldi) {
        RR_.set(fieldi,
                new volScalarField::Internal(
                    IOobject("RR." + Yvf[fieldi].name(),
                             this->mesh().time().timeName(),
                             this->mesh(),
                             IOobject::NO_READ,
                             IOobject::NO_WRITE),
                    thermo.mesh(),
                    dimensionedScalar(dimMass / dimVolume / dimTime, 0)));
    }
}
template <class ThermoType> label gpuChemistryModel<ThermoType>::nEqns() const {
    // nEqns = number of species + temperature + pressure
    return this->nSpecie() + 2;
}

template <class ThermoType>
label gpuChemistryModel<ThermoType>::nCells() const {
    return this->thermo().T().size();
}

template <class ThermoType>
label gpuChemistryModel<ThermoType>::nSpecie() const {
    return this->thermo().Y().size();
}

template <class ThermoType>
label gpuChemistryModel<ThermoType>::nReaction() const {
    return reactions_.size();
}

template <class ThermoType>
const PtrList<volScalarField::Internal>&
gpuChemistryModel<ThermoType>::RR() const {
    return RR_;
}

template <class ThermoType> void gpuChemistryModel<ThermoType>::calculate() {
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
PtrList<DimensionedField<scalar, volMesh>>
gpuChemistryModel<ThermoType>::reactionRR(const label reactioni) const {
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
scalar gpuChemistryModel<ThermoType>::solve(const scalar deltaT) {
    // Don't allow the time-step to change more than a factor of 2
    return min(
        // this->doSolve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        this->computeRRAndChemDeltaT(deltaT),
        2 * deltaT);
}

template <class ThermoType>
scalar gpuChemistryModel<ThermoType>::solve(const scalarField& deltaT) {
    throw std::logic_error("LTS time step not yet supported");
    // return this->doSolve<scalarField>(deltaT);
    return 0;
}

template <class ThermoType>
scalar
gpuChemistryModel<ThermoType>::computeRRAndChemDeltaT(const scalar& deltaT) {

    if (!this->chemistry_) { return great; }

    auto [RR, deltaTChem, minDeltaT] = evaluator_.computeRR(deltaT,
                                                            deltaTChemMax_,
                                                            getRho0(),
                                                            getDeltaTChem(),
                                                            getY0());

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
tmp<volScalarField> gpuChemistryModel<ThermoType>::tc() const {
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
tmp<volScalarField> gpuChemistryModel<ThermoType>::Qdot() const {
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

} // namespace Foam
