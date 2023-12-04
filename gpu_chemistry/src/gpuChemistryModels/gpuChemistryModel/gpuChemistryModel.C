#include "addToRunTimeSelectionTable.H"
#include "gpuChemistryModel.H"
#include "noChemistrySolver.H"
namespace Foam {

// Adding new models to runtime selection is horrible,
// the model / solver combination is chosen based on the
//  'runTimeName', the syntax of which depends on the gpuThermoType name.
using gpuModelType  = gpuChemistryModel;
using gpuSolverType = noChemistrySolver<gpuModelType>;
using gpuThermoType = typename gpuModelType::cpuThermoType;

defineTypeNameAndDebug(gpuModelType, 0);

static const word runTimeName = word(gpuSolverType::typeName_()) + "<" +
                                word(gpuModelType::typeName_()) + "<" +
                                gpuThermoType::typeName() + ">>";

defineTemplateTypeNameAndDebugWithName(gpuSolverType, runTimeName.c_str(), 0);

addToRunTimeSelectionTable(basicChemistryModel, gpuSolverType, thermo);

} // namespace Foam

namespace Foam {

std::vector<gScalar> gpuChemistryModel::getRho0() const {
    const volScalarField& rho = this->mesh()
                                    .template lookupObject<volScalarField>(
                                        this->thermo().phasePropertyName("rho"))
                                    .oldTime();
    return std::vector<gScalar>(rho.begin(), rho.end());
}

std::vector<gScalar> gpuChemistryModel::getP0() const {
    const volScalarField& p = this->thermo().p().oldTime();
    return std::vector<gScalar>(p.begin(), p.end());
}

std::vector<gScalar> gpuChemistryModel::getT0() const {
    const volScalarField& T = this->thermo().T().oldTime();
    return std::vector<gScalar>(T.begin(), T.end());
}

std::vector<gScalar> gpuChemistryModel::getDeltaTChem() const {
    return std::vector<gScalar>(deltaTChem_.begin(), deltaTChem_.end());
}

std::vector<gScalar> gpuChemistryModel::getY0() const {
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

FoamGpu::GpuKernelEvaluator gpuChemistryModel::makeEvaluator(
    label                                       nCells,
    label                                       nEqns,
    label                                       nSpecie,
    const ReactionList<cpuThermoType>&          reactions,
    const dictionary&                           physicalProperties,
    const dictionary&                           chemistryProperties,
    const multicomponentMixture<cpuThermoType>& mixture) {

    auto gpu_thermos =
        FoamGpu::makeGpuThermos(mixture.specieThermos(), physicalProperties);
    auto gpu_reactions = FoamGpu::makeGpuReactions(
        mixture.specieNames(), chemistryProperties, gpu_thermos, reactions);

    return FoamGpu::GpuKernelEvaluator(
        nCells,
        nEqns,
        nSpecie,
        gpu_thermos,
        gpu_reactions,
        FoamGpu::read_gpuODESolverInputs(
            chemistryProperties.subDict("odeCoeffs")));
}

gpuChemistryModel::gpuChemistryModel(const fluidMulticomponentThermo& thermo)
    : basicChemistryModel(thermo)
    , mixture_(dynamicCast<const multicomponentMixture<cpuThermoType>>(
          this->thermo()))
    , specieThermos_(mixture_.specieThermos())
    , reactions_(mixture_.specieNames(), specieThermos_, this->mesh(), *this)
    , RR_(this->thermo().Y().size())
    , evaluator_(makeEvaluator(
          nCells(),
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

label gpuChemistryModel::nEqns() const {
    // nEqns = number of species + temperature + pressure
    return this->nSpecie() + 2;
}

label gpuChemistryModel::nCells() const { return this->thermo().T().size(); }

label gpuChemistryModel::nSpecie() const { return this->thermo().Y().size(); }

label gpuChemistryModel::nReaction() const { return reactions_.size(); }

const PtrList<volScalarField::Internal>& gpuChemistryModel::RR() const {
    return RR_;
}

void gpuChemistryModel::calculate() {
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

PtrList<DimensionedField<scalar, volMesh>>
gpuChemistryModel::reactionRR(const label reactioni) const {
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

scalar gpuChemistryModel::solve(const scalar deltaT) {
    // Don't allow the time-step to change more than a factor of 2
    return min(
        // this->doSolve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        this->computeRRAndChemDeltaT(deltaT),
        2 * deltaT);
}

scalar gpuChemistryModel::solve(const scalarField& deltaT) {
    throw std::logic_error("LTS time step not yet supported");
    // return this->doSolve<scalarField>(deltaT);
    return 0;
}

scalar gpuChemistryModel::computeRRAndChemDeltaT(const scalar& deltaT) {

    if (!this->chemistry_) { return great; }

    auto [RR, deltaTChem, minDeltaT] = evaluator_.computeRR(
        deltaT, deltaTChemMax_, getRho0(), getDeltaTChem(), getY0());

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

tmp<volScalarField> gpuChemistryModel::tc() const {
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
            const Reaction<cpuThermoType>& R = reactions_[i];
            scalar                         omegaf, omegar;
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

tmp<volScalarField> gpuChemistryModel::Qdot() const {
    tmp<volScalarField> tQdot(volScalarField::New(
        "Qdot",
        this->mesh_,
        dimensionedScalar(dimEnergy / dimVolume / dimTime, 0)));

    if (!this->chemistry_) { return tQdot; }

    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    scalarField& Qdot = tQdot.ref();

    forAll(Yvf, i) {
        forAll(Qdot, celli) {
            const scalar hi = specieThermos_[i].hf();
            Qdot[celli] -= hi * RR_[i][celli];
        }
    }

    return tQdot;
}

} // namespace Foam
