


#include "UniformField.H"
#include "localEulerDdtScheme.H"
#include "clockTime.H"
#include "addToRunTimeSelectionTable.H"

#include <vector>
#include <algorithm>



template<class ThermoType>
void Foam::gpuChemistryModel<ThermoType>::calculate()
{
    if (!this->chemistry_)
    {
        return;
    }

    tmp<volScalarField> trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);

    const PtrList<volScalarField>& Yvf
        = this->thermo().Y();

    scalarField c(this->nSpecie());

    List<label> cTos; //Note! not allocated!


    forAll(rhovf, celli)
    {
        const scalar rho = rhovf[celli];
        const scalar T = Tvf[celli];
        const scalar p = pvf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            const scalar Yi = Yvf[i][celli];
            c[i] = rho*Yi/specieThermos_[i].W();
        }

        dNdtByV = Zero;

        forAll(reactions_, ri)
        {
            reactions_[ri].dNdtByV
            (
                p,
                T,
                c,
                celli,
                dNdtByV,
                false,
                cTos,
                0
            );
        }

        for (label i=0; i<this->nSpecie(); i++)
        {
            RR_[i][celli] = dNdtByV[i]*specieThermos_[i].W();
        }

    }
}

template<class ThermoType>
Foam::PtrList<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::gpuChemistryModel<ThermoType>::reactionRR
(
    const label reactioni
) const
{
    const PtrList<volScalarField>& Yvf
        = this->thermo().Y();

    PtrList<volScalarField::Internal> RR(this->nSpecie());
    for (label i=0; i<this->nSpecie(); i++)
    {
        RR.set
        (
            i,
            volScalarField::Internal::New
            (
                "RR." + Yvf[i].name(),
                this->mesh(),
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            ).ptr()
        );
    }

    if (!this->chemistry_)
    {
        return RR;
    }

    tmp<volScalarField> trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);


    scalarField c(this->nSpecie());

    List<label> cTos; //Note! not allocated!


    const auto& R = reactions_[reactioni];

    forAll(rhovf, celli)
    {
        const scalar rho = rhovf[celli];
        const scalar T = Tvf[celli];
        const scalar p = pvf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            const scalar Yi = Yvf[i][celli];
            c[i] = rho*Yi/specieThermos_[i].W();
        }

        dNdtByV = Zero;

        R.dNdtByV
        (
            p,
            T,
            c,
            celli,
            dNdtByV,
            false,
            cTos,
            0
        );

        for (label i=0; i<this->nSpecie(); i++)
        {
            RR[i][celli] = dNdtByV[i]*specieThermos_[i].W();
        }
    }

    return RR;
}

template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solve
(
    const scalar deltaT
)
{
    // Don't allow the time-step to change more than a factor of 2
    return min
    (
        //this->doSolve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        this->doSolve(deltaT),
        2*deltaT
    );
}


template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solve
(
    const scalarField& deltaT
)
{
    throw std::logic_error("LTS time step not yet supported");
    //return this->doSolve<scalarField>(deltaT);
    return 0;
}


#ifdef __NVIDIA_COMPILER__

template<class T>
__global__
void kernel(int n, T op)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) op(i);
}


#else

template<class T>
void kernel(int n, T op)
{
    for (int i = 0; i < n; ++i)
        op(i);
}


#endif

template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::doSolve
(
    const scalar& deltaT
)
{

    if (!this->chemistry_)
    {
        return great;
    }


    auto [RR, deltaTChem, minDeltaT] = evaluator_.computeRR
    (
        deltaT,
        deltaTChemMax_,
        device_rho0(),
        device_p0(),
        device_T0(),
        device_deltaTChem(),
        device_Y0()
    );

    auto RRs = make_mdspan(RR, extents<2>{nCells(), nSpecie()});

    for (label celli = 0; celli < nCells(); ++celli)
    {
        for (label i=0; i<this->nSpecie(); ++i)
        {
            RR_[i][celli] = RRs(celli, i);
        }
    }

    for (label celli = 0; celli < nCells(); ++celli)
    {
        deltaTChem_[celli] = deltaTChem[celli];
    }


    return minDeltaT;

    /*

    const auto d_rho0vf = device_rho0();
    const auto d_p0vf = device_p0();
    const auto d_T0vf = device_T0();
    const auto d_Yvf = device_Y0();


    auto d_deltaTChem = device_deltaTChem();
    auto d_RR = device_RR();

    device_vector<scalar> d_deltaTMin(nCells(), 0);


    device_vector<scalar> Js(nCells()*nEqns()*nEqns());

    auto op =
    [
        nSpecie = nSpecie(),
        deltaT = deltaT,
        deltaTChemMax = deltaTChemMax_,
        rho0vf = make_mdspan(d_rho0vf, extents<1>{nCells()}),
        p0vf = make_mdspan(d_p0vf, extents<1>{nCells()}),
        T0vf = make_mdspan(d_T0vf, extents<1>{nCells()}),
        deltaTChem = make_mdspan(d_deltaTChem, extents<1>{nCells()}),
        deltaTMin = make_mdspan(d_deltaTMin, extents<1>{nCells()}),
        Yvf = make_mdspan(d_Yvf, extents<2>{nCells(), nSpecie()}),
        RR = make_mdspan(d_RR, extents<2>{nCells(), nSpecie()}),
        Js = make_mdspan(Js, extents<3>{nCells(), nEqns(), nEqns()}),
        gpuOde = gpuOde_
    ](int celli)
    {

        constexpr int N = gpuConstants::SPECIE_MAX + 2;

        std::array<scalar, N> Y_arr{};
        auto Y = make_mdspan(Y_arr, extents<1>{nSpecie + 2});

        std::array<scalar, N> Y0_arr{};
        auto Y0 = make_mdspan(Y0_arr, extents<1>{nSpecie + 2});

        scalar* ptr = &Js(celli, 0, 0);
        auto J
        = mdspan<scalar, 2>(ptr, extents<2>{nSpecie+2, nSpecie+2});

        const scalar rho0 = rho0vf[celli];
        scalar p = p0vf[celli];
        scalar T = T0vf[celli];

        for (label i=0; i<nSpecie; ++i)
        {
            Y[i] = Y0[i] = Yvf(celli, i); //Yvf[i].oldTime()[celli];
        }

        Y[nSpecie]   = T;
        Y[nSpecie+1] = p;

        // Initialise time progress
        scalar timeLeft = deltaT;

        constexpr label li = 0;

        // Calculate the chemical source terms
        while (timeLeft > gpuConstants::gpuSmall)
        {
            scalar dt = timeLeft;

            gpuOde.solve(0, dt, Y, li, deltaTChem[celli], J);

            for (int i=0; i<nSpecie; i++)
            {
                Y[i] = std::max(0.0, Y[i]);
            }

            timeLeft -= dt;
        }

        //deltaTMin = min(deltaTChem_[celli], deltaTMin);
        deltaTMin[celli] = deltaTChem[celli];
        deltaTChem[celli] = std::min(deltaTChem[celli], deltaTChemMax);

        // Set the RR vector (used in the solver)
        for (label i=0; i<nSpecie; ++i)
        {
            RR(celli,i) = rho0*(Y[i] - Y0[i])/deltaT;
        }

    };

#ifdef __NVIDIA_COMPILER__
    label NTHREADS = 32;
    label NBLOCKS = (nCells() + NTHREADS - 1)/ NTHREADS;
    //kernel<<<(nCells()+255)/256, 256>>>(nCells(), op);
    kernel<<<NBLOCKS, NTHREADS>>>(nCells(), op);
    cudaDeviceSynchronize();
#else
    kernel(nCells(), op);
#endif

    {

        host_vector<scalar> temp = d_RR;

        auto RR = make_mdspan(temp, extents<2>{nCells(), nSpecie()});

        for (label celli = 0; celli < nCells(); ++celli)
        {
            for (label i=0; i<this->nSpecie(); ++i)
            {
                RR_[i][celli] = RR(celli, i);
            }
        }

    }

    {

        host_vector<scalar> temp = d_deltaTChem;

        for (label celli = 0; celli < nCells(); ++celli)
        {
            deltaTChem_[celli] = temp[celli];
        }

    }


    return *std::min_element(d_deltaTMin.begin(), d_deltaTMin.end());
    */



    /*

    const auto d_rho0vf = device_rho0();
    const auto rho0vf = make_mdspan(d_rho0vf, extents<1>{nCells()});

    const auto d_p0vf = device_p0();
    const auto p0vf = make_mdspan(d_p0vf, extents<1>{nCells()});

    const auto d_T0vf = device_T0();
    const auto T0vf = make_mdspan(d_T0vf, extents<1>{nCells()});

    const auto d_Yvf = device_Y0();
    const auto Yvf = make_mdspan(d_Yvf, extents<2>{nCells(), nSpecie()});

    auto d_RR = device_RR();
    auto RR = make_mdspan(d_RR, extents<2>{nCells(), nSpecie()});


    static constexpr int N = gpuConstants::SPECIE_MAX + 2;

    std::array<scalar, N> Y_arr;
    auto Y = make_mdspan(Y_arr, extents<1>{this->nSpecie() + 2});

    std::array<scalar, N> Y0_arr;
    auto Y0 = make_mdspan(Y0_arr, extents<1>{this->nSpecie() + 2});


    // Minimum chemical timestep
    scalar deltaTMin = great;

    for (label celli = 0; celli < nCells(); ++celli)
    {
        const scalar rho0 = rho0vf[celli];
        const scalar p = p0vf[celli];
        const scalar T = T0vf[celli];

        for (label i=0; i<this->nSpecie(); ++i)
        {
            Y[i] = Y0[i] = Yvf(celli, i); //Yvf[i].oldTime()[celli];
        }

        Y[this->nSpecie()]   = T;
        Y[this->nSpecie()+1] = p;

        // Initialise time progress
        scalar timeLeft = deltaT;

        constexpr label li = 0;

        // Calculate the chemical source terms
        while (timeLeft > small)
        {
            scalar dt = timeLeft;

            gpuOde_->solve(0, dt, Y, li, deltaTChem_[celli]);

            for (int i=0; i<this->nSpecie(); i++)
            {
                Y[i] = max(0.0, Y[i]);
            }

            timeLeft -= dt;
        }

        deltaTMin = min(deltaTChem_[celli], deltaTMin);
        deltaTChem_[celli] = min(deltaTChem_[celli], deltaTChemMax_);

        // Set the RR vector (used in the solver)
        for (label i=0; i<this->nSpecie(); ++i)
        {
            RR(celli,i) = rho0*(Y[i] - Y0[i])/deltaT;
        }

    }


    for (label celli = 0; celli < nCells(); ++celli)
    {
        for (label i=0; i<this->nSpecie(); ++i)
        {
            RR_[i][celli] = RR(celli, i);
        }
    }

    return deltaTMin;
    */
}

/*

template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::doSolve
(
    const scalar& deltaT
)
{

    if (!this->chemistry_)
    {
        return great;
    }


    const volScalarField& rho0vf =
        this->mesh().template lookupObject<volScalarField>
        (
            this->thermo().phasePropertyName("rho")
        ).oldTime();

    const volScalarField& T0vf = this->thermo().T().oldTime();
    const volScalarField& p0vf = this->thermo().p().oldTime();


    auto d_p0vf = device_p0();
    auto d_rho0vf = device_rho0();
    auto d_T0vf = device_T0();
    auto d_Yvf = device_Y0();



    // Minimum chemical timestep
    scalar deltaTMin = great;

    const PtrList<volScalarField>& Yvf
        = this->thermo().composition().Y();

    static constexpr int N = gpuConstants::SPECIE_MAX + 2;

    std::array<scalar, N> Y_arr;
    auto Y = make_mdspan(Y_arr, extents<1>{this->nSpecie() + 2});

    std::array<scalar, N> Y0_arr;
    auto Y0 = make_mdspan(Y0_arr, extents<1>{this->nSpecie() + 2});


    forAll(rho0vf, celli)
    {
        const scalar rho0 = rho0vf[celli];

        scalar p = p0vf[celli];
        scalar T = T0vf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            Y[i] = Y0[i] = Yvf[i].oldTime()[celli];
        }

        Y[this->nSpecie()]   = T;
        Y[this->nSpecie()+1] = p;

        // Initialise time progress
        scalar timeLeft = deltaT;

        constexpr label li = 0;

        // Calculate the chemical source terms
        while (timeLeft > small)
        {
            scalar dt = timeLeft;

            gpuOde_->solve(0, dt, Y, li, deltaTChem_[celli]);

            for (int i=0; i<this->nSpecie(); i++)
            {
                Y[i] = max(0.0, Y[i]);
            }

            timeLeft -= dt;
        }

        deltaTMin = min(deltaTChem_[celli], deltaTMin);
        deltaTChem_[celli] = min(deltaTChem_[celli], deltaTChemMax_);

        // Set the RR vector (used in the solver)
        for (label i=0; i<this->nSpecie(); i++)
        {
            RR_[i][celli] = rho0*(Y[i] - Y0[i])/deltaT;
        }

    }



    return deltaTMin;
}


*/






template<class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::gpuChemistryModel<ThermoType>::tc() const
{
    tmp<volScalarField> ttc
    (
        volScalarField::New
        (
            "tc",
            this->mesh(),
            dimensionedScalar(dimTime, small),
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
    );
    scalarField& tc = ttc.ref();

    if (!this->chemistry_)
    {
        ttc.ref().correctBoundaryConditions();
        return ttc;
    }

    tmp<volScalarField> trhovf(this->thermo().rho());
    const volScalarField& rhovf = trhovf();

    const volScalarField& Tvf = this->thermo().T();
    const volScalarField& pvf = this->thermo().p();

    scalarField dNdtByV(this->nSpecie() + 2);

    const PtrList<volScalarField>& Yvf
        = this->thermo().Y();

    scalarField c(this->nSpecie());

    List<label> cTos; //Note! not allocated!

    forAll(rhovf, celli)
    {
        const scalar rho = rhovf[celli];
        const scalar T = Tvf[celli];
        const scalar p = pvf[celli];

        for (label i=0; i<this->nSpecie(); i++)
        {
            c[i] = rho*Yvf[i][celli]/specieThermos_[i].W();
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
        forAll(reactions_, i)
        {
            const Reaction<ThermoType>& R = reactions_[i];
            scalar omegaf, omegar;
            R.omega(p, T, c, celli, omegaf, omegar);

            scalar wf = 0;
            forAll(R.rhs(), s)
            {
                wf += R.rhs()[s].stoichCoeff*omegaf;
            }
            sumW += wf;
            sumWRateByCTot += sqr(wf);

            scalar wr = 0;
            forAll(R.lhs(), s)
            {
                wr += R.lhs()[s].stoichCoeff*omegar;
            }
            sumW += wr;
            sumWRateByCTot += sqr(wr);
        }

        tc[celli] =
            sumWRateByCTot == 0 ? vGreat : sumW/sumWRateByCTot*sum(c);
    }

    ttc.ref().correctBoundaryConditions();
    return ttc;
}


template<class ThermoType>
Foam::tmp<Foam::volScalarField>
Foam::gpuChemistryModel<ThermoType>::Qdot() const
{
    tmp<volScalarField> tQdot
    (
        volScalarField::New
        (
            "Qdot",
            this->mesh_,
            dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
        )
    );

    if (!this->chemistry_)
    {
        return tQdot;
    }

    const PtrList<volScalarField>& Yvf
        = this->thermo().Y();

    scalarField& Qdot = tQdot.ref();

    forAll(Yvf, i)
    {
        forAll(Qdot, celli)
        {
            const scalar hi = specieThermos_[i].Hf();
            Qdot[celli] -= hi*RR_[i][celli];
        }
    }

    return tQdot;
}




// ************************************************************************* //
