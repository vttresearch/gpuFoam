/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Copyright (C) 2022 VTT Technical Research Centre of Finland Ltd
-------------------------------------------------------------------------------
License
    This file is a derivative work of OpenFOAM.

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
    Foam::gpuChemistryModel

Description
    Extends the standard chemistryModel with a load balancing
    functionality.



SourceFiles
    gpuChemistryModelI.H
    gpuChemistryModel.C

\*---------------------------------------------------------------------------*/

#ifndef gpuChemistryModel_H
#define gpuChemistryModel_H

#include <vector>

#include "ODESystem.H"
#include "ReactionList.H"
#include "volFields.H"
#include "multicomponentMixture.H"
#include "chemistryReductionMethod.H"
#include "chemistryTabulationMethod.H"
#include "DynamicField.H"
#include "janafThermo.H"
#include "perfectGas.H"
#include "gpuThermo.H"
#include "gpuReaction.H"
#include "gpuChemTests.H"
#include "gpuODESystem.H"
#include "gpuRosenbrock34.H"
#include "gpuConstants.H"
#include "topaz.hpp"
#include "utilities.H"

#include "gpuKernelEvaluator.H"


template<class T>
using device_vector = std::vector<T>;

template<class T>
using host_vector = std::vector<T>;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


/*---------------------------------------------------------------------------*\
                      Class gpuChemistryModel Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class gpuChemistryModel
:
    public basicChemistryModel
{
    // Private members






        //- Reference to the multi component mixture
        const multicomponentMixture<ThermoType>& mixture_;

        //- Thermodynamic data of the species
        const PtrList<ThermoType>& specieThermos_;

        //- Reactions
        const ReactionList<ThermoType> reactions_;


        //- List of reaction rate per specie [kg/m^3/s]
        PtrList<volScalarField::Internal> RR_;


    //Gpu fields


        std::vector<FoamGpu::gpuThermo> temp_gpuThermos_;
        std::vector<FoamGpu::gpuReaction> temp_gpuReactions_;

        FoamGpu::GpuKernelEvaluator evaluator_;

        template<class T>
        device_vector<T> toDevice(const std::vector<T>& v)
        {
            host_vector<T> temp(v.begin(), v.end());
            return device_vector<T>(temp);
        }


        auto device_rho0() const
        {
            const volScalarField& rho =
                this->mesh().template lookupObject<volScalarField>
                (
                    this->thermo().phasePropertyName("rho")
                ).oldTime();
            host_vector<scalar> temp(rho.begin(), rho.end());

            return device_vector<scalar>(temp);
            //return temp;
        }

        auto device_p0() const
        {
            const volScalarField& p = this->thermo().p().oldTime();
            host_vector<scalar> temp(p.begin(), p.end());
            return device_vector<scalar>(temp);
            //return temp;
        }

        auto device_T0() const
        {
            const volScalarField& T = this->thermo().T().oldTime();
            host_vector<scalar> temp(T.begin(), T.end());
            return device_vector<scalar>(temp);
            //return temp;
        }

        auto device_Y0() const
        {
            const PtrList<volScalarField>& Yvf
                = this->thermo().composition().Y();

            const volScalarField& p = this->thermo().p().oldTime();
            const volScalarField& T = this->thermo().T().oldTime();

            host_vector<scalar> temp(nCells() * nEqns());

            auto s = make_mdspan(temp, extents<2>{nCells(), nEqns()});

            for (label celli = 0; celli < nCells(); ++celli)
            {
                for (label i = 0; i < nSpecie(); ++i)
                {
                    s(celli, i) = Yvf[i].oldTime()[celli];
                }
                s(celli, nSpecie()) = T[celli];
                s(celli, nSpecie() + 1) = p[celli];
            }

            return device_vector<scalar>(temp);
            //return temp;
        }

        auto device_RR() const
        {
            const PtrList<volScalarField>& Yvf
                = this->thermo().composition().Y();

            label nCells = Yvf[0].size();
            label nFields = Yvf.size();
            host_vector<scalar> temp(nCells * nFields);
            return device_vector<scalar>(temp);
            //return temp;

        }

        auto device_deltaTChem() const
        {
            host_vector<scalar> temp(deltaTChem_.begin(), deltaTChem_.end());
            return device_vector<scalar>(temp);
            //return temp;
        }



protected:




    static auto makeGpuReactions
    (
        const speciesTable& species,
        const PtrList<ThermoType> cpuThermos,
        const dictionary& dict,
        const std::vector<FoamGpu::gpuThermo>& gpuThermos,
        const ReactionList<ThermoType>& cpuReactions
    )
    {
        using namespace FoamGpu;

        auto gpuReactions =
        make_reaction_list
        (
            species,
            cpuThermos,
            dict,
            gpuThermos,
            cpuReactions
        );

        device_vector<gpuReaction> temp = gpuReactions;

        testReactions(cpuReactions, gpuReactions);

        return gpuReactions;
    }











        //This is the actual solve call
        //- Solve the reaction system for the given time step
        //  of given type and return the characteristic time
        //  Variable number of species added
        scalar doSolve(const scalar& deltaT);



public:

    //- Runtime type information
    TypeName("gpuChemistryModel");


    // Constructors

        //- Construct from thermo
        gpuChemistryModel(const fluidMulticomponentThermo& thermo) :
        basicChemistryModel(thermo),
        mixture_(refCast<const multicomponentMixture<ThermoType>>(this->thermo())),
        specieThermos_(mixture_.specieThermos()),
        reactions_(mixture_.species(), specieThermos_, this->mesh(), *this),
        RR_(this->thermo().composition().Y().size()),
        temp_gpuThermos_
        (
            FoamGpu::make_gpu_thermos
            (
                specieThermos_,
                this->mesh().lookupObject<dictionary>("physicalProperties")
            )
        ),
        temp_gpuReactions_
        (
            FoamGpu::make_gpu_reactions
            (
                mixture_.species(),
                *this,
                temp_gpuThermos_,
                reactions_
            )
        ),
        evaluator_
        (
            this->nCells(),
            this->nEqns(),
            this->nSpecie(),
            temp_gpuThermos_,
            temp_gpuReactions_,
            FoamGpu::read_Rosenbrock34_inputs(this->subDict("odeCoeffs"))
        )
        {


            const PtrList<volScalarField>& Yvf
                = this->thermo().composition().Y();

            // Create the fields for the chemistry sources
            forAll(RR_, fieldi)
            {
                RR_.set
                (
                    fieldi,
                    new volScalarField::Internal
                    (
                        IOobject
                        (
                            "RR." + Yvf[fieldi].name(),
                            this->mesh().time().timeName(),
                            this->mesh(),
                            IOobject::NO_READ,
                            IOobject::NO_WRITE
                        ),
                        thermo.mesh(),
                        dimensionedScalar(dimMass/dimVolume/dimTime, 0)
                    )
                );
            }

            Info << "============= All OK ===============" << endl;

        }

        //- Disallow default bitwise copy construction
        gpuChemistryModel(const gpuChemistryModel&) = delete;


    //- Destructor
    virtual ~gpuChemistryModel() {}

    // Member Functions

        //- The number of species
        label nSpecie() const override
        {
            return this->thermo().composition().Y().size();
        }

        inline label nEqns() const
        {
            // nEqns = number of species + temperature + pressure
            return this->nSpecie() + 2;
        }

        //- The number of reactions
        label nReaction() const override
        {
            return reactions_.size();
        }

        label nCells() const
        {
            return this->thermo().T().size();
        }

        //- Return reaction rates of the species [kg/m^3/s]
        const PtrList<volScalarField::Internal>& RR() const override
        {
            return RR_;
        }

        //- Return reaction rates of the species in reactioni [kg/m^3/s]
        PtrList<volScalarField::Internal> reactionRR
        (
            const label reactioni
        ) const override;

        //- Calculates the reaction rates
        void calculate() override;

        //- Solve the reaction system for the given time step
        //  and return the characteristic time
        scalar solve(const scalar deltaT) override;

        //- Solve the reaction system for the given time step
        //  and return the characteristic time
        scalar solve(const scalarField& deltaT) override;

        //- Return the chemical time scale
        tmp<volScalarField> tc() const override;

        //- Return the heat release rate [kg/m/s^3]
        tmp<volScalarField> Qdot() const override;






    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const gpuChemistryModel&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//#include "gpuChemistryModelI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "gpuChemistryModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
