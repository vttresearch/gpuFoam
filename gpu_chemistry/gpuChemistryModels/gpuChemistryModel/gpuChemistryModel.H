#pragma once

#include <vector>

#include "DynamicField.H"
#include "ODESystem.H"
#include "ReactionList.H"
#include "chemistryReductionMethod.H"
#include "chemistryTabulationMethod.H"
#include "gpuConstants.H"
#include "gpuODESystem.H"
#include "gpuReaction.H"
#include "gpuRosenbrock34.H"
#include "read_ode_solver_inputs.H"
#include "gpuThermo.H"
#include "janafThermo.H"
#include "multicomponentMixture.H"
#include "perfectGas.H"
#include "utilities.H"
#include "volFields.H"

#include "gpuKernelEvaluator.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam {

template <class ThermoType>
class gpuChemistryModel : public basicChemistryModel {

    //- Reference to the multi component mixture
    const multicomponentMixture<ThermoType>& mixture_;

    //- Thermodynamic data of the species
    const PtrList<ThermoType>& specieThermos_;

    //- Reactions
    const ReactionList<ThermoType> reactions_;

    //- List of reaction rate per specie [kg/m^3/s]
    PtrList<volScalarField::Internal> RR_;

    //- The gpu kernel
    FoamGpu::GpuKernelEvaluator evaluator_;



private:

    /// @brief Get a vector of density values at old time.
    /// @return Vector of density values.
    std::vector<gScalar> getRho0() const;

    /// @brief Get a vector of pressure values at old time.
    /// @return Vector of pressure values.
    std::vector<gScalar> getP0() const;

    /// @brief Get a vector of temperature values at old time.
    /// @return Vector of temperature values.
    std::vector<gScalar> getT0() const;

    /// @brief Get a vector of chemistry time step values at old time.
    /// @return Vector of chemistry time step values.
    std::vector<gScalar> getDeltaTChem() const;

    /// @brief Get a flat vector of size nCells x (nSpecie + 2) corresponding to
    /// specie mass fraction values + temperature + pressure in each cell. The
    /// layout of the vector is [Y0_1, Y0_2 ...], where each Y0_i corresponds to
    /// the solution vector of cell i.
    /// @return Flat vector mass fractions, temperature and pressure in each
    /// cell.
    std::vector<gScalar> getY0() const;

    /// @brief Builds the gpuKernel from the corresponding user inputs and cpu
    /// thermos and reactions.
    /// @param nEqns Number of equations.
    /// @param nSpecie  Number of species.
    /// @param reactions The reactions constructed on cpu memory.
    /// @param physicalProperties  The physicalProperties dictionary.
    /// @param chemistryProperties The chemistryProperties dictionary.
    /// @param mixture The mixture containing cpu thermos and compositions.
    /// @return A gpu kernel evaluator object which can be called to compute the
    /// reaction rate on gpu.
    static FoamGpu::GpuKernelEvaluator
    makeEvaluator(label                                    nEqns,
                   label                                    nSpecie,
                   const ReactionList<ThermoType>&          reactions,
                   const dictionary&                        physicalProperties,
                   const dictionary&                        chemistryProperties,
                   const multicomponentMixture<ThermoType>& mixture);

    /// @brief Computes and updates the reaction rates (RR_) as well as
    /// chemistry time steps (chemDeltaT_).
    /// @param deltaT The current flow time step.
    /// @return The minimum of the computed chemistry time steps.
    scalar computeRRAndChemDeltaT(const scalar& deltaT);

public:
    //- Runtime type information
    TypeName("gpuChemistryModel");

    // Constructors

    /// @brief Construct a gpuChemistryModel based on the multicomponent thermo.
    /// @param thermo a fully constructed multicomponent thermo.
    gpuChemistryModel(const fluidMulticomponentThermo& thermo);

    gpuChemistryModel(const gpuChemistryModel&) = delete;
    virtual ~gpuChemistryModel() {}

    /// @brief Get the number of equations in the chemistry system (nSpecie +
    /// 2).
    /// @return Number of equations in the chemistry system.
    label nEqns() const;

    /// @brief Get the current cell count.
    /// @return Current cell count.
    label nCells() const;

    /// @brief Get the number of species in the chemistry system.
    /// @return Number of species in the chemistry system.
    label nSpecie() const override;

    /// @brief Get the number of reactions in the chemistry system.
    /// @return Number of reactions.
    label nReaction() const override;

    /// @brief Get the current reaction rates of the species [kg/m^3/s].
    /// @return The current reaction rates.
    const PtrList<volScalarField::Internal>& RR() const override;

    /// @brief Get the reaction rates of the species involved in the reaction
    /// corresponding to the input reaction index reactioni.
    /// @param reactioni The reaction index to query the specie reaction rates
    /// for.
    /// @return Reaction rates of the species in reactioni [kg/m^3/s]
    PtrList<volScalarField::Internal>
    reactionRR(const label reactioni) const override;

    /// @brief Calculates and updates the reaction rates (RR_) based on the
    /// current values of Y. Note that his runs on the cpu since no ode solve
    /// calls are required.
    void calculate() override;


    /// @brief Solve the reaction system for the input scalar (flow) time step.
    /// @param deltaT The current flow time step.
    /// @return The minimum chemistry time step.
    scalar solve(const scalar deltaT) override;

    /// @brief Solve the reaction system for the input scalarField (flow) time steps.
    /// @param deltaT The current cell-specific flow time steps.
    /// @return The minimum chemistry time step.
    scalar solve(const scalarField& deltaT) override;


    /// @brief Get the current cell-specific chemical time scales.
    /// @return Field of cell-specific chemical time scales.
    tmp<volScalarField> tc() const override;


    /// @brief Return the current heat release rates corresponding to the reaction rates (RR_).
    /// @return The heat release rate [kg/m/s^3].
    tmp<volScalarField> Qdot() const override;

    // Member Operators

    //- Disallow default bitwise assignment
    void operator=(const gpuChemistryModel&) = delete;
};


} // End namespace Foam

#ifdef NoRepository
#include "gpuChemistryModel.C"
#endif
