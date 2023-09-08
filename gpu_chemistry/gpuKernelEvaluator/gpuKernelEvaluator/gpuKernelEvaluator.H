#pragma once

#include <tuple>
#include <vector>

#include "gpuPerfectGas.H"
#include "gpuThermo.H"

#include "gpuReactionRate.H"

#include "gpuODESolver.H"
#include "gpuODESystem.H"
#include "gpuReaction.H"
#include "gpuRosenbrock34.H"

namespace FoamGpu {

struct GpuKernelEvaluator {

private:
    /// @brief Given an input solution ode vectors Y and time steps, compute the
    /// new solution vectors and time step field by calling the an ode.
    /// @param deltaT Current flow time step.
    /// @param deltaTChemMax Maximum chemistry time step..
    /// @param deltaTChem Current cell-specific chemistry time steps.
    /// @param Y The current cell-specific solution vectors [Y, T, p].
    /// @return New solution vectors and chemistry time steps.
    std::pair<std::vector<gScalar>, std::vector<gScalar>>
    computeYNew(gScalar                     deltaT,
                gScalar                     deltaTChemMax,
                const std::vector<gScalar>& deltaTChem,
                const std::vector<gScalar>& Y);

public:
    GpuKernelEvaluator(gLabel                         nEqns,
                       gLabel                         nSpecie,
                       const std::vector<gpuThermo>&  thermos,
                       const std::vector<gpuReaction> reactions,
                       gpuRosenbrock34Inputs          odeInputs)
        : nEqns_(nEqns)
        , nSpecie_(nSpecie)
        , hThermos_(thermos)
        , hReactions_(reactions)
        , odeInputs_(odeInputs) {}

    /// @brief Computes the reaction rate based on the input flow properties.
    /// @param deltaT Current flow time step.
    /// @param deltaTChemMax Maximum chemistry time step.
    /// @param rho The density field.
    /// @param deltaTChem Current cell-specific chemistry time steps.
    /// @param Yvf The current cell-specific solution vectors [Y, T, p].
    /// @return Vector of new reaction rates of size nCells x nSpecie, new
    /// chemistry time steps and the minimum new chemistry time step.
    std::tuple<std::vector<gScalar>, std::vector<gScalar>, gScalar>
    computeRR(gScalar                    deltaT,
              gScalar                    deltaTChemMax,
              const std::vector<gScalar> rho,
              const std::vector<gScalar> deltaTChem,
              const std::vector<gScalar> Yvf);

private:
    gLabel nEqns_, nSpecie_;

    std::vector<gpuThermo>   hThermos_;
    std::vector<gpuReaction> hReactions_;
    gpuRosenbrock34Inputs    odeInputs_;
};

} // namespace FoamGpu