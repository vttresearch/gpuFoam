#pragma once

#include <tuple>
#include <vector>

#include "gpuPerfectGas.H"
#include "gpuThermo.H"

#include "gpuReactionRate.H"
#include "host_device_vectors.H"
#include "gpuODESolverInputs.H"
#include "makeGpuOdeSolver.H"
#include "gpuODESystem.H"
#include "gpuReaction.H"
#include "error_handling.H"
#include "thermosAndReactions.H"

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
                const std::vector<gScalar>& Y) const;


public:

    GpuKernelEvaluator() = default;

    GpuKernelEvaluator(gLabel                         nEqns,
                       gLabel                         nSpecie,
                       const std::vector<gpuThermo>&  thermos,
                       const std::vector<gpuReaction>& reactions,
                       gpuODESolverInputs          odeInputs);


    GpuKernelEvaluator(const GpuKernelEvaluator& other) = delete;

    ~GpuKernelEvaluator() = default;
    

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
              const std::vector<gScalar> Yvf) const;

private:
    gLabel nEqns_, nSpecie_, nReactions_;
    thermosAndReactions     thermosReactions_;
    gpuODESystem             system_;
    gpuODESolver             solver_;


    void allocate();
    void deallocate();


};

} // namespace FoamGpu