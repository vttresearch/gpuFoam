#pragma once

#include <vector>
#include <tuple>

#include "gpuPerfectGas.H"
#include "gpuThermo.H"

#include "gpuReactionRate.H"

#include "gpuReaction.H"
#include "gpuODESystem.H"
#include "gpuODESolver.H"
#include "gpuRosenbrock34.H"



namespace FoamGpu
{

struct GpuKernelEvaluator
{


    GpuKernelEvaluator
    (
        gLabel nEqns,
        gLabel nSpecie,
        const std::vector<gpuThermo>& thermos,
        const std::vector<gpuReaction> reactions,
        gpuRosenbrock34Inputs odeInputs
    )
    :
    nEqns_(nEqns),
    nSpecie_(nSpecie),
    h_thermos_(thermos),
    h_reactions_(reactions),
    odeInputs_(odeInputs)
    {

        std::cout << "Neqns " << nEqns_ << std::endl;
        std::cout << "Nspecie " << nSpecie_ << std::endl;
    }


    std::pair<std::vector<gScalar>, std::vector<gScalar>> computeYNew
    (
        gScalar deltaT,
        gScalar deltaTChemMax,
        const std::vector<gScalar>& deltaTChem,
        const std::vector<gScalar>& Y
    );


    std::tuple<std::vector<gScalar>, std::vector<gScalar>, gScalar>
    computeRR
    (
        gScalar deltaT,
        gScalar deltaTChemMax,
        const std::vector<gScalar> rho,
        const std::vector<gScalar> p,
        const std::vector<gScalar> T,
        const std::vector<gScalar> deltaTChem,
        const std::vector<gScalar> Yvf
    );







private:

    gLabel nEqns_, nSpecie_;

    std::vector<gpuThermo> h_thermos_;
    std::vector<gpuReaction> h_reactions_;
    gpuRosenbrock34Inputs odeInputs_;

};



}