#pragma once

#include "gpuRosenbrock12.H"
#include "gpuRosenbrock23.H"
#include "gpuRosenbrock34.H"

#include "gpuODESolver.H"


namespace FoamGpu{


static inline gpuODESolver
make_gpuODESolver(const gpuODESystem& system, const gpuODESolverInputs& inputs)
{

    if (inputs.name == "Rosenbrock12"){
        return gpuODESolver(gpuRosenbrock12<gpuODESystem>(system, inputs));
    }
    if (inputs.name == "Rosenbrock23"){
        return gpuODESolver(gpuRosenbrock23<gpuODESystem>(system, inputs));
    }
    if (inputs.name == "Rosenbrock34"){
        return gpuODESolver(gpuRosenbrock34<gpuODESystem>(system, inputs));
    }


    throw std::logic_error("Invalid ODE Solver type");


}


}