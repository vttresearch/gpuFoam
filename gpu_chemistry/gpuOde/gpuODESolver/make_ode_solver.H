#pragma once

#include "gpuRosenbrock23.H"
#include "gpuRosenbrock34.H"

#include "gpuODESolver.H"


namespace FoamGpu{


static inline gpuODESolver
make_gpuODESolver(const gpuODESystem& ode, const gpuODESolverInputs& inputs)
{
    

    if (inputs.name == "Rosenbrock23"){
        return gpuODESolver(gpuRosenbrock23(ode, inputs));
    }
    if (inputs.name == "Rosenbrock34"){
        return gpuODESolver(gpuRosenbrock34(ode, inputs));
    }
    
    throw std::logic_error("Invalid ODE Solver type");


}


}