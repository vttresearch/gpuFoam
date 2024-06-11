#pragma once

#include "results.H"
#include "mechanisms.H"
#include "gpuThermo.H"

namespace TestData{


double kernel_test();

constantResults constant_results_gpu();



perfectGasResult perfect_gas_results_gpu
(
    gScalar p, gScalar T, gScalar Y,
    gScalar molWeight
);

thermoResults thermo_results_gpu(Mechanism mech);
reactionResults reaction_results_gpu(Mechanism mech);

std::tuple<std::vector<gScalar>, std::vector<gLabel>, std::vector<gScalar>>
lu_results_gpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals);


odeSystemResults odesystem_results_gpu(Mechanism mech);

std::vector<gScalar> ode_results_gpu(Mechanism mech, std::string solver_name, gScalar xStart, gScalar xEnd, gScalar dxTry);

bool test_for_each_index();
bool test_evaluator(gLabel nCells);


}