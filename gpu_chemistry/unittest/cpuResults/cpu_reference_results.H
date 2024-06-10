#pragma once

#include "mechanisms.H"
#include "results.H"

#include <utility>
#include <vector>
#include <string>



namespace TestData{

constantResults constant_results_cpu();



perfectGasResult perfect_gas_results_cpu
(
    gScalar p, gScalar T, gScalar Y,
    gScalar molWeight
);

thermoResults thermo_results_cpu(Mechanism mech);
reactionResults reaction_results_cpu(Mechanism mech);

std::tuple<std::vector<gScalar>, std::vector<gLabel>, std::vector<gScalar>>
lu_results_cpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals);


odeSystemResults odesystem_results_cpu(Mechanism mech);


std::vector<gScalar> ode_results_cpu(Mechanism mech, std::string solver_name, gScalar xStart, gScalar xEnd, gScalar dxTry);


}

