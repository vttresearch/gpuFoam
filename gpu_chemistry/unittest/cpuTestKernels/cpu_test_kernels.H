#pragma once

#include "gpuThermo.H"
#include "mechanisms.H"
#include "results.H"

namespace CpuTestKernels {

TestData::constantResults constants();

TestData::perfectGasResult
perfect_gas(gScalar p, gScalar T, gScalar Y, gScalar molWeight);

TestData::thermoResults   thermo(TestData::Mechanism mech);
TestData::reactionResults reaction(TestData::Mechanism mech);

std::tuple<std::vector<gScalar>,
           std::vector<gLabel>,
           std::vector<gScalar>>
lu(const std::vector<gScalar>& m_vals,
   const std::vector<gScalar>& s_vals);

TestData::odeSystemResults odesystem(TestData::Mechanism mech);

std::vector<gScalar> ode_solve(TestData::Mechanism mech,
                               std::string         solver_name,
                               gScalar             xStart,
                               gScalar             xEnd,
                               gScalar             dxTry);

} // namespace CpuTestKernels