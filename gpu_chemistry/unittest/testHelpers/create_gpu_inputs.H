#pragma once


#include "mechanisms.H"

#include "gpuPerfectGas.H"
#include "gpuReaction.H"
#include "gpuThermo.H"
#include "gpuODESolverInputs.H"


namespace TestData {

std::vector<FoamGpu::gpuThermo>   makeGpuThermos_h(Mechanism m);
std::vector<FoamGpu::gpuThermo>   makeGpuThermos_e(Mechanism m);
std::vector<FoamGpu::gpuReaction> makeGpuReactions(Mechanism m);
FoamGpu::gpuODESolverInputs makeGpuODEInputs(std::string odeName, Mechanism m);


} // namespace TestData