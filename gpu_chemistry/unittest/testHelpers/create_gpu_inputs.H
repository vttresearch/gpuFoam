#pragma once

#include "IOmanip.H"

#include "mechanisms.H"

#include "gpuPerfectGas.H"
#include "gpuReaction.H"
#include "gpuThermo.H"

namespace TestData {

std::vector<FoamGpu::gpuThermo>   makeGpuThermos(Mechanism m);
std::vector<FoamGpu::gpuReaction> makeGpuReactions(Mechanism m);

} // namespace TestData