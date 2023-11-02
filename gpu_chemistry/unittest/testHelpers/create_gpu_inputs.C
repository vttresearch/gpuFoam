#include "create_gpu_inputs.H"

#include "create_foam_inputs.H"
#include "makeGpuThermo.H"
#include "makeGpuReactions.H"

namespace TestData {

std::vector<FoamGpu::gpuThermo> makeGpuThermos(Mechanism m) {

    auto thermoDict = makeThermoDict(m);
    auto cpuThermos = makeCpuThermos(m);

    std::vector<FoamGpu::gpuThermo> ret;
    for (int i = 0; i < cpuThermos.size(); ++i) {
        auto subDict = thermoDict.subDict(cpuThermos[i].name());

        auto gpuThermo =
            FoamGpu::makeGpuThermo(cpuThermos[i], subDict);
        ret.push_back(gpuThermo);
    }

    return ret;
}

std::vector<FoamGpu::gpuReaction> makeGpuReactions(Mechanism m) {

    auto                   thermos      = makeCpuThermos(m);
    auto                   thermoDict   = makeThermoDict(m);
    auto                   reactionDict = makeReactionDict(m);
    Foam::List<Foam::word> s_list = thermoDict.lookup("species");
    Foam::speciesTable     species(s_list);

    auto gpu_thermos   = makeGpuThermos(m);
    auto cpu_reactions = makeCpuReactions(m);
    auto ret           = FoamGpu::makeGpuReactions(
        species, reactionDict, gpu_thermos, cpu_reactions);

    return ret;
}



} // namespace TestData