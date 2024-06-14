#include "create_gpu_inputs.H"

#include "create_foam_inputs.H"
#include "makeGpuReactions.H"
#include "makeGpuThermo.H"

namespace TestData {

std::vector<FoamGpu::gpuThermo> makeGpuThermos_h(Mechanism m) {

    auto thermoDict = makeThermoDict(m);
    auto cpuThermos = makeCpuThermos_h(m);

    std::vector<FoamGpu::gpuThermo> ret;
    for (int i = 0; i < cpuThermos.size(); ++i) {
        auto subDict = thermoDict.subDict(cpuThermos[i].name());

        auto gpuThermo =
            FoamGpu::makeGpuThermo(cpuThermos[i], subDict);
        ret.push_back(gpuThermo);
    }

    return ret;
}

std::vector<FoamGpu::gpuThermo> makeGpuThermos_e(Mechanism m) {

    auto thermoDict = makeThermoDict(m);
    auto cpuThermos = makeCpuThermos_e(m);

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

    auto                   thermos      = makeCpuThermos_h(m);
    auto                   thermoDict   = makeThermoDict(m);
    auto                   reactionDict = makeReactionDict(m);
    Foam::List<Foam::word> s_list = thermoDict.lookup("species");
    Foam::speciesTable     species(s_list);

    auto gpu_thermos   = makeGpuThermos_h(m);
    auto cpu_reactions = makeCpuReactions(m);
    auto ret           = FoamGpu::makeGpuReactions(
        species, reactionDict, gpu_thermos, cpu_reactions);

    return ret;
}

FoamGpu::gpuODESolverInputs makeGpuODEInputs(std::string odeName,
                                             Mechanism   m) {
    (void)m;
    FoamGpu::gpuODESolverInputs ret;

    ret.name = odeName;

    // These are same for gri and h2 tutorials
    ret.absTol = 1E-12;
    ret.relTol = 1E-1;

    ret.maxSteps  = 10000;
    ret.safeScale = 0.9;
    ret.alphaInc  = 0.2;
    ret.alphaDec  = 0.25;
    ret.minScale  = 0.2;
    ret.maxScale  = 10;

    return ret;
}

} // namespace TestData